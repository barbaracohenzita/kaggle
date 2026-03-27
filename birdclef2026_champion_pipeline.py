#!/usr/bin/env python3
"""
BirdCLEF 2026 end-to-end training + inference pipeline.

Goal: provide a stronger baseline with the main ingredients used in high-ranking
solutions: robust CV, multi-backbone training, augmentations, EMA, weighted
ensembling, and test-time augmentation (TTA).

Usage (Kaggle):
  python birdclef2026_champion_pipeline.py --mode train
  python birdclef2026_champion_pipeline.py --mode infer
  python birdclef2026_champion_pipeline.py --mode train_infer
"""

import argparse
import gc
import glob
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class CFG:
    seed: int = 42
    sr: int = 32_000
    dur: int = 5
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    f_min: int = 20
    f_max: int = 16_000

    folds: int = 5
    folds_to_train: tuple = (0, 1, 2, 3, 4)
    backbones: tuple = ("tf_efficientnet_b0_ns", "mobilenetv3_large_100")

    epochs: int = 20
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.02

    use_mixup: bool = True
    mixup_alpha: float = 0.25
    use_specaugment: bool = True
    tta_passes: int = 3

    min_rating: float = 1.5
    grad_clip: float = 5.0
    ema_decay: float = 0.999

    models_dir: str = "/kaggle/working/models_champion"


CFG = CFG()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUDIO_LEN = CFG.sr * CFG.dur


# ---------- Data discovery ----------
def find_comp_dir() -> str:
    candidates = [
        "/kaggle/input/birdclef-2026",
        "/kaggle/input/competitions/birdclef-2026",
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "sample_submission.csv")):
            return p
    for root, _, files in os.walk("/kaggle/input"):
        if "sample_submission.csv" in files and ("train.csv" in files or "test_soundscapes" in root):
            return root
    raise FileNotFoundError("Competition data not found under /kaggle/input")


# ---------- Metrics ----------
def padded_cmap(y_true: np.ndarray, y_pred: np.ndarray, pad: int = 5) -> float:
    aps = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        yt = np.concatenate([y_true[:, c], np.zeros(pad, dtype=np.float32)])
        yp = np.concatenate([y_pred[:, c], np.zeros(pad, dtype=np.float32)])
        aps.append(average_precision_score(yt, yp))
    return float(np.mean(aps)) if aps else 0.0


# ---------- Model ----------
class BirdModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.head = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


class FocalBCEWithLogits(nn.Module):
    def __init__(self, gamma: float = 1.5, smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = targets * p + (1.0 - targets) * (1.0 - p)
        focal = (1.0 - pt).pow(self.gamma) * bce
        return focal.mean()


# ---------- Dataset ----------
class BirdDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str,
        label_col: str,
        num_classes: int,
        is_train: bool,
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.label_col = label_col
        self.num_classes = num_classes
        self.is_train = is_train

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=CFG.sr,
            n_fft=CFG.n_fft,
            hop_length=CFG.hop_length,
            n_mels=CFG.n_mels,
            f_min=CFG.f_min,
            f_max=CFG.f_max,
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def _read_wave(self, fp: str) -> torch.Tensor:
        info = torchaudio.info(fp)
        req = int(AUDIO_LEN * info.sample_rate / CFG.sr)
        if info.num_frames > req:
            if self.is_train:
                offset = random.randint(0, info.num_frames - req)
            else:
                offset = (info.num_frames - req) // 2
            wave, sr = torchaudio.load(fp, frame_offset=offset, num_frames=req)
        else:
            wave, sr = torchaudio.load(fp)

        if sr != CFG.sr:
            wave = torchaudio.functional.resample(wave, sr, CFG.sr)
        if wave.shape[0] > 1:
            wave = wave.mean(0, keepdim=True)

        if wave.shape[1] < AUDIO_LEN:
            wave = F.pad(wave, (0, AUDIO_LEN - wave.shape[1]))
        else:
            wave = wave[:, :AUDIO_LEN]
        return wave

    def _specaugment(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.is_train or not CFG.use_specaugment:
            return spec
        t = spec.shape[-1]
        f = spec.shape[-2]
        time_mask = random.randint(max(1, t // 20), max(2, t // 8))
        freq_mask = random.randint(max(1, f // 20), max(2, f // 8))

        ts = random.randint(0, max(0, t - time_mask))
        fs = random.randint(0, max(0, f - freq_mask))

        spec[:, fs : fs + freq_mask, :] = 0.0
        spec[:, :, ts : ts + time_mask] = 0.0
        return spec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = os.path.join(self.data_dir, "train_audio", row["filename"])

        try:
            wave = self._read_wave(fp)
        except Exception:
            wave = torch.zeros((1, AUDIO_LEN), dtype=torch.float32)

        spec = self.db(self.mel(wave))
        spec = self._specaugment(spec)
        spec = spec.expand(3, -1, -1)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        y = torch.zeros(self.num_classes, dtype=torch.float32)
        y[int(row[self.label_col])] = 1.0
        return spec, y


# ---------- Train utils ----------
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    y_mix = lam * y + (1.0 - lam) * y[idx]
    return x_mix, y_mix


def evaluate(model: nn.Module, val_loader: DataLoader) -> float:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE, non_blocking=True)
            logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy()
            preds.append(p)
            trues.append(y.numpy())
    return padded_cmap(np.vstack(trues), np.vstack(preds))


def train_one_fold(df, fold, data_dir, num_classes, backbone, model_dir):
    train_df = df[df.fold != fold].reset_index(drop=True)
    valid_df = df[df.fold == fold].reset_index(drop=True)

    tr_ds = BirdDataset(train_df, data_dir, "label_idx", num_classes, is_train=True)
    va_ds = BirdDataset(valid_df, data_dir, "label_idx", num_classes, is_train=False)

    tr_dl = DataLoader(
        tr_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    va_dl = DataLoader(
        va_ds,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    model = BirdModel(backbone, num_classes).to(DEVICE)
    ema_model = AveragedModel(model, avg_fn=lambda avg, cur, n: CFG.ema_decay * avg + (1 - CFG.ema_decay) * cur)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CFG.lr,
        steps_per_epoch=len(tr_dl),
        epochs=CFG.epochs,
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )
    criterion = FocalBCEWithLogits(gamma=1.5, smoothing=CFG.label_smoothing)

    best = -1.0
    ckpt_path = os.path.join(model_dir, f"best_{backbone}_fold{fold}.pth")

    for epoch in range(CFG.epochs):
        model.train()
        run_loss = 0.0
        pbar = tqdm(tr_dl, desc=f"fold={fold} {backbone} ep={epoch+1}/{CFG.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            if CFG.use_mixup:
                x, y = mixup_batch(x, y, CFG.mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()
            scheduler.step()
            ema_model.update_parameters(model)

            run_loss += loss.item()
            pbar.set_postfix(loss=f"{run_loss / max(1, pbar.n):.4f}")

        score = evaluate(ema_model.module, va_dl)
        print(f"fold={fold} backbone={backbone} epoch={epoch+1} loss={run_loss/len(tr_dl):.4f} cMAP={score:.5f}")

        if score > best:
            best = score
            torch.save(
                {
                    "state_dict": ema_model.module.state_dict(),
                    "backbone": backbone,
                    "fold": fold,
                    "val_cmap": best,
                    "num_classes": num_classes,
                },
                ckpt_path,
            )
            print(f"  saved -> {ckpt_path} (cMAP={best:.5f})")

    del model, ema_model, optimizer, scheduler, tr_dl, va_dl, tr_ds, va_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best


# ---------- Inference ----------
def wav_to_spec_tensor(chunk_wav: np.ndarray, mel_tf, db_tf) -> torch.Tensor:
    t = torch.tensor(chunk_wav, device=DEVICE, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    spec = db_tf(mel_tf(t)).expand(1, 3, -1, -1)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec


def infer_submission(data_dir: str, species: list[str], model_dir: str):
    weight_paths = sorted(glob.glob(f"{model_dir}/best_*.pth"))
    if not weight_paths:
        raise FileNotFoundError(f"No weights found in {model_dir}")

    ensemble = []
    for path in weight_paths:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        m = BirdModel(ckpt["backbone"], len(species)).to(DEVICE)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        w = float(ckpt.get("val_cmap", 1.0))
        ensemble.append((m, max(w, 1e-6)))
        print(f"Loaded {os.path.basename(path)} weight={w:.5f}")
    total_w = sum(w for _, w in ensemble)

    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=CFG.sr,
        n_fft=CFG.n_fft,
        hop_length=CFG.hop_length,
        n_mels=CFG.n_mels,
        f_min=CFG.f_min,
        f_max=CFG.f_max,
    ).to(DEVICE)
    db_tf = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

    test_files = sorted(glob.glob(f"{data_dir}/test_soundscapes/*.ogg"))
    if not test_files:
        sub = pd.read_csv(f"{data_dir}/sample_submission.csv")
        sub.iloc[:, 1:] = 0.0
        sub.to_csv("submission.csv", index=False)
        print("No test files -> dummy submission.csv")
        return

    rows = []
    with torch.inference_mode():
        for fp in tqdm(test_files, desc="inference"):
            prefix = os.path.basename(fp).replace(".ogg", "")
            wav, sr = torchaudio.load(fp)
            if sr != CFG.sr:
                wav = torchaudio.functional.resample(wav, sr, CFG.sr)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            wav = wav[0].cpu().numpy()
            n_bins = max(1, int(np.ceil(len(wav) / AUDIO_LEN)))

            for i in range(n_bins):
                start = i * AUDIO_LEN
                chunk = wav[start : start + AUDIO_LEN]
                if len(chunk) < AUDIO_LEN:
                    chunk = np.pad(chunk, (0, AUDIO_LEN - len(chunk)))

                pred = np.zeros(len(species), dtype=np.float32)
                for _ in range(CFG.tta_passes):
                    spec = wav_to_spec_tensor(chunk, mel_tf, db_tf)
                    if random.random() < 0.5:
                        spec = torch.flip(spec, dims=[-1])
                    for model, w in ensemble:
                        p = torch.sigmoid(model(spec)).cpu().numpy()[0].astype(np.float32)
                        pred += (w / total_w) * p / CFG.tta_passes

                row_id = f"{prefix}_{(i + 1) * 5}"
                rows.append([row_id] + pred.tolist())

    sub = pd.DataFrame(rows, columns=["row_id"] + species)
    sub.to_csv("submission.csv", index=False)
    print(f"submission.csv generated with shape={sub.shape}")


# ---------- Main ----------
def main(mode: str):
    seed_everything(CFG.seed)
    data_dir = find_comp_dir()
    os.makedirs(CFG.models_dir, exist_ok=True)

    sub_df = pd.read_csv(f"{data_dir}/sample_submission.csv")
    species = list(sub_df.columns)[1:]
    label2idx = {str(s): i for i, s in enumerate(species)}
    num_classes = len(species)

    print(f"data_dir={data_dir}")
    print(f"device={DEVICE}")
    print(f"num_classes={num_classes}")
    print(f"models_dir={CFG.models_dir}")

    if mode in ("train", "train_infer"):
        train_df = pd.read_csv(f"{data_dir}/train.csv")
        train_df["primary_label"] = train_df["primary_label"].astype(str)
        train_df["label_idx"] = train_df["primary_label"].map(label2idx)
        train_df = train_df.dropna(subset=["label_idx"]).copy()
        train_df["label_idx"] = train_df["label_idx"].astype(int)
        if "rating" in train_df.columns:
            train_df = train_df[train_df["rating"].fillna(0.0) >= CFG.min_rating].reset_index(drop=True)

        skf = StratifiedKFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
        train_df["fold"] = -1
        for f, (_, vi) in enumerate(skf.split(train_df, train_df["label_idx"])):
            train_df.loc[vi, "fold"] = f

        all_scores = []
        for backbone in CFG.backbones:
            for fold in CFG.folds_to_train:
                score = train_one_fold(train_df, fold, data_dir, num_classes, backbone, CFG.models_dir)
                all_scores.append((backbone, fold, score))

        print("=== CV summary ===")
        for bb, fd, sc in all_scores:
            print(f"{bb} fold{fd}: {sc:.5f}")
        print(f"mean cMAP = {np.mean([x[2] for x in all_scores]):.5f}")

    if mode in ("infer", "train_infer"):
        infer_submission(data_dir, species, CFG.models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="train_infer",
        choices=["train", "infer", "train_infer"],
        help="train only, infer only, or both",
    )
    args = parser.parse_args()
    main(args.mode)
