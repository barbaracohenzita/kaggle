import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Codex API",
    description="API HTTP para executar prompts de código com os modelos OpenAI.",
    version="1.0.0",
)


class CodexRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt enviado ao modelo")
    model: str = Field(default="gpt-5", description="Modelo OpenAI")
    instructions: Optional[str] = Field(
        default="Você é um assistente especialista em programação.",
        description="Instruções de sistema/developer",
    )
    max_output_tokens: int = Field(
        default=800,
        ge=64,
        le=4000,
        description="Limite de tokens de saída",
    )


class CodexResponse(BaseModel):
    model: str
    output_text: str
    usage: Optional[Dict[str, Any]] = None
    created_at: str


class BatchCodexRequest(BaseModel):
    items: List[CodexRequest] = Field(..., min_length=1, max_length=32)


class BatchCodexResponse(BaseModel):
    total: int
    results: List[CodexResponse]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Defina a variável de ambiente OPENAI_API_KEY antes de usar a API.",
        )
    return OpenAI(api_key=api_key)


def run_codex(request: CodexRequest) -> CodexResponse:
    client = get_client()

    response = client.responses.create(
        model=request.model,
        instructions=request.instructions,
        input=request.prompt,
        max_output_tokens=request.max_output_tokens,
    )

    usage: Optional[Dict[str, Any]] = None
    if getattr(response, "usage", None) is not None:
        usage = response.usage.model_dump()

    return CodexResponse(
        model=response.model,
        output_text=response.output_text,
        usage=usage,
        created_at=utc_now_iso(),
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "codex-api",
        "version": "1.0.0",
        "docs": "/docs",
        "routes": ["/health", "/v1/codex/run", "/v1/codex/run-batch"],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "codex-api",
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "timestamp": utc_now_iso(),
    }


@app.post("/v1/codex/run", response_model=CodexResponse)
def codex_run(request: CodexRequest) -> CodexResponse:
    try:
        return run_codex(request)
    except HTTPException:
        raise
    except Exception as error:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Falha na chamada OpenAI: {error}") from error


@app.post("/v1/codex/run-batch", response_model=BatchCodexResponse)
def codex_run_batch(request: BatchCodexRequest) -> BatchCodexResponse:
    results: List[CodexResponse] = []
    for item in request.items:
        try:
            results.append(run_codex(item))
        except HTTPException:
            raise
        except Exception as error:  # pragma: no cover
            raise HTTPException(status_code=502, detail=f"Falha na chamada OpenAI: {error}") from error

    return BatchCodexResponse(total=len(results), results=results)
