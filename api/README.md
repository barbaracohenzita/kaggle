# Codex API

API em **FastAPI** para expor o OpenAI via HTTP (estilo "API do Codex").

## 1) Configuração

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt
export OPENAI_API_KEY="sua-chave"
```

## 2) Executar

```bash
uvicorn api.main:app --reload
```

Docs interativas: `http://127.0.0.1:8000/docs`

## Endpoints

- `GET /`: metadados e rotas disponíveis.
- `GET /health`: status + se `OPENAI_API_KEY` está configurada.
- `POST /v1/codex/run`: executa um prompt único.
- `POST /v1/codex/run-batch`: executa vários prompts em lote.

## Exemplo (prompt único)

### Request

```json
{
  "prompt": "Escreva uma função Python para inverter uma string.",
  "model": "gpt-5",
  "instructions": "Responda com código e uma explicação curta.",
  "max_output_tokens": 500
}
```

### Response (exemplo)

```json
{
  "model": "gpt-5",
  "output_text": "def reverse_string(s: str) -> str:\n    return s[::-1]",
  "usage": {
    "input_tokens": 42,
    "output_tokens": 32,
    "total_tokens": 74
  },
  "created_at": "2026-03-27T12:00:00+00:00"
}
```

## Observações

- A API usa o endpoint **Responses API** do SDK oficial da OpenAI.
- Se a chave `OPENAI_API_KEY` não estiver definida, a API retorna erro `500` explicando como corrigir.
