"""Optional FastAPI wrapper.

The CLI is the primary interface; this exists to show the pipeline is callable
from a service without restructuring it. The import is guarded so the base
install stays free of web dependencies.

    pip install -e ".[api]"
    uvicorn src.api.app:app --reload

    POST /extract  {"id": "Q-1", "text": "..."}   -> one extracted record
    GET  /health
"""

from __future__ import annotations

try:
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise ImportError(
        'FastAPI is not installed. Run `pip install -e ".[api]"`, '
        "or use the CLI: python main.py --input quotes.json"
    ) from exc

from pydantic import BaseModel

from src.config import Settings
from src.llm import get_adapter
from src.models import QuoteInput
from src.pipeline import process_quote

app = FastAPI(title="Quote Extraction", version="0.1.0")


class ExtractRequest(BaseModel):
    id: str = "adhoc"
    text: str


class ExtractResponse(BaseModel):
    quote_id: str
    needs_review: bool
    result: dict | None
    validation_errors: list[str]
    review_reasons: list[str]
    status: str


@app.get("/health")
def health() -> dict[str, str]:
    settings = Settings.from_env()
    return {"status": "ok", "provider": settings.provider, "model": settings.model}


@app.post("/extract", response_model=ExtractResponse)
def extract_quote(request: ExtractRequest) -> ExtractResponse:
    """Run one quote through the same stages the CLI uses.

    Nothing is written to disk here -- the caller gets the record directly.
    """
    settings = Settings.from_env()
    adapter = get_adapter(settings)
    outcome = process_quote(QuoteInput(id=request.id, text=request.text), adapter)
    return ExtractResponse(
        quote_id=outcome.quote_id,
        needs_review=outcome.needs_review,
        result=outcome.result.model_dump() if outcome.result else None,
        validation_errors=outcome.validation_errors,
        review_reasons=outcome.review_reasons,
        status=outcome.status,
    )
