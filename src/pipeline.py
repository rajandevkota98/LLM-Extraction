"""Stage orchestration.

    LOAD_INPUT -> LLM_EXTRACTION -> SCHEMA_VALIDATION -> NORMALIZATION
                                 -> REVIEW_DECISION -> RESULTS_WRITTEN

This module is the only place the stages are composed. Each stage is a plain
function taking data and returning data, so any of them can be exercised in a
test without a model, a filesystem, or the others.
"""

from __future__ import annotations

from pydantic import ValidationError

from src.components import normalizer, reviewer, validator, writer
from src.components.extractor import extract
from src.components.loader import load_quotes
from src.config import Settings
from src.llm import CallLog, get_adapter
from src.llm.base import LLMAdapter
from src.models import CallStatus, ExtractionResult, PipelineOutcome, QuoteInput

RESULT_KEYS = (
    "supplier_name",
    "currency",
    "items",
    "quote_expiry",
    "shipping_included",
    "notes",
    "assumptions",
)


def run(settings: Settings, adapter: LLMAdapter | None = None) -> list[PipelineOutcome]:
    """Run every quote through the pipeline and write all artifacts."""
    quotes = load_quotes(settings.input_path)  # LOAD_INPUT

    adapter = adapter or get_adapter(settings)
    call_log = CallLog(settings.call_log_path)
    call_log.reset()

    outcomes: list[PipelineOutcome] = []
    for quote in quotes:
        outcome = process_quote(quote, adapter)
        final_path, raw_path = writer.write_outputs(outcome, settings.output_dir)  # RESULTS_WRITTEN
        call_log.record(
            quote_id=quote.id,
            provider=getattr(adapter, "provider", "unknown"),
            model=getattr(adapter, "model", "unknown"),
            input_artifact=str(settings.input_path),
            output_artifact=str(raw_path),
            status=outcome.status,
        )
        outcomes.append(outcome)

    writer.write_review_summary(outcomes, settings.review_summary_path)
    return outcomes


def process_quote(quote: QuoteInput, adapter: LLMAdapter) -> PipelineOutcome:
    """Run one quote through stages 2-5. Never raises."""
    attempt = extract(adapter, quote.id, quote.text)  # LLM_EXTRACTION

    if attempt.payload is None:
        return _failed_extraction(quote, attempt.raw_response, attempt.error)

    raw_errors = validator.validate_payload(attempt.payload)  # SCHEMA_VALIDATION
    normalized, derived = normalizer.normalize(attempt.payload, quote.text)  # NORMALIZATION
    if derived:
        normalized["assumptions"] = normalized.get("assumptions", []) + derived

    # Validate again after normalization, and let *that* result drive the review
    # decision. Normalization is the sanctioned repair step: a model reporting
    # "£" is a schema fault before it runs and a resolved GBP afterwards. Holding
    # the pre-normalization error against the record would flag clean quotes for
    # a problem that no longer exists. Faults normalization could not fix survive
    # both passes and still count.
    validation_errors = validator.validate_payload(normalized)

    repaired = len(raw_errors) - len(validation_errors)
    if repaired > 0:
        normalized["assumptions"] = normalized.get("assumptions", []) + [
            f"Normalization repaired {repaired} schema fault"
            f"{'s' if repaired != 1 else ''} in the model output."
        ]

    result, build_errors = _build_result(normalized)
    validation_errors.extend(build_errors)

    reasons = reviewer.decide(  # REVIEW_DECISION
        normalized,
        quote.text,
        validation_errors,
        model_flagged=attempt.payload.get("needs_review"),
    )
    result.needs_review = reviewer.needs_review(reasons, validation_errors)

    return PipelineOutcome(
        quote_id=quote.id,
        raw_response=attempt.raw_response,
        raw_payload=attempt.payload,
        result=result,
        validation_errors=validation_errors,
        review_reasons=reasons,
        status=_status(validation_errors),
    )


def _failed_extraction(quote: QuoteInput, raw: str, error: str | None) -> PipelineOutcome:
    """The fallback path: the model gave us nothing usable.

    We still emit a full record so the quote appears in every artifact, marked
    for review with the reason stated plainly.
    """
    message = error or "Model response could not be parsed."
    reasons = reviewer.decide({}, quote.text, [message])
    return PipelineOutcome(
        quote_id=quote.id,
        raw_response=raw,
        raw_payload=None,
        result=ExtractionResult(needs_review=True, notes=[message]),
        validation_errors=[message],
        review_reasons=reasons,
        status="parse_error",
    )


def _build_result(normalized: dict) -> tuple[ExtractionResult, list[str]]:
    """Coerce the normalized payload into the typed result.

    A failure here is recorded as a validation error and replaced with an empty
    flagged record, rather than allowed to abort the run.
    """
    candidate = {key: normalized.get(key) for key in RESULT_KEYS}
    try:
        return ExtractionResult(**candidate), []
    except ValidationError as exc:
        details = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return ExtractionResult(needs_review=True), [f"result: {d}" for d in details]


def _status(validation_errors: list[str]) -> CallStatus:
    """Describes the call, not the review decision.

    A quote can be perfectly extracted and still need a human -- that is
    `success` with `needs_review: true`, not a failure.
    """
    return "validation_failed" if validation_errors else "success"


def summarize(outcomes: list[PipelineOutcome]) -> dict[str, int]:
    """Counts for the CLI's closing line."""
    flagged = sum(1 for o in outcomes if o.needs_review)
    return {
        "total": len(outcomes),
        "needs_review": flagged,
        "clean": len(outcomes) - flagged,
        "parse_errors": sum(1 for o in outcomes if o.status == "parse_error"),
    }


def default_settings(**overrides: object) -> Settings:
    """Convenience for callers that are not the CLI (tests, the API wrapper)."""
    return Settings.from_env(**overrides)  # type: ignore[arg-type]


__all__ = ["default_settings", "process_quote", "run", "summarize"]
