"""Pydantic models for the extraction pipeline.

Fully implemented.

Boundary note: `ExtractionResult` describes the *final, normalized* record —
the thing another service could store. Raw LLM output deliberately does NOT go
through this model on the way in; it stays a plain `dict` until
`components.validator` has accumulated its errors and `components.normalizer`
has coerced values. Parsing model output straight into a strict model would
turn a recoverable data problem into an exception and lose the error detail we
are required to report.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

CallStatus = Literal["success", "parse_error", "validation_failed"]


class QuoteInput(BaseModel):
    """One record from `quotes.json`."""

    id: str
    text: str


class LineItem(BaseModel):
    """A single priced line on a quote, after normalization.

    `description`, `quantity` and `unit_price` are nullable even though a good
    record always has them. Refusing to represent an incomplete line would mean
    either dropping it from the output or raising, and both hide the problem —
    whereas a null here is always accompanied by a review reason naming it.
    """

    sku: str | None = None
    description: str | None = None
    quantity: int | None = None
    unit_price: float | None = None
    lead_time_days: int | None = None


class ExtractionResult(BaseModel):
    """The normalized per-quote output written to `outputs/{quote_id}.json`."""

    supplier_name: str | None = None
    currency: str | None = None
    items: list[LineItem] = Field(default_factory=list)
    quote_expiry: str | None = None  # ISO `YYYY-MM-DD`, or null when unresolvable.
    shipping_included: bool | None = None
    notes: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    needs_review: bool = True


class ReviewSummaryEntry(BaseModel):
    """One row of `review_summary.json`."""

    quote_id: str
    needs_review: bool
    validation_errors: list[str] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)


class PipelineOutcome(BaseModel):
    """Everything one quote produced, carried between stages and out to the writer."""

    quote_id: str
    raw_response: str = ""
    raw_payload: dict[str, Any] | None = None
    result: ExtractionResult | None = None
    validation_errors: list[str] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    status: CallStatus = "success"

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons or self.validation_errors or self.result is None)

    def to_summary_entry(self) -> ReviewSummaryEntry:
        return ReviewSummaryEntry(
            quote_id=self.quote_id,
            needs_review=self.needs_review,
            validation_errors=list(self.validation_errors),
            review_reasons=list(self.review_reasons),
        )


class LLMCallRecord(BaseModel):
    """One line of `llm_calls.jsonl`."""

    quote_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    provider: str
    model: str
    input_artifact: str
    output_artifact: str
    status: CallStatus
