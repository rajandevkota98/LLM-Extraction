"""Stage composition: what one quote's record looks like after everything runs.

The cases here are the ones where a stage boundary is what makes the answer
right — a stage that must not raise, and the accounting of what normalization
changed. No network, no filesystem, no keys.
"""

from __future__ import annotations

import json

from src.llm.base import LLMError
from src.models import QuoteInput
from src.pipeline import process_quote


class _Adapter:
    """A stand-in model returning a fixed payload."""

    provider = "test"
    model = "test-1"

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, system: str, user: str) -> str:
        return json.dumps(self.payload)


class _Raises:
    provider = "test"
    model = "test-1"

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def complete(self, system: str, user: str) -> str:
        raise self.exc


def _payload(**overrides) -> dict:
    payload = {
        "supplier_name": "Beta Metals",
        "currency": "EUR",
        "items": [
            {
                "sku": None,
                "description": "Copper tubing coils",
                "quantity": 12,
                "unit_price": 73.0,
                "lead_time_days": None,
            }
        ],
        "quote_expiry": None,
        "shipping_included": None,
        "notes": [],
        "assumptions": [],
        "needs_review": False,
    }
    payload.update(overrides)
    return payload


def _run(payload: dict, text: str = "12 copper coils at EUR 73 each."):
    return process_quote(QuoteInput(id="Q-1", text=text), _Adapter(payload))


# --- failure never escapes the stage --------------------------------------- #


def test_a_declared_provider_failure_becomes_a_review_flag():
    outcome = process_quote(QuoteInput(id="Q-1", text="widgets"), _Raises(LLMError("429")))
    assert outcome.status == "parse_error"
    assert outcome.needs_review is True


def test_an_undeclared_adapter_exception_also_becomes_a_review_flag():
    """Adapters are contracted to raise LLMError, but they wrap third-party SDKs.

    One surprise must cost one quote, not the run — `review_summary.json` is
    written after the loop, so an escaping exception takes every quote's summary
    with it.
    """
    outcome = process_quote(QuoteInput(id="Q-1", text="widgets"), _Raises(AttributeError("boom")))
    assert outcome.status == "parse_error"
    assert outcome.needs_review is True
    assert "AttributeError" in outcome.validation_errors[0]


def test_unparsable_model_output_still_produces_a_record():
    outcome = process_quote(QuoteInput(id="Q-1", text="widgets"), _Raises(LLMError("no")))
    assert outcome.quote_id == "Q-1"
    assert outcome.result is not None
    assert outcome.review_reasons


# --- what normalization changed -------------------------------------------- #


def test_repairs_are_named_rather_than_counted():
    """Two faults repaired, one introduced — a count subtraction reports "1"."""
    outcome = _run(_payload(supplier_name="...", currency="£", notes="not a list"))
    assumptions = " ".join(outcome.result.assumptions)

    assert "repaired 2 schema faults" in assumptions
    assert "currency: '£' is not a 3-letter code." in assumptions
    assert "notes: expected a list of strings, got str." in assumptions
    # The fault normalization created is attributed to normalization, not left
    # sitting anonymously in validation_errors.
    assert "could not produce a usable value for 1 field" in assumptions
    assert "supplier_name: missing." in assumptions


def test_a_clean_quote_claims_no_repairs():
    outcome = _run(_payload())
    assert not any("Normalization" in a for a in outcome.result.assumptions)
    assert outcome.needs_review is False


# --- expiry the model returned but we could not read ------------------------ #


def test_an_unreadable_expiry_is_dropped_and_flagged():
    """`quote_expiry` is typed as ISO-or-null, so free text cannot stay in it.

    Dropping it is only safe because the drop reaches the operator: silently
    nulling would turn a hallucinated date into a quote that looks clean.
    """
    outcome = _run(_payload(quote_expiry="sometime in Q3"))
    assert outcome.result.quote_expiry is None
    assert outcome.needs_review is True
    assert any("sometime in Q3" in r for r in outcome.review_reasons)
    assert any("sometime in Q3" in a for a in outcome.result.assumptions)


def test_a_real_iso_date_survives_untouched():
    outcome = _run(_payload(quote_expiry="2026-09-30"))
    assert outcome.result.quote_expiry == "2026-09-30"
    assert outcome.needs_review is False


def test_an_impossible_calendar_date_is_dropped():
    """2026-02-31 matches the ISO pattern and is not a day."""
    outcome = _run(_payload(quote_expiry="2026-02-31"))
    assert outcome.result.quote_expiry is None
    assert outcome.needs_review is True


# --- the model's own flag escalates, never clears --------------------------- #


def test_the_model_flag_can_escalate_a_clean_quote():
    outcome = _run(_payload(needs_review=True))
    assert outcome.needs_review is True
    assert any("model flagged" in r for r in outcome.review_reasons)


def test_the_model_flag_cannot_clear_a_dirty_quote():
    outcome = _run(_payload(needs_review=False, supplier_name=None))
    assert outcome.needs_review is True
