"""Stage 5 -- REVIEW_DECISION.

The rules that decide whether a human has to look at a quote. All deterministic,
all in code. The model's own `needs_review` flag is read here as one input signal
among several: it can escalate a quote to review, it can never clear one.

Every rule returns a sentence a human can act on. "currency: missing" tells an
operator what to fix; "rule 3 failed" does not.
"""

from __future__ import annotations

import re
from typing import Any

from src.components.normalizer import KNOWN_CODES, has_relative_expiry

CURRENCY_CODE_RE = re.compile(r"^[A-Z]{3}$")


def decide(
    payload: dict,
    quote_text: str,
    validation_errors: list[str],
    *,
    model_flagged: Any = None,
    discarded_expiry: str | None = None,
) -> list[str]:
    """Return every reason this quote needs a human. Empty list means clean.

    `discarded_expiry` is the second signal the pipeline passes in alongside
    `model_flagged`: the expiry the model returned, when normalization could not
    read it as a calendar date and dropped it. Without it, a dropped value is
    indistinguishable here from a quote that never stated an expiry at all.
    """
    reasons: list[str] = []

    reasons.extend(_supplier_reasons(payload.get("supplier_name")))
    reasons.extend(_currency_reasons(payload.get("currency")))
    reasons.extend(_item_reasons(payload.get("items")))
    reasons.extend(_expiry_reasons(payload.get("quote_expiry"), quote_text, discarded_expiry))

    if validation_errors:
        count = len(validation_errors)
        reasons.append(
            f"Model output failed {count} schema check{'s' if count != 1 else ''}; "
            "see validation_errors."
        )

    if model_flagged is True:
        reasons.append("The extraction model flagged this quote as uncertain.")

    return reasons


def needs_review(reasons: list[str], validation_errors: list[str]) -> bool:
    """Review is required if anything at all is outstanding."""
    return bool(reasons or validation_errors)


def _supplier_reasons(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return ["Supplier name is missing from the extracted quote."]
    return []


def _currency_reasons(value: Any) -> list[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return ["Currency is missing from the extracted quote."]
    if not isinstance(value, str):
        return ["Currency is not a usable value."]

    text = value.strip()
    if not CURRENCY_CODE_RE.match(text.upper()):
        return [
            f"Currency is ambiguous: '{text}' could not be resolved to a 3-letter code "
            "from the quote text."
        ]
    if text.upper() not in KNOWN_CODES:
        return [f"Currency '{text.upper()}' is not a recognised code; confirm it before storing."]
    return []


def _item_reasons(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["No line items could be extracted from the quote."]

    reasons: list[str] = []
    for index, item in enumerate(value):
        label = f"Item {index + 1}"
        if not isinstance(item, dict):
            reasons.append(f"{label} is not a usable record.")
            continue

        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            reasons.append(f"{label} is missing a description.")

        quantity = item.get("quantity")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            reasons.append(f"{label} is missing a usable quantity.")

        price = item.get("unit_price")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or price < 0:
            reasons.append(f"{label} is missing a usable unit price.")
    return reasons


def _expiry_reasons(value: Any, quote_text: str, discarded: str | None = None) -> list[str]:
    """Only unresolved counts.

    A quote with no expiry mentioned at all is not a problem; a quote whose
    expiry we could see but could not safely pin to a date is.

    A discarded value and a relative phrase are usually the same fact seen twice,
    so the discard wins -- it is the more specific of the two and names the text
    that was thrown away.
    """
    if isinstance(value, str) and value.strip():
        return []
    if discarded:
        return [
            f"Quote expiry '{discarded}' could not be read as a calendar date and was left "
            "unset; confirm the expiry by hand."
        ]
    if has_relative_expiry(quote_text):
        return [
            "Quote expiry is expressed relative to an unknown date and could not be "
            "resolved safely."
        ]
    return []
