"""Stage 3 -- SCHEMA_VALIDATION.

Deterministic checks over the raw model payload. No LLM involvement, and no
exceptions: every problem becomes a string in the returned list so the full set
of faults reaches `review_summary.json` instead of only the first one.

This runs *before* normalization on purpose. It describes what the model actually
returned, which is what we want on record when something goes wrong.
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any

REQUIRED_KEYS = (
    "supplier_name",
    "currency",
    "items",
    "quote_expiry",
    "shipping_included",
    "notes",
    "assumptions",
    "needs_review",
)

ITEM_KEYS = ("sku", "description", "quantity", "unit_price", "lead_time_days")

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CURRENCY_CODE_RE = re.compile(r"^[A-Za-z]{3}$")


def validate_payload(payload: Any) -> list[str]:
    """Return every schema problem found, newest concern last. Never raises."""
    if not isinstance(payload, dict):
        return [f"payload: expected a JSON object, got {type(payload).__name__}."]

    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in payload:
            errors.append(f"{key}: required key is missing.")

    errors.extend(_check_supplier(payload.get("supplier_name")))
    errors.extend(_check_currency(payload.get("currency")))
    errors.extend(_check_items(payload.get("items")))
    errors.extend(_check_expiry(payload.get("quote_expiry")))
    errors.extend(_check_shipping(payload.get("shipping_included")))
    errors.extend(_check_string_list("notes", payload.get("notes")))
    errors.extend(_check_string_list("assumptions", payload.get("assumptions")))
    errors.extend(_check_needs_review(payload.get("needs_review")))
    return errors


def _check_supplier(value: Any) -> list[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return ["supplier_name: missing."]
    if not isinstance(value, str):
        return [f"supplier_name: expected a string, got {type(value).__name__}."]
    return []


def _check_currency(value: Any) -> list[str]:
    """A non-code currency is flagged, not rejected -- the normalizer may resolve it."""
    if value is None:
        return ["currency: missing."]
    if not isinstance(value, str):
        return [f"currency: expected a string, got {type(value).__name__}."]
    if not CURRENCY_CODE_RE.match(value.strip()):
        return [f"currency: '{value.strip()}' is not a 3-letter code."]
    return []


def _check_items(value: Any) -> list[str]:
    if value is None:
        return ["items: missing."]
    if not isinstance(value, list):
        return [f"items: expected a list, got {type(value).__name__}."]
    if not value:
        return ["items: must not be empty."]

    errors: list[str] = []
    for index, item in enumerate(value):
        errors.extend(_check_item(index, item))
    return errors


def _check_item(index: int, item: Any) -> list[str]:
    where = f"items[{index}]"
    if not isinstance(item, dict):
        return [f"{where}: expected an object, got {type(item).__name__}."]

    errors: list[str] = []
    for key in ITEM_KEYS:
        if key not in item:
            errors.append(f"{where}.{key}: required key is missing.")

    description = item.get("description")
    if description is None or (isinstance(description, str) and not description.strip()):
        errors.append(f"{where}.description: missing.")
    elif not isinstance(description, str):
        errors.append(f"{where}.description: expected a string, got {type(description).__name__}.")

    errors.extend(_check_quantity(where, item.get("quantity")))
    errors.extend(_check_unit_price(where, item.get("unit_price")))
    errors.extend(_check_lead_time(where, item.get("lead_time_days")))

    sku = item.get("sku")
    if sku is not None and not isinstance(sku, str):
        errors.append(f"{where}.sku: expected a string or null, got {type(sku).__name__}.")
    return errors


def _check_quantity(where: str, value: Any) -> list[str]:
    if value is None:
        return [f"{where}.quantity: missing."]
    # bool is a subclass of int; True is not a quantity.
    if isinstance(value, bool):
        return [f"{where}.quantity: expected an integer, got boolean."]
    if isinstance(value, float):
        if not value.is_integer():
            return [f"{where}.quantity: {value} is not a whole number."]
        value = int(value)
    if not isinstance(value, int):
        return [f"{where}.quantity: expected an integer, got {type(value).__name__}."]
    if value <= 0:
        return [f"{where}.quantity: must be greater than 0, got {value}."]
    return []


def _check_unit_price(where: str, value: Any) -> list[str]:
    if value is None:
        return [f"{where}.unit_price: missing."]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [f"{where}.unit_price: expected a number, got {type(value).__name__}."]
    if math.isnan(value) or math.isinf(value):
        return [f"{where}.unit_price: must be a finite number."]
    if value < 0:
        return [f"{where}.unit_price: must not be negative, got {value}."]
    return []


def _check_lead_time(where: str, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return [f"{where}.lead_time_days: expected an integer or null, got boolean."]
    if isinstance(value, float):
        if not value.is_integer():
            return [f"{where}.lead_time_days: {value} is not a whole number."]
        value = int(value)
    if not isinstance(value, int):
        return [f"{where}.lead_time_days: expected an integer or null, got {type(value).__name__}."]
    if value < 0:
        return [f"{where}.lead_time_days: must not be negative, got {value}."]
    return []


def _check_expiry(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, str):
        return [f"quote_expiry: expected an ISO date or null, got {type(value).__name__}."]
    text = value.strip()
    if not ISO_DATE_RE.match(text):
        return [f"quote_expiry: '{text}' is not an ISO YYYY-MM-DD date."]
    try:
        # Pattern alone would accept 2026-02-31.
        date.fromisoformat(text)
    except ValueError:
        return [f"quote_expiry: '{text}' is not a real calendar date."]
    return []


def _check_shipping(value: Any) -> list[str]:
    """Null is allowed: the quote genuinely may not say."""
    if value is None or isinstance(value, bool):
        return []
    return [f"shipping_included: expected a boolean or null, got {type(value).__name__}."]


def _check_string_list(key: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [f"{key}: expected a list of strings, got {type(value).__name__}."]
    return [
        f"{key}[{i}]: expected a string, got {type(entry).__name__}."
        for i, entry in enumerate(value)
        if not isinstance(entry, str)
    ]


def _check_needs_review(value: Any) -> list[str]:
    if value is None or isinstance(value, bool):
        return []
    return [f"needs_review: expected a boolean, got {type(value).__name__}."]
