"""Stage 4 -- NORMALIZATION.

Deterministic coercion of the model's payload into storable values. This is the
half of the boundary that makes the prompt's restraint safe: `prompts.py` tells
the model *not* to turn "around 3 weeks" into a number, because turning it into
21 is this module's job and needs to be reproducible.

Note the signature: normalization needs the original `quote_text`, not just the
payload. Resolving a lead-time phrase or a bare currency symbol means reading the
source, because those facts are in the text and absent from the model's output.

Nothing here invents data. Where the text does not support a value, the value
stays null and `reviewer.py` decides what that means.
"""

from __future__ import annotations

import re
from typing import Any

DAYS_PER_WEEK = 7
DAYS_PER_MONTH = 30

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}  # fmt: skip

# Symbols that map to exactly one currency. `$` and `¥` are deliberately absent:
# they are shared by several currencies, so a bare one is genuinely ambiguous.
UNAMBIGUOUS_SYMBOLS = {"€": "EUR", "£": "GBP", "₹": "INR", "₩": "KRW", "₺": "TRY"}
AMBIGUOUS_SYMBOLS = {"$", "¥"}

KNOWN_CODES = frozenset(
    {
        "USD", "EUR", "GBP", "AED", "INR", "JPY", "CAD", "AUD", "CHF",
        "CNY", "SGD", "SEK", "NOK", "DKK", "PLN", "ZAR", "KRW", "TRY",
        "MXN", "BRL", "HKD", "NZD", "SAR", "QAR", "THB", "MYR",
    }
)  # fmt: skip

# `$` resolves only when the text also names a dollar currency outright.
DOLLAR_CONTEXT = (
    (re.compile(r"\bUSD\b|\bUS\s*\$|\bU\.S\.\s*dollars?\b|\bUS\s+dollars?\b", re.I), "USD"),
    (re.compile(r"\bCAD\b|\bCanadian\s+dollars?\b|\bC\$", re.I), "CAD"),
    (re.compile(r"\bAUD\b|\bAustralian\s+dollars?\b|\bA\$", re.I), "AUD"),
    (re.compile(r"\bSGD\b|\bSingapore\s+dollars?\b|\bS\$", re.I), "SGD"),
)
YEN_CONTEXT = (
    (re.compile(r"\bJPY\b|\bJapanese\s+yen\b", re.I), "JPY"),
    (re.compile(r"\bCNY\b|\bRMB\b|\byuan\b|\brenminbi\b", re.I), "CNY"),
)

_CODE_IN_TEXT_RE = re.compile(r"\b([A-Z]{3})\b")

# Lead time expressed as a phrase. The approximation marker is captured so we can
# say in `assumptions` that the number is derived rather than stated.
LEAD_TIME_RE = re.compile(
    r"(?P<approx>\b(?:around|approx\.?|approximately|about|roughly|up\s+to)\b\s*|~\s*)?"
    r"(?P<count>\d{1,3}|" + "|".join(NUMBER_WORDS) + r")"
    r"\s*(?:-|\s)?\s*"
    r"(?P<unit>day|week|month)s?\b",
    re.I,
)

# Words that mean a following duration is about the quote's validity, not delivery.
# The optional filler lets "valid FOR 30 days" match as well as "expires 30 days".
EXPIRY_CONTEXT_RE = re.compile(
    r"(?:expir\w*|valid|good\s+(?:through|until)|quote|offer)"
    r"(?:\s+(?:for|until|through|till|up\s+to|within))?\W*$",
    re.I,
)

RELATIVE_EXPIRY_RE = re.compile(
    r"\b(?:expires?|expiry|valid(?:\s+(?:until|through|till))?|good\s+(?:through|until))"
    r"\s+(?P<phrase>(?:next|this|last|end\s+of|in|within|by)\b[^.\n]{0,40})",
    re.I,
)

SHIPPING_EXCLUDED_RE = re.compile(
    r"(?:shipping|freight|delivery|carriage)\s+(?:is\s+|are\s+)?"
    r"(?:extra|additional|not\s+included|excluded|separate)"
    r"|\bfreight\s+(?:charges?\s+)?(?:apply|applies)\b"
    r"|\bex[-\s]?works\b|\bFOB\b",
    re.I,
)
SHIPPING_INCLUDED_RE = re.compile(
    r"(?:shipping|freight|delivery|carriage)\s+(?:is\s+|are\s+)?included"
    r"|\bfree\s+shipping\b|\bincl\.?\s+(?:shipping|freight)\b|\bdelivered\s+duty\s+paid\b|\bDDP\b",
    re.I,
)


def normalize(payload: dict, quote_text: str) -> tuple[dict, list[str]]:
    """Return a normalized copy of `payload` plus any assumptions we had to make.

    The returned assumptions are ours, not the model's -- the caller merges them
    so a human can see which values were derived rather than stated.
    """
    result = _tidy_strings(payload)
    assumptions: list[str] = []

    result["supplier_name"] = _normalize_supplier(result.get("supplier_name"))
    result["currency"], currency_note = _normalize_currency(result.get("currency"), quote_text)
    if currency_note:
        assumptions.append(currency_note)

    result["items"], item_notes = _normalize_items(result.get("items"), quote_text)
    assumptions.extend(item_notes)

    result["quote_expiry"], expiry_note = _normalize_expiry(result.get("quote_expiry"), quote_text)
    if expiry_note:
        assumptions.append(expiry_note)

    result["shipping_included"] = _normalize_shipping(result.get("shipping_included"), quote_text)
    result["notes"] = _clean_string_list(result.get("notes"))
    result["assumptions"] = _clean_string_list(result.get("assumptions"))
    return result, assumptions


# --------------------------------------------------------------------------- #
# whitespace and casing
# --------------------------------------------------------------------------- #


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tidy_strings(value: Any) -> Any:
    """Trim and collapse whitespace in every string, at every depth."""
    if isinstance(value, str):
        return _collapse(value)
    if isinstance(value, list):
        return [_tidy_strings(entry) for entry in value]
    if isinstance(value, dict):
        return {key: _tidy_strings(entry) for key, entry in value.items()}
    return value


def _normalize_supplier(value: Any) -> Any:
    """Trim only.

    No case folding: supplier names are proper nouns, and title-casing turns
    'ACME GmbH' into 'Acme Gmbh'. Mangled data is worse than untouched data.
    """
    if not isinstance(value, str):
        return value
    cleaned = _collapse(value).strip(" .,;:-")
    return cleaned or None


def _clean_string_list(value: Any) -> list[str]:
    """Drop blanks and duplicates while preserving order."""
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            continue
        text = _collapse(entry)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


# --------------------------------------------------------------------------- #
# currency
# --------------------------------------------------------------------------- #


def _normalize_currency(value: Any, quote_text: str) -> tuple[Any, str | None]:
    """Resolve to an uppercase 3-letter code, or leave it unresolved.

    An unresolvable symbol is returned as-is rather than guessed at, so
    `reviewer.py` can flag it. Assuming `$` means USD is exactly the kind of
    silent invention this pipeline exists to prevent.
    """
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[A-Za-z]{3}", text):
            return text.upper(), None

        symbol = text or None
        if symbol in UNAMBIGUOUS_SYMBOLS:
            code = UNAMBIGUOUS_SYMBOLS[symbol]
            return code, f"Currency symbol '{symbol}' maps to {code}."
        if symbol in AMBIGUOUS_SYMBOLS:
            resolved = _resolve_ambiguous_symbol(symbol, quote_text)
            if resolved:
                return resolved, (
                    f"Currency symbol '{symbol}' resolved to {resolved} "
                    "from a currency named in the quote text."
                )
            return symbol, None

    # Nothing usable on the payload -- fall back to a code spelled out in the text.
    for match in _CODE_IN_TEXT_RE.finditer(quote_text):
        if match.group(1) in KNOWN_CODES:
            code = match.group(1)
            return code, f"Currency {code} taken from the quote text."
    return value if isinstance(value, str) and value.strip() else None, None


def _resolve_ambiguous_symbol(symbol: str, quote_text: str) -> str | None:
    context = DOLLAR_CONTEXT if symbol == "$" else YEN_CONTEXT
    matches = {code for pattern, code in context if pattern.search(quote_text)}
    # Two dollar currencies named in one quote is ambiguity, not evidence.
    return matches.pop() if len(matches) == 1 else None


# --------------------------------------------------------------------------- #
# items and lead time
# --------------------------------------------------------------------------- #


def _normalize_items(value: Any, quote_text: str) -> tuple[Any, list[str]]:
    if not isinstance(value, list):
        return value, []

    notes: list[str] = []
    phrase = parse_lead_time(quote_text)
    items = []
    for entry in value:
        if not isinstance(entry, dict):
            items.append(entry)
            continue
        item = dict(entry)

        if isinstance(item.get("sku"), str):
            sku = _collapse(item["sku"]).upper()
            item["sku"] = sku or None

        if isinstance(item.get("description"), str):
            item["description"] = _collapse(item["description"]).strip(" .,;:-") or None

        item["quantity"] = _to_int(item.get("quantity"))
        item["unit_price"] = _to_float(item.get("unit_price"))
        item["lead_time_days"] = _to_int(item.get("lead_time_days"))

        # Only fill a gap -- never overwrite a number the model read off the page.
        if item["lead_time_days"] is None and phrase is not None:
            item["lead_time_days"] = phrase.days
            notes.append(
                f"Lead time {phrase.days} days derived from '{phrase.source}' in the quote text."
            )
        items.append(item)
    return items, list(dict.fromkeys(notes))


class LeadTime:
    """A lead-time phrase resolved to whole days."""

    __slots__ = ("days", "source", "approximate")

    def __init__(self, days: int, source: str, approximate: bool) -> None:
        self.days = days
        self.source = source
        self.approximate = approximate


def parse_lead_time(text: str) -> LeadTime | None:
    """Find a lead-time phrase and convert it to days.

    Weeks are x7 and months are x30. Durations attached to the quote's validity
    ('valid for 30 days') are skipped -- that is an expiry, not a delivery time.
    """
    for match in LEAD_TIME_RE.finditer(text or ""):
        preceding = text[max(0, match.start() - 24) : match.start()]
        if EXPIRY_CONTEXT_RE.search(preceding):
            continue

        raw_count = match.group("count").lower()
        count = NUMBER_WORDS.get(raw_count)
        if count is None:
            try:
                count = int(raw_count)
            except ValueError:
                continue
        if count <= 0:
            continue

        unit = match.group("unit").lower()
        days = count * {"day": 1, "week": DAYS_PER_WEEK, "month": DAYS_PER_MONTH}[unit]
        return LeadTime(
            days=days,
            source=_collapse(match.group(0)),
            approximate=bool(match.group("approx")),
        )
    return None


def _to_int(value: Any) -> Any:
    """Coerce clean integer-ish values; leave anything doubtful for the validator."""
    if isinstance(value, bool) or value is None:
        return None if isinstance(value, bool) else value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if re.fullmatch(r"-?\d+", text):
            return int(text)
    return value


def _to_float(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return None if isinstance(value, bool) else value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = re.sub(r"[^\d.\-]", "", value.strip().replace(",", ""))
        if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            return float(text)
    return value


# --------------------------------------------------------------------------- #
# expiry and shipping
# --------------------------------------------------------------------------- #


def _normalize_expiry(value: Any, quote_text: str) -> tuple[Any, str | None]:
    """Keep a real ISO date; refuse to resolve a relative one.

    'Next Friday' has no safe answer: the quote's send date is unknown, so any
    date we produced would be fabricated. Detect it, null it, and say why.
    """
    if isinstance(value, str) and value.strip():
        return value.strip(), None

    relative = RELATIVE_EXPIRY_RE.search(quote_text or "")
    if relative:
        phrase = _collapse(relative.group("phrase")).strip(" .,;:")
        return None, (
            f"Quote expiry is stated relatively as '{phrase}'. It cannot be resolved to a "
            "calendar date without the quote's send date, so it is left unset."
        )
    return None, None


def has_relative_expiry(quote_text: str) -> bool:
    """True when the text dates the quote relative to an unknown moment."""
    return bool(RELATIVE_EXPIRY_RE.search(quote_text or ""))


def _normalize_shipping(value: Any, quote_text: str) -> Any:
    """Infer from the text only when the model left it unset."""
    if isinstance(value, bool):
        return value
    if SHIPPING_EXCLUDED_RE.search(quote_text or ""):
        return False
    if SHIPPING_INCLUDED_RE.search(quote_text or ""):
        return True
    return None
