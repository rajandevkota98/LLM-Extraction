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

from src.components.validator import is_iso_date

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

# A three-letter code only counts as a currency when the text uses it as one:
# next to an amount, or introduced by a word like "prices in". A bare uppercase
# triple is just as likely to be part of a supplier's address ("ACME SAR OFFICE").
_CODE_IN_TEXT_RE = re.compile(
    r"\b(?P<before>[A-Z]{3})\s*\d"
    r"|\d\s*(?P<after>[A-Z]{3})\b"
    r"|(?i:\b(?:currency|prices?|priced|quoted|payable|billed|invoiced)\b[^.\n]{0,12}?)"
    r"\b(?P<named>[A-Z]{3})\b"
)

# Lead time expressed as a phrase. The approximation marker is captured so we can
# say in `assumptions` that the number is derived rather than stated.
LEAD_TIME_RE = re.compile(
    r"(?P<approx>\b(?:around|approx\.?|approximately|about|roughly|up\s+to)\b\s*|~\s*)?"
    r"(?P<count>\d{1,3}|" + "|".join(NUMBER_WORDS) + r")"
    r"\s*(?:-|\s)?\s*"
    r"(?P<unit>day|week|month)s?\b",
    re.I,
)

# Words that mean a following duration is about something other than delivery:
# the quote's validity ("valid for 30 days"), the payment terms ("Net 30 days"),
# or a warranty ("warranty 12 months"). Payment terms in particular appear on a
# large share of real quotes, and reading one as a lead time is silently wrong --
# an integer lead time passes every downstream check.
#
# Only the text *before* the duration is examined. Looking after it would break
# the common "ships in 3 weeks, payment net 30" ordering, where the trailing
# payment term has nothing to do with the delivery promise that precedes it.
NON_DELIVERY_CONTEXT_RE = re.compile(
    r"(?:expir\w*|valid|good\s+(?:through|until)|quote|offer"
    r"|nett?|payment|terms|invoiced?|warrant\w*|guarantee\w*)"
    r"(?:\s+(?:for|until|through|till|up\s+to|within|of|is|are))?\W*$",
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
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if re.fullmatch(r"[A-Za-z]{3}", text):
            return text.upper(), None

        if text in UNAMBIGUOUS_SYMBOLS:
            code = UNAMBIGUOUS_SYMBOLS[text]
            return code, f"Currency symbol '{text}' maps to {code}."
        if text in AMBIGUOUS_SYMBOLS:
            resolved = _resolve_ambiguous_symbol(text, quote_text)
            if resolved:
                return resolved, (
                    f"Currency symbol '{text}' resolved to {resolved} "
                    "from a currency named in the quote text."
                )
            return text, None

        # Something the model read off the page that we cannot resolve. It is kept
        # so the reviewer can name what was unusable. Replacing it with a code
        # found elsewhere in the text would assert something the model did not.
        return text, None

    # Nothing usable on the payload -- fall back to a code spelled out in the text.
    for match in _CODE_IN_TEXT_RE.finditer(quote_text or ""):
        code = match.group("before") or match.group("after") or match.group("named")
        if code in KNOWN_CODES:
            return code, f"Currency {code} taken from the quote text."
    return None, None


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
    # A lead time found in the body of the quote belongs to a line only when
    # there is one line to attribute it to. Spreading a single "around 3 weeks"
    # across every item invents an association the text never made -- and the
    # result passes validation, so nothing downstream would catch it.
    attributable = phrase is not None and sum(isinstance(e, dict) for e in value) == 1

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
        if item["lead_time_days"] is None and attributable:
            item["lead_time_days"] = phrase.days
            notes.append(
                f"Lead time {phrase.days} days derived from '{phrase.source}' in the quote text."
            )
        items.append(item)

    if (
        phrase is not None
        and not attributable
        and any(isinstance(i, dict) and i.get("lead_time_days") is None for i in items)
    ):
        notes.append(
            f"The quote text states a lead time of '{phrase.source}', but has more than one "
            "line item and does not say which it applies to. Lead times were left unset."
        )
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

    Weeks are x7 and months are x30. Durations attached to something other than
    delivery are skipped: 'valid for 30 days' is an expiry and 'Net 30 days' is a
    payment term, and neither is a delivery promise.
    """
    for match in LEAD_TIME_RE.finditer(text or ""):
        preceding = text[max(0, match.start() - 24) : match.start()]
        if NON_DELIVERY_CONTEXT_RE.search(preceding):
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


# One numeric token: digits, optionally broken up by `.` or `,`. Deliberately not
# a "strip everything that is not a digit" pass -- that turns "12 units @ 5.00"
# into 125.0 by concatenating two unrelated numbers.
_NUMBER_TOKEN_RE = re.compile(r"-?\d[\d.,]*\d|-?\d")


def _to_int(value: Any) -> Any:
    """Coerce clean integer-ish values; leave anything doubtful for the validator."""
    if isinstance(value, bool) or value is None:
        return None if isinstance(value, bool) else value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        number = _to_float(value)
        if isinstance(number, float) and number.is_integer():
            return int(number)
    return value


def _to_float(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return None if isinstance(value, bool) else value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        tokens = _NUMBER_TOKEN_RE.findall(value)
        # Exactly one number, or we do not know which one is the price.
        if len(tokens) == 1:
            number = _parse_number(tokens[0])
            if number is not None:
                return number
    return value


def _parse_number(token: str) -> float | None:
    """Read one numeric token, resolving `.`/`,` without guessing at a locale.

    Getting this wrong is the worst failure the pipeline has: a mis-read separator
    produces a finite, non-negative number that passes every validation and review
    check, so a price wrong by a factor of 1000 is written out as clean.

    Where both separators appear, the last one is the decimal point -- that settles
    "1.234,56" and "1,234.56" as 1234.56 without knowing the supplier's country.
    A lone comma groups thousands when exactly three digits follow it ("1,200") and
    is a decimal point when one or two do ("18,50" is 18.5, not 1850). Repeated
    separators are grouping. Anything else is genuinely ambiguous and returns None,
    so the value survives as a string and the validator reports it.
    """
    sign = -1.0 if token.startswith("-") else 1.0
    digits = token.lstrip("-")
    dot, comma = digits.rfind("."), digits.rfind(",")

    if dot != -1 and comma != -1:
        split_at = max(dot, comma)
    elif comma != -1:
        groups = digits.split(",")
        trailing = len(groups[-1])
        if len(groups) > 2 or trailing == 3:
            split_at = -1
        elif trailing in (1, 2):
            split_at = comma
        else:
            return None
    elif dot != -1:
        split_at = -1 if digits.count(".") > 1 else dot
    else:
        split_at = -1

    whole = re.sub(r"[.,]", "", digits if split_at == -1 else digits[:split_at])
    fraction = "" if split_at == -1 else digits[split_at + 1 :]
    if not whole.isdigit() or (fraction and not fraction.isdigit()):
        return None
    return sign * float(f"{whole}.{fraction or 0}")


# --------------------------------------------------------------------------- #
# expiry and shipping
# --------------------------------------------------------------------------- #


def _normalize_expiry(value: Any, quote_text: str) -> tuple[Any, str | None]:
    """Keep a real ISO date; refuse to resolve or keep anything else.

    'Next Friday' has no safe answer: the quote's send date is unknown, so any
    date we produced would be fabricated. Detect it, null it, and say why.

    A model that ignores the prompt and answers "next Friday" in this field gets
    the same treatment. `quote_expiry` is typed as an ISO date or null, and free
    text sitting in it would be parsed as a date by whatever stores the record.
    Dropping it is only safe because the drop is named, both in `assumptions` and
    -- via `reviewer.py` -- as a reason a human has to look at the quote.
    """
    discarded = None
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if is_iso_date(text):
            return text, None
        discarded = text

    relative = RELATIVE_EXPIRY_RE.search(quote_text or "")
    phrase = _collapse(relative.group("phrase")).strip(" .,;:") if relative else None

    if discarded and phrase:
        return None, (
            f"Quote expiry '{discarded}' is not a calendar date; the text states it relatively "
            f"as '{phrase}', which cannot be resolved without the quote's send date. Left unset."
        )
    if discarded:
        return None, (
            f"Quote expiry '{discarded}' is not a calendar date and could not be resolved "
            "from the text, so it is left unset."
        )
    if phrase:
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
