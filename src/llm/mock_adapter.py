"""A deterministic stand-in for a real model.

This exists so the pipeline runs end-to-end on a clean checkout with no API key.
It is NOT a parser we would ship, and it is not the solution to the exercise --
it is a crude bundle of regexes standing where a language model would be.

Two properties matter:

1. It is not keyed to the sample quote ids or their exact wording. Give it
   different supplier text of the same shape and it produces the same quality of
   guess. Hardcoding the samples would make the downstream stages untestable.

2. Its output is deliberately imperfect, in the same ways a real model's output
   is imperfect. It leaves "around 3 weeks" as an unresolved null rather than
   computing 21, reports a bare "$" as the currency when that is all the text
   shows, and never resolves a relative expiry. Those gaps are what
   `src/components/normalizer.py` and `reviewer.py` exist to close, so a run
   against this adapter exercises them for real.
"""

from __future__ import annotations

import json
import re

# Codes we will recognise if the text spells one out. Not exhaustive by design --
# an unknown code simply falls through to the review path.
CURRENCY_CODES = frozenset(
    {
        "USD", "EUR", "GBP", "AED", "INR", "JPY", "CAD", "AUD",
        "CHF", "CNY", "SGD", "SEK", "NOK", "DKK", "PLN", "ZAR",
    }
)  # fmt: skip

CURRENCY_SYMBOLS = "$€£₹¥"

_CLAUSE_SPLIT_RE = re.compile(r"\.\s+|\n+|;\s*")
_CODE_RE = re.compile(r"\b([A-Z]{3})\b")
_SYMBOL_RE = re.compile(f"[{re.escape(CURRENCY_SYMBOLS)}]")
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_EXPLICIT_DAYS_RE = re.compile(r"\b(\d+)\s*(?:calendar\s+|business\s+|working\s+)?days?\b", re.I)
_SKU_RE = re.compile(r"\bSKU\s*(?:number|no\.?|#|:)?\s*([A-Z0-9][A-Z0-9\-_/]{2,})\b", re.I)
_NO_SKU_RE = re.compile(r"\bno\s+sku\b|\bsku\s+not\s+(?:provided|given|available)\b", re.I)

_SUPPLIER_LABELLED_RE = re.compile(r"\b(?:supplier|vendor|from)\s*:\s*([^.\n]+)", re.I)
_SUPPLIER_VERB_RE = re.compile(
    r"^([A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,3})\s+"
    r"(?:offers?|quotes?|quotation|proposes?|pricing)\b"
)

# Quantity, in descending order of how explicit the text is being.
_QTY_PATTERNS = (
    re.compile(r"\b(?:qty|quantity)\s*[:=]?\s*(\d[\d,]*)\b", re.I),
    re.compile(r"\b(\d[\d,]*)\s+(?:units?|pcs?|pieces?|nos?|items?)\b", re.I),
    re.compile(r"\b(\d[\d,]*)\s+(?=[a-z])"),
)

# Unit price. Each pattern yields a numeric group named `amount`.
_PRICE_PATTERNS = (
    re.compile(
        r"\b[A-Z]{3}\s*(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?:/\s*(?:unit|ea|pc)|per\s+unit)",
    ),
    re.compile(
        rf"(?:@|\bat\b|\bfor\b)\s*[{re.escape(CURRENCY_SYMBOLS)}]?\s*"
        rf"(?P<amount>\d[\d,]*(?:\.\d+)?)",
        re.I,
    ),
    re.compile(rf"[{re.escape(CURRENCY_SYMBOLS)}]\s*(?P<amount>\d[\d,]*(?:\.\d+)?)"),
    re.compile(
        r"\b(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?:/\s*(?:unit|ea|pc)|per\s+unit|each)\b",
        re.I,
    ),
)

_SHIPPING_EXCLUDED_RE = re.compile(
    r"(?:shipping|freight|delivery)\s+(?:is\s+)?(?:extra|not included|excluded)"
    r"|\b(?:freight|shipping)\s+(?:charges?\s+)?(?:apply|applies)\b",
    re.I,
)
_SHIPPING_INCLUDED_RE = re.compile(
    r"(?:shipping|freight|delivery)\s+(?:is\s+)?included|\bfree\s+shipping\b",
    re.I,
)

# Noise to strip out of a description once quantity and price have been lifted.
_DESC_NOISE_RE = re.compile(
    r"\b(?:units?|pcs?|pieces?|nos?|items?)\s+of\b|\b(?:qty|quantity)\b|\beach\b"
    r"|\bper\s+unit\b|/\s*(?:unit|ea|pc)\b|\bat\b|\bfor\b",
    re.I,
)
_LEADING_LABEL_RE = re.compile(
    r"^[^:]{0,60}\b(?:offers?|quotes?|quotation|proposes?|pricing|supplier|vendor)\b[^:]{0,20}:\s*",
    re.I,
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t-–—,:")


def _find_currency(text: str) -> str | None:
    """A spelled-out code wins; otherwise report the bare symbol, unresolved."""
    for match in _CODE_RE.finditer(text):
        if match.group(1) in CURRENCY_CODES:
            return match.group(1)
    symbol = _SYMBOL_RE.search(text)
    # Deliberately NOT mapped to a code here -- a symbol alone is ambiguous, and
    # deciding what it means is deterministic work, not the model's call.
    return symbol.group(0) if symbol else None


def _find_supplier(text: str, clauses: list[str]) -> str | None:
    labelled = _SUPPLIER_LABELLED_RE.search(text)
    if labelled:
        return _clean(labelled.group(1))
    for clause in clauses:
        verb = _SUPPLIER_VERB_RE.match(clause.strip())
        if verb:
            return _clean(verb.group(1))
    return None


def _find_quantity(clause: str) -> int | None:
    for pattern in _QTY_PATTERNS:
        match = pattern.search(clause)
        if match:
            try:
                return int(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _find_price(clause: str) -> tuple[float, tuple[int, int]] | None:
    for pattern in _PRICE_PATTERNS:
        match = pattern.search(clause)
        if match:
            try:
                return float(match.group("amount").replace(",", "")), match.span()
            except ValueError:
                continue
    return None


def _describe(clause: str, qty: int | None, price_span: tuple[int, int] | None) -> str:
    text = clause
    if price_span:
        text = text[: price_span[0]] + " " + text[price_span[1] :]
    text = _LEADING_LABEL_RE.sub("", text)
    if qty is not None:
        text = re.sub(rf"\b{qty:,}\b|\b{qty}\b", " ", text, count=1)
    text = _DESC_NOISE_RE.sub(" ", text)
    text = _SKU_RE.sub(" ", text)
    text = re.sub(rf"\b(?:{'|'.join(CURRENCY_CODES)})\b", " ", text)
    text = re.sub(rf"[{re.escape(CURRENCY_SYMBOLS)}]", " ", text)
    return _clean(text)


def _extract_items(text: str, clauses: list[str]) -> list[dict]:
    items: list[dict] = []
    for clause in clauses:
        price = _find_price(clause)
        if price is None:
            continue
        unit_price, span = price
        qty = _find_quantity(clause)
        description = _describe(clause, qty, span)
        if not description:
            continue

        sku_match = None if _NO_SKU_RE.search(clause) else _SKU_RE.search(clause)
        days = _EXPLICIT_DAYS_RE.search(clause)
        items.append(
            {
                "sku": sku_match.group(1).upper() if sku_match else None,
                "description": description,
                "quantity": qty,
                "unit_price": unit_price,
                # Only an explicit day count. "3 weeks" is left for the normalizer.
                "lead_time_days": int(days.group(1)) if days else None,
            }
        )

    # Document-level SKU and lead time only apply when there is exactly one line;
    # spreading them across several items would be inventing an association.
    if len(items) == 1:
        item = items[0]
        if item["sku"] is None and not _NO_SKU_RE.search(text):
            doc_sku = _SKU_RE.search(text)
            if doc_sku:
                item["sku"] = doc_sku.group(1).upper()
        if item["lead_time_days"] is None:
            doc_days = _EXPLICIT_DAYS_RE.search(text)
            if doc_days:
                item["lead_time_days"] = int(doc_days.group(1))
    return items


def _find_shipping(text: str) -> bool | None:
    if _SHIPPING_EXCLUDED_RE.search(text):
        return False
    if _SHIPPING_INCLUDED_RE.search(text):
        return True
    return None


def build_payload(quote_text: str) -> dict:
    """Produce the shape a model would return, warts and all."""
    text = quote_text.strip()
    clauses = [c for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]

    supplier = _find_supplier(text, clauses)
    currency = _find_currency(text)
    items = _extract_items(text, clauses)
    expiry = _ISO_DATE_RE.search(text)
    shipping = _find_shipping(text)

    notes: list[str] = []
    assumptions: list[str] = []

    if _NO_SKU_RE.search(text):
        notes.append("Supplier states no SKU was provided.")
    if re.search(r"\burgent\b|\brush\b|\bexpedite", text, re.I):
        notes.append("Order flagged as urgent in the quote text.")

    # Surface the phrase without resolving it -- resolution is deterministic work.
    relative_expiry = re.search(
        r"\b(?:expires?|valid(?:\s+until)?|good\s+(?:through|until))\s+"
        r"((?:next|this|end of|in)\b[^.\n]{0,30})",
        text,
        re.I,
    )
    if expiry is None and relative_expiry:
        assumptions.append(
            f"Expiry is stated relatively as '{_clean(relative_expiry.group(1))}'; "
            "no calendar date could be taken from the text alone."
        )
    if any(i["lead_time_days"] is None for i in items) and re.search(
        r"\b(?:week|month)s?\b", text, re.I
    ):
        assumptions.append("Lead time is given as a phrase rather than a number of days.")
    if currency and currency not in CURRENCY_CODES:
        assumptions.append(f"Currency appears only as the symbol '{currency}'.")

    # Mirrors rule 7 of the system prompt: a KEY field missing or ambiguous.
    # Having merely recorded an assumption is not by itself a reason to flag --
    # if it were, the stub's own chatter would drown out the deterministic rules.
    incomplete = (
        not supplier
        or currency is None
        or currency not in CURRENCY_CODES
        or not items
        or any(
            i["quantity"] is None or i["unit_price"] is None or not i["description"] for i in items
        )
    )

    return {
        "supplier_name": supplier,
        "currency": currency,
        "items": items,
        "quote_expiry": expiry.group(1) if expiry else None,
        "shipping_included": shipping,
        "notes": notes,
        "assumptions": assumptions,
        "needs_review": bool(incomplete),
    }


class MockAdapter:
    """Satisfies `LLMAdapter`. Same input, same output, every time."""

    provider = "mock"
    model = "regex-stub-v1"

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002 - system is unused here
        """Return a JSON string, mimicking a model that respects the format."""
        quote_text = _unwrap_quote_text(user)
        return json.dumps(build_payload(quote_text), ensure_ascii=False, indent=2)


def _unwrap_quote_text(user_prompt: str) -> str:
    """Pull the quote back out of the rendered user prompt.

    A real provider gets the whole prompt and figures this out itself; the stub
    needs the raw text, so it reads its own delimiters back.
    """
    match = re.search(r"<quote_text>\s*(.*?)\s*</quote_text>", user_prompt, re.S)
    return match.group(1) if match else user_prompt
