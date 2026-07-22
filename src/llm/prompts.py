"""Extraction prompt templates.

Fully implemented — this is the actual text sent to the model.

Deliberately, the prompt asks ONLY for extraction. It does not ask the model to
validate, normalize units, resolve relative dates, or decide review outcomes:
those are deterministic rules in `src/components/`. Do not move a rule in here
to make a test pass.
"""

from __future__ import annotations

OUTPUT_SCHEMA = """{
  "supplier_name": "string | null",
  "currency": "string | null",
  "items": [
    {
      "sku": "string | null",
      "description": "string",
      "quantity": number,
      "unit_price": number,
      "lead_time_days": number | null
    }
  ],
  "quote_expiry": "YYYY-MM-DD | null",
  "shipping_included": true | false | null,
  "notes": ["string"],
  "assumptions": ["string"],
  "needs_review": true | false
}"""

SYSTEM_PROMPT = f"""You extract structured pricing data from messy supplier quote text.

Return a single JSON object and nothing else. No prose, no markdown fences, no \
explanation before or after. The object must use exactly these keys:

{OUTPUT_SCHEMA}

Rules:
1. Extract only what is present in the provided quote text. Never use outside \
knowledge about the supplier, the product, or typical prices.
2. If a value is not stated, use null (or an empty list for `notes` and \
`assumptions`). Do not guess.
3. Never invent a SKU, an expiry date, or shipping terms. If the text says a SKU \
was not provided, `sku` is null.
4. Record every uncertain interpretation in `assumptions` as a short sentence, \
for example "Assumed the quantity 12 applies to the copper tubing line".
5. Copy quantities and unit prices exactly as written. Do not convert currencies, \
do not compute totals, and do not convert lead-time phrases into days — if the \
text says "around 3 weeks", report what it says rather than computing a number.
6. If the expiry is relative ("expires next Friday") and cannot be stated as a \
calendar date from the text alone, set `quote_expiry` to null and note the phrase \
in `assumptions`.
7. Set `needs_review` to true when any key field is missing or ambiguous — \
supplier name, currency, an item's description, quantity, or unit price.
8. Put anything else worth a human's attention (urgency, freight terms, partial \
information) in `notes`.
"""

USER_PROMPT_TEMPLATE = """Extract the quote below.

Quote id: {quote_id}

<quote_text>
{quote_text}
</quote_text>

Respond with the JSON object only."""


def build_user_prompt(quote_id: str, quote_text: str) -> str:
    """Render the per-quote user message."""
    return USER_PROMPT_TEMPLATE.format(quote_id=quote_id, quote_text=quote_text)
