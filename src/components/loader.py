"""Stage 1 -- LOAD_INPUT.

Reads `quotes.json` and shape-checks it. This is the one stage allowed to fail
hard: if we cannot read the input there is no pipeline to run. Everything after
this point degrades into a review flag instead of raising.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models import QuoteInput


class InputError(RuntimeError):
    """The input file is missing, unreadable, or not the expected shape."""


def load_quotes(path: Path) -> list[QuoteInput]:
    """Read a list of `{"id": ..., "text": ...}` records.

    Records missing an id or with empty text are rejected here rather than
    silently producing an unattributable output file later.

    Raises:
        InputError: the file is missing, is not valid JSON, or is not a list.
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InputError(f"Input file not found: {path}") from exc
    except OSError as exc:
        raise InputError(f"Could not read {path}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise InputError(f"{path} must contain a JSON array of quote objects.")

    quotes: list[QuoteInput] = []
    seen: set[str] = set()
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise InputError(f"{path}[{index}] is not an object.")

        quote_id = str(record.get("id") or "").strip()
        text = str(record.get("text") or "").strip()
        if not quote_id:
            raise InputError(f"{path}[{index}] is missing an 'id'.")
        if not text:
            raise InputError(f"{path}[{index}] ('{quote_id}') has no 'text'.")
        if quote_id in seen:
            raise InputError(f"{path} contains duplicate id '{quote_id}'.")

        seen.add(quote_id)
        quotes.append(QuoteInput(id=quote_id, text=text))

    if not quotes:
        raise InputError(f"{path} contains no quotes.")
    return quotes
