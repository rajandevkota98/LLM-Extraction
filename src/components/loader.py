"""Stage 1 -- LOAD_INPUT.

Reads `quotes.json` and shape-checks it. This is the one stage allowed to fail
hard: if we cannot read the input there is no pipeline to run. Everything after
this point degrades into a review flag instead of raising.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.models import QuoteInput

# An id becomes a filename in `outputs/`, so it has to be one ordinary path
# segment. `../q` would write outside the output directory and `A/B` would fail
# mid-run on a directory that does not exist, taking `review_summary.json` --
# written only after the loop -- down with it. Rejecting beats sanitizing: an
# operator looks for the output file by the id they put in, so the two must match.
UNSAFE_ID_RE = re.compile(r"[/\\]|[\x00-\x1f]|^\.+$")
MAX_ID_LENGTH = 120


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
        if UNSAFE_ID_RE.search(quote_id) or len(quote_id) > MAX_ID_LENGTH:
            raise InputError(
                f"{path}[{index}] has an unusable 'id' ('{quote_id}'): an id is written out as "
                f"a filename, so it must be at most {MAX_ID_LENGTH} characters and must not "
                "contain path separators, control characters, or be '.' or '..'."
            )

        seen.add(quote_id)
        quotes.append(QuoteInput(id=quote_id, text=text))

    if not quotes:
        raise InputError(f"{path} contains no quotes.")
    return quotes
