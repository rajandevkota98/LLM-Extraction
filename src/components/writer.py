"""Stage 6 -- RESULTS_WRITTEN.

Every quote produces a final record and a raw record, whatever happened upstream.
A quote that failed to parse still gets both files: silence is the one outcome a
downstream operator cannot act on.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models import PipelineOutcome, ReviewSummaryEntry


def write_outputs(outcome: PipelineOutcome, output_dir: Path) -> tuple[Path, Path]:
    """Write `{id}.json` and `{id}_raw.json`. Returns both paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / f"{outcome.quote_id}.json"
    raw_path = output_dir / f"{outcome.quote_id}_raw.json"

    result = outcome.result.model_dump() if outcome.result else {}
    _write_json(final_path, result)
    _write_json(raw_path, _raw_document(outcome))
    return final_path, raw_path


def write_review_summary(outcomes: list[PipelineOutcome], path: Path) -> Path:
    """Write one summary entry per quote, in input order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[ReviewSummaryEntry] = [o.to_summary_entry() for o in outcomes]
    _write_json(path, [entry.model_dump() for entry in entries])
    return path


def _raw_document(outcome: PipelineOutcome) -> dict | list:
    """The model's output, untouched where possible.

    When the response parsed we store the object itself. When it did not, we
    cannot store invalid JSON in a `.json` file, so the unparsable text is
    preserved verbatim inside an envelope that says so.
    """
    if outcome.raw_payload is not None:
        return outcome.raw_payload
    return {
        "_parse_error": True,
        "_note": "The model response was not valid JSON. Original text preserved below.",
        "_raw_response": outcome.raw_response,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
