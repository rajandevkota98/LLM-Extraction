"""Append-only audit log of every extraction call.

One JSON object per line in `llm_calls.jsonl`. This is deliberately separate from
the adapters: adapters make calls, this records that they happened. A logging
failure must never take down a pipeline run, so writes are best-effort.
"""

from __future__ import annotations

from pathlib import Path

from src.models import CallStatus, LLMCallRecord


class CallLog:
    """Writes one `LLMCallRecord` per line, creating the file on first use."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def record(
        self,
        *,
        quote_id: str,
        provider: str,
        model: str,
        input_artifact: str,
        output_artifact: str,
        status: CallStatus,
    ) -> None:
        """Append one call record. Never raises."""
        entry = LLMCallRecord(
            quote_id=quote_id,
            provider=provider,
            model=model,
            input_artifact=input_artifact,
            output_artifact=output_artifact,
            status=status,
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(entry.model_dump_json() + "\n")
        except OSError:  # pragma: no cover - the audit log is not worth failing a run over
            pass

    def reset(self) -> None:
        """Truncate the log so a run starts clean. Never raises."""
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:  # pragma: no cover
            pass
