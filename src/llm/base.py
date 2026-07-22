"""The LLM boundary.

Everything downstream of this module treats model output as untrusted text.
An adapter's only job is: take a prompt, return a string. It does not parse,
validate, or repair — those are deterministic concerns owned by
`src/components/`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Raised when the provider call itself fails (network, auth, quota)."""


@runtime_checkable
class LLMAdapter(Protocol):
    """Minimal interface every provider adapter satisfies.

    `provider` and `model` are metadata only — they are recorded in
    `llm_calls.jsonl` and never influence validation or review decisions.
    """

    provider: str
    model: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text response.

        Implementations must not attempt to parse or fix up the response;
        return exactly what came back so it can be persisted verbatim to
        `outputs/{quote_id}_raw.json`.

        Raises:
            LLMError: if the provider call fails.
        """
        ...
