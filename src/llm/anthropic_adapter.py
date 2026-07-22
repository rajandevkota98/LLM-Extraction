"""Real provider adapter.

The `anthropic` import is guarded so the CLI runs on a clean checkout with only
the base dependencies installed. Selecting this provider without the package (or
without a key) is a configuration error we surface immediately rather than a
crash halfway through a run.
"""

from __future__ import annotations

from src.llm.base import LLMError

MAX_TOKENS = 2048


class AnthropicAdapter:
    """Calls the Anthropic Messages API and returns the raw text block.

    No parsing, no repair, no retries. `src/components/extractor.py` owns the
    decision of what to do with a bad response.
    """

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise LLMError(
                'The anthropic package is not installed. Run `pip install -e ".[llm]"`, '
                "or use the mock adapter with `python main.py --mock`."
            ) from exc

        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Use `--mock` to run without a key.")

        self._client = anthropic.Anthropic(api_key=api_key)
        self.provider = "anthropic"
        self.model = model

    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text response verbatim."""
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise a wide variety
            raise LLMError(f"Anthropic call failed: {exc}") from exc

        return "".join(block.text for block in message.content if block.type == "text")
