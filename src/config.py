"""Environment-driven settings.

Fully implemented. The only interesting rule here: the mock adapter is the
default. We only select a real provider when an API key is actually present, so
a clean checkout runs end-to-end with no credentials.

Provider precedence when nothing is stated explicitly: OpenRouter, then
Anthropic, then mock. OpenRouter comes first because it is the gateway this
project is configured around -- one key reaches every model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is a declared dependency, but never let a missing .env break startup.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - trivial fallback

    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


PROVIDER_MOCK = "mock"
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_ANTHROPIC = "anthropic"

DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Provider routing preference, the modern replacement for the old `:nitro` /
# `:floor` model suffixes. Sent as `provider.sort` when set.
OPENROUTER_SORTS = ("throughput", "price", "latency")

# `OPENROUTER_API` is the name used in .env.example; the longer form is what most
# tooling exports, so accept either rather than making anyone rename a key.
OPENROUTER_KEY_VARS = ("OPENROUTER_API", "OPENROUTER_API_KEY")
# `MODEL` is provider-agnostic and wins; `LLM_MODEL` is kept for older configs.
MODEL_VARS = ("MODEL", "LLM_MODEL")


def _first_env(*names: str) -> str | None:
    """Return the first of `names` set to a non-empty value."""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration for one pipeline run."""

    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    site_url: str | None
    app_name: str | None
    sort: str | None
    input_path: Path
    output_dir: Path
    review_summary_path: Path
    call_log_path: Path

    @property
    def use_mock(self) -> bool:
        """True when the deterministic fixture adapter should be used."""
        return self.provider == PROVIDER_MOCK

    @classmethod
    def from_env(
        cls,
        *,
        input_path: str | Path = "quotes.json",
        output_dir: str | Path = "outputs",
        review_summary_path: str | Path = "review_summary.json",
        call_log_path: str | Path = "llm_calls.jsonl",
        provider: str | None = None,
        model: str | None = None,
        force_mock: bool = False,
    ) -> Settings:
        """Build settings from the environment, with explicit CLI overrides winning.

        Resolution order for `provider`: explicit argument -> `LLM_PROVIDER` env
        var -> whichever API key is present (OpenRouter first) -> "mock".
        `force_mock` short-circuits everything, so `--mock` always works even
        with keys configured.
        """
        load_dotenv()

        openrouter_key = _first_env(*OPENROUTER_KEY_VARS)
        anthropic_key = _first_env("ANTHROPIC_API_KEY")
        resolved = _resolve_provider(
            requested=provider or _first_env("LLM_PROVIDER"),
            force_mock=force_mock,
            openrouter_key=openrouter_key,
            anthropic_key=anthropic_key,
        )

        if resolved == PROVIDER_OPENROUTER:
            api_key = openrouter_key
            resolved_model = model or _first_env(*MODEL_VARS) or DEFAULT_OPENROUTER_MODEL
        elif resolved == PROVIDER_ANTHROPIC:
            api_key = anthropic_key
            # `MODEL` is shared with OpenRouter, whose ids are always
            # "provider/model". Handing one of those to the Anthropic API is a
            # guaranteed 404 that surfaces as every quote failing to extract, so a
            # slug-shaped value is ignored here rather than sent. An explicit
            # `--model` still wins: that is a choice, not a leftover .env line.
            generic = _first_env(*MODEL_VARS)
            resolved_model = (
                model
                or _first_env("ANTHROPIC_MODEL")
                or (generic if generic and "/" not in generic else None)
                or DEFAULT_ANTHROPIC_MODEL
            )
        else:
            api_key = None
            resolved_model = model or PROVIDER_MOCK

        return cls(
            provider=resolved,
            model=resolved_model,
            api_key=api_key,
            base_url=_first_env("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL,
            site_url=_first_env("OPENROUTER_SITE_URL"),
            app_name=_first_env("OPENROUTER_APP_NAME"),
            sort=_normalize_sort(_first_env("OPENROUTER_SORT")),
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            review_summary_path=Path(review_summary_path),
            call_log_path=Path(call_log_path),
        )


def _normalize_sort(value: str | None) -> str | None:
    """Accept a routing preference, ignoring anything OpenRouter would reject.

    An unrecognised value is dropped rather than passed through: a bad `sort`
    fails the whole request, and a routing preference is never worth losing an
    extraction over.
    """
    if not value:
        return None
    lowered = value.strip().lower()
    return lowered if lowered in OPENROUTER_SORTS else None


def _resolve_provider(
    *,
    requested: str | None,
    force_mock: bool,
    openrouter_key: str | None,
    anthropic_key: str | None,
) -> str:
    """Pick a provider, degrading to the mock rather than failing on a missing key."""
    if force_mock:
        return PROVIDER_MOCK

    keys = {PROVIDER_OPENROUTER: openrouter_key, PROVIDER_ANTHROPIC: anthropic_key}

    if requested:
        requested = requested.strip().lower()
        if requested == PROVIDER_MOCK:
            return PROVIDER_MOCK
        # A real provider asked for without credentials degrades to the mock
        # instead of crashing, so a misconfigured .env still produces output.
        return requested if keys.get(requested) else PROVIDER_MOCK

    if openrouter_key:
        return PROVIDER_OPENROUTER
    if anthropic_key:
        return PROVIDER_ANTHROPIC
    return PROVIDER_MOCK
