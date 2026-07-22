"""The LLM boundary: adapters, prompts, and the call audit log.

`get_adapter` is the only place a provider is chosen. Everything downstream
depends on the `LLMAdapter` protocol, never on a concrete provider.
"""

from __future__ import annotations

from src.config import PROVIDER_ANTHROPIC, PROVIDER_OPENROUTER, Settings
from src.llm.base import LLMAdapter, LLMError
from src.llm.call_log import CallLog
from src.llm.mock_adapter import MockAdapter

__all__ = ["CallLog", "LLMAdapter", "LLMError", "MockAdapter", "get_adapter"]


def get_adapter(settings: Settings) -> LLMAdapter:
    """Return the adapter named by settings.

    Defaults to the mock: `Settings.from_env` only resolves a real provider when
    an API key is actually present, so a clean checkout always runs.
    """
    if settings.provider == PROVIDER_OPENROUTER:
        from src.llm.openrouter_adapter import OpenRouterAdapter

        return OpenRouterAdapter(
            api_key=settings.api_key or "",
            model=settings.model,
            base_url=settings.base_url or "",
            site_url=settings.site_url,
            app_name=settings.app_name,
            sort=settings.sort,
        )
    if settings.provider == PROVIDER_ANTHROPIC:
        from src.llm.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(api_key=settings.api_key or "", model=settings.model)
    return MockAdapter()
