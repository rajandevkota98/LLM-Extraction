"""Provider selection has to be predictable, and it has to fail soft.

A misconfigured .env should still produce output to look at, not a traceback.
These tests touch no network and construct no clients.
"""

from __future__ import annotations

import pytest

from src.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    PROVIDER_ANTHROPIC,
    PROVIDER_MOCK,
    PROVIDER_OPENROUTER,
    Settings,
)
from src.llm.base import LLMError
from src.llm.openrouter_adapter import validate_model_id

ENV_VARS = (
    "OPENROUTER_API",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "LLM_PROVIDER",
    "MODEL",
    "LLM_MODEL",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_SITE_URL",
    "OPENROUTER_APP_NAME",
    "OPENROUTER_SORT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from a known-empty environment and no .env on disk."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("src.config.load_dotenv", lambda *a, **k: False)


# --- provider selection ---------------------------------------------------- #


def test_no_keys_falls_back_to_mock():
    settings = Settings.from_env()
    assert settings.provider == PROVIDER_MOCK
    assert settings.use_mock is True


def test_openrouter_key_selects_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API", "sk-or-test")
    settings = Settings.from_env()
    assert settings.provider == PROVIDER_OPENROUTER
    assert settings.api_key == "sk-or-test"


def test_long_form_openrouter_key_is_also_accepted(monkeypatch):
    """People paste OPENROUTER_API_KEY; do not make them rename it."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert Settings.from_env().provider == PROVIDER_OPENROUTER


def test_openrouter_wins_when_both_keys_are_present(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API", "sk-or-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert Settings.from_env().provider == PROVIDER_OPENROUTER


def test_anthropic_key_alone_selects_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert Settings.from_env().provider == PROVIDER_ANTHROPIC


def test_force_mock_beats_every_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API", "sk-or-test")
    assert Settings.from_env(force_mock=True).provider == PROVIDER_MOCK


def test_requesting_a_provider_without_its_key_degrades_to_mock(monkeypatch):
    """A misconfigured .env still produces output rather than a crash."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert Settings.from_env().provider == PROVIDER_MOCK


def test_explicit_provider_argument_beats_the_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API", "sk-or-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert Settings.from_env(provider="anthropic").provider == PROVIDER_ANTHROPIC


# --- model and headers ----------------------------------------------------- #


def test_model_env_var_is_used(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API", "sk-or-test")
    monkeypatch.setenv("MODEL", "anthropic/claude-opus-4.8")
    assert Settings.from_env().model == "anthropic/claude-opus-4.8"


def test_model_defaults_when_unset(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API", "sk-or-test")
    assert Settings.from_env().model == DEFAULT_OPENROUTER_MODEL


def test_anthropic_ignores_an_openrouter_slug_left_in_model(monkeypatch):
    """`MODEL` is shared with OpenRouter, whose ids are always 'provider/model'.

    Sending one to the Anthropic API is a guaranteed 404 that surfaces as every
    quote failing to extract, which reads like a broken pipeline rather than a
    stale .env line.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("MODEL", "openai/gpt-oss-120b")
    assert Settings.from_env().model == DEFAULT_ANTHROPIC_MODEL


def test_anthropic_still_honours_a_plain_model_name(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("MODEL", "claude-sonnet-5")
    assert Settings.from_env().model == "claude-sonnet-5"


def test_anthropic_model_var_and_cli_override_both_win_over_the_generic_one(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-8")
    assert Settings.from_env().model == "claude-opus-4-8"
    # An explicit --model is a choice, not a leftover, even when slug-shaped.
    assert Settings.from_env(model="anthropic/claude-opus-4-8").model == "anthropic/claude-opus-4-8"


def test_cli_model_override_wins(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API", "sk-or-test")
    monkeypatch.setenv("MODEL", "openai/gpt-oss-120b:nitro")
    assert Settings.from_env(model="meta-llama/llama-3.3-70b").model == "meta-llama/llama-3.3-70b"


def test_attribution_headers_are_none_when_unset(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API", "sk-or-test")
    settings = Settings.from_env()
    assert settings.site_url is None
    assert settings.app_name is None
    assert settings.sort is None
    assert settings.base_url == "https://openrouter.ai/api/v1"


def test_routing_sort_is_accepted_case_insensitively(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_SORT", "Throughput")
    assert Settings.from_env().sort == "throughput"


def test_unknown_routing_sort_is_dropped_not_forwarded(monkeypatch):
    """A bad sort fails the whole request; never lose an extraction over it."""
    monkeypatch.setenv("OPENROUTER_API", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_SORT", "fastest-please")
    assert Settings.from_env().sort is None


# --- model id validation --------------------------------------------------- #


def test_bare_model_name_is_rejected_with_a_useful_message():
    """OpenRouter ids are provider/model; a bare name 404s confusingly."""
    with pytest.raises(LLMError) as exc:
        validate_model_id("gpt-oss-120b")
    assert "openai/gpt-oss-120b" in str(exc.value)


def test_retired_nitro_suffix_is_rejected_with_the_replacement():
    """`:nitro` is no longer a published id; routing moved to provider.sort."""
    with pytest.raises(LLMError) as exc:
        validate_model_id("openai/gpt-oss-120b:nitro")
    message = str(exc.value)
    assert "openai/gpt-oss-120b" in message
    assert "OPENROUTER_SORT=throughput" in message


def test_published_model_ids_pass():
    validate_model_id("openai/gpt-oss-120b")
    validate_model_id("openai/gpt-oss-20b:free")
    validate_model_id("anthropic/claude-opus-4.8")
