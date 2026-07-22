"""OpenRouter provider adapter.

OpenRouter exposes an OpenAI-compatible Chat Completions API at
`https://openrouter.ai/api/v1`, so we drive it with the `openai` SDK pointed at
that base URL rather than hand-rolling HTTP. The import is guarded so the base
install stays free of provider SDKs.

Same contract as every adapter: take a prompt, return the raw text. No parsing,
no repair, no retries -- `src/components/extractor.py` owns what happens to a bad
response.
"""

from __future__ import annotations

from src.llm.base import LLMError

MAX_TOKENS = 2048
MODELS_URL = "https://openrouter.ai/models"


class OpenRouterAdapter:
    """Calls one model through OpenRouter and returns its raw text."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        *,
        site_url: str | None = None,
        app_name: str | None = None,
        sort: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise LLMError(
                'The openai package is not installed. Run `pip install -e ".[openrouter]"`, '
                "or use the mock adapter with `python main.py --mock`."
            ) from exc

        if not api_key:
            raise LLMError(
                "OPENROUTER_API is not set. Add it to .env, or use `--mock` to run without a key."
            )

        validate_model_id(model)

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        # Optional attribution headers. They put the app on OpenRouter's
        # leaderboards and have no effect on extraction, so they are omitted
        # entirely when unset rather than sent empty.
        self._headers = {
            key: value
            for key, value in (("HTTP-Referer", site_url), ("X-Title", app_name))
            if value
        }
        # Provider routing preference, e.g. sort by throughput. Omitted when
        # unset so the request body stays exactly what OpenRouter expects.
        self._extra_body = {"provider": {"sort": sort}} if sort else None
        self.provider = "openrouter"
        self.model = model

    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text response verbatim."""
        # Reading the response is inside the try on purpose: OpenRouter fronts many
        # upstream providers and their response shapes vary, so an unexpected one
        # has to surface as LLMError rather than as a raw AttributeError escaping
        # the adapter contract and aborting the whole run.
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                extra_headers=self._headers or None,
                extra_body=self._extra_body,
            )
            if not response.choices:
                # OpenRouter surfaces upstream provider failures this way rather
                # than as an HTTP error, so an empty choices list is a real outcome.
                raise LLMError(f"OpenRouter returned no choices for model '{self.model}'.")
            return response.choices[0].message.content or ""
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise a wide variety
            raise LLMError(f"OpenRouter call failed: {exc}") from exc


# Suffixes OpenRouter no longer publishes as model ids. Routing preference moved
# to the `provider.sort` field, so these now 404 instead of doing anything.
RETIRED_VARIANTS = {"nitro": "throughput", "floor": "price"}


def validate_model_id(model: str) -> None:
    """Reject a slug that OpenRouter cannot route.

    Ids are `provider/model`, optionally with a published variant suffix such as
    `:free`. Catching a bad id here turns a confusing 404 from the API into a
    message that says what to fix.
    """
    if "/" not in model:
        raise LLMError(
            f"'{model}' is not a valid OpenRouter model id. Ids are "
            f"'provider/model' (for example 'openai/{model}'). "
            f"Browse the exact slugs at {MODELS_URL}."
        )

    suffix = model.rsplit(":", 1)[-1] if ":" in model else ""
    if suffix in RETIRED_VARIANTS:
        base = model.rsplit(":", 1)[0]
        raise LLMError(
            f"'{model}' uses the retired ':{suffix}' suffix, which OpenRouter no "
            f"longer publishes as a model id. Use '{base}' and set "
            f"OPENROUTER_SORT={RETIRED_VARIANTS[suffix]} to get the same routing."
        )
