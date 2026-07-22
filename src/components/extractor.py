"""Stage 2 -- LLM_EXTRACTION.

One call per quote, then a best-effort attempt to get JSON out of whatever came
back. Nothing here interprets the *content* of the response: a value being wrong,
missing, or absurd is the validator's problem. This module only answers "did we
get a JSON object at all?".
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.llm.base import LLMAdapter, LLMError
from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass(frozen=True)
class ExtractionAttempt:
    """What one call produced."""

    raw_response: str
    payload: dict | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.payload is not None


def extract(adapter: LLMAdapter, quote_id: str, quote_text: str) -> ExtractionAttempt:
    """Call the model once and try to recover a JSON object from the response."""
    user_prompt = build_user_prompt(quote_id, quote_text)
    try:
        raw = adapter.complete(SYSTEM_PROMPT, user_prompt)
    except LLMError as exc:
        return ExtractionAttempt(raw_response="", payload=None, error=f"LLM call failed: {exc}")

    payload, error = parse_response(raw)
    return ExtractionAttempt(raw_response=raw, payload=payload, error=error)


def parse_response(raw: str) -> tuple[dict | None, str | None]:
    """Recover a JSON object from raw model text.

    The ladder, cheapest first:
      1. parse the whole response
      2. strip a markdown code fence and parse again
      3. isolate the first balanced `{...}` span and parse that
    """
    if not raw or not raw.strip():
        return None, "Model returned an empty response."

    for candidate in _candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, None
        return None, f"Model returned JSON {type(parsed).__name__}, expected an object."

    return None, "Model response did not contain a parsable JSON object."


def _candidates(raw: str) -> list[str]:
    text = raw.strip()
    candidates = [text]

    fenced = _strip_code_fence(text)
    if fenced != text:
        candidates.append(fenced)

    span = _first_balanced_object(text)
    if span is not None:
        candidates.append(span)
    return candidates


def _strip_code_fence(text: str) -> str:
    """Drop a surrounding ``` or ```json fence if one is present."""
    if not text.startswith("```"):
        return text
    body = text[3:]
    if body[:4].lower() == "json":
        body = body[4:]
    closing = body.rfind("```")
    return (body[:closing] if closing != -1 else body).strip()


def _first_balanced_object(text: str) -> str | None:
    """Return the first balanced `{...}` span, ignoring braces inside strings.

    Deliberately a scan and not a regex: regular expressions cannot match
    balanced delimiters, and the usual `\\{.*\\}` shortcut swallows or truncates
    anything nested.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
