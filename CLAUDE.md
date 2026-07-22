# CLAUDE.md

Notes for working in this repo.

## What this is

A small service that extracts structured pricing from messy supplier quote text
using an LLM, then validates and normalizes the result in deterministic Python.
Built as a ~60-minute take-home; an evaluator runs it from a clean checkout and
may swap `quotes.json` for different quote text of the same shape.

What is being judged is the boundary between model output and deterministic code,
plus validation discipline and handling of ambiguous data — not feature count.

## The rule that governs everything

**The LLM extracts. Deterministic Python decides.**

The prompt in `src/llm/prompts.py` asks only for extraction. It explicitly tells
the model *not* to convert `"around 3 weeks"` into a number, because that
conversion belongs in `normalizer.py` where it is reproducible and testable.

Never move a rule into the prompt to make a test pass. If a check belongs to
validation, normalization, or the review decision, it lives in `src/components/`
and runs without a model. The model's own `needs_review` flag is read as one
input signal among several — it can escalate a quote to review, it can never
clear one.

## Pipeline

    LOAD_INPUT -> LLM_EXTRACTION -> SCHEMA_VALIDATION -> NORMALIZATION
                                 -> REVIEW_DECISION -> RESULTS_WRITTEN

| Stage | Module |
|---|---|
| LOAD_INPUT | `src/components/loader.py` |
| LLM_EXTRACTION | `src/components/extractor.py` (+ `src/llm/`) |
| SCHEMA_VALIDATION | `src/components/validator.py` |
| NORMALIZATION | `src/components/normalizer.py` |
| REVIEW_DECISION | `src/components/reviewer.py` |
| RESULTS_WRITTEN | `src/components/writer.py` |

`src/pipeline.py` is the only place the stages are composed. Each stage is a
plain function from data to data, so any one can be tested without a model, a
filesystem, or the others.

Validation runs twice: once on the raw payload (the required stage) and again
after normalization. The post-normalization pass drives the review decision,
because normalization is the sanctioned repair step — a model returning `"£"` is
a schema fault before it runs and a resolved `GBP` afterwards. Faults that
normalization cannot fix survive both passes and still count.

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/python main.py --input quotes.json    # mock unless a key is configured
.venv/bin/python main.py --mock                 # force mock even with keys set
.venv/bin/python main.py --provider openrouter  # OPENROUTER_API + [openrouter] extra
.venv/bin/python main.py --provider anthropic   # ANTHROPIC_API_KEY + [anthropic] extra

.venv/bin/pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format .
.venv/bin/pre-commit install && .venv/bin/pre-commit run --all-files

.venv/bin/pip install -e ".[api]" && .venv/bin/uvicorn src.api.app:app --reload

docker compose up api                       # API on :8000
docker compose run --rm cli                 # one-shot pipeline into ./outputs
LLM_PROVIDER=mock docker compose run --rm cli   # force the offline path
```

Code must pass `ruff check` and `ruff format --check` before commit; the
pre-commit hooks enforce both.

## Layout

```
main.py                      CLI entry point
src/config.py                env-driven Settings; mock is the default
src/models.py                Pydantic models for every boundary
src/pipeline.py              stage orchestration
src/components/              one module per pipeline stage
src/llm/base.py               LLMAdapter protocol — the provider boundary
src/llm/prompts.py            the actual prompt text
src/llm/mock_adapter.py       offline stand-in model
src/llm/openrouter_adapter.py default real provider, guarded import
src/llm/anthropic_adapter.py  direct Anthropic, guarded import
src/llm/call_log.py           llm_calls.jsonl
src/api/app.py               optional FastAPI wrapper
tests/                       validator, normalizer, reviewer, config
Dockerfile                   multi-stage, non-root, venv-only runtime
docker-compose.yml           `api` service; `cli` behind a profile
.github/workflows/ci.yml     ruff + pytest on push and PR
```

## Conventions

- Pydantic models at every boundary; raw model output stays a plain `dict` until
  it has been validated and normalized. Parsing untrusted output straight into a
  strict model turns a recoverable data problem into an exception and destroys
  the error detail we have to report.
- Validation accumulates `list[str]` and never raises. Every fault reaches
  `review_summary.json`, not just the first one. Errors carry a path prefix
  (`items[1].quantity: ...`).
- `loader.py` is the only stage allowed to fail hard — with no input there is no
  run. Everything after it degrades into a review flag.
- Review reasons are sentences an operator can act on, not rule ids.
- Nothing invents data. Where the text does not support a value it stays null and
  the reviewer decides what that means. Two standing examples: a bare `$` is not
  assumed to be USD (four currencies use it), and a relative expiry is never
  resolved against wall-clock time (the quote's send date is unknown).
- Tests make no network calls and touch no API keys.
- Optional dependencies (`fastapi`, `openai`, `anthropic`) are imported behind
  guards so the base install always runs.
- Provider selection is key-driven, not flag-driven: OpenRouter if
  `OPENROUTER_API` is set, else Anthropic, else mock. A provider requested
  without its key degrades to the mock rather than crashing — a half-configured
  `.env` should still produce output to look at.
- Model ids are passed to the provider verbatim, never rewritten. Bad ids are
  rejected up front with a message naming the fix (missing `provider/` prefix,
  or a retired `:nitro` / `:floor` suffix that `OPENROUTER_SORT` now replaces).
- `.dockerignore` must keep excluding `.env`. Docker does not read `.gitignore`,
  and a key copied into a layer survives `docker history` even if a later layer
  removes the file.
- `[tool.setuptools] py-modules = ["main"]` is load-bearing: listing `packages`
  explicitly disables auto-discovery, so without it a non-editable install ships
  no `main` module and the `llm-extraction` console script breaks. Only
  `pip install -e .` hides that.

## Do not

- Hardcode answers for the sample quotes, or key any logic to `Q-100x`. The
  evaluator will swap the file.
- Let the LLM make validation, normalization, or review decisions.
- Require a paid API to run or understand the project — the mock adapter is the
  default whenever no provider key is set, and `.env.example` ships every key
  commented out so copying it cannot select a provider by accident.
- Add a workflow engine, a database, retry/backoff, or auth. Out of scope.
- Treat `needs_review: true` as a failure. It is a normal outcome; the CLI still
  exits 0.

## Current state

Everything described above is implemented and verified: `main.py` runs
end-to-end on the mock, 89 tests pass, `ruff check` and `ruff format --check` are
clean. The three deterministic stages are fully implemented, not stubbed.

Two invariants the tests exist to hold, both learned from bugs that shipped once:
a quote id is written out as a filename, so `loader.py` rejects anything that is
not one plain path segment; and a number the normalizer cannot read unambiguously
stays a string rather than becoming a plausible wrong number, because a finite,
non-negative price passes every check downstream of it.

`src/llm/mock_adapter.py` is a crude regex stand-in for a model, not a parser to
build on. It is deliberately imperfect — leaving `"3 weeks"` unresolved, letting
a bare `$` through — so the downstream stages do real work on every run. If you
extend the pipeline, do not improve the mock to make an output look better; fix
the deterministic stage that should have handled it.

The OpenRouter path is verified against a local stub server (correct endpoint,
bearer auth, attribution headers, `provider.sort` body field, system+user
messages) but has not been run against the live API. The direct `anthropic` path
is written and untested.
