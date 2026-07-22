# Supplier quote extraction

Extracts structured pricing from free-form supplier quote text using an LLM, then
validates and normalizes the result in deterministic Python.

The model extracts. Code decides.

## Run it

No API key required — the pipeline defaults to an offline mock adapter.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python main.py --input quotes.json
```

```
provider: mock  model: regex-stub-v1
note: running on the offline mock adapter (no API key required).

  Q-1001  clean         success
  Q-1002  clean         success
  Q-1003  NEEDS REVIEW  success
            - Quote expiry is expressed relative to an unknown date and could not be resolved safely.

3 quote(s): 2 clean, 1 needing review, 0 unparsable.
```

Exit code is 0 whenever the pipeline ran. Needing review is a normal outcome, not
a failure. Non-zero means the input was unreadable (2) or the provider could not
be reached (3).

### With a real model (OpenRouter)

```bash
.venv/bin/pip install -e ".[openrouter]"
cp .env.example .env          # set OPENROUTER_API and MODEL
.venv/bin/python main.py --input quotes.json
```

`.env`:

```ini
OPENROUTER_API="sk-or-..."
MODEL="openai/gpt-oss-120b"
OPENROUTER_SORT=throughput          # optional: throughput | price | latency
OPENROUTER_SITE_URL=https://...     # optional attribution
OPENROUTER_APP_NAME=quote-extraction
```

OpenRouter speaks the OpenAI Chat Completions API, so the adapter drives it with
the `openai` SDK pointed at `https://openrouter.ai/api/v1`. Anthropic is still
supported directly — `pip install -e ".[anthropic]"` and set `ANTHROPIC_API_KEY`.

Provider selection lives in `src/config.py` and needs no flags: OpenRouter if
`OPENROUTER_API` is set, else Anthropic if `ANTHROPIC_API_KEY` is set, else the
mock. Asking for a provider whose key is missing degrades to the mock rather than
crashing, so a half-configured `.env` still produces output to look at.
`--mock` forces the mock even with keys present.

Two model-id guards fail fast with an actionable message instead of a 404:

- a bare name (`gpt-oss-120b`) — OpenRouter ids are always `provider/model`
- a retired variant suffix (`:nitro`, `:floor`) — routing preference moved to the
  `provider.sort` field, which is what `OPENROUTER_SORT` sets

### Optional HTTP interface

```bash
.venv/bin/pip install -e ".[api]"
.venv/bin/uvicorn src.api.app:app --reload
curl -s localhost:8000/extract -H 'content-type: application/json' \
  -d '{"id":"Q-1","text":"Acme Ltd. 10 widgets at EUR 4.50 each. Ships in 2 weeks."}'
```

The CLI is the primary interface; the API is a thin wrapper over the same
`process_quote` function.

### Docker

Nothing to install but Docker, and no API key needed — without one the container
falls back to the same offline mock adapter.

```bash
docker compose up api                  # HTTP interface on :8000
docker compose run --rm cli            # one-shot pipeline, writes ./outputs
```

Force the offline path regardless of what is in `.env`:

```bash
LLM_PROVIDER=mock docker compose run --rm cli
```

The `cli` service writes to a bind-mounted `./outputs`, so it runs as uid
`1000:1000` to keep the files yours rather than root's. If your uid differs:

```bash
export DOCKER_USER="$(id -u):$(id -g)"
```

Two details worth knowing. The container steers `review_summary.json` and
`llm_calls.jsonl` into `outputs/` as well, so a single mount carries every
artefact. And `.dockerignore` excludes `.env` — Docker does not read
`.gitignore`, and a key copied into a layer survives `docker history` even if a
later layer deletes the file.

## Output

| Artifact | Contents |
|---|---|
| `outputs/{id}.json` | final normalized record |
| `outputs/{id}_raw.json` | the model's output, verbatim |
| `review_summary.json` | one entry per quote: `needs_review`, `validation_errors`, `review_reasons` |
| `llm_calls.jsonl` | one audit record per extraction call |

Every quote produces every artifact, including quotes whose model response could
not be parsed. Silence is the one outcome an operator cannot act on.

## Pipeline

    LOAD_INPUT -> LLM_EXTRACTION -> SCHEMA_VALIDATION -> NORMALIZATION
                                 -> REVIEW_DECISION -> RESULTS_WRITTEN

One module per stage under `src/components/`, composed only in `src/pipeline.py`.
Each stage is a plain function from data to data, so any one is testable without
a model, a filesystem, or the others.

## Design notes

**The boundary.** The prompt asks only for extraction. It explicitly instructs the
model *not* to convert `"around 3 weeks"` into a number — that conversion lives in
`normalizer.py`, where it is deterministic, inspectable and unit-tested. Every
validation and review rule is ordinary Python. The model's own `needs_review` flag
is treated as one input signal: it can escalate a quote to review, never clear one.

**Untrusted until proven otherwise.** Model output stays a plain `dict` until it
has been validated and normalized. Parsing it straight into a strict Pydantic model
would turn a recoverable data problem into an exception and destroy exactly the
error detail we are required to report. Validation accumulates a list of errors and
never raises, so the summary shows every fault rather than the first one.

**Validation runs twice.** Once on the raw payload — the required stage — and again
after normalization, and it is the second pass that drives the review decision.
Normalization is the sanctioned repair step: a model returning `"£"` is a schema
fault before it runs and a resolved `GBP` afterwards. Holding the pre-normalization
error against the record would flag clean quotes for a problem that no longer
exists. Faults normalization cannot fix survive both passes and still count.

**Two deliberate refusals.** These are the judgment calls that matter most:

- A bare `$` is *not* assumed to be USD. Four currencies use that symbol, so it
  resolves only when the text also names one outright (`Currency USD`, `CAD`, …),
  and stays unresolved and flagged otherwise. `€`, `£` and `₹` do resolve, because
  each belongs to exactly one currency.
- A relative expiry is *never* resolved against today's date. "Expires next Friday"
  has no safe answer without the quote's send date, and any date we produced would
  be fabricated. It is detected, left null, and flagged.

**Failure has a path.** Malformed model output goes through a fallback ladder —
parse, strip code fences, then isolate the first balanced `{...}` — before being
recorded as `parse_error`. The balanced-brace step is a counting scan rather than a
regex, because regular expressions cannot match balanced delimiters and the usual
`\{.*\}` shortcut breaks on nested objects.

**The mock adapter is honest about what it is.** It is a crude bundle of regexes
standing where a model would be, not a parser to build on, and it is not keyed to
the sample quotes. It is deliberately imperfect in the same ways a model is —
leaving `"3 weeks"` unresolved, reporting a bare `$` — so the deterministic stages
do real work on every run rather than rubber-stamping pre-cleaned input.

**Known trade-off.** Because the model's uncertainty flag escalates, a quote can be
marked for review with "the extraction model flagged this quote as uncertain" as its
only reason, even after normalization resolved the underlying ambiguity. In a
sourcing workflow, erring toward a two-second human glance beats silently storing a
value the extractor doubted — but it is a choice, and dropping that rule is a
one-line change in `reviewer.py`.

## Tests

```bash
.venv/bin/pytest -q      # 89 tests, no network, no API keys
```

Covering the cases a naive implementation gets wrong: `True` passing as a quantity
because `bool` subclasses `int`; `2026-02-31` passing a regex-only date check; a
lead-time phrase overwriting a number the model actually read off the page;
`"valid for 30 days"` being mistaken for a delivery time.

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pre-commit install
.venv/bin/pre-commit run --all-files
.venv/bin/ruff check . && .venv/bin/ruff format .
```

CI (`.github/workflows/ci.yml`) runs two jobs on every push to `main` and every
pull request: `ruff check` + `ruff format --check`, and the test suite. Both
install from the `dev` extra rather than a prebuilt action, so the linter version
in CI is the one pinned in `pyproject.toml` and cannot drift from pre-commit.
No secrets are configured — the suite is offline by design, and a job that needed
a provider to be up would fail for reasons unrelated to the commit.

## Scope

No workflow engine, database, retry/backoff, or auth. Six stages, one module each,
kept to what the problem actually requires.
