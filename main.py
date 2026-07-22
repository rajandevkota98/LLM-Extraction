"""CLI entry point.

    python main.py --input quotes.json

Exits 0 when the pipeline ran, even if every quote needs review -- needing review
is a normal outcome, not a failure. A non-zero exit means the input could not be
read or the provider could not be reached.
"""

from __future__ import annotations

import argparse
import sys

from src.components.loader import InputError
from src.config import Settings
from src.llm import get_adapter
from src.llm.base import LLMError
from src.models import PipelineOutcome
from src.pipeline import run, summarize

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_PROVIDER_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Extract structured pricing from supplier quote text.",
    )
    parser.add_argument("--input", default="quotes.json", help="Path to quotes.json.")
    parser.add_argument("--output-dir", default="outputs", help="Where per-quote JSON is written.")
    parser.add_argument(
        "--review-summary", default="review_summary.json", help="Path for the review summary."
    )
    parser.add_argument("--call-log", default="llm_calls.jsonl", help="Path for the call log.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force the offline mock adapter, even if an API key is present.",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "openrouter", "anthropic"],
        help="Override provider selection. Default: whichever API key is present.",
    )
    parser.add_argument(
        "--model",
        help="Override the model id, e.g. openai/gpt-oss-120b:nitro for OpenRouter.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = Settings.from_env(
        input_path=args.input,
        output_dir=args.output_dir,
        review_summary_path=args.review_summary,
        call_log_path=args.call_log,
        provider=args.provider,
        model=args.model,
        force_mock=args.mock,
    )

    try:
        adapter = get_adapter(settings)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_PROVIDER_ERROR

    print(f"provider: {settings.provider}  model: {getattr(adapter, 'model', '-')}")
    if settings.use_mock:
        print("note: running on the offline mock adapter (no API key required).\n")

    try:
        outcomes = run(settings, adapter=adapter)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    _report(outcomes, settings)
    return EXIT_OK


def _report(outcomes: list[PipelineOutcome], settings: Settings) -> None:
    width = max((len(o.quote_id) for o in outcomes), default=8)
    for outcome in outcomes:
        verdict = "NEEDS REVIEW" if outcome.needs_review else "clean"
        print(f"  {outcome.quote_id:<{width}}  {verdict:<12}  {outcome.status}")
        for reason in outcome.review_reasons:
            print(f"  {'':<{width}}    - {reason}")

    counts = summarize(outcomes)
    print(
        f"\n{counts['total']} quote(s): {counts['clean']} clean, "
        f"{counts['needs_review']} needing review, {counts['parse_errors']} unparsable."
    )
    print(f"wrote {settings.output_dir}/, {settings.review_summary_path}, {settings.call_log_path}")


if __name__ == "__main__":
    raise SystemExit(main())
