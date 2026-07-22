"""The loader is the one stage allowed to fail hard, so what it lets through
matters. An id it accepts becomes a filename in `outputs/` without further
checking, and the evaluator supplies the file.
"""

from __future__ import annotations

import json

import pytest

from src.components.loader import InputError, load_quotes


def _write(tmp_path, records) -> object:
    path = tmp_path / "quotes.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_ordinary_ids_are_accepted(tmp_path):
    path = _write(tmp_path, [{"id": "Q-1001", "text": "10 widgets at EUR 4.50 each."}])
    quotes = load_quotes(path)
    assert [q.id for q in quotes] == ["Q-1001"]


@pytest.mark.parametrize("bad_id", ["../pwned", "outputs/../../etc/x", "A/B", "A\\B", "..", "."])
def test_ids_that_would_escape_the_output_directory_are_rejected(tmp_path, bad_id):
    """`../pwned` writes outside `outputs/`; `A/B` fails mid-run on a missing dir."""
    path = _write(tmp_path, [{"id": bad_id, "text": "10 widgets at EUR 4.50 each."}])
    with pytest.raises(InputError, match="unusable 'id'"):
        load_quotes(path)


def test_control_characters_and_overlong_ids_are_rejected(tmp_path):
    path = _write(tmp_path, [{"id": "Q\x00-1", "text": "widgets"}])
    with pytest.raises(InputError, match="unusable 'id'"):
        load_quotes(path)

    path = _write(tmp_path, [{"id": "Q" * 200, "text": "widgets"}])
    with pytest.raises(InputError, match="unusable 'id'"):
        load_quotes(path)


def test_duplicate_ids_are_rejected(tmp_path):
    """Two quotes writing to one filename would leave only the second."""
    path = _write(tmp_path, [{"id": "Q-1", "text": "a"}, {"id": "Q-1", "text": "b"}])
    with pytest.raises(InputError, match="duplicate id"):
        load_quotes(path)
