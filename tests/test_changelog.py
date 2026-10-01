from datetime import date

import pytest

from crag.loaders.changelog import parse_changelog

CHANGELOG = """# Changelog — Sample Person

## 2026-03-02 — Acme Corp Platform Engineer: interview prep

Prep folder created. Cheat sheet attached to TRACK-7.

## 2026-03-01 — Acme Corp Platform Engineer: applied

Applied with the EN CV. TRACK-7 set to Applied.

## 2026-03-01 — Globex Data Lead: rejected

Rejected after the first call.
"""


def test_one_document_per_entry_with_date_title_and_issue_key() -> None:
    documents = parse_changelog(CHANGELOG)
    assert [(d.document_date, d.title, d.issue_key) for d in documents] == [
        (date(2026, 3, 2), "Acme Corp Platform Engineer: interview prep", "TRACK-7"),
        (date(2026, 3, 1), "Acme Corp Platform Engineer: applied", "TRACK-7"),
        (date(2026, 3, 1), "Globex Data Lead: rejected", None),
    ]
    assert documents[2].text == "Rejected after the first call."
    assert {d.collection for d in documents} == {"private"}
    assert {d.source_type for d in documents} == {"changelog"}


def test_source_keys_are_unique_and_stable() -> None:
    keys = [d.source_key for d in parse_changelog(CHANGELOG)]
    assert len(set(keys)) == len(keys)
    assert keys == [d.source_key for d in parse_changelog(CHANGELOG)]


def test_text_without_entries_gives_nothing() -> None:
    assert parse_changelog("# Changelog\n\nNothing yet.\n") == []


def test_malformed_date_is_rejected() -> None:
    with pytest.raises(ValueError, match="2026-13-01"):
        parse_changelog("## 2026-13-01 — Broken\n\nText.\n")
