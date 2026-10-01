"""Sensitive-looking values are assembled at runtime so this file passes its own scan."""

import subprocess
from pathlib import Path

import pytest

from crag.guard import compile_denylist, load_denylist, main, scan_text

AT = "@"
PLUS = "+"


def rules(text: str, terms: list[str] | None = None) -> list[str]:
    return [finding.rule for finding in scan_text(text, "sample", compile_denylist(terms or []))]


def test_flags_private_email() -> None:
    assert rules("write to jane.doe" + AT + "mailbox.test") == ["email address"]


@pytest.mark.parametrize(
    ("local_part", "domain"),
    [
        ("someone", "example.com"),
        ("someone", "example.org"),
        ("someone", "sub.example.net"),
        ("123+someone", "users.noreply.github.com"),
    ],
)
def test_allows_placeholder_and_noreply_emails(local_part: str, domain: str) -> None:
    assert rules("contact: " + local_part + AT + domain) == []


@pytest.mark.parametrize("number", ["43 660 1234567", "49 (30) 123-4567", "1 555 0100 200"])
def test_flags_international_phone_numbers(number: str) -> None:
    assert rules("call " + PLUS + number) == ["phone number"]


def test_ignores_short_numbers_and_versions() -> None:
    assert rules("fastapi==0.142.2, +3 retries, port 5432") == []


def test_flags_iban() -> None:
    assert rules("account " + "AT61" + " 1904 3002 3457 3201") == ["IBAN"]


def test_denylist_matches_whole_words_case_insensitively() -> None:
    assert rules("Met with ACME Corp today", ["acme corp"]) == ["private term from denylist"]
    assert rules("acmecorporation is fine", ["acme corp"]) == []


def test_reports_line_number_and_never_the_matched_value() -> None:
    secret = "jane.doe" + AT + "mailbox.test"
    [finding] = scan_text("first line\nmail " + secret, "notes.md", None)
    assert str(finding) == "notes.md:2: email address"
    assert secret not in str(finding)


def test_denylist_file_skips_blanks_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "denylist.txt"
    path.write_text("# companies\nAcme Corp\n\n  Globex  \n", encoding="utf-8")
    assert load_denylist(path) == ["Acme Corp", "Globex"]


def test_missing_denylist_means_no_private_terms(tmp_path: Path) -> None:
    assert load_denylist(tmp_path / "absent.txt") == []
    assert compile_denylist([]) is None


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    return tmp_path


def git(*args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t" + AT + "example.com", *args], check=True
    )


def test_staged_scan_blocks_personal_data(repo: Path) -> None:
    (repo / "clean.txt").write_text("nothing here\n", encoding="utf-8")
    (repo / "leak.txt").write_text("phone " + PLUS + "43 660 1234567\n", encoding="utf-8")
    git("add", ".")
    assert main(["staged"]) == 1


def test_staged_scan_passes_clean_changes(repo: Path) -> None:
    (repo / "clean.txt").write_text("nothing here\n", encoding="utf-8")
    git("add", ".")
    assert main(["staged"]) == 0


def test_staged_scan_uses_local_denylist(repo: Path) -> None:
    (repo / "data" / "pii").mkdir(parents=True)
    (repo / "data" / "pii" / "denylist.txt").write_text("Acme Corp\n", encoding="utf-8")
    (repo / "notes.md").write_text("Interview at Acme Corp\n", encoding="utf-8")
    git("add", "notes.md")
    assert main(["staged"]) == 1


def test_history_scan_checks_commit_messages(repo: Path) -> None:
    git("commit", "-q", "--allow-empty", "-m", "chore: fine")
    assert main(["history"]) == 0
    git("commit", "-q", "--allow-empty", "-m", "fix: mail jane" + AT + "mailbox.test")
    assert main(["history"]) == 1


def test_tracked_scan_ignores_binary_files(repo: Path) -> None:
    (repo / "image.bin").write_bytes(b"\0\1jane" + AT.encode() + b"mailbox.test")
    git("add", ".")
    assert main(["tracked"]) == 0


def test_message_scan(tmp_path: Path) -> None:
    message = tmp_path / "COMMIT_EDITMSG"
    message.write_text("feat: add search\n", encoding="utf-8")
    assert main(["message", str(message)]) == 0
    message.write_text("feat: call " + PLUS + "43 660 1234567\n", encoding="utf-8")
    assert main(["message", str(message)]) == 1
