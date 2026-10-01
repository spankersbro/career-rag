"""Block personal data and private terms from entering this public repository.

Findings report the rule and location only, never the matched text, because CI logs are public.
"""

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

DENYLIST_PATH = Path("data/pii/denylist.txt")

ALLOWED_EMAIL_DOMAINS = (
    "users.noreply.github.com",
    "example.com",
    "example.org",
    "example.net",
)

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})")
INTERNATIONAL_PHONE = re.compile(r"(?<![\w+])\+\d{1,3}[\s/()-]*\d(?:[\s/()-]*\d){6,}")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b")


@dataclass(frozen=True)
class Finding:
    location: str
    line_number: int
    rule: str

    def __str__(self) -> str:
        return f"{self.location}:{self.line_number}: {self.rule}"


def load_denylist(path: Path = DENYLIST_PATH) -> list[str]:
    if not path.exists():
        return []
    terms = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [term for term in terms if term and not term.startswith("#")]


def compile_denylist(terms: Iterable[str]) -> re.Pattern[str] | None:
    escaped = [re.escape(term) for term in terms]
    if not escaped:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)", re.IGNORECASE)


def _is_allowed_email_domain(domain: str) -> bool:
    domain = domain.lower()
    return any(
        domain == allowed or domain.endswith("." + allowed) for allowed in ALLOWED_EMAIL_DOMAINS
    )


def _line_rules(line: str, denylist: re.Pattern[str] | None) -> Iterator[str]:
    if any(not _is_allowed_email_domain(match.group(1)) for match in EMAIL.finditer(line)):
        yield "email address"
    if INTERNATIONAL_PHONE.search(line):
        yield "phone number"
    if IBAN.search(line):
        yield "IBAN"
    if denylist is not None and denylist.search(line):
        yield "private term from denylist"


def scan_text(text: str, location: str, denylist: re.Pattern[str] | None) -> list[Finding]:
    return [
        Finding(location, line_number, rule)
        for line_number, line in enumerate(text.splitlines(), start=1)
        for rule in _line_rules(line, denylist)
    ]


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout  # noqa: S603, S607


def _decode(blob: bytes) -> str | None:
    if b"\0" in blob:
        return None
    return blob.decode("utf-8", errors="replace")


def _scan_blobs(
    paths_and_blobs: Iterable[tuple[str, bytes]], denylist: re.Pattern[str] | None
) -> list[Finding]:
    findings: list[Finding] = []
    for path, blob in paths_and_blobs:
        text = _decode(blob)
        if text is not None:
            findings.extend(scan_text(text, path, denylist))
    return findings


def scan_tracked_files(denylist: re.Pattern[str] | None) -> list[Finding]:
    paths = [p for p in _git("ls-files", "-z").decode().split("\0") if p]
    return _scan_blobs(((p, Path(p).read_bytes()) for p in paths), denylist)


def scan_staged_files(denylist: re.Pattern[str] | None) -> list[Finding]:
    output = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode()
    paths = [p for p in output.split("\0") if p]
    return _scan_blobs(((p, _git("show", f":{p}")) for p in paths), denylist)


def scan_commit_messages(denylist: re.Pattern[str] | None) -> list[Finding]:
    findings: list[Finding] = []
    for record in _git("log", "--format=%H%x00%B%x1e").decode().split("\x1e"):
        commit, _, message = record.strip().partition("\0")
        if commit:
            findings.extend(scan_text(message, f"commit {commit[:7]}", denylist))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m crag.guard", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("tracked", help="scan every tracked file")
    subcommands.add_parser("staged", help="scan staged file contents (pre-commit)")
    message_parser = subcommands.add_parser(
        "message", help="scan a commit message file (commit-msg)"
    )
    message_parser.add_argument("path", type=Path)
    subcommands.add_parser("history", help="scan every commit message")
    arguments = parser.parse_args(argv)

    terms = load_denylist()
    denylist = compile_denylist(terms)
    if not terms:
        print(f"guard: no denylist at {DENYLIST_PATH}, private-term check skipped", file=sys.stderr)

    if arguments.command == "tracked":
        findings = scan_tracked_files(denylist)
    elif arguments.command == "staged":
        findings = scan_staged_files(denylist)
    elif arguments.command == "message":
        findings = scan_text(arguments.path.read_text(encoding="utf-8"), "commit message", denylist)
    else:
        findings = scan_commit_messages(denylist)

    for finding in findings:
        print(finding, file=sys.stderr)
    if findings:
        print(
            f"guard: {len(findings)} finding(s). This repository is public; remove them.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
