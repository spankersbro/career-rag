import re
from collections import Counter
from datetime import date

from crag.documents import Document

ENTRY_HEADING = re.compile(r"^## (\d{4}-\d{2}-\d{2}) — (.+)$", re.MULTILINE)
ISSUE_KEY = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")


def parse_changelog(markdown: str, label: str) -> list[Document]:
    """One document per `## YYYY-MM-DD — Title` entry, newest first as written.

    `label` names the changelog, so entries from different files never share a key.
    """
    headings = list(ENTRY_HEADING.finditer(markdown))
    remaining_per_date = Counter(heading.group(1) for heading in headings)
    documents = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        remaining_per_date[heading.group(1)] -= 1
        sequence = remaining_per_date[heading.group(1)]
        body = markdown[heading.end() : end].strip()
        documents.append(
            _entry_document(heading, body, f"{key_prefix(label)}{heading.group(1)}:{sequence}")
        )
    return documents


def key_prefix(label: str) -> str:
    return f"changelog:{label}:"


def _entry_document(heading: re.Match[str], body: str, source_key: str) -> Document:
    """The source key holds no title text: titles can name people, and keys are not masked.

    Its sequence counts from the oldest entry of that date, so a new entry added on top
    does not change the keys of the entries below it.
    """
    raw_date, title = heading.group(1), heading.group(2).strip()
    try:
        entry_date = date.fromisoformat(raw_date)
    except ValueError as error:
        raise ValueError(f"changelog entry has an invalid date: {raw_date}") from error
    issue_key = ISSUE_KEY.search(body)
    return Document(
        collection="private",
        source_type="changelog",
        source_key=source_key,
        title=title,
        text=body,
        issue_key=issue_key.group(0) if issue_key else None,
        document_date=entry_date,
    )
