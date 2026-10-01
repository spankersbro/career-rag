import re
from datetime import date

from crag.documents import Document

ENTRY_HEADING = re.compile(r"^## (\d{4}-\d{2}-\d{2}) — (.+)$", re.MULTILINE)
ISSUE_KEY = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")


def parse_changelog(markdown: str) -> list[Document]:
    """One document per `## YYYY-MM-DD — Title` entry, newest first as written."""
    headings = list(ENTRY_HEADING.finditer(markdown))
    documents = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        documents.append(_entry_document(heading, markdown[heading.end() : end].strip(), index))
    return documents


def _entry_document(heading: re.Match[str], body: str, index: int) -> Document:
    raw_date, title = heading.group(1), heading.group(2).strip()
    try:
        entry_date = date.fromisoformat(raw_date)
    except ValueError as error:
        raise ValueError(f"changelog entry has an invalid date: {raw_date}") from error
    issue_key = ISSUE_KEY.search(body)
    return Document(
        collection="private",
        source_type="changelog",
        source_key=f"changelog:{raw_date}:{index}:{title}",
        title=title,
        text=body,
        issue_key=issue_key.group(0) if issue_key else None,
        document_date=entry_date,
    )
