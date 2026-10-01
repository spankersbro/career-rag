from dataclasses import dataclass
from datetime import date
from typing import Literal

Collection = Literal["private", "esco"]


@dataclass(frozen=True)
class Document:
    collection: Collection
    source_type: str
    source_key: str
    title: str
    text: str
    url: str | None = None
    issue_key: str | None = None
    document_date: date | None = None
