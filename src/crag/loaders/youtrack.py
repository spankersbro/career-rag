import io
import json
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from crag.documents import Document

ISSUE_FIELDS = (
    "idReadable,summary,description,created,"
    "customFields(name,value(name)),"
    "comments(id,text,created),"
    "attachments(id,name,url,mimeType)"
)
PAGE_SIZE = 100
TIMEOUT_SECONDS = 30


class YouTrackClient:
    def __init__(self, base_url: str, token: str) -> None:
        if urllib.parse.urlparse(base_url).scheme != "https":
            raise ValueError(f"YouTrack base URL must use https: {base_url}")
        self.base_url = base_url.rstrip("/")
        self._token = token

    def issues(self, query: str) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        while True:
            params = {
                "query": query,
                "fields": ISSUE_FIELDS,
                "$top": PAGE_SIZE,
                "$skip": len(issues),
            }
            page = json.loads(self._get("/api/issues?" + urllib.parse.urlencode(params)))
            issues.extend(page)
            if len(page) < PAGE_SIZE:
                return issues

    def attachment(self, url: str) -> bytes:
        return self._get(attachment_path(url, self.base_url))

    def _get(self, path: str) -> bytes:
        request = urllib.request.Request(  # noqa: S310 - scheme is checked in __init__
            self.base_url + path,
            headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            body: bytes = response.read()
            return body


def attachment_path(url: str, base_url: str) -> str:
    """Only file downloads on the tracker's own host are followed; anything else is refused."""
    parsed = urllib.parse.urlparse(url)
    base_host = urllib.parse.urlparse(base_url).hostname
    relative = parsed.scheme == "" and parsed.netloc == ""
    same_host = relative or (parsed.scheme == "https" and parsed.hostname == base_host)
    in_files = parsed.path.startswith("/api/files/") and ".." not in parsed.path.split("/")
    if not same_host or not in_files:
        raise ValueError(f"refusing attachment URL outside {base_host}/api/files/")
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


def pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as error:
        raise ValueError("attachment is not a readable PDF") from error
    return "\n\n".join(page.strip() for page in pages if page.strip())


def issue_documents(
    issue: dict[str, Any], base_url: str, attachment_texts: dict[str, str]
) -> list[Document]:
    key = issue.get("idReadable")
    if not key:
        raise ValueError("issue has no idReadable")
    summary = issue.get("summary") or ""
    url = f"{base_url.rstrip('/')}/issue/{key}"
    documents = [
        Document(
            collection="private",
            source_type="issue",
            source_key=key,
            title=f"{key} {summary}",
            text=_description_text(issue),
            url=url,
            issue_key=key,
            document_date=_date(issue.get("created")),
        )
    ]
    for comment in issue.get("comments") or []:
        if (comment.get("text") or "").strip():
            documents.append(
                Document(
                    collection="private",
                    source_type="issue_comment",
                    source_key=f"{key}#comment-{comment['id']}",
                    title=f"{key} {summary}",
                    text=comment["text"].strip(),
                    url=url,
                    issue_key=key,
                    document_date=_date(comment.get("created")),
                )
            )
    for attachment in issue.get("attachments") or []:
        text = attachment_texts.get(attachment["id"], "")
        if text.strip():
            documents.append(
                Document(
                    collection="private",
                    source_type="issue_attachment",
                    source_key=f"{key}#attachment-{attachment['id']}",
                    title=f"{key} {attachment['name']}",
                    text=text.strip(),
                    url=url,
                    issue_key=key,
                    document_date=_date(issue.get("created")),
                )
            )
    return documents


def _description_text(issue: dict[str, Any]) -> str:
    parts = [issue.get("summary") or ""]
    state = _state(issue)
    if state:
        parts.append(f"State: {state}")
    if (issue.get("description") or "").strip():
        parts.append(issue["description"].strip())
    return "\n\n".join(part for part in parts if part)


def _state(issue: dict[str, Any]) -> str | None:
    for field in issue.get("customFields") or []:
        if field.get("name") == "State" and isinstance(field.get("value"), dict):
            name: str | None = field["value"].get("name")
            return name
    return None


def _date(epoch_millis: int | None) -> date | None:
    if epoch_millis is None:
        return None
    return datetime.fromtimestamp(epoch_millis / 1000, tz=UTC).date()
