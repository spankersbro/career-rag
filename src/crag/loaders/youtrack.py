import io
import json
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime
from typing import Any, Protocol

from pypdf import PdfReader

from crag.documents import Document

ISSUE_FIELDS = (
    "idReadable,summary,description,created,"
    "customFields(name,value(name)),"
    "comments(id,text,created),"
    "attachments(id,name,url,mimeType)"
)
PAGE_SIZE = 100
TIMEOUT_SECONDS = 30


class Opener(Protocol):
    def open(self, request: urllib.request.Request, timeout: float) -> Any: ...


class RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """urllib copies the Authorization header onto redirects, to any host and over http."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        raise ValueError(f"refusing HTTP {code} redirect from the tracker")


class YouTrackClient:
    def __init__(self, base_url: str, token: str, opener: Opener | None = None) -> None:
        if urllib.parse.urlparse(base_url).scheme != "https":
            raise ValueError(f"YouTrack base URL must use https: {base_url}")
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._opener = opener or urllib.request.build_opener(RefuseRedirects)

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
            if not isinstance(page, list):
                raise ValueError("tracker returned something other than a list of issues")
            issues.extend(page)
            if len(page) < PAGE_SIZE:
                return issues

    def attachment(self, url: str) -> bytes:
        return self._get(attachment_path(url, self.base_url))

    def _get(self, path_and_query: str) -> bytes:
        base_path = urllib.parse.urlparse(self.base_url).path
        request = urllib.request.Request(  # noqa: S310 - scheme is checked in __init__
            self.base_url + path_and_query.removeprefix(base_path),
            headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
        )
        with self._opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            body: bytes = response.read()
            return body


def attachment_path(url: str, base_url: str) -> str:
    """Only file downloads under the tracker's own /api/files/ are followed."""
    parsed = urllib.parse.urlparse(url)
    base = urllib.parse.urlparse(base_url)
    relative = parsed.scheme == "" and parsed.netloc == ""
    same_origin = relative or (
        parsed.scheme == "https"
        and parsed.hostname == base.hostname
        and (parsed.port or 443) == (base.port or 443)
    )
    path = urllib.parse.unquote(parsed.path)
    files_prefix = base.path.rstrip("/") + "/api/files/"
    in_files = path.startswith(files_prefix) and ".." not in path.split("/")
    if not same_origin or not in_files:
        raise ValueError(f"refusing attachment URL outside {base.hostname}{files_prefix}")
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


def pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:  # pypdf raises many unrelated types on malformed files
        raise ValueError("attachment is not a readable PDF") from error
    return "\n\n".join(page.strip() for page in pages if page.strip())


def pdf_attachments(issue: dict[str, Any]) -> list[dict[str, str]]:
    return [
        attachment
        for attachment in _entries(issue, "attachments", ("id", "name", "url"))
        if attachment.get("mimeType") == "application/pdf"
    ]


def issue_documents(
    issue: dict[str, Any], base_url: str, attachment_texts: dict[str, str]
) -> list[Document]:
    key = issue.get("idReadable")
    if not isinstance(key, str) or not key:
        raise ValueError("issue has no idReadable")
    summary = _text(issue.get("summary"))
    url = f"{base_url.rstrip('/')}/issue/{key}"
    created = _date(issue.get("created"), key)
    documents = [
        Document(
            collection="private",
            source_type="issue",
            source_key=key,
            title=f"{key} {summary}",
            text=_description_text(issue),
            url=url,
            issue_key=key,
            document_date=created,
        )
    ]
    for comment in _entries(issue, "comments", ("id",)):
        if _text(comment.get("text")).strip():
            documents.append(
                Document(
                    collection="private",
                    source_type="issue_comment",
                    source_key=f"{key}#comment-{comment['id']}",
                    title=f"{key} {summary}",
                    text=_text(comment.get("text")).strip(),
                    url=url,
                    issue_key=key,
                    document_date=_date(comment.get("created"), key),
                )
            )
    for attachment in _entries(issue, "attachments", ("id", "name")):
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
                    document_date=created,
                )
            )
    return documents


def _entries(issue: dict[str, Any], field: str, required: tuple[str, ...]) -> list[dict[str, Any]]:
    """Comments or attachments carrying the string fields we use; malformed ones are dropped."""
    return [
        entry
        for entry in issue.get(field) or []
        if isinstance(entry, dict) and all(isinstance(entry.get(name), str) for name in required)
    ]


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _description_text(issue: dict[str, Any]) -> str:
    parts = [_text(issue.get("summary"))]
    state = _state(issue)
    if state:
        parts.append(f"State: {state}")
    description = _text(issue.get("description")).strip()
    if description:
        parts.append(description)
    return "\n\n".join(part for part in parts if part)


def _state(issue: dict[str, Any]) -> str | None:
    for field in issue.get("customFields") or []:
        if isinstance(field, dict) and field.get("name") == "State":
            value = field.get("value")
            name = value.get("name") if isinstance(value, dict) else None
            return name if isinstance(name, str) else None
    return None


def _date(epoch_millis: object, issue_key: str) -> date | None:
    if epoch_millis is None:
        return None
    if not isinstance(epoch_millis, int) or isinstance(epoch_millis, bool):
        raise ValueError(f"{issue_key}: timestamp is not epoch milliseconds")
    return datetime.fromtimestamp(epoch_millis / 1000, tz=UTC).date()
