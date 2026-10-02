import json
import threading
import urllib.parse
import urllib.request
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from crag.loaders.youtrack import (
    PAGE_SIZE,
    RefuseRedirects,
    YouTrackClient,
    attachment_path,
    issue_documents,
    pdf_attachments,
    pdf_text,
)

BASE_URL = "https://tracker.example.com"

ISSUE = {
    "idReadable": "TRACK-7",
    "summary": "Acme Corp — Platform Engineer",
    "description": "**Role:** Platform Engineer\n\n**Notes:** Band A to B.",
    "created": 1790000000000,
    "customFields": [{"name": "State", "value": {"name": "Rejected"}}],
    "comments": [
        {"id": "4-1", "text": "Call went well.", "created": 1790100000000},
        {"id": "4-2", "text": "", "created": 1790200000000},
    ],
    "attachments": [],
}


def test_description_document_carries_key_state_and_url() -> None:
    [description, comment] = issue_documents(ISSUE, BASE_URL, {})
    assert description.source_key == "TRACK-7"
    assert description.source_type == "issue"
    assert description.collection == "private"
    assert description.issue_key == "TRACK-7"
    assert description.title == "TRACK-7 Acme Corp — Platform Engineer"
    assert description.url == "https://tracker.example.com/issue/TRACK-7"
    assert description.document_date == date(2026, 9, 21)
    assert description.text == (
        "Acme Corp — Platform Engineer\n\nState: Rejected\n\n"
        "**Role:** Platform Engineer\n\n**Notes:** Band A to B."
    )
    assert comment.source_key == "TRACK-7#comment-4-1"
    assert comment.source_type == "issue_comment"
    assert comment.text == "Call went well."


def test_attachment_texts_become_documents() -> None:
    issue = {**ISSUE, "comments": [], "attachments": [{"id": "9-1", "name": "posting.pdf"}]}
    documents = issue_documents(issue, BASE_URL, {"9-1": "Posting text."})
    attachment = documents[-1]
    assert attachment.source_key == "TRACK-7#attachment-9-1"
    assert attachment.source_type == "issue_attachment"
    assert attachment.title == "TRACK-7 posting.pdf"
    assert attachment.text == "Posting text."


def test_issue_without_description_or_state() -> None:
    issue = {"idReadable": "TRACK-8", "summary": "Lead", "created": 1790000000000}
    [description] = issue_documents(issue, BASE_URL, {})
    assert description.text == "Lead"


def test_issue_without_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="idReadable"):
        issue_documents({"summary": "x"}, BASE_URL, {})


def test_attachment_path_accepts_same_host_file_urls() -> None:
    url = "https://tracker.example.com:443/api/files/12-10?sign=abc&updated=1"
    assert attachment_path(url, BASE_URL) == "/api/files/12-10?sign=abc&updated=1"
    assert attachment_path("/api/files/12-11?sign=x", BASE_URL) == "/api/files/12-11?sign=x"


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.net/api/files/12-10",
        "https://tracker.example.com/api/admin/users",
        "https://tracker.example.com/api/files/../admin/users",
        "http://tracker.example.com/api/files/12-10",
        "file:///etc/passwd",
        "",
    ],
)
def test_attachment_path_rejects_other_hosts_and_paths(url: str) -> None:
    with pytest.raises(ValueError, match="attachment URL"):
        attachment_path(url, BASE_URL)


def minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = b"%PDF-1.4\n"
    offsets = []
    for number, content in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n".encode() + content + b"\nendobj\n"
    xref_offset = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    body += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    body += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    body += f"startxref\n{xref_offset}\n%%EOF\n".encode()
    return body


def test_pdf_text_extracts_page_text() -> None:
    assert pdf_text(minimal_pdf("Senior Platform Engineer")) == "Senior Platform Engineer"


def test_pdf_text_rejects_non_pdf_bytes() -> None:
    with pytest.raises(ValueError, match="PDF"):
        pdf_text(b"not a pdf")


@pytest.mark.parametrize(
    "url",
    [
        "https://tracker.example.com/api/files/%2e%2e/admin/users",
        "https://tracker.example.com:8443/api/files/12-10",
    ],
)
def test_attachment_path_rejects_encoded_traversal_and_other_ports(url: str) -> None:
    with pytest.raises(ValueError, match="attachment URL"):
        attachment_path(url, BASE_URL)


def test_attachment_path_honours_a_context_path() -> None:
    base = "https://tracker.example.com/youtrack"
    url = "https://tracker.example.com/youtrack/api/files/12-10?sign=a"
    assert attachment_path(url, base) == "/youtrack/api/files/12-10?sign=a"
    with pytest.raises(ValueError, match="attachment URL"):
        attachment_path("https://tracker.example.com/api/files/12-10", base)


def test_malformed_comments_and_attachments_are_dropped() -> None:
    issue = {
        **ISSUE,
        "comments": [{"text": "no id"}, "not a dict", {"id": "4-3", "text": "kept"}],
        "attachments": [{"id": "9-1"}, {"id": "9-2", "name": "cv.pdf"}],
    }
    documents = issue_documents(issue, BASE_URL, {"9-1": "orphan", "9-2": "CV text"})
    assert [d.source_key for d in documents] == [
        "TRACK-7",
        "TRACK-7#comment-4-3",
        "TRACK-7#attachment-9-2",
    ]


def test_non_numeric_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="TRACK-7: timestamp"):
        issue_documents({**ISSUE, "created": "2026-03-01"}, BASE_URL, {})


def test_only_complete_pdf_attachments_are_downloaded() -> None:
    issue = {
        "attachments": [
            {"id": "9-1", "name": "a.pdf", "url": "/api/files/9-1", "mimeType": "application/pdf"},
            {"id": "9-2", "name": "b.png", "url": "/api/files/9-2", "mimeType": "image/png"},
            {"id": "9-3", "name": "c.pdf", "mimeType": "application/pdf"},
        ]
    }
    assert [a["id"] for a in pdf_attachments(issue)] == ["9-1"]


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class FakeOpener:
    def __init__(self, bodies: list[bytes]) -> None:
        self.bodies = bodies
        self.requests: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
        self.requests.append(request)
        return FakeResponse(self.bodies.pop(0))


def test_issues_are_paged_until_a_short_page() -> None:
    full_page = json.dumps([{"idReadable": f"T-{i}"} for i in range(PAGE_SIZE)]).encode()
    opener = FakeOpener([full_page, json.dumps([{"idReadable": "T-last"}]).encode()])
    client = YouTrackClient(BASE_URL, "secret", opener)
    issues = client.issues("project: T")
    assert len(issues) == PAGE_SIZE + 1
    skips = [
        urllib.parse.parse_qs(urllib.parse.urlparse(r.full_url).query)["$skip"]
        for r in opener.requests
    ]
    assert skips == [["0"], [str(PAGE_SIZE)]]
    assert opener.requests[0].get_header("Authorization") == "Bearer secret"


def test_issues_must_be_a_list() -> None:
    client = YouTrackClient(BASE_URL, "secret", FakeOpener([b'{"error": "denied"}']))
    with pytest.raises(ValueError, match="list of issues"):
        client.issues("project: T")


def test_attachment_download_uses_the_checked_path() -> None:
    opener = FakeOpener([b"%PDF"])
    client = YouTrackClient("https://tracker.example.com/youtrack", "secret", opener)
    assert client.attachment("https://tracker.example.com/youtrack/api/files/9-1?sign=a") == b"%PDF"
    assert (
        opener.requests[0].full_url == "https://tracker.example.com/youtrack/api/files/9-1?sign=a"
    )


def test_base_url_must_be_https() -> None:
    with pytest.raises(ValueError, match="https"):
        YouTrackClient("http://tracker.example.com", "secret")


class Redirecting(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(302)
        self.send_header("Location", "http://127.0.0.1:9/steal")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return None


def test_redirects_are_refused_so_the_token_is_never_forwarded() -> None:
    server = HTTPServer(("127.0.0.1", 0), Redirecting)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        opener = urllib.request.build_opener(RefuseRedirects)
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/files/1",
            headers={"Authorization": "Bearer secret"},
        )
        with pytest.raises(ValueError, match="redirect"):
            opener.open(request, timeout=5)
    finally:
        server.shutdown()


def test_default_client_refuses_redirects() -> None:
    client = YouTrackClient(BASE_URL, "secret")
    handlers = getattr(client._opener, "handlers", [])
    assert any(isinstance(handler, RefuseRedirects) for handler in handlers)
