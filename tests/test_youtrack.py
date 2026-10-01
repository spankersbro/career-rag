from datetime import date

import pytest

from crag.loaders.youtrack import attachment_path, issue_documents, pdf_text

BASE_URL = "https://tracker.example.com"

ISSUE = {
    "idReadable": "TRACK-7",
    "summary": "Acme Corp — Platform Engineer",
    "description": "**Role:** Platform Engineer\n\n**Notes:** Band 5,000-6,000.",
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
        "**Role:** Platform Engineer\n\n**Notes:** Band 5,000-6,000."
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
