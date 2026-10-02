"""Fixtures use invented words (Quuxdb, Springfield) so nothing resembles a real record."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import pytest
from fastapi.testclient import TestClient

from crag.api import ALLOWED_HOSTS, MAX_QUERY_CHARS, create_app
from crag.embedding import HashEmbedder
from crag.ingest import ingest
from crag.masking import Masker
from crag.search import Hit
from crag.store import Connection
from tests.conftest import private_document

QUESTION = "Which roles lacked Quuxdb?"


class FakeLlm:
    def __init__(self, error: Exception | None = None) -> None:
        self.prompts: list[str] = []
        self.error = error

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return "TRACK-24 lacked Quuxdb [1]."


@pytest.fixture
def seeded(conn: Connection) -> Connection:
    ingest(
        conn,
        [
            private_document("TRACK-24", "Declined after the call: no Quuxdb skills."),
            private_document("TRACK-9", "Roles in Springfield, too far away."),
        ],
        HashEmbedder(),
        Masker([]),
    )
    conn.commit()
    return conn


def connections_to(
    connection: Connection | None,
) -> Callable[[], AbstractContextManager[Connection]]:
    @contextmanager
    def open_connection() -> Iterator[Connection]:
        if connection is None:
            raise ConnectionError("database down")
        yield connection

    return open_connection


@pytest.fixture
def client(seeded: Connection) -> Iterator[TestClient]:
    app = create_app(connections_to(seeded), HashEmbedder(), llm=None)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


def test_health() -> None:
    app = create_app(connections_to(None), HashEmbedder(), llm=None)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_ranked_citations(client: TestClient) -> None:
    response = client.get("/search", params={"q": QUESTION, "collection": "private"})
    assert response.status_code == 200
    [first, *_] = response.json()["hits"]
    assert first["issue_key"] == "TRACK-24"
    assert first["url"] == "https://tracker.example.com/issue/TRACK-24"
    assert first["document_date"] == "2026-03-01"
    assert set(first) == set(Hit.__dataclass_fields__) - {"chunk_id"}


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},
        {"q": " "},
        {"q": "x" * (MAX_QUERY_CHARS + 1)},
        {"q": "quuxdb", "limit": "0"},
        {"q": "quuxdb", "limit": "51"},
        {"q": "quuxdb", "collection": "secret"},
        {},
    ],
)
def test_search_rejects_invalid_input(client: TestClient, params: dict[str, str]) -> None:
    assert client.get("/search", params=params).status_code == 422


def test_long_pasted_text_is_searchable(client: TestClient) -> None:
    pasted = "We need Quuxdb experience. " * 60
    assert len(pasted) <= MAX_QUERY_CHARS
    assert client.get("/search", params={"q": pasted}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example.com", "127.0.0.1.nip.io", "192.168.1.5"])
def test_requests_for_other_hosts_are_refused(seeded: Connection, host: str) -> None:
    app = create_app(connections_to(seeded), HashEmbedder(), llm=None)
    with TestClient(app, base_url=f"http://{host}") as test_client:
        assert test_client.get("/search", params={"q": QUESTION}).status_code == 400


def test_allowed_hosts_are_loopback_only() -> None:
    assert set(ALLOWED_HOSTS) == {"127.0.0.1", "localhost", "[::1]"}


def test_ask_without_llm_is_extractive(client: TestClient) -> None:
    response = client.get("/ask", params={"q": QUESTION})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["answer_error"] is None
    assert body["citations"][0]["number"] == 1
    assert body["citations"][0]["issue_key"] == "TRACK-24"


def test_ask_with_local_llm_answers_from_the_passages(seeded: Connection) -> None:
    llm = FakeLlm()
    app = create_app(connections_to(seeded), HashEmbedder(), llm=llm)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        body = test_client.get("/ask", params={"q": QUESTION}).json()
    assert body["answer"] == "TRACK-24 lacked Quuxdb [1]."
    assert "[1] TRACK-24" in llm.prompts[0]
    assert "no Quuxdb skills" in llm.prompts[0]


@pytest.mark.parametrize(
    "error", [TimeoutError("timed out"), ConnectionRefusedError(), ValueError("bad json")]
)
def test_ask_keeps_citations_when_the_llm_fails(seeded: Connection, error: Exception) -> None:
    app = create_app(connections_to(seeded), HashEmbedder(), llm=FakeLlm(error))
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        response = test_client.get("/ask", params={"q": QUESTION})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["answer_error"] == "language model unavailable"
    assert body["citations"][0]["issue_key"] == "TRACK-24"


def test_database_down_gives_503() -> None:
    app = create_app(connections_to(None), HashEmbedder(), llm=None)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        assert test_client.get("/search", params={"q": QUESTION}).status_code == 503
