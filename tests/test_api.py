from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from crag.api import create_app
from crag.embedding import HashEmbedder
from crag.ingest import ingest
from crag.masking import Masker
from crag.search import Hit
from crag.store import Connection
from tests.conftest import private_document


class FakeLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "TRACK-24 was not put forward [1]."


@pytest.fixture
def seeded(conn: Connection) -> Connection:
    ingest(
        conn,
        [
            private_document("TRACK-24", "Not put forward: no RAG experience."),
            private_document("TRACK-9", "Roles in Linz, too far away."),
        ],
        HashEmbedder(),
        Masker([]),
    )
    conn.commit()
    return conn


@pytest.fixture
def client(seeded: Connection) -> Iterator[TestClient]:
    with TestClient(create_app(seeded, HashEmbedder(), llm=None)) as test_client:
        yield test_client


def test_health() -> None:
    with TestClient(create_app(None, HashEmbedder(), llm=None)) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_returns_ranked_citations(client: TestClient) -> None:
    response = client.get(
        "/search", params={"q": "Which roles failed on RAG?", "collection": "private"}
    )
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
        {"q": "rag", "limit": "0"},
        {"q": "rag", "limit": "51"},
        {"q": "rag", "collection": "secret"},
        {},
    ],
)
def test_search_rejects_invalid_input(client: TestClient, params: dict[str, str]) -> None:
    assert client.get("/search", params=params).status_code == 422


def test_ask_without_llm_is_extractive(client: TestClient) -> None:
    response = client.get("/ask", params={"q": "Which roles failed on RAG?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["citations"][0]["number"] == 1
    assert body["citations"][0]["issue_key"] == "TRACK-24"


def test_ask_with_local_llm_answers_from_the_passages(seeded: Connection) -> None:
    llm = FakeLlm()
    with TestClient(create_app(seeded, HashEmbedder(), llm=llm)) as test_client:
        body = test_client.get("/ask", params={"q": "Which roles failed on RAG?"}).json()
    assert body["answer"] == "TRACK-24 was not put forward [1]."
    assert "[1] TRACK-24" in llm.prompts[0]
    assert "no RAG experience" in llm.prompts[0]


def test_ask_rejects_a_blank_question(client: TestClient) -> None:
    assert client.get("/ask", params={"q": " "}).status_code == 422
