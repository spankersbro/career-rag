"""Runs against a real Postgres with pgvector (CRAG__DATABASE__URL).

Each test gets its own schema."""

import os
import uuid
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import psycopg
import pytest

from crag.documents import Document
from crag.embedding import HashEmbedder
from crag.ingest import ingest, main
from crag.masking import Masker
from crag.store import EMBEDDING_DIMENSION, connect, migrate

AT = "@"


@pytest.fixture
def database_url() -> str:
    url = os.environ.get("CRAG__DATABASE__URL")
    if not url:
        pytest.fail(
            "CRAG__DATABASE__URL is not set; start the database with `docker compose up -d db`"
        )
    return url


@pytest.fixture
def schema(database_url: str) -> Iterator[str]:
    name = "test_" + uuid.uuid4().hex[:12]
    with psycopg.connect(database_url, autocommit=True) as admin:
        admin.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        admin.execute(psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(name)))
    yield name
    with psycopg.connect(database_url, autocommit=True) as admin:
        admin.execute(
            psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(name))
        )


@pytest.fixture
def conn(database_url: str, schema: str) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with connect(database_url, schema=schema) as connection:
        migrate(connection)
        yield connection


def private_document(source_key: str, text: str) -> Document:
    return Document(
        collection="private",
        source_type="issue",
        source_key=source_key,
        title=f"{source_key} Acme Corp — Platform Engineer",
        text=text,
        issue_key=source_key,
        document_date=date(2026, 3, 1),
    )


def rows(conn: psycopg.Connection[tuple[object, ...]], query: str) -> list[tuple[object, ...]]:
    return conn.execute(query).fetchall()


def test_ingest_stores_masked_chunks_with_embeddings(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    document = private_document(
        "TRACK-7", "Contact Jane Roe at jane" + AT + "mailbox.test.\n\nRejected: no RAG."
    )
    result = ingest(conn, [document], HashEmbedder(), Masker(["Jane Roe"]), max_chars=40)

    assert result.sources == 1
    assert result.chunks == 2
    stored = rows(
        conn,
        "SELECT c.ordinal, c.text, c.embedding_model, s.issue_key FROM chunks c "
        "JOIN sources s ON s.id = c.source_id ORDER BY c.ordinal",
    )
    assert stored == [
        (0, "Contact [person] at [email].", "hash-test", "TRACK-7"),
        (1, "Rejected: no RAG.", "hash-test", "TRACK-7"),
    ]


def test_reingesting_a_source_replaces_its_chunks(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    ingest(
        conn,
        [private_document("TRACK-7", "Old text.\n\nMore old text.")],
        HashEmbedder(),
        Masker([]),
        max_chars=20,
    )
    ingest(
        conn, [private_document("TRACK-7", "New text.")], HashEmbedder(), Masker([]), max_chars=20
    )
    assert rows(conn, "SELECT count(*) FROM sources") == [(1,)]
    assert rows(conn, "SELECT text FROM chunks") == [("New text.",)]


def test_public_documents_are_not_masked(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    skill = Document(
        collection="esco",
        source_type="esco_skill",
        source_key="http://data.europa.eu/esco/skill/1",
        title="manage Jane Roe",
        text="manage Jane Roe",
    )
    ingest(conn, [skill], HashEmbedder(), Masker(["Jane Roe"]))
    assert rows(conn, "SELECT text FROM chunks") == [("manage Jane Roe",)]


def test_private_documents_require_a_local_embedder(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    with pytest.raises(ValueError, match="local embedding model"):
        ingest(conn, [private_document("TRACK-7", "text")], HashEmbedder(local=False), Masker([]))
    assert rows(conn, "SELECT count(*) FROM sources") == [(0,)]


def test_embedding_dimension_must_match_the_schema(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    with pytest.raises(ValueError, match="dimension"):
        ingest(
            conn,
            [private_document("TRACK-7", "text")],
            HashEmbedder(dimension=EMBEDDING_DIMENSION + 1),
            Masker([]),
        )


def test_blank_documents_are_skipped(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    result = ingest(conn, [private_document("TRACK-7", "   ")], HashEmbedder(), Masker([]))
    assert (result.sources, result.chunks) == (0, 0)


def test_migrate_twice_is_harmless(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    migrate(conn)
    assert rows(conn, "SELECT count(*) FROM sources") == [(0,)]


def test_cli_ingests_a_changelog_file(
    database_url: str, schema: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "## 2026-03-01 — Acme Corp: applied\n\nApplied, TRACK-7.\n", encoding="utf-8"
    )
    people = tmp_path / "people.txt"
    people.write_text("Jane Roe\n", encoding="utf-8")
    monkeypatch.setenv("CRAG__DATABASE__SCHEMA", schema)
    monkeypatch.setenv("CRAG__PII__PEOPLE_FILE", str(people))
    monkeypatch.setenv("CRAG__EMBEDDING__MODEL", "hash-test")

    assert main(["changelog", str(changelog)]) == 0

    with connect(database_url, schema=schema) as connection:
        assert rows(
            connection,
            "SELECT s.issue_key, s.document_date, c.text FROM chunks c "
            "JOIN sources s ON s.id = c.source_id",
        ) == [("TRACK-7", date(2026, 3, 1), "Applied, TRACK-7.")]
