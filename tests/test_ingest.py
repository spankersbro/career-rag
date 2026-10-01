"""Runs against a real Postgres with pgvector (CRAG__DATABASE__URL).

Each test gets its own schema."""

import os
import uuid
from collections.abc import Iterator
from dataclasses import replace
from datetime import date
from http.client import IncompleteRead
from pathlib import Path

import psycopg
import pytest

from crag.documents import Document
from crag.embedding import HashEmbedder
from crag.ingest import Batch, ingest, ingest_batch, main, youtrack_batch
from crag.masking import Masker
from crag.store import EMBEDDING_DIMENSION, connect, migrate
from tests.test_youtrack import minimal_pdf

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

    assert main(["changelog", str(changelog), "--label", "cv"]) == 0

    with connect(database_url, schema=schema) as connection:
        assert rows(
            connection,
            "SELECT s.issue_key, s.document_date, c.text FROM chunks c "
            "JOIN sources s ON s.id = c.source_id",
        ) == [("TRACK-7", date(2026, 3, 1), "Applied, TRACK-7.")]


def test_titles_are_masked_before_storage_and_embedding(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    document = replace(private_document("TRACK-7", "Body."), title="Call with Jane Roe")
    seen: list[str] = []

    class RecordingEmbedder(HashEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            seen.extend(texts)
            return super().embed(texts)

    ingest(conn, [document], RecordingEmbedder(), Masker(["Jane Roe"]))
    assert rows(conn, "SELECT title FROM sources") == [("Call with [person]",)]
    assert seen == ["Call with [person]\n\nBody."]


def test_batch_removes_sources_missing_from_a_complete_load(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    types = frozenset({"issue"})
    first = Batch([private_document("TRACK-7", "a"), private_document("TRACK-8", "b")], types)
    ingest_batch(conn, first, HashEmbedder(), Masker([]))
    result = ingest_batch(
        conn, Batch([private_document("TRACK-7", "a")], types), HashEmbedder(), Masker([])
    )
    assert result.removed == 1
    assert rows(conn, "SELECT source_key FROM sources") == [("TRACK-7",)]


def test_batch_keeps_unavailable_sources_and_other_types(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    def attachment(key: str) -> Document:
        return replace(private_document(key, "cv"), source_type="issue_attachment")

    changelog = replace(
        private_document("changelog:cv:2026-03-01:0", "log"), source_type="changelog"
    )
    ingest(conn, [attachment("TRACK-7#attachment-9-1"), changelog], HashEmbedder(), Masker([]))
    batch = Batch(
        [attachment("TRACK-8#attachment-9-2")],
        frozenset({"issue_attachment"}),
        frozenset({"TRACK-7#attachment-9-1"}),
    )
    assert ingest_batch(conn, batch, HashEmbedder(), Masker([])).removed == 0
    assert rows(conn, "SELECT count(*) FROM sources") == [(3,)]


def test_document_that_became_blank_is_removed(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    types = frozenset({"issue"})
    ingest_batch(
        conn, Batch([private_document("TRACK-7", "text")], types), HashEmbedder(), Masker([])
    )
    ingest_batch(
        conn, Batch([private_document("TRACK-7", "  ")], types), HashEmbedder(), Masker([])
    )
    assert rows(conn, "SELECT count(*) FROM sources") == [(0,)]


def test_vectors_from_two_models_never_mix(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    ingest(conn, [private_document("TRACK-7", "text")], HashEmbedder(), Masker([]))

    class OtherModel(HashEmbedder):
        name = "other-model"

    with pytest.raises(ValueError, match="hash-test"):
        ingest(conn, [private_document("TRACK-8", "text")], OtherModel(), Masker([]))


class FakeClient:
    base_url = "https://tracker.example.com"

    def __init__(self, issues: list[dict[str, object]], files: dict[str, bytes]) -> None:
        self._issues = issues
        self._files = files

    def issues(self, query: str) -> list[dict[str, object]]:
        return self._issues

    def attachment(self, url: str) -> bytes:
        return self._files[url]


def test_one_unreadable_pdf_does_not_stop_the_load() -> None:
    issue: dict[str, object] = {
        "idReadable": "TRACK-7",
        "summary": "Acme Corp",
        "attachments": [
            {
                "id": "9-1",
                "name": "broken.pdf",
                "url": "/api/files/9-1",
                "mimeType": "application/pdf",
            },
            {
                "id": "9-2",
                "name": "posting.pdf",
                "url": "/api/files/9-2",
                "mimeType": "application/pdf",
            },
        ],
    }
    client = FakeClient(
        [issue], {"/api/files/9-1": b"not a pdf", "/api/files/9-2": minimal_pdf("Posting")}
    )
    batch = youtrack_batch(client, "project: TRACK")  # type: ignore[arg-type]
    assert [d.source_key for d in batch.documents] == ["TRACK-7", "TRACK-7#attachment-9-2"]
    assert batch.unavailable_keys == {"TRACK-7#attachment-9-1"}


def test_schema_dimension_matches_the_code() -> None:
    schema_sql = (Path(__file__).parents[1] / "src" / "crag" / "schema.sql").read_text()
    assert f"vector({EMBEDDING_DIMENSION})" in schema_sql


def test_cli_refuses_private_sources_without_a_people_file(
    schema: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("## 2026-03-01 — Acme Corp: applied\n\nApplied.\n", encoding="utf-8")
    monkeypatch.setenv("CRAG__DATABASE__SCHEMA", schema)
    monkeypatch.setenv("CRAG__PII__PEOPLE_FILE", str(tmp_path / "absent.txt"))
    monkeypatch.setenv("CRAG__EMBEDDING__MODEL", "hash-test")
    with pytest.raises(ValueError, match="people file"):
        main(["changelog", str(changelog), "--label", "cv"])


def test_cli_ingests_esco_without_a_people_file(
    database_url: str, schema: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "skills_en.csv"
    csv_path.write_text(
        "conceptUri,preferredLabel,altLabels,description\n"
        "http://data.europa.eu/esco/skill/1,use databases,,Manage data.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CRAG__DATABASE__SCHEMA", schema)
    monkeypatch.setenv("CRAG__PII__PEOPLE_FILE", str(tmp_path / "absent.txt"))
    monkeypatch.setenv("CRAG__EMBEDDING__MODEL", "hash-test")
    assert main(["esco", str(csv_path)]) == 0
    with connect(database_url, schema=schema) as connection:
        assert rows(
            connection, "SELECT collection, text FROM chunks JOIN sources s ON s.id = source_id"
        ) == [("esco", "use databases\n\nManage data.")]


def test_empty_load_never_prunes(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    types = frozenset({"issue"})
    ingest_batch(conn, Batch([private_document("TRACK-7", "a")], types), HashEmbedder(), Masker([]))
    with pytest.raises(ValueError, match="empty"):
        ingest_batch(conn, Batch([], types), HashEmbedder(), Masker([]))
    assert rows(conn, "SELECT source_key FROM sources") == [("TRACK-7",)]


def test_load_that_would_remove_most_sources_is_refused(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    types = frozenset({"issue"})
    everything = [private_document(f"TRACK-{n}", "text") for n in range(4)]
    ingest_batch(conn, Batch(everything, types), HashEmbedder(), Masker([]))
    with pytest.raises(ValueError, match="3 of 4"):
        ingest_batch(conn, Batch(everything[:1], types), HashEmbedder(), Masker([]))
    assert rows(conn, "SELECT count(*) FROM sources") == [(4,)]


def test_prune_is_limited_to_the_key_prefix(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    types = frozenset({"changelog"})

    def entry(key: str) -> Document:
        return replace(private_document(key, "log"), source_type="changelog")

    ingest(conn, [entry("changelog:site:2026-03-01:0")], HashEmbedder(), Masker([]))
    batch = Batch([entry("changelog:cv:2026-03-01:0")], types, key_prefix="changelog:cv:")
    assert ingest_batch(conn, batch, HashEmbedder(), Masker([])).removed == 0
    assert rows(conn, "SELECT count(*) FROM sources") == [(2,)]


@pytest.mark.parametrize("error", [TimeoutError("read timed out"), IncompleteRead(b"")])
def test_network_failures_on_one_attachment_are_skipped(error: Exception) -> None:
    class FailingClient(FakeClient):
        def attachment(self, url: str) -> bytes:
            raise error

    issue: dict[str, object] = {
        "idReadable": "TRACK-7",
        "attachments": [
            {"id": "9-1", "name": "a.pdf", "url": "/api/files/9-1", "mimeType": "application/pdf"}
        ],
    }
    batch = youtrack_batch(FailingClient([issue], {}), "q")  # type: ignore[arg-type]
    assert batch.unavailable_keys == {"TRACK-7#attachment-9-1"}


def test_masker_from_file_requires_the_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="people file"):
        Masker.from_file(tmp_path / "absent.txt")
    (tmp_path / "people.txt").write_text("Jane Roe\n", encoding="utf-8")
    assert Masker.from_file(tmp_path / "people.txt").mask("Jane Roe") == "[person]"
