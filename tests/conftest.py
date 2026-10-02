"""Shared fixtures for tests that run against a real Postgres with pgvector.

Each test gets its own schema, so tests are independent and never touch a real index.
"""

import os
import uuid
from collections.abc import Iterator
from datetime import date

import psycopg
import pytest

from crag.documents import Document
from crag.store import Connection, connect, migrate


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
def conn(database_url: str, schema: str) -> Iterator[Connection]:
    with connect(database_url, schema=schema) as connection:
        migrate(connection)
        yield connection


def private_document(source_key: str, text: str, title: str | None = None) -> Document:
    return Document(
        collection="private",
        source_type="issue",
        source_key=source_key,
        title=title or f"{source_key} Acme Corp — Platform Engineer",
        text=text,
        url=f"https://tracker.example.com/issue/{source_key}",
        issue_key=source_key,
        document_date=date(2026, 3, 1),
    )


def rows(conn: Connection, query: str) -> list[tuple[object, ...]]:
    return conn.execute(query).fetchall()
