from importlib.resources import files

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from crag.documents import Document

EMBEDDING_DIMENSION = 384  # must match vector(384) in schema.sql; a test checks both
MIGRATION_LOCK_ID = 7_310_001  # arbitrary; only has to be unique among this database's lock users

Connection = psycopg.Connection[tuple[object, ...]]


def connect(url: str, schema: str | None = None) -> Connection:
    connection: Connection = psycopg.connect(url)
    if schema:
        connection.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
    connection.commit()
    register_vector(connection)
    return connection


def migrate(connection: Connection) -> None:
    """Applies the schema under an advisory lock, so concurrent starts do not race."""
    schema_sql = files("crag").joinpath("schema.sql").read_text(encoding="utf-8")
    with connection.transaction():
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_ID,))
        connection.execute(schema_sql)


def replace_source(
    connection: Connection,
    document: Document,
    chunks: list[str],
    embeddings: list[list[float]],
    embedding_model: str,
) -> None:
    """Upserts the source and swaps all its chunks in one transaction."""
    with connection.transaction():
        row = connection.execute(
            """
            INSERT INTO sources
                (collection, source_type, source_key, title, url, issue_key, document_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (collection, source_key) DO UPDATE SET
                source_type = EXCLUDED.source_type, title = EXCLUDED.title, url = EXCLUDED.url,
                issue_key = EXCLUDED.issue_key, document_date = EXCLUDED.document_date,
                ingested_at = now()
            RETURNING id
            """,
            (
                document.collection,
                document.source_type,
                document.source_key,
                document.title,
                document.url,
                document.issue_key,
                document.document_date,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"upsert returned no id for {document.source_key}")
        source_id = row[0]
        connection.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO chunks (source_id, ordinal, text, embedding, embedding_model) "
                "VALUES (%s, %s, %s, %s, %s)",
                [
                    (source_id, ordinal, text, np.array(vector, dtype=np.float32), embedding_model)
                    for ordinal, (text, vector) in enumerate(zip(chunks, embeddings, strict=True))
                ],
            )


def embedding_models(connection: Connection) -> set[str]:
    rows = connection.execute("SELECT DISTINCT embedding_model FROM chunks").fetchall()
    return {str(row[0]) for row in rows}


def stale_sources(
    connection: Connection,
    source_types: frozenset[str],
    key_prefix: str,
    keep_keys: frozenset[str],
) -> tuple[int, int]:
    """Counts (stale, total) sources in scope: these types, keys under this prefix."""
    row = connection.execute(
        """
        SELECT count(*) FILTER (WHERE NOT (source_key = ANY(%s))), count(*)
        FROM sources WHERE source_type = ANY(%s) AND starts_with(source_key, %s)
        """,
        (list(keep_keys), list(source_types), key_prefix),
    ).fetchone()
    if row is None:
        raise RuntimeError("count query returned no row")
    return int(str(row[0])), int(str(row[1]))


def delete_stale_sources(
    connection: Connection,
    source_types: frozenset[str],
    key_prefix: str,
    keep_keys: frozenset[str],
) -> int:
    with connection.transaction():
        cursor = connection.execute(
            """
            DELETE FROM sources
            WHERE source_type = ANY(%s) AND starts_with(source_key, %s)
              AND NOT (source_key = ANY(%s))
            """,
            (list(source_types), key_prefix, list(keep_keys)),
        )
        return cursor.rowcount
