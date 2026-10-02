"""Hybrid retrieval: vector similarity and IDF-weighted keyword matches, fused by rank."""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Literal, get_args

import numpy as np

from crag.config import load_settings
from crag.embedding import Embedder
from crag.ingest import embedder_from_settings
from crag.store import Connection, connect, embedding_models

CollectionFilter = Literal["private", "esco", "all"]

MAX_LIMIT = 50
CANDIDATES_PER_RANKING = MAX_LIMIT  # each ranking must reach the deepest page a caller can ask for
RRF_K = 60  # standard constant from Cormack et al. (2009); damps the weight of top ranks


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    source_key: str
    source_type: str
    issue_key: str | None
    title: str
    url: str | None
    document_date: date | None
    text: str
    score: float


def search(
    connection: Connection,
    embedder: Embedder,
    query: str,
    collection: CollectionFilter = "all",
    limit: int = 10,
) -> list[Hit]:
    if not query.strip():
        raise ValueError("query must not be blank")
    if collection not in get_args(CollectionFilter):
        raise ValueError(
            f"collection must be one of {get_args(CollectionFilter)}, got {collection!r}"
        )
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}, got {limit}")

    [query_vector] = embedder.embed([query_for_embedding(query, embedder)])
    rankings = [
        _vector_ranking(connection, query_vector, collection),
        keyword_ranking(connection, keyword_terms(connection, query), collection),
    ]
    fused = fuse(rankings)[:limit]
    return _hits(connection, fused)


def query_for_embedding(query: str, embedder: Embedder) -> str:
    """The longest word prefix that fits the model; the keyword side still sees every word."""
    fitted = ""
    for word in query.split():
        candidate = f"{fitted} {word}" if fitted else word
        if not embedder.fits(candidate):
            break
        fitted = candidate
    return fitted


def check_index_model(connection: Connection, embedder: Embedder) -> None:
    """Query vectors from another model than the index's give meaningless distances."""
    other = embedding_models(connection) - {embedder.name}
    if other:
        raise ValueError(
            f"index was built with {sorted(other)}, but queries use {embedder.name}; "
            "set CRAG__EMBEDDING__MODEL to match or re-ingest"
        )


def keyword_terms(connection: Connection, query: str) -> list[str]:
    """Query words as the index stores them, minus English and German stopwords."""
    rows = connection.execute(
        """
        SELECT DISTINCT word
        FROM unnest(tsvector_to_array(to_tsvector('simple', %s))) AS word
        WHERE ts_lexize('english_stem', word) <> '{}' AND ts_lexize('german_stem', word) <> '{}'
        ORDER BY word
        """,
        (query,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def fuse(rankings: list[list[int]]) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: a chunk found high in either ranking, or in both, rises."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _vector_ranking(
    connection: Connection, query_vector: list[float], collection: CollectionFilter
) -> list[int]:
    """Nearest chunks, re-sorted after the index scan.

    The connection enables pgvector's iterative HNSW scan, so a collection filter cannot leave
    the page short; its relaxed ordering is why the outer query sorts again.
    """
    vector = np.array(query_vector, dtype=np.float32)
    if collection == "all":
        rows = connection.execute(
            """
            WITH nearest AS MATERIALIZED (
                SELECT c.id, c.embedding <=> %(vector)s AS distance FROM chunks c
                ORDER BY c.embedding <=> %(vector)s LIMIT %(candidates)s
            )
            SELECT id FROM nearest ORDER BY distance, id
            """,
            {"vector": vector, "candidates": CANDIDATES_PER_RANKING},
        ).fetchall()
    else:
        rows = connection.execute(
            """
            WITH nearest AS MATERIALIZED (
                SELECT c.id, c.embedding <=> %(vector)s AS distance
                FROM chunks c JOIN sources s ON s.id = c.source_id
                WHERE s.collection = %(collection)s
                ORDER BY c.embedding <=> %(vector)s LIMIT %(candidates)s
            )
            SELECT id FROM nearest ORDER BY distance, id
            """,
            {"vector": vector, "collection": collection, "candidates": CANDIDATES_PER_RANKING},
        ).fetchall()
    return [int(str(row[0])) for row in rows]


def keyword_ranking(
    connection: Connection, terms: list[str], collection: CollectionFilter
) -> list[int]:
    """Scores a chunk by the summed IDF of the query terms it contains, within the collection.

    Postgres ranking functions have no IDF, so a chunk repeating a common word would beat one
    containing the rare word the question is about. Every lookup goes through the GIN index:
    the work grows with the chunks that match a term, not with the size of the table.
    """
    if not terms:
        return []
    rows = connection.execute(
        """
        WITH total AS MATERIALIZED (
            SELECT count(*)::float8 AS n FROM chunks c JOIN sources s ON s.id = c.source_id
            WHERE %(collection)s = 'all' OR s.collection = %(collection)s
        ),
        terms AS MATERIALIZED (
            SELECT q.lexeme::tsquery AS query, matches.frequency
            FROM unnest(%(lexemes)s::text[]) AS q(lexeme)
            CROSS JOIN LATERAL (
                SELECT count(*) AS frequency FROM chunks c JOIN sources s ON s.id = c.source_id
                WHERE c.search_text @@ q.lexeme::tsquery
                  AND (%(collection)s = 'all' OR s.collection = %(collection)s)
            ) AS matches
        )
        SELECT c.id, sum(ln(1 + total.n / greatest(terms.frequency, 1))) AS score
        FROM terms
        JOIN chunks c ON c.search_text @@ terms.query
        JOIN sources s ON s.id = c.source_id
        CROSS JOIN total
        WHERE %(collection)s = 'all' OR s.collection = %(collection)s
        GROUP BY c.id
        ORDER BY score DESC, c.id
        LIMIT %(candidates)s
        """,
        {
            "lexemes": [_lexeme_query(term) for term in terms],
            "collection": collection,
            "candidates": CANDIDATES_PER_RANKING,
        },
    ).fetchall()
    return [int(str(row[0])) for row in rows]


def _lexeme_query(term: str) -> str:
    """A tsquery that matches exactly this lexeme; quoting stops ':' or '&' being read as syntax."""
    return "'" + term.replace("\\", "\\\\").replace("'", "''") + "'"


def _hits(connection: Connection, fused: list[tuple[int, float]]) -> list[Hit]:
    if not fused:
        return []
    rows = connection.execute(
        """
        SELECT c.id, s.source_key, s.source_type, s.issue_key, s.title, s.url,
               s.document_date, c.text
        FROM chunks c JOIN sources s ON s.id = c.source_id
        WHERE c.id = ANY(%s)
        """,
        ([chunk_id for chunk_id, _ in fused],),
    ).fetchall()
    by_id = {int(str(row[0])): row for row in rows}
    return [
        Hit(
            chunk_id=chunk_id,
            source_key=str(row[1]),
            source_type=str(row[2]),
            issue_key=None if row[3] is None else str(row[3]),
            title=str(row[4]),
            url=None if row[5] is None else str(row[5]),
            document_date=row[6] if isinstance(row[6], date) else None,
            text=str(row[7]),
            score=score,
        )
        for chunk_id, score in fused
        if (row := by_id.get(chunk_id)) is not None
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m crag.search", description="Search the index.")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--collection", choices=get_args(CollectionFilter), default="all")
    arguments = parser.parse_args(argv)

    settings = load_settings()
    embedder = embedder_from_settings(settings)
    with connect(settings.database_url, schema=settings.database_schema) as connection:
        check_index_model(connection, embedder)
        hits = search(
            connection,
            embedder,
            arguments.query,
            arguments.collection,
            arguments.limit,
        )
    for number, hit in enumerate(hits, start=1):
        dated = f" ({hit.document_date.isoformat()})" if hit.document_date else ""
        print(f"{number}. {hit.title}{dated} — {hit.url or hit.source_key}")
        print("   " + " ".join(hit.text.split())[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
