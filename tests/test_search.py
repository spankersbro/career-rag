"""Fixtures use invented words (Quuxdb, Springfield) so nothing resembles a real record."""

from datetime import date

import pytest

from crag.documents import Document
from crag.embedding import HashEmbedder
from crag.ingest import ingest
from crag.masking import Masker
from crag.search import (
    RRF_K,
    check_index_model,
    fuse,
    keyword_ranking,
    keyword_terms,
    main,
    query_for_embedding,
    search,
)
from crag.store import Connection
from tests.conftest import private_document

QUESTION = "Which roles lacked Quuxdb?"


def esco_skill(number: int, text: str) -> Document:
    return Document(
        collection="esco",
        source_type="esco_skill",
        source_key=f"http://data.europa.eu/esco/skill/{number}",
        title=text,
        text=text,
        url=f"http://data.europa.eu/esco/skill/{number}",
    )


def seed(conn: Connection) -> None:
    """Without IDF, TRACK-2 (two common terms) would beat TRACK-24 (one rare term).

    TRACK-24 is inserted last, so the id tie-break cannot rescue it either.
    """
    documents = [
        private_document("TRACK-2", "Widget roles lacked depth."),
        private_document("TRACK-3", "Gadget roles lacked focus in Springfield."),
        private_document("TRACK-4", "Roles in Shelbyville lacked a budget."),
        private_document("TRACK-5", "Roles lacked a sponsor. Band A to B per month."),
        private_document("TRACK-24", "Declined after the call: no Quuxdb skills."),
        esco_skill(1, "use Quuxdb to store gadgets"),
    ]
    ingest(conn, documents, HashEmbedder(), Masker([]))


def issue_key_of(conn: Connection, chunk_id: int) -> object:
    row = conn.execute(
        "SELECT s.issue_key FROM chunks c JOIN sources s ON s.id = c.source_id WHERE c.id = %s",
        (chunk_id,),
    ).fetchone()
    return row[0] if row else None


def test_keyword_terms_drop_english_and_german_stopwords(conn: Connection) -> None:
    assert keyword_terms(conn, QUESTION) == ["lacked", "quuxdb", "roles"]
    german = keyword_terms(conn, "Welche Rollen wegen Quuxdb-Erfahrung?")
    assert "welche" not in german
    assert "quuxdb" in german


def test_keyword_terms_of_stopwords_only_is_empty(conn: Connection) -> None:
    assert keyword_terms(conn, "which on the") == []


def test_fuse_rewards_documents_found_by_both_rankings() -> None:
    fused = fuse([[1, 2, 3], [3, 4]])
    assert fused[0][0] == 3
    assert fused[0][1] == pytest.approx(1 / (RRF_K + 3) + 1 / (RRF_K + 1))
    assert [chunk_id for chunk_id, _ in fused] == [3, 1, 2, 4]


def test_fuse_of_nothing_is_empty() -> None:
    assert fuse([[], []]) == []


def test_keyword_ranking_puts_a_rare_term_above_two_common_ones(conn: Connection) -> None:
    seed(conn)
    ranking = keyword_ranking(conn, ["lacked", "quuxdb", "roles"], "private")
    assert issue_key_of(conn, ranking[0]) == "TRACK-24"


def test_idf_is_measured_within_the_collection(conn: Connection) -> None:
    seed(conn)
    many_esco = [esco_skill(n, f"Quuxdb skill number {n}") for n in range(10, 40)]
    ingest(conn, many_esco, HashEmbedder(), Masker([]))
    ranking = keyword_ranking(conn, ["lacked", "quuxdb", "roles"], "private")
    assert issue_key_of(conn, ranking[0]) == "TRACK-24"


def test_keyword_ranking_without_terms_is_empty(conn: Connection) -> None:
    seed(conn)
    assert keyword_ranking(conn, [], "all") == []


def test_terms_with_tsquery_syntax_characters_are_matched_literally(conn: Connection) -> None:
    ingest(conn, [private_document("TRACK-1", "notes")], HashEmbedder(), Masker([]))
    assert keyword_ranking(conn, ["a:b", "c&d", "e'f", "g|h", "!i", "(j)", "k\\l"], "all") == []


def test_hits_carry_citation_fields(conn: Connection) -> None:
    seed(conn)
    [hit, *_] = search(conn, HashEmbedder(), "Quuxdb skills", collection="private")
    assert hit.source_key == "TRACK-24"
    assert hit.source_type == "issue"
    assert hit.title == "TRACK-24 Acme Corp — Platform Engineer"
    assert hit.url == "https://tracker.example.com/issue/TRACK-24"
    assert hit.document_date == date(2026, 3, 1)
    assert "no Quuxdb skills" in hit.text
    assert hit.score > 0


def test_collection_filter(conn: Connection) -> None:
    seed(conn)
    esco = search(conn, HashEmbedder(), "Quuxdb", collection="esco")
    assert {hit.source_type for hit in esco} == {"esco_skill"}
    everything = search(conn, HashEmbedder(), "Quuxdb", collection="all", limit=10)
    assert {hit.source_type for hit in everything} == {"issue", "esco_skill"}


def test_filtered_vector_search_still_fills_the_page_through_the_hnsw_index(
    conn: Connection,
) -> None:
    private = [private_document(f"TRACK-{n}", f"private note {n}") for n in range(5)]
    crowd = [esco_skill(n, f"private note crowd {n}") for n in range(100, 400)]
    ingest(conn, private + crowd, HashEmbedder(), Masker([]))
    conn.execute("SET enable_seqscan = off")
    conn.execute("SET hnsw.ef_search = 10")
    hits = search(conn, HashEmbedder(), "private note", collection="private", limit=5)
    assert len(hits) == 5


def test_connections_enable_iterative_hnsw_scans(conn: Connection) -> None:
    assert conn.execute("SHOW hnsw.iterative_scan").fetchone() == ("relaxed_order",)


def test_limit_is_respected_and_validated(conn: Connection) -> None:
    seed(conn)
    assert len(search(conn, HashEmbedder(), "roles", collection="all", limit=2)) == 2
    with pytest.raises(ValueError, match="limit"):
        search(conn, HashEmbedder(), "roles", limit=0)
    with pytest.raises(ValueError, match="limit"):
        search(conn, HashEmbedder(), "roles", limit=51)


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_is_rejected(conn: Connection, query: str) -> None:
    with pytest.raises(ValueError, match="query"):
        search(conn, HashEmbedder(), query)


def test_unknown_collection_is_rejected(conn: Connection) -> None:
    with pytest.raises(ValueError, match="collection"):
        search(conn, HashEmbedder(), "roles", collection="secret")  # type: ignore[arg-type]


def test_stopword_only_query_still_uses_vectors(conn: Connection) -> None:
    seed(conn)
    assert search(conn, HashEmbedder(), "which on the", collection="all")


def test_empty_index_gives_no_hits(conn: Connection) -> None:
    assert search(conn, HashEmbedder(), "Quuxdb") == []


def test_query_longer_than_the_window_is_cut_for_the_vector_side(conn: Connection) -> None:
    embedder = HashEmbedder(max_words=5)
    long_query = "Quuxdb " + "posting text " * 50
    assert query_for_embedding(long_query, embedder) == "Quuxdb posting text posting text"
    seed(conn)
    assert search(conn, embedder, long_query, collection="private")


def test_index_model_must_match_the_embedder(conn: Connection) -> None:
    seed(conn)
    check_index_model(conn, HashEmbedder())

    class OtherModel(HashEmbedder):
        name = "other-model"

    with pytest.raises(ValueError, match="hash-test"):
        check_index_model(conn, OtherModel())


def test_cli_prints_ranked_citations(
    conn: Connection,
    schema: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed(conn)
    conn.commit()
    monkeypatch.setenv("CRAG__DATABASE__SCHEMA", schema)
    monkeypatch.setenv("CRAG__EMBEDDING__MODEL", "hash-test")
    assert main(["Quuxdb skills", "--collection", "private", "--limit", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("1. TRACK-24 ")
    assert "https://tracker.example.com/issue/TRACK-24" in lines[0]


def bulk_esco_chunks(conn: Connection, count: int) -> None:
    """Fast synthetic rows straight into the tables; ingest() would spend minutes on HNSW."""
    conn.execute(
        """
        WITH new_sources AS (
            INSERT INTO sources (collection, source_type, source_key, title)
            SELECT 'esco', 'esco_skill', 'bulk:' || n, 'skill ' || n
            FROM generate_series(1, %(count)s) AS n
            RETURNING id, title
        )
        INSERT INTO chunks (source_id, ordinal, text, embedding, embedding_model, search_text)
        SELECT id, 0, title || ' manage data team', array_fill(0.01, ARRAY[384])::vector,
               'hash-test', to_tsvector('simple', title || ' manage data team')
        FROM new_sources
        """,
        {"count": count},
    )


def test_keyword_ranking_scales_with_matches_not_rows_squared(conn: Connection) -> None:
    seed(conn)
    bulk_esco_chunks(conn, 3000)
    conn.execute("ANALYZE chunks")
    conn.execute("SET statement_timeout = '2s'")
    ranking = keyword_ranking(conn, ["data", "manage", "quuxdb", "team"], "all")
    assert len(ranking) == 50
    private = keyword_ranking(conn, ["lacked", "quuxdb", "roles"], "private")
    assert issue_key_of(conn, private[0]) == "TRACK-24"
