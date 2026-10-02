from datetime import date

import pytest

from crag.documents import Document
from crag.embedding import HashEmbedder
from crag.ingest import ingest
from crag.masking import Masker
from crag.search import RRF_K, fuse, keyword_ranking, keyword_terms, main, search
from crag.store import Connection
from tests.conftest import private_document


def seed(conn: Connection) -> None:
    documents = [
        private_document(
            "TRACK-24", "Recruiter call went well. Not put forward: no RAG experience."
        ),
        private_document("TRACK-9", "Roles in Linz and roles in Vienna. Many roles, many roles."),
        private_document("TRACK-8", "Salary band 5,500-6,400 per month. Roles for architects."),
        private_document("TRACK-3", "Applied with the EN CV. Roles in AI transformation."),
        Document(
            collection="esco",
            source_type="esco_skill",
            source_key="http://data.europa.eu/esco/skill/1",
            title="use retrieval-augmented generation",
            text="use retrieval-augmented generation (RAG) to ground answers",
            url="http://data.europa.eu/esco/skill/1",
        ),
    ]
    ingest(conn, documents, HashEmbedder(), Masker([]))


def test_keyword_terms_drop_english_and_german_stopwords(conn: Connection) -> None:
    assert keyword_terms(conn, "Which roles failed on RAG?") == ["failed", "rag", "roles"]
    assert "welche" not in keyword_terms(conn, "Welche Bewerbungen wegen RAG-Erfahrung?")
    assert "rag" in keyword_terms(conn, "Welche Bewerbungen wegen RAG-Erfahrung?")


def test_keyword_terms_of_stopwords_only_is_empty(conn: Connection) -> None:
    assert keyword_terms(conn, "which on the") == []


def test_fuse_rewards_documents_found_by_both_rankings() -> None:
    fused = fuse([[1, 2, 3], [3, 4]])
    assert fused[0][0] == 3
    assert fused[0][1] == pytest.approx(1 / (RRF_K + 3) + 1 / (RRF_K + 1))
    assert [chunk_id for chunk_id, _ in fused] == [3, 1, 2, 4]


def test_fuse_of_nothing_is_empty() -> None:
    assert fuse([[], []]) == []


def test_keyword_ranking_puts_a_rare_term_above_a_repeated_common_one(conn: Connection) -> None:
    seed(conn)
    ranking = keyword_ranking(conn, ["failed", "rag", "roles"], "private")
    top = conn.execute(
        "SELECT s.issue_key FROM chunks c JOIN sources s ON s.id = c.source_id WHERE c.id = %s",
        (ranking[0],),
    ).fetchone()
    assert top == ("TRACK-24",)


def test_keyword_ranking_without_terms_is_empty(conn: Connection) -> None:
    seed(conn)
    assert keyword_ranking(conn, [], "all") == []


def test_terms_with_tsquery_syntax_characters_are_matched_literally(conn: Connection) -> None:
    ingest(conn, [private_document("TRACK-1", "notes")], HashEmbedder(), Masker([]))
    assert keyword_ranking(conn, ["a:b", "c&d", "e'f", "g|h", "!i", "(j)", "k\\l"], "all") == []


def test_hits_carry_citation_fields(conn: Connection) -> None:
    seed(conn)
    [hit, *_] = search(conn, HashEmbedder(), "RAG experience", collection="private")
    assert hit.source_key == "TRACK-24"
    assert hit.source_type == "issue"
    assert hit.title == "TRACK-24 Acme Corp — Platform Engineer"
    assert hit.url == "https://tracker.example.com/issue/TRACK-24"
    assert hit.document_date == date(2026, 3, 1)
    assert "no RAG experience" in hit.text
    assert hit.score > 0


def test_collection_filter(conn: Connection) -> None:
    seed(conn)
    esco = search(conn, HashEmbedder(), "RAG", collection="esco")
    assert {hit.source_type for hit in esco} == {"esco_skill"}
    everything = search(conn, HashEmbedder(), "RAG", collection="all", limit=10)
    assert {hit.source_type for hit in everything} == {"issue", "esco_skill"}


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
    assert search(conn, HashEmbedder(), "RAG") == []


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
    assert main(["Which roles failed on RAG?", "--limit", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("1. TRACK-24 ")
    assert "https://tracker.example.com/issue/TRACK-24" in lines[0]
