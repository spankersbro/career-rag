import math
import os

import pytest

from crag.embedding import FastEmbedder, HashEmbedder, cosine


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashEmbedder(dimension=16)
    [first, second] = embedder.embed(["vector database", "vector database"])
    assert first == second
    assert len(first) == 16
    assert math.isclose(sum(value * value for value in first), 1.0)


def test_hash_embedder_ranks_shared_words_higher() -> None:
    [query, related, unrelated] = HashEmbedder().embed(
        ["quuxdb gap", "declined for quuxdb gap", "weather report"]
    )
    assert cosine(query, related) > cosine(query, unrelated)


def test_hash_embedder_handles_empty_text() -> None:
    [vector] = HashEmbedder(dimension=8).embed([""])
    assert vector == [0.0] * 8


@pytest.mark.skipif(
    not os.environ.get("CRAG__TEST__FASTEMBED"),
    reason="downloads a 0.22 GB model; CI sets CRAG__TEST__FASTEMBED=1 with a model cache",
)
def test_local_multilingual_model_matches_across_languages() -> None:
    embedder = FastEmbedder(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir=os.environ.get("CRAG__EMBEDDING__CACHE_DIR", "data/models"),
    )
    assert embedder.local
    assert embedder.dimension == 384
    [english, german, unrelated] = embedder.embed(
        [
            "experience with vector databases",
            "Erfahrung mit Vektordatenbanken",
            "sunny weather today",
        ]
    )
    assert cosine(english, german) > cosine(english, unrelated)


@pytest.mark.skipif(
    not os.environ.get("CRAG__TEST__FASTEMBED"),
    reason="downloads a 0.22 GB model; CI sets CRAG__TEST__FASTEMBED=1 with a model cache",
)
def test_local_model_refuses_text_beyond_its_token_window() -> None:
    embedder = FastEmbedder(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir=os.environ.get("CRAG__EMBEDDING__CACHE_DIR", "data/models"),
    )
    assert embedder.max_tokens == 128
    assert embedder.fits("word " * 100)
    assert not embedder.fits("word " * 200)
    with pytest.raises(ValueError, match="128-token window"):
        embedder.embed(["word " * 200])


def test_hash_embedder_word_budget() -> None:
    assert HashEmbedder(max_words=2).fits("one two")
    assert not HashEmbedder(max_words=2).fits("one two three")
    assert HashEmbedder().fits("word " * 10_000)


def test_hash_embedder_refuses_text_beyond_its_window_like_the_real_model() -> None:
    with pytest.raises(ValueError, match="2-word window"):
        HashEmbedder(max_words=2).embed(["one two three"])
