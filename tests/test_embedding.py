import math
import os
from collections.abc import Iterator

import pytest

from crag.embedding import FastEmbedder, HashEmbedder, cosine


@pytest.fixture(scope="module")
def local_model() -> Iterator[FastEmbedder]:
    """One model for the module, released before interpreter shutdown.

    An ONNX session still alive at exit, with test threads around, aborts the process on macOS.
    """
    if not os.environ.get("CRAG__TEST__FASTEMBED"):
        pytest.skip("downloads a 0.22 GB model; CI sets CRAG__TEST__FASTEMBED=1 with a model cache")
    model = FastEmbedder(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir=os.environ.get("CRAG__EMBEDDING__CACHE_DIR", "data/models"),
    )
    yield model
    model.close()


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


def test_local_multilingual_model_matches_across_languages(local_model: FastEmbedder) -> None:
    embedder = local_model
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


def test_local_model_refuses_text_beyond_its_token_window(local_model: FastEmbedder) -> None:
    embedder = local_model
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


def test_closed_local_model_refuses_to_embed() -> None:
    if not os.environ.get("CRAG__TEST__FASTEMBED"):
        pytest.skip("downloads a 0.22 GB model; CI sets CRAG__TEST__FASTEMBED=1 with a model cache")
    model = FastEmbedder(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir=os.environ.get("CRAG__EMBEDDING__CACHE_DIR", "data/models"),
    )
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.embed(["text"])
