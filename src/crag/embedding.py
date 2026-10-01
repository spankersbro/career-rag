import hashlib
import math
import re
from typing import Protocol

from fastembed import TextEmbedding

WORD = re.compile(r"\w+")


class Embedder(Protocol):
    name: str
    dimension: int
    local: bool

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """Runs the model on this machine, so private text never leaves it."""

    local = True

    def __init__(self, model: str, cache_dir: str) -> None:
        self.name = model
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        self.dimension = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]


class HashEmbedder:
    """Bag-of-words hashing: deterministic, instant, no model. For tests only."""

    name = "hash-test"

    def __init__(self, dimension: int = 384, local: bool = True) -> None:
        self.dimension = dimension
        self.local = local

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in WORD.findall(text.lower()):
            bucket = (
                int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % self.dimension
            )
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0
