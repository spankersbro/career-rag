import hashlib
import math
import re
from typing import Protocol

from fastembed import TextEmbedding
from tokenizers import Tokenizer

WORD = re.compile(r"\w+")


class Embedder(Protocol):
    name: str
    dimension: int
    local: bool

    def fits(self, text: str) -> bool: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """Runs the model on this machine, so private text never leaves it.

    The model reads a fixed number of tokens and silently drops the rest, so `embed` refuses
    text that does not fit; chunking uses `fits` to stay inside the window.
    """

    local = True

    def __init__(self, model: str, cache_dir: str) -> None:
        self.name = model
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        model_tokenizer = self._model.model.tokenizer  # type: ignore[attr-defined]
        self.max_tokens: int = model_tokenizer.truncation["max_length"]
        self._tokenizer = Tokenizer.from_str(model_tokenizer.to_str())
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self.dimension = len(next(iter(self._model.embed(["dimension probe"]))))

    def token_count(self, text: str) -> int:
        return len(self._tokenizer.encode(text).ids)

    def fits(self, text: str) -> bool:
        return self.token_count(text) <= self.max_tokens

    def embed(self, texts: list[str]) -> list[list[float]]:
        too_long = [text[:60] for text in texts if not self.fits(text)]
        if too_long:
            raise ValueError(
                f"{len(too_long)} text(s) exceed the {self.max_tokens}-token window of {self.name}"
            )
        return [vector.tolist() for vector in self._model.embed(texts)]


class HashEmbedder:
    """Bag-of-words hashing: deterministic, instant, no model. For tests only."""

    name = "hash-test"

    def __init__(
        self, dimension: int = 384, local: bool = True, max_words: int | None = None
    ) -> None:
        self.dimension = dimension
        self.local = local
        self.max_words = max_words

    def fits(self, text: str) -> bool:
        return self.max_words is None or len(WORD.findall(text)) <= self.max_words

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
