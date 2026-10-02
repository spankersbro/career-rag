import re
from collections.abc import Callable

DEFAULT_MAX_CHARS = 1200

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

Fits = Callable[[str], bool]


def chunk_text(
    text: str, max_chars: int = DEFAULT_MAX_CHARS, fits: Fits | None = None
) -> list[str]:
    """Packs whole paragraphs into chunks; splits a paragraph only when it alone is too long.

    A chunk is at most `max_chars` long and, when given, satisfies `fits` (for example the
    embedding model's token window), splitting down to sentences, words and finally characters.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    accepts = _limit(max_chars, fits)
    pieces = [
        piece
        for paragraph in PARAGRAPH_BREAK.split(text.strip())
        if paragraph.strip()
        for piece in _split(paragraph.strip(), accepts)
    ]
    return _pack(pieces, "\n\n", accepts)


def _limit(max_chars: int, fits: Fits | None) -> Fits:
    if fits is None:
        return lambda piece: len(piece) <= max_chars
    return lambda piece: len(piece) <= max_chars and fits(piece)


def _split(paragraph: str, accepts: Fits) -> list[str]:
    if accepts(paragraph):
        return [paragraph]
    sentences = [
        part
        for sentence in SENTENCE_END.split(paragraph)
        for part in _split_sentence(sentence, accepts)
    ]
    return _pack(sentences, " ", accepts)


def _split_sentence(sentence: str, accepts: Fits) -> list[str]:
    if accepts(sentence):
        return [sentence]
    words = [part for word in sentence.split() for part in _cut_word(word, accepts)]
    return _pack(words, " ", accepts)


def _cut_word(word: str, accepts: Fits) -> list[str]:
    """Last resort for a single token-heavy string: the longest prefixes that still fit."""
    parts = []
    while word:
        end = _longest_fitting_prefix(word, accepts)
        parts.append(word[:end])
        word = word[end:]
    return parts


def _longest_fitting_prefix(word: str, accepts: Fits) -> int:
    """Binary search: tokenising every prefix length would be quadratic on long blobs."""
    low, high = 1, len(word)
    while low < high:
        middle = (low + high + 1) // 2
        if accepts(word[:middle]):
            low = middle
        else:
            high = middle - 1
    return low


def _pack(pieces: list[str], separator: str, accepts: Fits) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = current + separator + piece if current else piece
        if accepts(candidate):
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = piece
    if current:
        chunks.append(current)
    return chunks
