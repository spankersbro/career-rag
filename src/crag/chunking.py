import re

DEFAULT_MAX_CHARS = 1200

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Packs whole paragraphs into chunks; splits a paragraph only when it alone is too long."""
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    pieces = [
        piece
        for paragraph in PARAGRAPH_BREAK.split(text.strip())
        if paragraph.strip()
        for piece in _split_paragraph(paragraph.strip(), max_chars)
    ]
    return _pack(pieces, "\n\n", max_chars)


def _split_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = [
        part for sentence in SENTENCE_END.split(paragraph) for part in _cut(sentence, max_chars)
    ]
    return _pack(sentences, " ", max_chars)


def _cut(sentence: str, max_chars: int) -> list[str]:
    return [sentence[start : start + max_chars] for start in range(0, len(sentence), max_chars)]


def _pack(pieces: list[str], separator: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = current + separator + piece if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        current = piece
    if current:
        chunks.append(current)
    return chunks
