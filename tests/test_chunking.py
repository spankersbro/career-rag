from collections.abc import Callable

import pytest

from crag.chunking import chunk_text


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("First paragraph.\n\nSecond paragraph.", max_chars=100) == [
        "First paragraph.\n\nSecond paragraph."
    ]


def test_paragraphs_are_packed_up_to_the_limit() -> None:
    paragraphs = ["a" * 40, "b" * 40, "c" * 40]
    assert chunk_text("\n\n".join(paragraphs), max_chars=90) == [
        "a" * 40 + "\n\n" + "b" * 40,
        "c" * 40,
    ]


def test_long_paragraph_is_split_at_sentences() -> None:
    text = "One two three. Four five six. Seven eight nine."
    assert chunk_text(text, max_chars=30) == ["One two three. Four five six.", "Seven eight nine."]


def test_sentence_longer_than_limit_is_cut_hard() -> None:
    assert chunk_text("x" * 25, max_chars=10) == ["x" * 10, "x" * 10, "x" * 5]


def test_every_chunk_respects_the_limit() -> None:
    text = "\n\n".join(f"Sentence {i} has some words. Another one here." for i in range(50))
    chunks = chunk_text(text, max_chars=120)
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(
        " ", ""
    )


@pytest.mark.parametrize("text", ["", "   ", "\n\n\n"])
def test_blank_text_gives_no_chunks(text: str) -> None:
    assert chunk_text(text) == []


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        chunk_text("text", max_chars=0)


def word_budget(limit: int) -> Callable[[str], bool]:
    return lambda piece: len(piece.split()) <= limit


def test_fits_predicate_limits_chunks_below_max_chars() -> None:
    text = "one two three four. five six seven eight.\n\nnine ten eleven twelve."
    chunks = chunk_text(text, max_chars=1000, fits=word_budget(4))
    assert chunks == ["one two three four.", "five six seven eight.", "nine ten eleven twelve."]


def test_sentence_over_the_budget_is_split_at_words() -> None:
    assert chunk_text("a b c d e f g", fits=word_budget(3)) == ["a b c", "d e f", "g"]


def test_word_over_the_budget_is_cut_into_fitting_parts() -> None:
    def short(piece: str) -> bool:
        return len(piece) <= 4

    assert chunk_text("abcdefghij", fits=short) == ["abcd", "efgh", "ij"]
