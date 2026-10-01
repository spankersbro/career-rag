from pathlib import Path

from crag.masking import Masker, load_people

AT = "@"
PLUS = "+"


def test_masks_emails_including_placeholder_domains() -> None:
    masker = Masker([])
    text = "write to jane.doe" + AT + "mailbox.test or info" + AT + "example.com"
    assert masker.mask(text) == "write to [email] or [email]"


def test_masks_phone_numbers_and_iban() -> None:
    masker = Masker([])
    text = "call " + PLUS + "43 660 1234567, pay to " + "AT61" + " 1904 3002 3457 3201"
    assert masker.mask(text) == "call [phone], pay to [iban]"


def test_masks_people_as_whole_words_case_insensitively() -> None:
    masker = Masker(["Jane Roe", "Roe"])
    assert (
        masker.mask("Call with jane roe, then Roe again.")
        == "Call with [person], then [person] again."
    )
    assert masker.mask("Monroe is a street") == "Monroe is a street"


def test_longer_names_win_over_shorter_prefixes() -> None:
    masker = Masker(["Jane", "Jane Roe"])
    assert masker.mask("Jane Roe") == "[person]"


def test_leaves_company_names_and_numbers_alone() -> None:
    masker = Masker(["Jane Roe"])
    text = "Acme Corp, band 5,200-5,700 per month, version 1.2.3"
    assert masker.mask(text) == text


def test_empty_text() -> None:
    assert Masker(["Jane Roe"]).mask("") == ""


def test_people_file_skips_blanks_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "people.txt"
    path.write_text("# recruiters\nJane Roe\n\n  Max Mustermann \n", encoding="utf-8")
    assert load_people(path) == ["Jane Roe", "Max Mustermann"]


def test_missing_people_file_means_no_names(tmp_path: Path) -> None:
    assert load_people(tmp_path / "absent.txt") == []
