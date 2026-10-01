import re
from pathlib import Path

from crag.guard import EMAIL, IBAN, INTERNATIONAL_PHONE

NATIONAL_PHONE = re.compile(r"(?<![\w.])0\d{1,4}(?:[\s/-]?\d{2,4}){2,3}(?![\w.])")
SLASHED_OR_DASHED_DATE = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
MIN_PHONE_DIGITS = 7  # shortest national number with area code; also rules out "09 30 45" times


def load_people(path: Path) -> list[str]:
    if not path.exists():
        return []
    names = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [name for name in names if name and not name.startswith("#")]


def _mask_national_phone(match: re.Match[str]) -> str:
    candidate = match.group(0)
    digits = sum(character.isdigit() for character in candidate)
    if digits < MIN_PHONE_DIGITS or SLASHED_OR_DASHED_DATE.fullmatch(candidate):
        return candidate
    return "[phone]"


class Masker:
    """Replaces third-party personal data before text is chunked or embedded.

    Company names stay: they are what the archive is searched by, and they are not personal data.
    """

    def __init__(self, people: list[str]) -> None:
        longest_first = sorted(people, key=len, reverse=True)
        alternatives = "|".join(re.escape(name) for name in longest_first)
        self._people = (
            re.compile(r"(?<!\w)(?:" + alternatives + r")(?!\w)", re.IGNORECASE) if people else None
        )

    @classmethod
    def from_file(cls, path: Path) -> "Masker":
        """For every entry point that ingests private documents: a missing list is an error."""
        if not path.exists():
            raise ValueError(
                f"people file {path} is missing; private documents would keep third-party "
                "names. Create it (it may be empty) or set CRAG__PII__PEOPLE_FILE."
            )
        return cls(load_people(path))

    def mask(self, text: str) -> str:
        text = EMAIL.sub("[email]", text)
        text = IBAN.sub("[iban]", text)
        text = INTERNATIONAL_PHONE.sub("[phone]", text)
        text = NATIONAL_PHONE.sub(_mask_national_phone, text)
        if self._people is not None:
            text = self._people.sub("[person]", text)
        return text
