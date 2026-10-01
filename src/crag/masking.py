import re
from pathlib import Path

from crag.guard import EMAIL, IBAN, INTERNATIONAL_PHONE


def load_people(path: Path) -> list[str]:
    if not path.exists():
        return []
    names = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [name for name in names if name and not name.startswith("#")]


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

    def mask(self, text: str) -> str:
        text = EMAIL.sub("[email]", text)
        text = INTERNATIONAL_PHONE.sub("[phone]", text)
        text = IBAN.sub("[iban]", text)
        if self._people is not None:
            text = self._people.sub("[person]", text)
        return text
