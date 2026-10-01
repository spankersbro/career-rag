import csv
from pathlib import Path

from crag.documents import Document

REQUIRED_COLUMNS = {"conceptUri", "preferredLabel", "altLabels", "description"}


def parse_skills_csv(path: Path) -> list[Document]:
    """Reads an ESCO skills export (skills_<lang>.csv) into one public document per skill."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is not an ESCO skills export; missing columns: {sorted(missing)}"
            )
        return [_skill_document(row) for row in reader]


def _skill_document(row: dict[str, str]) -> Document:
    label = row["preferredLabel"].strip()
    parts = [label]
    if row["description"].strip():
        parts.append(row["description"].strip())
    alternatives = [alt.strip() for alt in row["altLabels"].splitlines() if alt.strip()]
    if alternatives:
        parts.append("Also known as: " + "; ".join(alternatives))
    return Document(
        collection="esco",
        source_type="esco_skill",
        source_key=row["conceptUri"],
        title=label,
        text="\n\n".join(parts),
        url=row["conceptUri"],
    )
