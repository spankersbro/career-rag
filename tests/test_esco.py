from pathlib import Path

import pytest

from crag.loaders.esco import parse_skills_csv

HEADER = (
    "conceptType,conceptUri,skillType,reuseLevel,preferredLabel,altLabels,hiddenLabels,"
    "status,modifiedDate,scopeNote,definition,inScheme,description\n"
)


def write_csv(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "skills_en.csv"
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def test_one_public_document_per_skill(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "KnowledgeSkillCompetence,http://data.europa.eu/esco/skill/1,skill/competence,"
        'cross-sector,use databases,"query databases\nwork with SQL",,released,,,,,'
        '"Use software tools to manage data."\n',
    )
    [document] = parse_skills_csv(path)
    assert document.collection == "esco"
    assert document.source_type == "esco_skill"
    assert document.source_key == "http://data.europa.eu/esco/skill/1"
    assert document.url == "http://data.europa.eu/esco/skill/1"
    assert document.title == "use databases"
    assert document.text == (
        "use databases\n\nUse software tools to manage data.\n\n"
        "Also known as: query databases; work with SQL"
    )


def test_skill_without_description_or_alt_labels(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "KnowledgeSkillCompetence,http://data.europa.eu/esco/skill/2,skill/competence,"
        "cross-sector,plan sprints,,,released,,,,,\n",
    )
    [document] = parse_skills_csv(path)
    assert document.text == "plan sprints"


def test_missing_columns_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_text("name,value\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conceptUri"):
        parse_skills_csv(path)


def test_short_row_without_label_is_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "KnowledgeSkillCompetence,http://data.europa.eu/esco/skill/3\n")
    with pytest.raises(ValueError, match="preferredLabel"):
        parse_skills_csv(path)
