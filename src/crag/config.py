import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str
    database_schema: str | None
    embedding_model: str
    embedding_cache_dir: str
    people_file: Path
    youtrack_base_url: str | None
    youtrack_token: str | None
    youtrack_query: str | None
    llm_ollama_url: str | None
    llm_model: str | None


def load_settings(environ: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ) if environ is None else environ
    return Settings(
        database_url=env.get("CRAG__DATABASE__URL", "postgresql://crag:crag@127.0.0.1:5433/crag"),
        database_schema=env.get("CRAG__DATABASE__SCHEMA") or None,
        embedding_model=env.get(
            "CRAG__EMBEDDING__MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        embedding_cache_dir=env.get("CRAG__EMBEDDING__CACHE_DIR", "data/models"),
        people_file=Path(env.get("CRAG__PII__PEOPLE_FILE", "data/pii/people.txt")),
        youtrack_base_url=env.get("CRAG__YOUTRACK__BASE_URL") or None,
        youtrack_token=env.get("CRAG__YOUTRACK__TOKEN") or None,
        youtrack_query=env.get("CRAG__YOUTRACK__QUERY") or None,
        llm_ollama_url=env.get("CRAG__LLM__OLLAMA_URL") or None,
        llm_model=env.get("CRAG__LLM__MODEL") or None,
    )
