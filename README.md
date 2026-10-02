# Career RAG

A retrieval-augmented assistant over one person's job search. It matches a new posting against
evidence from past CVs and cover letters, flags gaps that were never claimed, and checks salary
bands against a floor.

The corpus is private and stays local. This repository holds only code, synthetic test
fixtures and documentation.

- Stack: Python, FastAPI, Postgres + pgvector, Docker Compose, GitHub Actions.
- Decisions: [`docs/decisions/`](docs/decisions/).

## Development

```sh
uv sync
scripts/install-hooks.sh   # required: blocks personal data from being committed
uv run pytest
uv run ruff check . && uv run mypy
```

## Loading data

```sh
cp .env.example .env            # fill in the tracker URL, token and query; never commit .env
docker compose up -d db
set -a; . ./.env; set +a
uv run python -m crag.ingest youtrack              # issues, comments, PDF attachments
uv run python -m crag.ingest changelog path/to/CHANGELOG.md --label cv
uv run python -m crag.ingest esco path/to/skills_en.csv
```

- Third-party names to mask go in `data/pii/people.txt`, one per line. Emails, phone numbers
  and IBANs are masked automatically. Masking applies to the private collection only.
- The ESCO skills CSV is downloaded by hand from the ESCO portal (classification download,
  CSV format) and kept under `data/`.
- Integration tests need the database: `CRAG__DATABASE__URL=postgresql://crag:crag@127.0.0.1:5433/crag`.

## Searching

```sh
uv run python -m crag.search "Which roles failed on RAG?" --collection private
uv run uvicorn --factory crag.api:app_from_settings --host 127.0.0.1 --port 8000
curl -s "http://127.0.0.1:8000/search?q=RAG%20experience&collection=private&limit=5"
curl -s "http://127.0.0.1:8000/ask?q=Which%20roles%20failed%20on%20RAG%3F"
```

- The API serves private data and has no authentication: bind it to 127.0.0.1 only.
- `/ask` returns the top passages with numbered citations. With a local model it also writes
  an answer from those passages:
  `docker compose --profile llm up -d ollama`,
  `docker compose exec ollama ollama pull <model>`, then set `CRAG__LLM__OLLAMA_URL` and
  `CRAG__LLM__MODEL`. Remote model URLs are refused.

Status: phase 1 in progress.
