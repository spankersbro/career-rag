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
uv run python -m crag.ingest changelog path/to/CHANGELOG.md
uv run python -m crag.ingest esco path/to/skills_en.csv
```

- Third-party names to mask go in `data/pii/people.txt`, one per line. Emails, phone numbers
  and IBANs are masked automatically. Masking applies to the private collection only.
- The ESCO skills CSV is downloaded by hand from the ESCO portal (classification download,
  CSV format) and kept under `data/`.
- Integration tests need the database: `CRAG__DATABASE__URL=postgresql://crag:crag@127.0.0.1:5433/crag`.

Status: phase 1 in progress.
