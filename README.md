# Career RAG

A retrieval-augmented assistant over one person's job search. It matches a new posting against
evidence from past CVs and cover letters, flags gaps that were never claimed, and checks salary
bands against a floor.

The corpus is private and stays local. This repository holds only code, synthetic test
fixtures and documentation.

- Stack: Python, FastAPI, Postgres + pgvector, Docker Compose, GitHub Actions.
- Decisions: [`docs/decisions/`](docs/decisions/).

Status: setup.
