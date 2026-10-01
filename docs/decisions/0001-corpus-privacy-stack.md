# 0001 — Corpus, privacy model and stack

Date: 2026-10-01 · Status: accepted

## Context

AI architect roles ask for hands-on RAG and vector database work: chunking, embeddings, vector
stores, retrieval quality, model selection, cost and memory. Using a managed platform that does
retrieval for you does not show this. The project has to show work designed end to end, and it
has to be useful every day, not only as a demo.

## Decision

**Corpus: a personal job-search archive, plus ESCO.**
- Application records from an issue tracker: role, salary band, gaps, history, outcome.
- Postings, and the CVs and cover letters as sent.
- Change logs and interview notes from a CV repository.
- ESCO (EU skills and occupations catalogue, DE/EN, about 14,000 skills). It gives the vector
  side real scale and a shared vocabulary to map postings and evidence onto.

**Privacy model.**
- The repository is public and holds code, synthetic fixtures and documentation only.
- The archive and everything derived from it stay local in `data/`, which is git-ignored.
- Third-party personal data is masked at ingestion, before chunking.
- Private documents use a local embedding model. Only public data (ESCO) may use a hosted one.
- CI scans every commit for personal data and secrets and fails on a hit.

**Stack.**
- Python 3.12, FastAPI.
- Postgres with pgvector as the primary store. Qdrant is run against the same eval set later,
  to compare.
- Docker Compose for local runs. GitHub Actions for CI.
- Evaluation runs in CI on a synthetic fixture corpus and blocks merges that lower retrieval
  quality. The real eval set runs locally against `data/`.

**Phases.**
1. Ingestion and basic search with citations (pgvector).
2. Evaluation set and CI gate.
3. Hybrid search (BM25 + vectors), reranking, embedding and LLM comparisons.
4. MCP server, cost tracking per query, access control, write-up.

## Consequences

- The real eval set and its expected answers are private and stay in `data/`.
- CI cannot see the real corpus, so CI quality numbers come from synthetic fixtures. Real
  numbers are reported by hand, in aggregate only.
- At a few hundred private documents, scale says little. ESCO carries the scale argument.
