# 0002 — Phase 1 scope, answer mode and guards

Date: 2026-10-01 · Status: accepted

## Context

Phase 1 (CRAG-1) is too large to review as one unit. `/ask` needs a language model to write
answers, but ADR 0001 forbids sending private documents to a hosted API, and no local model
runs on the development machine yet. The repository is public, so a leak would be permanent.

## Decision

**Three pull requests on CRAG-1**, each green and reviewed before the next:
1. Skeleton and guards: project layout, guard, git hooks, CI.
2. Storage and ingestion: Postgres + pgvector, loaders, PII masking, chunking, embeddings.
3. `/search` and `/ask` with citations, and the phase 1 done check.

**Answer mode.** `/ask` is extractive in phase 1: it returns the best passages with citations
and no generated text. An Ollama service is added as an opt-in Compose profile; when a model is
available, `/ask` also writes an answer from the passages. Comparing language models is
phase 3.

**Guards.** A leak is stopped at three points:
- `pre-commit` hook: scans staged content for email addresses, international phone numbers,
  IBANs and a private denylist of names and companies. Runs gitleaks when installed.
- `commit-msg` hook: the same scan on the commit message.
- CI: the scan on every tracked file and every commit message, plus gitleaks over the full
  history.

The denylist lives in `data/pii/denylist.txt`, which is git-ignored, so CI cannot use it. CI
covers patterns; the hooks cover names. Findings print the rule and location, never the
matched text, because CI logs are public.

## Consequences

- The hooks are the only check for private names. They must be installed in every clone
  (`scripts/install-hooks.sh`) and must never be skipped with `--no-verify`.
- Test data that has to look sensitive is assembled at runtime, so test files pass the scan.
