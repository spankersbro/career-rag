# Career RAG — project context

**Standards and tracking.** Shared rules: @../../dev-standards/CLAUDE.md and the done checklist
@../../dev-standards/GATES.md. Issues live in YouTrack project **CRAG** (`youtrack` MCP server).
Commit subjects: `type: CRAG-123 - description`. Decisions: `docs/decisions/`.

**GitHub account: `spankersbro` only.** Commits, pushes, `gh` calls, issues and PRs all use it,
never any other account. The repo-local git config pins the author identity and a credential
helper that always takes the `spankersbro` token, whatever `gh` account is active. For `gh`
commands, set `GH_TOKEN="$(gh auth token --user spankersbro)"`. Do not change the local git
config.

**What this is.** A RAG system over a personal job-search archive plus the public ESCO skills
catalogue. It matches a posting against evidence from past applications, flags gaps that were
never claimed, and checks salary bands against a floor. See ADR 0001.

**This repository is public. No sensitive data may go into it at all.**
- Code, synthetic fixtures and documentation only. The archive and everything derived from it
  (chunks, embeddings, eval questions and answers, logs) stay in `data/`, which is git-ignored.
- Never commit, quote or paraphrase archive content: not in fixtures, tests, docs, commit
  messages, issue references or example output. Fixtures are invented from scratch.
- Never name real companies, recruiters, referees, employers, salary figures or application
  outcomes anywhere in the repo.
- Ingestion masks third-party personal data before anything is chunked or embedded.
- Private documents are embedded with a local model. Only public data (ESCO) may go to a hosted
  API.
- Before every commit, check the staged diff for any of the above. When in doubt, leave it out
  and ask.
