# 0003 — Hybrid search and token-window chunking

Date: 2026-10-02 · Status: accepted

## Context

With vectors only, the phase 1 done question ("Which roles failed on RAG?") did not return the
right record in the top 5. Two causes were found and measured:

- The embedding model reads 128 tokens and silently drops the rest. Chunks were sized at 1200
  characters, so most of each chunk was never embedded. Two texts that differ only after 150
  words got identical vectors (cosine 1.0).
- Short acronyms and names, the words such questions turn on, embed poorly in a small model.

## Decision

- **Chunk to the model's window.** The embedder exposes its token window and refuses text that
  does not fit; chunking packs paragraphs, then sentences, then words, so each chunk plus its
  title fits.
- **Hybrid retrieval.** A vector ranking and a keyword ranking, fused with reciprocal rank
  fusion (k = 60, the standard constant).
- **Keyword ranking** uses Postgres full-text search with the `simple` configuration, so
  acronyms match exactly and German and English share one index. English and German stopwords
  are dropped from the query. Chunks are scored by the summed IDF of matched terms, because the
  built-in ranking functions have no IDF and reward repeated common words.
- **Answers** are extractive by default. A language model writes an answer only when it runs
  locally (Ollama, opt-in Compose profile); remote URLs are refused.

## Consequences

- Real-data check: the done question returns JOBS-level evidence first in English and German.
  Questions about people or numbers ("who kept me on record", "pay below my salary") are still
  weak; measuring and tuning is phase 2 (evaluation set) and phase 3 (reranking, models).
- Re-embedding with a larger model needs a new window size and an empty index (mixing models
  is refused).
