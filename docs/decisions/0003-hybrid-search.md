# 0003 — Hybrid search and token-window chunking

Date: 2026-10-02 · Status: accepted

## Context

With vectors only, the phase 1 done question did not return the right record in the top 5.
Two causes were found and measured:

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

- Real-data check: the done question now returns the right record first, in English and German.
  Questions about people, or ones that need arithmetic over figures, are still weak; measuring
  and tuning is phase 2 (evaluation set) and phase 3 (reranking, models).
- IDF is computed within the searched collection, so a large public catalogue does not skew
  weights for the private archive. Every keyword lookup goes through the GIN index. Measured
  on 20,000 chunks: a full search takes about 70 ms when common words match nearly every
  chunk, and 15-20 ms for rarer terms. A test with 3,000 chunks and a 2-second statement
  timeout guards against the quadratic plan an earlier draft had.
- Loading through `ingest` updates the HNSW index row by row: about 10 minutes for 20,000
  chunks. A bulk path for the ESCO catalogue is a follow-up.
- The API opens a connection per request and allows only loopback Host headers. A local
  language model is reached without any proxy; if it fails, the citations are still returned.
- Re-embedding with a larger model needs a new window size and an empty index (mixing models
  is refused).
