CREATE TABLE IF NOT EXISTS sources (
    id bigserial PRIMARY KEY,
    collection text NOT NULL CHECK (collection IN ('private', 'esco')),
    source_type text NOT NULL,
    source_key text NOT NULL,
    title text NOT NULL,
    url text,
    issue_key text,
    document_date date,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (collection, source_key)
);

CREATE TABLE IF NOT EXISTS chunks (
    id bigserial PRIMARY KEY,
    source_id bigint NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    text text NOT NULL,
    embedding vector(384) NOT NULL,
    embedding_model text NOT NULL,
    UNIQUE (source_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS sources_issue_key ON sources (issue_key);

-- Keyword side of hybrid search. 'simple' keeps words as written: no stemming, so acronyms
-- like RAG match exactly, and German and English text share one configuration.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_text tsvector;
UPDATE chunks c SET search_text = to_tsvector('simple', s.title || E'\n' || c.text)
FROM sources s WHERE s.id = c.source_id AND c.search_text IS NULL;
DO $$
BEGIN
    -- Only once: SET NOT NULL takes an exclusive lock and rescans the table.
    IF EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'chunks'::regclass AND attname = 'search_text' AND NOT attnotnull
    ) THEN
        ALTER TABLE chunks ALTER COLUMN search_text SET NOT NULL;
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS chunks_search_text ON chunks USING gin (search_text);
