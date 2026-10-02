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
