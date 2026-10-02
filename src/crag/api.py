"""HTTP API over the index. It serves private data: bind it to 127.0.0.1 only."""

from dataclasses import asdict
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query

from crag.answer import LanguageModel, OllamaClient, build_prompt
from crag.config import load_settings
from crag.embedding import Embedder
from crag.ingest import embedder_from_settings
from crag.search import MAX_LIMIT, CollectionFilter, Hit, search
from crag.store import Connection, connect, migrate

ASK_PASSAGES = 5  # enough context for a short answer; more passages dilute a small local model


def create_app(
    connection: Connection | None, embedder: Embedder, llm: LanguageModel | None
) -> FastAPI:
    app = FastAPI(title="Career RAG")

    def require_connection() -> Connection:
        if connection is None:
            raise HTTPException(503, "no database connection")
        return connection

    def require_question(text: str) -> str:
        if not text.strip():
            raise HTTPException(422, "q must not be blank")
        return text

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/search")
    def search_endpoint(
        q: Annotated[str, Query(min_length=1)],
        collection: CollectionFilter = "all",
        limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 10,
    ) -> dict[str, Any]:
        hits = search(require_connection(), embedder, require_question(q), collection, limit)
        return {"query": q, "hits": [_public(hit) for hit in hits]}

    @app.get("/ask")
    def ask(
        q: Annotated[str, Query(min_length=1)],
        collection: CollectionFilter = "all",
    ) -> dict[str, Any]:
        hits = search(require_connection(), embedder, require_question(q), collection, ASK_PASSAGES)
        answer = llm.generate(build_prompt(q, hits)) if llm and hits else None
        citations = [{"number": n, **_public(hit)} for n, hit in enumerate(hits, start=1)]
        return {"question": q, "answer": answer, "citations": citations}

    return app


def _public(hit: Hit) -> dict[str, Any]:
    fields = asdict(hit)
    fields.pop("chunk_id")
    return fields


def app_from_settings() -> FastAPI:
    """Entry point: `uvicorn --factory crag.api:app_from_settings --host 127.0.0.1`."""
    settings = load_settings()
    connection = connect(settings.database_url, schema=settings.database_schema)
    migrate(connection)
    connection.autocommit = True
    llm = (
        OllamaClient(settings.llm_ollama_url, settings.llm_model)
        if settings.llm_ollama_url and settings.llm_model
        else None
    )
    return create_app(connection, embedder_from_settings(settings), llm)
