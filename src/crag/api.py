"""HTTP API over the index. It serves private data without authentication, so it is bound to
loopback and refuses requests addressed to any other host name (DNS rebinding)."""

import http.client
import json
import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict
from typing import Annotated, Any

import psycopg
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from crag.answer import LanguageModel, OllamaClient, build_prompt
from crag.config import Settings, load_settings
from crag.embedding import Embedder
from crag.ingest import embedder_from_settings
from crag.search import MAX_LIMIT, CollectionFilter, Hit, check_index_model, search
from crag.store import Connection, connect, migrate

log = logging.getLogger("crag.api")

ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
ASK_PASSAGES = 5  # enough context for a short answer; more passages dilute a small local model
MAX_QUERY_CHARS = 4000  # room for a pasted job posting; the vector side uses what fits the model

ConnectionFactory = Callable[[], AbstractContextManager[Connection]]
Question = Annotated[str, Query(min_length=1, max_length=MAX_QUERY_CHARS)]


def create_app(
    connections: ConnectionFactory, embedder: Embedder, llm: LanguageModel | None
) -> FastAPI:
    app = FastAPI(title="Career RAG")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(ALLOWED_HOSTS))

    @contextmanager
    def database() -> Iterator[Connection]:
        try:
            context = connections()
            connection = context.__enter__()
        except (psycopg.OperationalError, ConnectionError) as error:
            raise HTTPException(503, "database unavailable") from error
        try:
            yield connection
        finally:
            context.__exit__(None, None, None)

    def require_question(text: str) -> str:
        if not text.strip():
            raise HTTPException(422, "q must not be blank")
        return text

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/search")
    def search_endpoint(
        q: Question,
        collection: CollectionFilter = "all",
        limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 10,
    ) -> dict[str, Any]:
        question = require_question(q)
        with database() as connection:
            hits = search(connection, embedder, question, collection, limit)
        return {"query": q, "hits": [_public(hit) for hit in hits]}

    @app.get("/ask")
    def ask(q: Question, collection: CollectionFilter = "all") -> dict[str, Any]:
        question = require_question(q)
        with database() as connection:
            hits = search(connection, embedder, question, collection, ASK_PASSAGES)
        answer, answer_error = _answer(llm, question, hits)
        citations = [{"number": n, **_public(hit)} for n, hit in enumerate(hits, start=1)]
        return {
            "question": q,
            "answer": answer,
            "answer_error": answer_error,
            "citations": citations,
        }

    return app


def _answer(
    llm: LanguageModel | None, question: str, hits: list[Hit]
) -> tuple[str | None, str | None]:
    """The citations are useful without an answer, so a failing model never fails the request."""
    if llm is None or not hits:
        return None, None
    try:
        return llm.generate(build_prompt(question, hits)), None
    except (OSError, ValueError, http.client.HTTPException) as error:
        log.warning(json.dumps({"event": "llm_failed", "error": type(error).__name__}))
        return None, "language model unavailable"


def _public(hit: Hit) -> dict[str, Any]:
    fields = asdict(hit)
    fields.pop("chunk_id")
    return fields


def _connections(settings: Settings) -> ConnectionFactory:
    """A fresh connection per request: survives database restarts and is closed afterwards."""

    @contextmanager
    def open_connection() -> Iterator[Connection]:
        with connect(settings.database_url, schema=settings.database_schema) as connection:
            connection.autocommit = True
            yield connection

    return open_connection


def app_from_settings() -> FastAPI:
    """Entry point: `uvicorn --factory crag.api:app_from_settings --host 127.0.0.1`."""
    settings = load_settings()
    embedder = embedder_from_settings(settings)
    with connect(settings.database_url, schema=settings.database_schema) as connection:
        migrate(connection)
        check_index_model(connection, embedder)
    llm = (
        OllamaClient(settings.llm_ollama_url, settings.llm_model)
        if settings.llm_ollama_url and settings.llm_model
        else None
    )
    return create_app(_connections(settings), embedder, llm)
