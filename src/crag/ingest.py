import argparse
import json
import logging
import sys
import urllib.error
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from crag.chunking import DEFAULT_MAX_CHARS, chunk_text
from crag.config import Settings, load_settings
from crag.documents import Document
from crag.embedding import Embedder, FastEmbedder, HashEmbedder
from crag.loaders.changelog import parse_changelog
from crag.loaders.esco import parse_skills_csv
from crag.loaders.youtrack import YouTrackClient, issue_documents, pdf_attachments, pdf_text
from crag.masking import Masker, load_people
from crag.store import (
    EMBEDDING_DIMENSION,
    Connection,
    connect,
    delete_sources_except,
    embedding_models,
    migrate,
    replace_source,
)

log = logging.getLogger("crag.ingest")

SOURCE_TYPES = {
    "youtrack": frozenset({"issue", "issue_comment", "issue_attachment"}),
    "changelog": frozenset({"changelog"}),
    "esco": frozenset({"esco_skill"}),
}


@dataclass(frozen=True)
class IngestResult:
    sources: int
    chunks: int
    removed: int = 0
    stored_keys: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Batch:
    """A complete load of some source types; sources of those types not in it are stale."""

    documents: list[Document]
    source_types: frozenset[str]
    unavailable_keys: frozenset[str] = field(default_factory=frozenset)


def ingest(
    connection: Connection,
    documents: Iterable[Document],
    embedder: Embedder,
    masker: Masker,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> IngestResult:
    if embedder.dimension != EMBEDDING_DIMENSION:
        raise ValueError(
            f"embedding dimension {embedder.dimension} does not match "
            f"the schema ({EMBEDDING_DIMENSION})"
        )
    other_models = embedding_models(connection) - {embedder.name}
    if other_models:
        raise ValueError(
            f"index already holds vectors from {sorted(other_models)}; "
            f"re-embedding with {embedder.name} needs an empty index"
        )
    sources = chunks = 0
    stored_keys: set[str] = set()
    for document in documents:
        if document.collection == "private" and not embedder.local:
            raise ValueError(
                f"{document.source_key}: private documents need a local embedding model"
            )
        prepared = _masked(document, masker) if document.collection == "private" else document
        texts = chunk_text(prepared.text, max_chars)
        if not texts:
            continue
        embeddings = embedder.embed([f"{prepared.title}\n\n{text}" for text in texts])
        replace_source(connection, prepared, texts, embeddings, embedder.name)
        sources += 1
        chunks += len(texts)
        stored_keys.add(document.source_key)
    return IngestResult(sources, chunks, stored_keys=frozenset(stored_keys))


def ingest_batch(
    connection: Connection,
    batch: Batch,
    embedder: Embedder,
    masker: Masker,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> IngestResult:
    result = ingest(connection, batch.documents, embedder, masker, max_chars)
    keep = result.stored_keys | batch.unavailable_keys
    removed = delete_sources_except(connection, batch.source_types, keep)
    return replace(result, removed=removed)


def _masked(document: Document, masker: Masker) -> Document:
    return replace(document, title=masker.mask(document.title), text=masker.mask(document.text))


def youtrack_batch(client: YouTrackClient, query: str) -> Batch:
    documents: list[Document] = []
    unavailable: set[str] = set()
    for issue in client.issues(query):
        attachment_texts: dict[str, str] = {}
        for attachment in pdf_attachments(issue):
            try:
                attachment_texts[attachment["id"]] = pdf_text(client.attachment(attachment["url"]))
            except (ValueError, urllib.error.URLError) as error:
                key = f"{issue.get('idReadable')}#attachment-{attachment['id']}"
                unavailable.add(key)
                log.warning(
                    json.dumps(
                        {"event": "attachment_skipped", "source_key": key, "reason": str(error)}
                    )
                )
        documents.extend(issue_documents(issue, client.base_url, attachment_texts))
    return Batch(documents, SOURCE_TYPES["youtrack"], frozenset(unavailable))


def _youtrack_client(settings: Settings) -> tuple[YouTrackClient, str]:
    if not (settings.youtrack_base_url and settings.youtrack_token and settings.youtrack_query):
        raise ValueError(
            "set CRAG__YOUTRACK__BASE_URL, CRAG__YOUTRACK__TOKEN and CRAG__YOUTRACK__QUERY"
        )
    return YouTrackClient(
        settings.youtrack_base_url, settings.youtrack_token
    ), settings.youtrack_query


def _masker(settings: Settings) -> Masker:
    if not settings.people_file.exists():
        raise ValueError(
            f"people file {settings.people_file} is missing; private documents would keep "
            "third-party names. Create it (it may be empty) or set CRAG__PII__PEOPLE_FILE."
        )
    return Masker(load_people(settings.people_file))


def _embedder(settings: Settings) -> Embedder:
    if settings.embedding_model == HashEmbedder.name:
        return HashEmbedder()
    return FastEmbedder(settings.embedding_model, settings.embedding_cache_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crag.ingest", description="Load documents into the index."
    )
    sources = parser.add_subparsers(dest="source", required=True)
    sources.add_parser(
        "youtrack", help="issues, comments and PDF attachments matching CRAG__YOUTRACK__QUERY"
    )
    sources.add_parser("changelog", help="a Markdown changelog file").add_argument(
        "path", type=Path
    )
    sources.add_parser("esco", help="an ESCO skills CSV export").add_argument("path", type=Path)
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings()
    if arguments.source == "esco":
        batch = Batch(parse_skills_csv(arguments.path), SOURCE_TYPES["esco"])
        masker = Masker([])
    else:
        masker = _masker(settings)
        if arguments.source == "youtrack":
            batch = youtrack_batch(*_youtrack_client(settings))
        else:
            changelog = parse_changelog(arguments.path.read_text(encoding="utf-8"))
            batch = Batch(changelog, SOURCE_TYPES["changelog"])

    with connect(settings.database_url, schema=settings.database_schema) as connection:
        migrate(connection)
        result = ingest_batch(connection, batch, _embedder(settings), masker)
    log.info(
        json.dumps(
            {
                "event": "ingested",
                "source": arguments.source,
                "sources": result.sources,
                "chunks": result.chunks,
                "removed": result.removed,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
