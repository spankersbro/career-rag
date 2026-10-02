import argparse
import http.client
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from crag.chunking import DEFAULT_MAX_CHARS, Fits, chunk_text
from crag.config import Settings, load_settings
from crag.documents import Document
from crag.embedding import Embedder, FastEmbedder, HashEmbedder
from crag.loaders.changelog import key_prefix as changelog_key_prefix
from crag.loaders.changelog import parse_changelog
from crag.loaders.esco import parse_skills_csv
from crag.loaders.youtrack import YouTrackClient, issue_documents, pdf_attachments, pdf_text
from crag.masking import Masker
from crag.store import (
    EMBEDDING_DIMENSION,
    Connection,
    connect,
    delete_stale_sources,
    embedding_models,
    migrate,
    replace_source,
    stale_sources,
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
    key_prefix: str = ""
    allow_mass_removal: bool = False


MAX_REMOVAL_SHARE = 0.5  # a normal reload changes a few sources; losing half means the load broke


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
        if not embedder.fits(f"{prepared.title}\n\nx"):
            raise ValueError(f"{document.source_key}: title alone exceeds the embedding window")
        texts = chunk_text(prepared.text, max_chars, fits=_window(embedder, prepared.title))
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
    """Loads a complete batch, then removes sources in its scope that it no longer contains.

    An empty batch, or one that would remove more than MAX_REMOVAL_SHARE of the sources in
    scope, is refused before anything is written: a wrong query or a revoked token must not
    wipe the index.
    """
    if not batch.documents and not batch.allow_mass_removal:
        raise ValueError("refusing an empty load; check the query, token or input file")
    loaded_keys = frozenset(
        document.source_key for document in batch.documents if document.text.strip()
    )
    stale, total = stale_sources(
        connection, batch.source_types, batch.key_prefix, loaded_keys | batch.unavailable_keys
    )
    if not batch.allow_mass_removal and stale > max(1, total * MAX_REMOVAL_SHARE):
        raise ValueError(
            f"this load would remove {stale} of {total} sources; "
            "rerun with --allow-mass-removal if that is intended"
        )
    result = ingest(connection, batch.documents, embedder, masker, max_chars)
    keep = result.stored_keys | batch.unavailable_keys
    removed = delete_stale_sources(connection, batch.source_types, batch.key_prefix, keep)
    return replace(result, removed=removed)


def _window(embedder: Embedder, title: str) -> Fits:
    """Each chunk is embedded with its title in front, so both must fit the model together."""

    def fits(chunk: str) -> bool:
        return embedder.fits(f"{title}\n\n{chunk}")

    return fits


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
            except (ValueError, OSError, http.client.HTTPException) as error:
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


def embedder_from_settings(settings: Settings) -> Embedder:
    if settings.embedding_model == HashEmbedder.name:
        return HashEmbedder()
    return FastEmbedder(settings.embedding_model, settings.embedding_cache_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m crag.ingest", description="Load documents into the index."
    )
    parser.add_argument(
        "--allow-mass-removal",
        action="store_true",
        help=f"allow a load that removes more than {MAX_REMOVAL_SHARE:.0%} of existing sources",
    )
    sources = parser.add_subparsers(dest="source", required=True)
    sources.add_parser(
        "youtrack", help="issues, comments and PDF attachments matching CRAG__YOUTRACK__QUERY"
    )
    changelog_parser = sources.add_parser("changelog", help="a Markdown changelog file")
    changelog_parser.add_argument("path", type=Path)
    changelog_parser.add_argument(
        "--label", required=True, help="names this changelog; keeps its entries apart from others"
    )
    sources.add_parser("esco", help="an ESCO skills CSV export").add_argument("path", type=Path)
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings()
    if arguments.source == "esco":
        batch = Batch(parse_skills_csv(arguments.path), SOURCE_TYPES["esco"])
        masker = Masker([])
    else:
        masker = Masker.from_file(settings.people_file)
        if arguments.source == "youtrack":
            batch = youtrack_batch(*_youtrack_client(settings))
        else:
            batch = Batch(
                parse_changelog(arguments.path.read_text(encoding="utf-8"), arguments.label),
                SOURCE_TYPES["changelog"],
                key_prefix=changelog_key_prefix(arguments.label),
            )
    batch = replace(batch, allow_mass_removal=arguments.allow_mass_removal)

    embedder = embedder_from_settings(settings)
    try:
        with connect(settings.database_url, schema=settings.database_schema) as connection:
            migrate(connection)
            result = ingest_batch(connection, batch, embedder, masker)
    finally:
        embedder.close()
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
