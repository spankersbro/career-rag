import argparse
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from crag.chunking import DEFAULT_MAX_CHARS, chunk_text
from crag.config import Settings, load_settings
from crag.documents import Document
from crag.embedding import Embedder, FastEmbedder, HashEmbedder
from crag.loaders.changelog import parse_changelog
from crag.loaders.esco import parse_skills_csv
from crag.loaders.youtrack import YouTrackClient, issue_documents, pdf_text
from crag.masking import Masker, load_people
from crag.store import EMBEDDING_DIMENSION, Connection, connect, migrate, replace_source

log = logging.getLogger("crag.ingest")


@dataclass(frozen=True)
class IngestResult:
    sources: int
    chunks: int


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
    sources = chunks = 0
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
    return IngestResult(sources, chunks)


def _masked(document: Document, masker: Masker) -> Document:
    return Document(
        collection=document.collection,
        source_type=document.source_type,
        source_key=document.source_key,
        title=masker.mask(document.title),
        text=masker.mask(document.text),
        url=document.url,
        issue_key=document.issue_key,
        document_date=document.document_date,
    )


def youtrack_documents(settings: Settings) -> list[Document]:
    if not (settings.youtrack_base_url and settings.youtrack_token and settings.youtrack_query):
        raise ValueError(
            "set CRAG__YOUTRACK__BASE_URL, CRAG__YOUTRACK__TOKEN and CRAG__YOUTRACK__QUERY"
        )
    client = YouTrackClient(settings.youtrack_base_url, settings.youtrack_token)
    documents: list[Document] = []
    for issue in client.issues(settings.youtrack_query):
        attachment_texts = {
            attachment["id"]: pdf_text(client.attachment(attachment["url"]))
            for attachment in issue.get("attachments") or []
            if attachment.get("mimeType") == "application/pdf"
        }
        documents.extend(issue_documents(issue, client.base_url, attachment_texts))
    return documents


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
    if arguments.source == "youtrack":
        documents = youtrack_documents(settings)
    elif arguments.source == "changelog":
        documents = parse_changelog(arguments.path.read_text(encoding="utf-8"))
    else:
        documents = parse_skills_csv(arguments.path)

    with connect(settings.database_url, schema=settings.database_schema) as connection:
        migrate(connection)
        result = ingest(
            connection, documents, _embedder(settings), Masker(load_people(settings.people_file))
        )
    log.info(
        json.dumps(
            {
                "event": "ingested",
                "source": arguments.source,
                "sources": result.sources,
                "chunks": result.chunks,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
