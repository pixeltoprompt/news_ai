"""
Data ingestion pipeline — the piece that turns raw articles into a
searchable index.

Flow: load articles -> chunk with overlap -> fit/apply embeddings ->
store in ChromaDB (vectors) + build a BM25 index (keyword) alongside it.

Chunk size (400 chars ~ roughly 80-100 tokens for this corpus) and overlap
(50 chars) mirror the parameters described in the original RAG build:
small enough that retrieval doesn't pull in irrelevant surrounding
content, with enough overlap that a fact sitting on a chunk boundary
isn't split away from its context.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

from .embeddings import EmbeddingClient
from .schemas import Article, Chunk, IngestionReport

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma_store"
BM25_CORPUS_PATH = DATA_DIR / "bm25_corpus.json"


def load_articles(path: Path = DATA_DIR / "sample_articles" / "articles.json") -> list[Article]:
    with open(path) as f:
        raw = json.load(f)
    return [Article(**item) for item in raw]


def chunk_article(article: Article, chunk_size: int = 400, overlap: int = 50) -> list[Chunk]:
    """Sliding-window character chunking with overlap. Character-based
    rather than token-based to avoid pulling in a tokenizer dependency —
    documented here as a simplification versus a production system, which
    would chunk by token count using the target model's tokenizer."""
    text = article.body
    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                chunk_id=f"{article.article_id}_chunk_{index}",
                article_id=article.article_id,
                article_title=article.title,
                text=chunk_text,
                chunk_index=index,
                published_date=article.published_date,
                category=article.category,
            ))
            index += 1
        start += chunk_size - overlap

    return chunks


def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # embedding_function=None because we compute and pass embeddings
    # ourselves via EmbeddingClient — this is what lets the local TF-IDF
    # fallback and a real API backend both work through the same code path
    # without Chroma trying to download its own default embedding model.
    return client.get_or_create_collection(name="news_articles", embedding_function=None)


def ingest(articles_path: Path = DATA_DIR / "sample_articles" / "articles.json") -> IngestionReport:
    articles = load_articles(articles_path)

    all_chunks: list[Chunk] = []
    for article in articles:
        all_chunks.extend(chunk_article(article))

    embedder = EmbeddingClient()
    embedder.fit([c.text for c in all_chunks])  # no-op for API backends
    vectors = embedder.embed_documents([c.text for c in all_chunks])

    collection = get_chroma_collection()
    # Reset collection on re-ingestion so repeated demo runs are idempotent.
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    collection.add(
        ids=[c.chunk_id for c in all_chunks],
        embeddings=vectors,
        documents=[c.text for c in all_chunks],
        metadatas=[{
            "article_id": c.article_id,
            "article_title": c.article_title,
            "chunk_index": c.chunk_index,
            "published_date": str(c.published_date),
            "category": c.category,
        } for c in all_chunks],
    )

    # Persist the raw chunk corpus separately for the BM25 (keyword) index,
    # which retrieval.py builds fresh at query time — BM25 is cheap enough
    # that persisting a fitted index isn't necessary for a corpus this size.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_CORPUS_PATH, "w") as f:
        json.dump([c.model_dump(mode="json") for c in all_chunks], f)

    return IngestionReport(
        articles_ingested=len(articles),
        chunks_created=len(all_chunks),
        embedding_backend=embedder.backend,
    )


if __name__ == "__main__":
    report = ingest()
    print(f"Ingested {report.articles_ingested} articles into {report.chunks_created} chunks "
          f"using '{report.embedding_backend}' embeddings.")
