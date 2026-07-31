"""
Hybrid retrieval: vector similarity search (via ChromaDB) fused with BM25
keyword search (via rank_bm25).

Why hybrid rather than vector-only: pure semantic search can miss exact
keyword matches that matter a lot in news retrieval — a query for
"Ranji Trophy" should reliably surface the article containing that exact
phrase, not just semantically-similar sports content. BM25 catches that;
vector search catches paraphrase and conceptual matches BM25 would miss.
Fusing both is the standard fix, and is exactly what's described in the
"what I'd rebuild differently" answer for the original system.

Fusion method: reciprocal rank fusion (RRF), which combines two ranked
lists using only rank position (not raw scores, which aren't comparable
across a cosine-similarity scale and a BM25 scale). This avoids needing
to normalize two incompatible scoring systems.
"""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from .embeddings import EmbeddingClient
from .ingestion import BM25_CORPUS_PATH, get_chroma_collection
from .schemas import Chunk, RetrievedChunk

RRF_K = 60  # standard smoothing constant for reciprocal rank fusion


def _load_bm25_index() -> tuple[BM25Okapi, list[Chunk]]:
    with open(BM25_CORPUS_PATH) as f:
        raw = json.load(f)
    chunks = [Chunk(**item) for item in raw]
    tokenized = [c.text.lower().split() for c in chunks]
    return BM25Okapi(tokenized), chunks


def _vector_search(query: str, top_k: int) -> list[tuple[str, float]]:
    """Returns [(chunk_id, similarity_score), ...] ranked best-first."""
    embedder = EmbeddingClient()
    query_vector = embedder.embed_query(query)
    collection = get_chroma_collection()
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    ids = results["ids"][0]
    distances = results["distances"][0]  # chroma returns distance; smaller = closer
    return list(zip(ids, distances))


def _bm25_search(query: str, top_k: int) -> list[tuple[str, float]]:
    bm25, chunks = _load_bm25_index()
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip([c.chunk_id for c in chunks], scores), key=lambda x: -x[1])
    return ranked[:top_k]


def hybrid_retrieve(query: str, top_k: int = 5, candidate_pool: int = 10) -> list[RetrievedChunk]:
    vector_results = _vector_search(query, candidate_pool)
    bm25_results = _bm25_search(query, candidate_pool)

    # Reciprocal Rank Fusion
    fused_scores: dict[str, float] = {}
    for rank, (chunk_id, _) in enumerate(vector_results):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, (chunk_id, _) in enumerate(bm25_results):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    vector_score_map = dict(vector_results)
    bm25_score_map = dict(bm25_results)

    _, chunks_list = _load_bm25_index()
    chunk_lookup = {c.chunk_id: c for c in chunks_list}

    ranked_ids = sorted(fused_scores.keys(), key=lambda cid: -fused_scores[cid])[:top_k]

    retrieved: list[RetrievedChunk] = []
    for chunk_id in ranked_ids:
        if chunk_id not in chunk_lookup:
            continue
        retrieved.append(RetrievedChunk(
            chunk=chunk_lookup[chunk_id],
            vector_score=float(vector_score_map.get(chunk_id, 0.0)),
            bm25_score=float(bm25_score_map.get(chunk_id, 0.0)),
            fused_score=fused_scores[chunk_id],
        ))

    return retrieved
