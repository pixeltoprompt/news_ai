"""
Provider-agnostic embedding client. Same pattern as llm_client.py in the
PathPulse project: auto-detect a real embedding API key and use it, or
fall back to a genuinely-computed local embedding so the repo runs with
zero external dependencies and zero cost.

The local fallback is NOT a stub that returns random vectors — it fits a
real TF-IDF vectorizer over the ingested corpus and L2-normalizes the
output, so cosine similarity search behaves correctly and retrieval
quality is honestly comparable (if weaker than a trained neural embedding
model) rather than faked. This also means the "local" and "API" paths are
swappable without changing anything downstream — the retriever just
consumes fixed-dimension vectors either way.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

load_dotenv()

_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_COHERE_KEY = os.getenv("COHERE_API_KEY")

VECTORIZER_PATH = Path(__file__).parent.parent.parent / "data" / "tfidf_vectorizer.pkl"


class EmbeddingClient:
    def __init__(self, max_features: int = 512) -> None:
        self.backend = self._resolve_backend()
        self.max_features = max_features
        self._vectorizer: Optional[TfidfVectorizer] = None
        if self.backend == "local_tfidf":
            self._load_or_init_vectorizer()

    def _resolve_backend(self) -> str:
        if _OPENAI_KEY:
            return "openai"
        if _COHERE_KEY:
            return "cohere"
        return "local_tfidf"

    def _load_or_init_vectorizer(self) -> None:
        if VECTORIZER_PATH.exists():
            with open(VECTORIZER_PATH, "rb") as f:
                self._vectorizer = pickle.load(f)

    # -- Fitting (ingestion time only) --------------------------------------

    def fit(self, corpus: list[str]) -> None:
        """Fits the local TF-IDF vectorizer on the full chunk corpus. Must be
        called once during ingestion before embed_query() is used, when
        running in local_tfidf mode. Real API backends need no fitting."""
        if self.backend != "local_tfidf":
            return
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._vectorizer.fit(corpus)
        VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(self._vectorizer, f)

    # -- Embedding ------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "openai":
            return self._embed_openai(texts)
        if self.backend == "cohere":
            return self._embed_cohere(texts)
        return self._embed_local(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        if self._vectorizer is None:
            raise RuntimeError(
                "Local TF-IDF vectorizer is not fitted. Run ingestion "
                "(EmbeddingClient.fit()) before embedding queries."
            )
        matrix = self._vectorizer.transform(texts)
        dense = normalize(matrix, norm="l2").toarray()
        return dense.tolist()

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI(api_key=_OPENAI_KEY)
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in resp.data]

    def _embed_cohere(self, texts: list[str]) -> list[list[float]]:
        import cohere

        client = cohere.Client(_COHERE_KEY)
        resp = client.embed(texts=texts, model="embed-english-v3.0", input_type="search_document")
        return resp.embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)
