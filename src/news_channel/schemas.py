"""
Data models used across the ingestion, retrieval, summarization, and
generation stages. Kept centralized for the same reason as in the
PathPulse project: strict contracts at every pipeline boundary make
failures loud and localized instead of silent and downstream.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class Article(BaseModel):
    article_id: str
    title: str
    body: str
    category: str
    published_date: date
    source: str = "News Channel Newsroom"


class Chunk(BaseModel):
    chunk_id: str
    article_id: str
    article_title: str
    text: str
    chunk_index: int
    published_date: date
    category: str


class RetrievedChunk(BaseModel):
    chunk: Chunk
    vector_score: float
    bm25_score: float
    fused_score: float


class RAGResult(BaseModel):
    query: str
    answer: str
    sources: list[str]  # article titles cited
    retrieved_chunks: list[RetrievedChunk]
    faithfulness_flag: bool  # True if a lightweight overlap check passed
    faithfulness_note: str


class SummaryResult(BaseModel):
    article_id: str
    summary: str
    word_count: int


class PresenterScript(BaseModel):
    article_id: str
    script: str
    estimated_duration_seconds: float
    target_duration_seconds: float


class IngestionReport(BaseModel):
    articles_ingested: int
    chunks_created: int
    embedding_backend: str
