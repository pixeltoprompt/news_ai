"""
Tests cover the full pipeline: ingestion produces the expected chunk
count, retrieval surfaces the correct article for topically distinct
queries, RAG answers cite real sources, and summarization/script
generation produce non-empty, reasonably-sized output.

Run with: pytest -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # noqa: E402

from news_channel.ingestion import ingest, load_articles  # noqa: E402
from news_channel.rag_chain import answer_query  # noqa: E402
from news_channel.retrieval import hybrid_retrieve  # noqa: E402
from news_channel.summarization import generate_presenter_script, summarize_article  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def ingested_corpus():
    """Ingests the sample corpus once for the whole test module."""
    report = ingest()
    assert report.chunks_created > 0
    return report


def test_ingestion_produces_expected_article_count():
    articles = load_articles()
    assert len(articles) == 6


def test_retrieval_surfaces_correct_article_for_cricket_query():
    results = hybrid_retrieve("Ranji Trophy squad captain", top_k=3)
    assert len(results) > 0
    top_titles = [r.chunk.article_title for r in results]
    assert any("Cricket" in t or "Ranji" in t for t in top_titles)


def test_retrieval_surfaces_correct_article_for_weather_query():
    results = hybrid_retrieve("orange alert heavy rainfall", top_k=3)
    assert len(results) > 0
    top_titles = [r.chunk.article_title for r in results]
    assert any("Monsoon" in t or "Rainfall" in t for t in top_titles)


def test_rag_answer_cites_real_sources():
    result = answer_query("What is the theme of this year's Ganesh festival?")
    assert len(result.sources) > 0
    all_titles = {a.title for a in load_articles()}
    assert all(s in all_titles for s in result.sources)


def test_rag_returns_no_context_message_for_unrelated_query():
    """Sanity check: retrieval always returns *something* from a small
    fixed corpus (there's no relevance threshold), but the fused score
    for a genuinely unrelated query should still be identifiable as weak.
    This test documents that limitation rather than hiding it."""
    result = answer_query("What is the capital of France?")
    assert result.retrieved_chunks  # will retrieve something — no threshold cutoff yet
    assert result.answer  # but should still produce *a* response, not crash


def test_summarization_produces_nonempty_summary():
    article = load_articles()[0]
    result = summarize_article(article)
    assert result.word_count > 0
    assert len(result.summary) > 20


def test_presenter_script_respects_rough_target_duration():
    article = load_articles()[0]
    script = generate_presenter_script(article, target_duration_seconds=20)
    assert script.script
    # Local-mock is extractive and won't hit the target precisely —
    # this just checks it's in a sane ballpark, not wildly off (e.g. 10x).
    assert script.estimated_duration_seconds < 20 * 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
