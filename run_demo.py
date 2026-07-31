"""
End-to-end demo: ingest -> RAG query -> summarize -> presenter script.

Usage:
    python run_demo.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from news_channel.ingestion import ingest, load_articles  # noqa: E402
from news_channel.rag_chain import answer_query  # noqa: E402
from news_channel.summarization import generate_presenter_script, summarize_article  # noqa: E402


def main() -> None:
    print("=" * 72)
    print("News Channel AI — Demo Run")
    print("=" * 72)

    print("\n[1/4] Ingesting sample articles...")
    report = ingest()
    print(f"  -> {report.articles_ingested} articles, {report.chunks_created} chunks, "
          f"backend='{report.embedding_backend}'")

    print("\n[2/4] Running RAG queries...")
    queries = [
        "What did the traffic police do about parking near Sitabuldi market?",
        "Is there a flood risk in Nagpur right now?",
        "Who is leading the Vidarbha cricket squad this season?",
    ]
    for q in queries:
        result = answer_query(q)
        print(f"\n  Q: {q}")
        print(f"  A: {result.answer}")
        print(f"  Sources: {', '.join(result.sources)}")
        print(f"  Faithfulness check: {'PASS' if result.faithfulness_flag else 'FLAGGED'} — {result.faithfulness_note}")

    articles = load_articles()
    sample_article = articles[0]

    print(f"\n[3/4] Summarizing: \"{sample_article.title}\"")
    summary = summarize_article(sample_article)
    print(f"  Summary ({summary.word_count} words): {summary.summary}")

    print(f"\n[4/4] Generating 30-second presenter script for the same article...")
    script = generate_presenter_script(sample_article, target_duration_seconds=30)
    print(f"  Estimated duration: {script.estimated_duration_seconds}s (target: {script.target_duration_seconds}s)")
    print(f"  Script: {script.script}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
