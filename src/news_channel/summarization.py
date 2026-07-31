"""
Two content-generation capabilities, both mirroring the original
build: a short broadcast-style summary, and a longer News Presenter
script sized to a target on-air duration.
"""

from __future__ import annotations

from .llm_client import LLMClient
from .schemas import Article, PresenterScript, SummaryResult

SUMMARY_SYSTEM_PROMPT = (
    "You summarise news articles in exactly 3 sentences for a TV broadcast context. "
    "Maintain journalistic neutrality. Use present tense where natural. "
    "Do not editorialise or add information not present in the article."
)

PRESENTER_SYSTEM_PROMPT = (
    "You write on-air news presenter scripts in a natural spoken register — the way a "
    "news anchor would actually read a story aloud, not the way it would be written. "
    "Open with a brief lead-in and close with a short wrap-up line. Do not use bullet "
    "points, headers, or written-only phrasing."
)

WORDS_PER_SECOND_SPOKEN = 2.5  # ~150 words/minute, a standard broadcast speaking pace


def summarize_article(article: Article) -> SummaryResult:
    prompt = f"---ARTICLE---\n{article.body}\n---END---\n\nSummarise this in 3 sentences."
    llm = LLMClient()
    summary = llm.generate(prompt, system=SUMMARY_SYSTEM_PROMPT)

    return SummaryResult(
        article_id=article.article_id,
        summary=summary,
        word_count=len(summary.split()),
    )


def generate_presenter_script(article: Article, target_duration_seconds: float = 30) -> PresenterScript:
    target_words = int(target_duration_seconds * WORDS_PER_SECOND_SPOKEN)

    prompt = (
        f"---ARTICLE---\n{article.body}\n---END---\n\n"
        f"Write a presenter script for this story, targeting roughly {target_words} words "
        f"(about {target_duration_seconds:.0f} seconds of on-air reading time)."
    )
    llm = LLMClient()
    script = llm.generate(prompt, system=PRESENTER_SYSTEM_PROMPT, max_tokens=max(300, target_words * 2))

    estimated_duration = len(script.split()) / WORDS_PER_SECOND_SPOKEN

    return PresenterScript(
        article_id=article.article_id,
        script=script,
        estimated_duration_seconds=round(estimated_duration, 1),
        target_duration_seconds=target_duration_seconds,
    )
