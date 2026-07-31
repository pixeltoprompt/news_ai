"""
RAG chain: ties hybrid retrieval to generation, with a source-citation
requirement and a lightweight faithfulness check on the output.

The faithfulness check here is intentionally simple (word-overlap between
the generated answer and the retrieved context) — it's a stand-in for
what a RAGAS faithfulness score would give you properly. It's included
anyway because "how do you know retrieval/generation is actually working"
is exactly the kind of question this project is meant to be able to
answer concretely, not just in theory. See README for what a production
version of this check would use instead.
"""

from __future__ import annotations

from .llm_client import LLMClient
from .retrieval import hybrid_retrieve
from .schemas import RAGResult

SYSTEM_PROMPT = (
    "You are News Channel Newsroom's internal research assistant. Answer the question using "
    "ONLY the provided article excerpts. If the excerpts don't contain the answer, "
    "say so explicitly rather than guessing. Keep answers concise and cite which "
    "article(s) the information came from."
)


def _build_context_block(chunks) -> str:
    parts = []
    for rc in chunks:
        parts.append(f"[{rc.chunk.article_title}]: {rc.chunk.text}")
    return "\n\n".join(parts)


def _faithfulness_check(answer: str, context: str) -> tuple[bool, str]:
    """Lightweight lexical-overlap proxy for faithfulness. Flags likely
    hallucination if the answer shares very little vocabulary with the
    retrieved context — not a substitute for RAGAS, but catches the
    clearest failure case (the model answering from parametric knowledge
    instead of the provided context) without needing a second LLM call."""
    answer_words = set(w.lower() for w in answer.split() if len(w) > 4)
    context_words = set(w.lower() for w in context.split() if len(w) > 4)
    if not answer_words:
        return False, "Empty answer."

    overlap_ratio = len(answer_words & context_words) / len(answer_words)

    if overlap_ratio < 0.15:
        return False, (
            f"Low lexical overlap with retrieved context ({overlap_ratio:.0%}) — "
            "answer may not be grounded in the retrieved articles."
        )
    return True, f"Lexical overlap with context: {overlap_ratio:.0%}."


def answer_query(query: str, top_k: int = 4) -> RAGResult:
    retrieved = hybrid_retrieve(query, top_k=top_k)

    if not retrieved:
        return RAGResult(
            query=query,
            answer="No relevant articles were found in the index for this query.",
            sources=[],
            retrieved_chunks=[],
            faithfulness_flag=False,
            faithfulness_note="No context retrieved.",
        )

    context = _build_context_block(retrieved)
    prompt = f"Question: {query}\n\n---CONTEXT---\n{context}\n---END---"

    llm = LLMClient()
    answer = llm.generate(prompt, system=SYSTEM_PROMPT)

    faithful, note = _faithfulness_check(answer, context)
    sources = sorted(set(rc.chunk.article_title for rc in retrieved))

    return RAGResult(
        query=query,
        answer=answer,
        sources=sources,
        retrieved_chunks=retrieved,
        faithfulness_flag=faithful,
        faithfulness_note=note,
    )
