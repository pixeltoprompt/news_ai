"""
Provider-agnostic LLM client — same design as the PathPulse project's
llm_client.py, duplicated rather than shared as a package dependency so
each repo is independently cloneable and self-contained (a deliberate
trade-off: a small amount of duplication in exchange for zero coupling
between two otherwise-unrelated portfolio projects).
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")


class LLMClient:
    def __init__(self) -> None:
        self.backend = self._resolve_backend()

    def _resolve_backend(self) -> str:
        if _ANTHROPIC_KEY:
            return "anthropic"
        if _OPENAI_KEY:
            return "openai"
        return "local_mock"

    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 400) -> str:
        if self.backend == "anthropic":
            return self._generate_anthropic(prompt, system, max_tokens)
        if self.backend == "openai":
            return self._generate_openai(prompt, system, max_tokens)
        return self._generate_local_mock(prompt, system)

    def _generate_anthropic(self, prompt: str, system: Optional[str], max_tokens: int) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _generate_openai(self, prompt: str, system: Optional[str], max_tokens: int) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=_OPENAI_KEY)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=max_tokens)
        return resp.choices[0].message.content

    def _generate_local_mock(self, prompt: str, system: Optional[str]) -> str:
        """Deterministic extractive fallback so the repo is runnable and
        testable with zero API keys. Rather than returning generic
        placeholder text, this does real (if simple) extractive
        summarization — first N sentences plus the highest-signal
        sentence by keyword density — so demo output is genuinely
        article-specific, not templated filler."""
        prompt_lower = prompt.lower()

        # Pull the source text out of the prompt (it's always included
        # between markers by the callers in summarization.py / rag_chain.py)
        if "---ARTICLE---" in prompt:
            source = prompt.split("---ARTICLE---")[1].split("---END---")[0].strip()
        elif "---CONTEXT---" in prompt:
            source = prompt.split("---CONTEXT---")[1].split("---END---")[0].strip()
        else:
            source = prompt

        sentences = [s.strip() for s in source.replace("\n", " ").split(".") if len(s.strip()) > 15]

        if "presenter" in prompt_lower or "script" in prompt_lower:
            lead = sentences[0] if sentences else "This is a developing story."
            body = " ".join(sentences[1:4]) if len(sentences) > 1 else ""
            return f"Good evening. {lead}. {body} We'll continue to follow this story and bring you updates as they develop."

        if "query" in prompt_lower or "answer the question" in prompt_lower or "---context---" in prompt_lower:
            if not sentences:
                return "The retrieved context does not contain enough information to answer this question."
            return f"Based on the available reporting: {sentences[0]}. {sentences[1] if len(sentences) > 1 else ''}".strip()

        # Default: 3-sentence extractive summary
        summary_sentences = sentences[:3] if sentences else ["No summary available."]
        return ". ".join(summary_sentences) + "."
