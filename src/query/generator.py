import logging
from typing import Iterator

from anthropic import (
    APIConnectionError,
    APIStatusError,
    Anthropic,
    AuthenticationError,
)

logger = logging.getLogger(__name__)


class Generator:
    def __init__(
        self,
        client: Anthropic,
        model: str,
        max_tokens: int,
        system_prompt: str,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

    def generate_stream(self, query: str, chunks: list[dict]) -> Iterator[str]:
        """Generate a streaming answer using Claude with the given context chunks.

        Yields text fragments as they arrive, followed by a sources summary.
        On error, yields an error message instead of crashing.
        """
        context = self._build_context(chunks)
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    }
                ],
            ) as stream:
                for text in stream.text_stream:
                    yield text

            sources_summary = self._summarize_sources(chunks)
            if sources_summary:
                yield sources_summary

        except AuthenticationError:
            logger.error("Claude API authentication failed")
            yield "\n\n[Error: Claude API authentication failed. Check ANTHROPIC_API_KEY.]"
        except APIConnectionError as e:
            logger.error(f"Cannot connect to Claude API: {e}")
            yield "\n\n[Error: Cannot connect to Claude API. Check your internet connection.]"
        except APIStatusError as e:
            logger.error(f"Claude API error: {e}")
            yield f"\n\n[Error: Claude API error: {e.status_code}]"

    def _format_source(self, chunk: dict) -> str:
        parts = []
        if chunk.get("title"):
            parts.append(chunk["title"])
        if chunk.get("heading"):
            parts.append(chunk["heading"])
        if chunk.get("category"):
            parts.append(chunk["category"])
        if chunk.get("source"):
            parts.append(chunk["source"])
        return " | ".join(parts)

    def _build_context(self, chunks: list[dict]) -> str:
        context = ""
        for index, chunk in enumerate(chunks):
            context += f"[{index + 1}] {self._format_source(chunk)}\n{chunk['text']}\n\n"

        return context

    def _summarize_sources(self, chunks: list[dict]) -> str:
        if not chunks:
            return ""

        summary = "\n\n## Sources\n\n"
        for index, chunk in enumerate(chunks):
            summary += f"- [{index + 1}] {self._format_source(chunk)}\n"

        return summary
