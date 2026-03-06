from typing import Iterator
from anthropic import Anthropic
from anthropic.types import TextBlock

SYSTEM_PROMPT = """
You are a helpful assistant. Your job is to answer questions based only on my personal notes.

If there aren't any related notes, tell the user you couldn't find relevant context in the notes.

Cite sources using [1], [2], etc. when referencing information from the notes.
"""
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024


class Generator:
    def __init__(self, client: Anthropic) -> None:
        self._client = client

    def generate(self, query: str, chunks: list[dict]) -> str:
        context = self._build_context(chunks)
        response = self._client.messages.create(
            system=SYSTEM_PROMPT,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
        )

        block = response.content[0]
        if not isinstance(block, TextBlock):
            raise ValueError(f"Unexpected response type: {type(block)}")

        sources_summary = self._summarize_sources(chunks)

        return f"{block.text}{sources_summary}"

    def generate_stream(self, query: str, chunks: list[dict]) -> Iterator[str]:
        context = self._build_context(chunks)
        with self._client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
        ) as stream:
            for text in stream.text_stream:
                yield text

        sources_summary = self._summarize_sources(chunks)
        if sources_summary:
            yield sources_summary

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

        summary = "\n\nSources:\n"
        for index, chunk in enumerate(chunks):
            summary += f"[{index + 1}] {self._format_source(chunk)}\n"

        return summary
