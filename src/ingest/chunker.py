import logging
import re

from .models import Chunk, HeadingSection, ObsidianNote, ReadwiseHighlight

logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, max_chunk_size: int = 500, overlap: int = 50) -> None:
        self._max_chunk_size = max_chunk_size
        self._overlap = overlap

    def chunk_note(self, note: ObsidianNote) -> list[Chunk]:
        """Split an Obsidian note into chunks for embedding.

        First splits by markdown headings, then applies fixed-size chunking
        with overlap for sections exceeding max_chunk_size.
        """
        sections = self._split_headings(note)
        chunks = self._split_large_chunks(note, sections)
        logger.debug(f"Chunked '{note.title}' into {len(chunks)} chunks")
        return chunks

    def _split_headings(self, note: ObsidianNote) -> list[HeadingSection]:
        parts = re.split(r"^(#{1,6})\s(.+)$", note.content, flags=re.MULTILINE)

        pre_heading = parts[0]

        sections: list[HeadingSection] = (
            [HeadingSection(heading="", text=pre_heading)]
            if pre_heading and pre_heading.strip()
            else []
        )

        for i in range(1, len(parts), 3):
            heading = parts[i + 1]
            text = parts[i + 2]

            if text and text.strip():
                sections.append(HeadingSection(heading=heading, text=text))

        return sections

    def _split_large_chunks(
        self, note: ObsidianNote, sections: list[HeadingSection]
    ) -> list[Chunk]:
        chunks = []
        step = self._max_chunk_size - self._overlap
        for section in sections:
            for i in range(0, len(section.text), step):
                chunk_text = section.text[i : i + self._max_chunk_size]
                if chunk_text and chunk_text.strip():
                    chunks.append(
                        Chunk(
                            text=chunk_text.strip(),
                            source=note.path,
                            title=note.title,
                            heading=section.heading,
                            tags=note.tags,
                            category="notes",
                        )
                    )

        return chunks


def highlight_to_chunk(highlight: ReadwiseHighlight) -> Chunk:
    """Convert a Readwise highlight to a Chunk.

    Each highlight becomes a single chunk (no splitting needed).
    """
    return Chunk(
        text=highlight.text,
        source=highlight.readwise_url,
        title=highlight.title,
        heading="",
        tags=highlight.tags,
        category=highlight.category,
        author=highlight.author,
    )
