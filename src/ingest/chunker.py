from dataclasses import dataclass
import re
from typing import List, Tuple

from .obsidian import NoteData


@dataclass
class Chunk:
    text: str
    source: str
    note_title: str
    heading: str
    tags: List[str]


class Chunker:
    def __init__(self, max_chunk_size: int = 500, overlap: int = 50) -> None:
        self._max_chunk_size = max_chunk_size
        self._overlap = overlap

    def chunk_note(self, note: NoteData) -> List[Chunk]:
        tuples = self._split_headings(note)
        chunks = self._split_large_chunks(note, tuples)
        return chunks

    def _split_headings(self, note: NoteData) -> List[Tuple[str, str]]:
        parts = re.split(r"^(#{1,6})\s(.+)$", note.content, flags=re.MULTILINE)

        pre_heading = parts[0]

        tuples: List[Tuple[str, str]] = (
            [(pre_heading, "")] if pre_heading and pre_heading.strip() else []
        )

        for i in range(1, len(parts), 3):
            heading = parts[i + 1]
            text = parts[i + 2]

            if text and text.strip():
                tuples.append((text, heading))

        return tuples

    def _split_large_chunks(
        self, note: NoteData, tuples: List[Tuple[str, str]]
    ) -> List[Chunk]:
        chunks = []
        step = self._max_chunk_size - self._overlap
        for text, heading in tuples:
            for i in range(0, len(text), step):
                chunk_text = text[i : i + self._max_chunk_size]
                if chunk_text and chunk_text.strip():
                    chunks.append(
                        Chunk(
                            text=chunk_text.strip(),
                            source=note.path,
                            note_title=note.title,
                            heading=heading,
                            tags=note.tags,
                        )
                    )

        return chunks
