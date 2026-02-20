import pytest
from ingest.obsidian import NoteData
from ingest.chunker import Chunker, Chunk


def make_note(content, title="Test Note", path="test.md", tags=None):
    return NoteData(
        title=title,
        path=path,
        frontmatter={},
        tags=tags or [],
        content=content,
    )


class TestChunkDataclass:
    def test_chunk_has_expected_fields(self):
        chunk = Chunk(
            text="hello",
            source="notes/test.md",
            note_title="Test",
            heading="Intro",
            tags=["python"],
        )
        assert chunk.text == "hello"
        assert chunk.source == "notes/test.md"
        assert chunk.note_title == "Test"
        assert chunk.heading == "Intro"
        assert chunk.tags == ["python"]


class TestHeadingSplit:
    def test_single_section_no_headings(self):
        note = make_note("Just some plain text without headings.")
        chunker = Chunker()
        chunks = chunker.chunk_note(note)
        assert len(chunks) == 1
        assert "plain text" in chunks[0].text
        assert chunks[0].heading == ""

    def test_splits_by_h1(self):
        note = make_note("# First\nContent one\n# Second\nContent two")
        chunker = Chunker()
        chunks = chunker.chunk_note(note)
        assert len(chunks) == 2
        assert "Content one" in chunks[0].text
        assert "Content two" in chunks[1].text

    def test_splits_by_h2(self):
        note = make_note("## Intro\nHello\n## Details\nWorld")
        chunker = Chunker()
        chunks = chunker.chunk_note(note)
        assert len(chunks) == 2

    def test_splits_by_mixed_heading_levels(self):
        note = make_note("# Top\nA\n## Sub\nB\n### Deep\nC")
        chunker = Chunker()
        chunks = chunker.chunk_note(note)
        assert len(chunks) == 3

    def test_preserves_heading_text(self):
        note = make_note("# My Section\nSome content here")
        chunker = Chunker()
        chunks = chunker.chunk_note(note)
        assert chunks[0].heading == "My Section"

    def test_content_before_first_heading(self):
        note = make_note("Intro paragraph\n# First Heading\nHeading content")
        chunker = Chunker()
        chunks = chunker.chunk_note(note)
        assert len(chunks) == 2
        assert "Intro paragraph" in chunks[0].text
        assert chunks[0].heading == ""
        assert chunks[1].heading == "First Heading"


class TestFixedSizeFallback:
    def test_long_section_gets_split(self):
        long_text = "# Big Section\n" + "a" * 800
        note = make_note(long_text)
        chunker = Chunker(max_chunk_size=500, overlap=50)
        chunks = chunker.chunk_note(note)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= 500

    def test_short_section_stays_intact(self):
        note = make_note("# Short\nBrief content")
        chunker = Chunker(max_chunk_size=500)
        chunks = chunker.chunk_note(note)
        assert len(chunks) == 1

    def test_overlap_exists_between_fixed_chunks(self):
        content = "# Section\n" + "word " * 200  # ~1000 chars
        note = make_note(content)
        chunker = Chunker(max_chunk_size=500, overlap=50)
        chunks = chunker.chunk_note(note)
        assert len(chunks) >= 2
        # Last 50 chars of first chunk should appear at start of second
        end_of_first = chunks[0].text[-50:]
        assert end_of_first in chunks[1].text


class TestMetadataInheritance:
    def test_chunks_inherit_source(self):
        note = make_note("Some text", path="projects/my-note.md")
        chunker = Chunker()
        chunks = chunker.chunk_note(note)
        assert chunks[0].source == "projects/my-note.md"

    def test_chunks_inherit_title(self):
        note = make_note("Some text", title="My Great Note")
        chunker = Chunker()
        chunks = chunker.chunk_note(note)
        assert chunks[0].note_title == "My Great Note"

    def test_chunks_inherit_tags(self):
        note = make_note("Some text", tags=["python", "rag"])
        chunker = Chunker()
        chunks = chunker.chunk_note(note)
        assert chunks[0].tags == ["python", "rag"]


class TestEdgeCases:
    def test_empty_content_returns_no_chunks(self):
        note = make_note("")
        chunker = Chunker()
        chunks = chunker.chunk_note(note)
        assert chunks == []

    def test_whitespace_only_content_returns_no_chunks(self):
        note = make_note("   \n\n  ")
        chunker = Chunker()
        chunks = chunker.chunk_note(note)
        assert chunks == []

    def test_heading_with_no_body_is_skipped(self):
        note = make_note("# Empty Section\n\n# Has Content\nActual text")
        chunker = Chunker()
        chunks = chunker.chunk_note(note)
        assert len(chunks) == 1
        assert chunks[0].heading == "Has Content"

    def test_multiple_notes_independent(self):
        chunker = Chunker()
        note1 = make_note("# A\nFirst", title="Note1", path="n1.md")
        note2 = make_note("# B\nSecond", title="Note2", path="n2.md")
        chunks1 = chunker.chunk_note(note1)
        chunks2 = chunker.chunk_note(note2)
        assert chunks1[0].note_title == "Note1"
        assert chunks2[0].note_title == "Note2"
