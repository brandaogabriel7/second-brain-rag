import chromadb
import pytest
from ingest.chunker import Chunk
from storage.vector_store import VectorStore


@pytest.fixture
def store(request):
    """VectorStore backed by an ephemeral (in-memory) ChromaDB client."""
    client = chromadb.EphemeralClient()
    return VectorStore(client=client, collection_name=request.node.name)


def make_chunk(text="Some text", source="notes/test.md", title="Test Note",
               heading="Intro", tags=None):
    return Chunk(
        text=text,
        source=source,
        note_title=title,
        heading=heading,
        tags=tags or [],
    )


def fake_embedding(value=0.1, dims=8):
    """Return a simple embedding vector."""
    return [value] * dims


class TestAddChunks:
    def test_adds_single_chunk(self, store):
        chunks = [make_chunk(text="hello world")]
        embeddings = [fake_embedding(0.1)]
        store.add_chunks(chunks, embeddings)
        results = store.search(fake_embedding(0.1), top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "hello world"

    def test_adds_multiple_chunks(self, store):
        chunks = [
            make_chunk(text="first"),
            make_chunk(text="second"),
            make_chunk(text="third"),
        ]
        embeddings = [fake_embedding(0.1), fake_embedding(0.2), fake_embedding(0.3)]
        store.add_chunks(chunks, embeddings)
        results = store.search(fake_embedding(0.1), top_k=10)
        assert len(results) == 3

    def test_empty_chunks_is_noop(self, store):
        store.add_chunks([], [])
        results = store.search(fake_embedding(0.1), top_k=10)
        assert results == []


class TestSearch:
    def test_returns_most_similar_first(self, store):
        chunks = [
            make_chunk(text="far away"),
            make_chunk(text="very close"),
        ]
        embeddings = [[0.0] * 8, [0.9] * 8]
        store.add_chunks(chunks, embeddings)
        results = store.search([1.0] * 8, top_k=2)
        assert results[0]["text"] == "very close"

    def test_respects_top_k(self, store):
        chunks = [make_chunk(text=f"chunk {i}") for i in range(5)]
        embeddings = [fake_embedding(i * 0.1) for i in range(5)]
        store.add_chunks(chunks, embeddings)
        results = store.search(fake_embedding(0.1), top_k=3)
        assert len(results) == 3

    def test_search_empty_store(self, store):
        results = store.search(fake_embedding(0.1), top_k=5)
        assert results == []


class TestMetadata:
    def test_preserves_source(self, store):
        chunks = [make_chunk(source="projects/note.md")]
        store.add_chunks(chunks, [fake_embedding()])
        results = store.search(fake_embedding(), top_k=1)
        assert results[0]["source"] == "projects/note.md"

    def test_preserves_title(self, store):
        chunks = [make_chunk(title="My Note")]
        store.add_chunks(chunks, [fake_embedding()])
        results = store.search(fake_embedding(), top_k=1)
        assert results[0]["note_title"] == "My Note"

    def test_preserves_heading(self, store):
        chunks = [make_chunk(heading="Section One")]
        store.add_chunks(chunks, [fake_embedding()])
        results = store.search(fake_embedding(), top_k=1)
        assert results[0]["heading"] == "Section One"

    def test_preserves_tags_as_comma_separated(self, store):
        chunks = [make_chunk(tags=["python", "rag"])]
        store.add_chunks(chunks, [fake_embedding()])
        results = store.search(fake_embedding(), top_k=1)
        assert results[0]["tags"] == "python,rag"

    def test_empty_tags(self, store):
        chunks = [make_chunk(tags=[])]
        store.add_chunks(chunks, [fake_embedding()])
        results = store.search(fake_embedding(), top_k=1)
        assert results[0]["tags"] == ""

    def test_includes_distance(self, store):
        chunks = [make_chunk()]
        store.add_chunks(chunks, [fake_embedding()])
        results = store.search(fake_embedding(), top_k=1)
        assert "distance" in results[0]
