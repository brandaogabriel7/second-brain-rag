from unittest.mock import MagicMock

import pytest
from query.retriever import Retriever


@pytest.fixture
def embedder():
    mock = MagicMock()
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    mock.embed_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return mock


@pytest.fixture
def store():
    mock = MagicMock()
    mock.search.return_value = [
        {"text": "result one", "source": "a.md", "distance": 0.1},
        {"text": "result two", "source": "b.md", "distance": 0.3},
    ]
    return mock


@pytest.fixture
def retriever(embedder, store):
    return Retriever(embedder=embedder, store=store)


class TestSearch:
    def test_returns_search_results(self, retriever):
        results = retriever.search("my question", top_k=5)
        assert len(results) == 2
        assert results[0]["text"] == "result one"

    def test_embeds_query_text(self, retriever, embedder):
        retriever.search("my question", top_k=5)
        embedder.embed_query.assert_called_once_with("my question")

    def test_passes_embedding_to_store(self, retriever, store):
        retriever.search("my question", top_k=3)
        store.search.assert_called_once_with([0.1, 0.2, 0.3], top_k=3)

    def test_respects_top_k(self, retriever, store):
        retriever.search("query", top_k=10)
        store.search.assert_called_once_with([0.1, 0.2, 0.3], top_k=10)


class TestIngest:
    def test_embeds_chunks_and_stores(self, retriever, embedder, store):
        chunks = [MagicMock(), MagicMock()]
        chunks[0].text = "first"
        chunks[1].text = "second"
        retriever.ingest(chunks)
        embedder.embed_batch.assert_called_once_with(["first", "second"])
        store.add_chunks.assert_called_once_with(chunks, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    def test_empty_chunks_is_noop(self, retriever, embedder, store):
        retriever.ingest([])
        embedder.embed_batch.assert_not_called()
        store.add_chunks.assert_not_called()
