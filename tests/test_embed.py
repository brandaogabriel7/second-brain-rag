from unittest.mock import MagicMock, patch

import pytest
from embeddings.embed import Embedder


@pytest.fixture
def mock_openai():
    with patch("embeddings.embed.OpenAI") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


def fake_embedding_response(vectors):
    """Build a mock response matching OpenAI's embedding API shape."""
    response = MagicMock()
    response.data = [
        MagicMock(embedding=vec, index=i) for i, vec in enumerate(vectors)
    ]
    return response


class TestEmbedQuery:
    def test_returns_single_vector(self, mock_openai):
        mock_openai.embeddings.create.return_value = fake_embedding_response(
            [[0.1, 0.2, 0.3]]
        )
        embedder = Embedder()
        result = embedder.embed_query("hello world")
        assert result == [0.1, 0.2, 0.3]

    def test_calls_openai_with_correct_model(self, mock_openai):
        mock_openai.embeddings.create.return_value = fake_embedding_response(
            [[0.0]]
        )
        embedder = Embedder()
        embedder.embed_query("test")
        mock_openai.embeddings.create.assert_called_once_with(
            input=["test"], model="text-embedding-3-small"
        )


class TestEmbedBatch:
    def test_returns_list_of_vectors(self, mock_openai):
        mock_openai.embeddings.create.return_value = fake_embedding_response(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )
        embedder = Embedder()
        result = embedder.embed_batch(["a", "b", "c"])
        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        assert result[2] == [0.5, 0.6]

    def test_preserves_order(self, mock_openai):
        mock_openai.embeddings.create.return_value = fake_embedding_response(
            [[1.0], [2.0], [3.0]]
        )
        embedder = Embedder()
        result = embedder.embed_batch(["first", "second", "third"])
        assert result == [[1.0], [2.0], [3.0]]

    def test_single_item_batch(self, mock_openai):
        mock_openai.embeddings.create.return_value = fake_embedding_response(
            [[0.5, 0.5]]
        )
        embedder = Embedder()
        result = embedder.embed_batch(["only one"])
        assert len(result) == 1

    def test_empty_batch(self, mock_openai):
        embedder = Embedder()
        result = embedder.embed_batch([])
        assert result == []
        mock_openai.embeddings.create.assert_not_called()
