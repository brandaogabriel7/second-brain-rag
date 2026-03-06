from unittest.mock import Mock, MagicMock
import pytest
from query.generator import Generator


def make_search_result(
    text="Some content",
    source="notes/test.md",
    title="Test Note",
    heading="Introduction",
    tags="",
    category="notes",
    distance=0.1,
):
    return {
        "text": text,
        "source": source,
        "title": title,
        "heading": heading,
        "tags": tags,
        "category": category,
        "distance": distance,
    }


class TestGenerateStream:
    @pytest.fixture
    def streaming_client(self):
        """Mock client with streaming support."""
        client = Mock()

        # Create a mock stream context manager
        stream_mock = MagicMock()
        stream_mock.__enter__ = Mock(return_value=stream_mock)
        stream_mock.__exit__ = Mock(return_value=False)
        stream_mock.text_stream = iter(["This ", "is ", "streamed."])

        client.messages.stream.return_value = stream_mock
        return client

    def test_yields_text_chunks_and_sources(self, streaming_client):
        generator = Generator(
            client=streaming_client,
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system_prompt="Test prompt",
        )
        chunks = [make_search_result(title="Test Note", source="notes/test.md", heading="Introduction", category="notes")]

        result = list(generator.generate_stream("Question?", chunks))

        assert result[0] == "This "
        assert result[1] == "is "
        assert result[2] == "streamed."
        assert "## Sources" in result[3]
        assert "- [1] Test Note | Introduction | notes | notes/test.md" in result[3]

    def test_calls_stream_method(self, streaming_client):
        generator = Generator(
            client=streaming_client,
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system_prompt="Test prompt",
        )
        chunks = [make_search_result()]

        # Consume the generator
        list(generator.generate_stream("Question?", chunks))

        streaming_client.messages.stream.assert_called_once()
