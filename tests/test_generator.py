from unittest.mock import Mock, MagicMock
import pytest
from anthropic.types import TextBlock
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


@pytest.fixture
def mock_client():
    """Mock Anthropic client."""
    client = Mock()
    response = Mock()
    response.content = [TextBlock(type="text", text="This is the answer.")]
    client.messages.create.return_value = response
    return client


@pytest.fixture
def generator(mock_client):
    return Generator(client=mock_client)


class TestGenerate:
    def test_returns_response_with_sources(self, generator, mock_client):
        chunks = [make_search_result(title="Test Note", source="notes/test.md", heading="Introduction", category="notes")]
        result = generator.generate("What is RAG?", chunks)
        assert "This is the answer." in result
        assert "Sources:" in result
        assert "[1] Test Note | Introduction | notes | notes/test.md" in result

    def test_calls_client_with_messages(self, generator, mock_client):
        chunks = [make_search_result(text="RAG is retrieval augmented generation")]
        generator.generate("What is RAG?", chunks)

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs

        assert "messages" in call_kwargs
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_includes_query_in_message(self, generator, mock_client):
        chunks = [make_search_result()]
        generator.generate("What is RAG?", chunks)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_message = call_kwargs["messages"][0]["content"]
        assert "What is RAG?" in user_message

    def test_includes_chunk_text_in_message(self, generator, mock_client):
        chunks = [make_search_result(text="Important info here")]
        generator.generate("Tell me more", chunks)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_message = call_kwargs["messages"][0]["content"]
        assert "Important info here" in user_message

    def test_includes_chunk_source_in_message(self, generator, mock_client):
        chunks = [make_search_result(source="projects/rag.md")]
        generator.generate("Question?", chunks)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_message = call_kwargs["messages"][0]["content"]
        assert "projects/rag.md" in user_message

    def test_includes_chunk_heading_in_message(self, generator, mock_client):
        chunks = [make_search_result(heading="Architecture")]
        generator.generate("Question?", chunks)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_message = call_kwargs["messages"][0]["content"]
        assert "Architecture" in user_message

    def test_numbers_multiple_chunks(self, generator, mock_client):
        chunks = [
            make_search_result(text="First chunk"),
            make_search_result(text="Second chunk"),
            make_search_result(text="Third chunk"),
        ]
        generator.generate("Question?", chunks)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_message = call_kwargs["messages"][0]["content"]
        assert "[1]" in user_message
        assert "[2]" in user_message
        assert "[3]" in user_message

    def test_context_format_associates_number_with_source(self, generator, mock_client):
        chunks = [
            make_search_result(source="notes/first.md", text="First content"),
            make_search_result(source="notes/second.md", text="Second content"),
        ]
        generator.generate("Question?", chunks)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_message = call_kwargs["messages"][0]["content"]

        # [1] should appear before [2]
        assert user_message.index("[1]") < user_message.index("[2]")
        # Each source should appear near its number
        assert user_message.index("[1]") < user_message.index("notes/first.md") < user_message.index("[2]")
        assert user_message.index("[2]") < user_message.index("notes/second.md")

    def test_uses_system_prompt(self, generator, mock_client):
        chunks = [make_search_result()]
        generator.generate("Question?", chunks)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 0

    def test_specifies_model(self, generator, mock_client):
        chunks = [make_search_result()]
        generator.generate("Question?", chunks)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "model" in call_kwargs

    def test_specifies_max_tokens(self, generator, mock_client):
        chunks = [make_search_result()]
        generator.generate("Question?", chunks)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "max_tokens" in call_kwargs

    def test_with_empty_chunks_omits_sources(self, generator, mock_client):
        # Should still work - Claude will say it has no context
        result = generator.generate("What is RAG?", [])
        assert result == "This is the answer."

    def test_includes_category_in_sources(self, generator, mock_client):
        chunks = [
            make_search_result(title="Deep Work", heading="", category="books", source="https://readwise.io/open/123"),
        ]
        result = generator.generate("Question?", chunks)
        assert "[1] Deep Work | books | https://readwise.io/open/123" in result

    def test_mixed_sources_with_different_categories(self, generator, mock_client):
        chunks = [
            make_search_result(title="My Note", heading="Intro", category="notes", source="notes/test.md"),
            make_search_result(title="Atomic Habits", heading="", category="books", source="https://readwise.io/open/456"),
        ]
        result = generator.generate("Question?", chunks)
        assert "[1] My Note | Intro | notes | notes/test.md" in result
        assert "[2] Atomic Habits | books | https://readwise.io/open/456" in result


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
        generator = Generator(client=streaming_client)
        chunks = [make_search_result(title="Test Note", source="notes/test.md", heading="Introduction", category="notes")]

        result = list(generator.generate_stream("Question?", chunks))

        assert result[0] == "This "
        assert result[1] == "is "
        assert result[2] == "streamed."
        assert "Sources:" in result[3]
        assert "[1] Test Note | Introduction | notes | notes/test.md" in result[3]

    def test_calls_stream_method(self, streaming_client):
        generator = Generator(client=streaming_client)
        chunks = [make_search_result()]

        # Consume the generator
        list(generator.generate_stream("Question?", chunks))

        streaming_client.messages.stream.assert_called_once()
