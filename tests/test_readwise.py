from unittest.mock import MagicMock, patch

import pytest
from ingest.readwise import ReadwiseClient, Highlight, highlight_to_chunk
from ingest.chunker import Chunk


@pytest.fixture
def mock_httpx():
    with patch("ingest.readwise.httpx") as mock:
        yield mock


def make_api_response(results, next_cursor=None):
    """Build a mock response matching Readwise API shape."""
    return {
        "results": results,
        "nextPageCursor": next_cursor,
    }


def make_highlight(
    id="123",
    content="highlighted text",
    title="Book Title",
    author="Author Name",
    category="books",
    url="https://readwise.io/open/123",
    tags=None,
):
    """Build a highlight dict matching Readwise API shape."""
    return {
        "id": id,
        "content": content,
        "title": title,
        "author": author,
        "category": category,
        "url": url,
        "tags": tags or [],
    }


class TestFetchHighlights:
    def test_returns_list_of_highlights(self, mock_httpx):
        mock_httpx.get.return_value.json.return_value = make_api_response(
            [make_highlight(content="test highlight")]
        )
        mock_httpx.get.return_value.raise_for_status = MagicMock()

        client = ReadwiseClient(token="test-token")
        result = client.fetch_highlights()

        assert len(result) == 1
        assert isinstance(result[0], Highlight)
        assert result[0].content == "test highlight"

    def test_sends_auth_header(self, mock_httpx):
        mock_httpx.get.return_value.json.return_value = make_api_response([])
        mock_httpx.get.return_value.raise_for_status = MagicMock()

        client = ReadwiseClient(token="my-secret-token")
        client.fetch_highlights()

        mock_httpx.get.assert_called()
        call_kwargs = mock_httpx.get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Token my-secret-token"

    def test_parses_highlight_fields(self, mock_httpx):
        mock_httpx.get.return_value.json.return_value = make_api_response(
            [
                make_highlight(
                    content="The best time to plant a tree was 20 years ago.",
                    title="Atomic Habits",
                    author="James Clear",
                    category="books",
                    url="https://readwise.io/open/456",
                    tags=["productivity", "habits"],
                )
            ]
        )
        mock_httpx.get.return_value.raise_for_status = MagicMock()

        client = ReadwiseClient(token="test-token")
        result = client.fetch_highlights()

        h = result[0]
        assert h.content == "The best time to plant a tree was 20 years ago."
        assert h.title == "Atomic Habits"
        assert h.author == "James Clear"
        assert h.category == "books"
        assert h.url == "https://readwise.io/open/456"
        assert h.tags == ["productivity", "habits"]

    def test_handles_empty_results(self, mock_httpx):
        mock_httpx.get.return_value.json.return_value = make_api_response([])
        mock_httpx.get.return_value.raise_for_status = MagicMock()

        client = ReadwiseClient(token="test-token")
        result = client.fetch_highlights()

        assert result == []


class TestPagination:
    def test_fetches_multiple_pages(self, mock_httpx):
        # First call returns page 1 with cursor, second call returns page 2
        mock_httpx.get.return_value.raise_for_status = MagicMock()
        mock_httpx.get.return_value.json.side_effect = [
            make_api_response(
                [make_highlight(id="1", content="first")],
                next_cursor="cursor123",
            ),
            make_api_response(
                [make_highlight(id="2", content="second")],
                next_cursor=None,
            ),
        ]

        client = ReadwiseClient(token="test-token")
        result = client.fetch_highlights()

        assert len(result) == 2
        assert result[0].content == "first"
        assert result[1].content == "second"

    def test_passes_cursor_to_next_request(self, mock_httpx):
        mock_httpx.get.return_value.raise_for_status = MagicMock()
        mock_httpx.get.return_value.json.side_effect = [
            make_api_response([make_highlight()], next_cursor="abc123"),
            make_api_response([], next_cursor=None),
        ]

        client = ReadwiseClient(token="test-token")
        client.fetch_highlights()

        # Second call should include the cursor
        second_call = mock_httpx.get.call_args_list[1]
        assert second_call.kwargs["params"]["pageCursor"] == "abc123"


class TestHighlightToChunk:
    def test_converts_highlight_to_chunk(self):
        highlight = Highlight(
            id="abc123",
            content="This is the highlight text",
            title="Deep Work",
            author="Cal Newport",
            category="books",
            url="https://readwise.io/open/abc123",
            tags=["focus", "productivity"],
        )

        chunk = highlight_to_chunk(highlight)

        assert isinstance(chunk, Chunk)
        assert chunk.text == "This is the highlight text"
        assert chunk.title == "Deep Work"
        assert chunk.author == "Cal Newport"
        assert chunk.category == "books"
        assert chunk.tags == ["focus", "productivity"]

    def test_uses_readwise_url_as_source(self):
        highlight = Highlight(
            id="xyz789",
            content="text",
            title="Title",
            author="Author",
            category="articles",
            url="https://readwise.io/open/xyz789",
            tags=[],
        )

        chunk = highlight_to_chunk(highlight)

        assert chunk.source == "https://readwise.io/open/xyz789"

    def test_sets_heading_to_empty_string(self):
        highlight = Highlight(
            id="123",
            content="text",
            title="Title",
            author="Author",
            category="books",
            url="https://readwise.io/open/123",
            tags=[],
        )

        chunk = highlight_to_chunk(highlight)

        assert chunk.heading == ""

    def test_handles_none_author_and_category(self):
        highlight = Highlight(
            id="123",
            content="text",
            title="Title",
            author=None,
            category=None,
            url="https://readwise.io/open/123",
            tags=[],
        )

        chunk = highlight_to_chunk(highlight)

        assert chunk.author is None
        assert chunk.category is None
