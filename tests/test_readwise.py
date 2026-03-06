from unittest.mock import MagicMock, patch

import pytest
from ingest.chunker import highlight_to_chunk
from ingest.models import Chunk, ReadwiseHighlight
from ingest.readwise import ReadwiseClient


@pytest.fixture
def mock_httpx():
    with patch("ingest.readwise.httpx") as mock:
        yield mock


def make_api_response(results, next_cursor=None):
    """Build a mock response matching Readwise v2 Export API shape."""
    return {
        "count": len(results),
        "results": results,
        "nextPageCursor": next_cursor,
    }


def make_book(
    user_book_id=123,
    title="Book Title",
    author="Author Name",
    category="books",
    highlights=None,
):
    """Build a book dict matching Readwise v2 Export API shape."""
    return {
        "user_book_id": user_book_id,
        "title": title,
        "author": author,
        "category": category,
        "highlights": highlights or [],
    }


def make_highlight(
    id=456,
    text="highlighted text",
    tags=None,
    readwise_url="https://readwise.io/open/456",
):
    """Build a highlight dict matching Readwise v2 Export API shape."""
    # Tags in v2 are objects with "name" key
    tag_objects = [{"name": tag} for tag in (tags or [])]
    return {
        "id": id,
        "text": text,
        "tags": tag_objects,
        "readwise_url": readwise_url,
    }


class TestIterHighlightPages:
    def test_yields_multiple_pages(self, mock_httpx):
        mock_httpx.get.return_value.raise_for_status = MagicMock()
        mock_httpx.get.return_value.json.side_effect = [
            make_api_response(
                [make_book(highlights=[make_highlight(id=1, text="first")])],
                next_cursor="cursor123",
            ),
            make_api_response(
                [make_book(highlights=[make_highlight(id=2, text="second")])],
                next_cursor=None,
            ),
        ]

        client = ReadwiseClient(token="test-token")
        pages = list(client.iter_highlight_pages())

        assert len(pages) == 2
        assert pages[0][0].text == "first"
        assert pages[1][0].text == "second"

    def test_passes_cursor_to_next_request(self, mock_httpx):
        mock_httpx.get.return_value.raise_for_status = MagicMock()
        mock_httpx.get.return_value.json.side_effect = [
            make_api_response(
                [make_book(highlights=[make_highlight()])],
                next_cursor="abc123",
            ),
            make_api_response([], next_cursor=None),
        ]

        client = ReadwiseClient(token="test-token")
        list(client.iter_highlight_pages())

        # Second call should include the cursor
        second_call = mock_httpx.get.call_args_list[1]
        assert second_call.kwargs["params"]["pageCursor"] == "abc123"


class TestHighlightToChunk:
    def test_converts_highlight_to_chunk(self):
        highlight = ReadwiseHighlight(
            id="abc123",
            text="This is the highlight text",
            title="Deep Work",
            author="Cal Newport",
            category="books",
            readwise_url="https://readwise.io/open/abc123",
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
        highlight = ReadwiseHighlight(
            id="xyz789",
            text="text",
            title="Title",
            author="Author",
            category="articles",
            readwise_url="https://readwise.io/open/xyz789",
            tags=[],
        )

        chunk = highlight_to_chunk(highlight)

        assert chunk.source == "https://readwise.io/open/xyz789"

    def test_sets_heading_to_empty_string(self):
        highlight = ReadwiseHighlight(
            id="123",
            text="text",
            title="Title",
            author="Author",
            category="books",
            readwise_url="https://readwise.io/open/123",
            tags=[],
        )

        chunk = highlight_to_chunk(highlight)

        assert chunk.heading == ""

    def test_handles_none_author_and_category(self):
        highlight = ReadwiseHighlight(
            id="123",
            text="text",
            title="Title",
            author=None,
            category=None,
            readwise_url="https://readwise.io/open/123",
            tags=[],
        )

        chunk = highlight_to_chunk(highlight)

        assert chunk.author is None
        assert chunk.category is None
