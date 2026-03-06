from dataclasses import dataclass
from functools import wraps
import httpx
import time

from ingest.chunker import Chunk

READWISE_BASE_URL = "https://readwise.io/api"


@dataclass
class Highlight:
    id: str
    text: str
    title: str
    author: str
    category: str
    tags: list[str]
    readwise_url: str


def retry_with_backoff(max_retries=5, default_backoff=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 429:
                        raise
                    retries += 1
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        sleep_time = int(retry_after)
                    else:
                        sleep_time = default_backoff * (2 ** (retries - 1))
                    print(f"Rate limited. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
            print(f"Failed after {max_retries} retries. Returning partial results.")
            return None

        return wrapper

    return decorator


REQUEST_DELAY = 3  # seconds between requests (20 req/min limit)


class ReadwiseClient:
    def __init__(self, token: str) -> None:
        self._token = token

    def fetch_highlights(self) -> list[Highlight]:
        next_cursor = ""
        highlights = []
        is_first_request = True

        while next_cursor is not None:
            if not is_first_request:
                time.sleep(REQUEST_DELAY)
            is_first_request = False

            result = self._fetch_highlights_page(next_cursor)

            if result is None:
                break

            highlights_page, next_cursor = result
            highlights.extend(highlights_page)

        return highlights

    @retry_with_backoff()
    def _fetch_highlights_page(
        self, page_cursor: str
    ) -> tuple[list[Highlight], str | None]:
        params = {"pageCursor": page_cursor} if page_cursor else {}
        response = httpx.get(
            f"{READWISE_BASE_URL}/v2/export/",
            headers={"Authorization": f"Token {self._token}"},
            params=params,
        )
        response.raise_for_status()

        response_json = response.json()

        highlights = []
        for book in response_json.get("results", []):
            book_title = book.get("title", "")
            book_author = book.get("author", "")
            book_category = book.get("category", "")

            for highlight in book.get("highlights", []):
                # Tags in v2 are objects with "name" key
                tags = [tag.get("name", "") for tag in highlight.get("tags", [])]
                tags = [t for t in tags if t]  # filter empty

                highlights.append(
                    Highlight(
                        id=str(highlight.get("id", "")),
                        text=highlight.get("text", ""),
                        title=book_title,
                        author=book_author,
                        category=book_category,
                        tags=tags,
                        readwise_url=highlight.get("readwise_url", ""),
                    )
                )

        next_cursor = response_json.get("nextPageCursor")

        return highlights, next_cursor


def highlight_to_chunk(highlight: Highlight) -> Chunk:
    return Chunk(
        text=highlight.text,
        source=highlight.readwise_url,
        title=highlight.title,
        heading="",
        tags=highlight.tags,
        category=highlight.category,
        author=highlight.author,
    )
