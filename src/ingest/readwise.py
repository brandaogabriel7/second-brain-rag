import logging
import time
from functools import wraps
from typing import Iterator

import httpx

from .models import ReadwiseHighlight

logger = logging.getLogger(__name__)

READWISE_BASE_URL = "https://readwise.io/api"


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
                    logger.warning(f"Rate limited. Retrying in {sleep_time}s (attempt {retries}/{max_retries})...")
                    time.sleep(sleep_time)
            logger.error(f"Failed after {max_retries} retries. Returning partial results.")
            return None

        return wrapper

    return decorator


REQUEST_DELAY = 3  # seconds between requests (20 req/min limit)


class ReadwiseClient:
    def __init__(self, token: str) -> None:
        self._token = token

    def iter_highlight_pages(self) -> Iterator[list[ReadwiseHighlight]]:
        """Yield each page of highlights for progress tracking."""
        next_cursor = ""
        is_first_request = True

        while next_cursor is not None:
            if not is_first_request:
                time.sleep(REQUEST_DELAY)
            is_first_request = False

            result = self._fetch_highlights_page(next_cursor)
            if result is None:
                break

            highlights_page, next_cursor = result
            yield highlights_page

    @retry_with_backoff()
    def _fetch_highlights_page(
        self, page_cursor: str
    ) -> tuple[list[ReadwiseHighlight], str | None]:
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
                tags = [t for t in tags if t]

                highlights.append(
                    ReadwiseHighlight(
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
