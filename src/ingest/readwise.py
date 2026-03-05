from dataclasses import dataclass
import httpx

READWISE_BASE_URL = "https://readwise.io/api"


@dataclass
class Highlight:
    id: str
    content: str
    title: str
    author: str
    category: str
    tags: list[str]


class ReadwiseClient:
    def __init__(self, token: str) -> None:
        self._token = token

    def fetch_highlights(self) -> list[Highlight]:
        next_cursor = ""
        highlights = []

        while next_cursor is not None:
            params = {}
            if next_cursor:
                params["pageCursor"] = next_cursor
            response = httpx.get(
                f"{READWISE_BASE_URL}/v3/list/",
                headers={"Authorization": f"Token {self._token}"},
                params=params,
            )
            response.raise_for_status()

            response_json = response.json()

            highlights = highlights + [
                Highlight(
                    id=result["id"],
                    content=result["content"],
                    title=result["title"],
                    author=result.get("author"),
                    category=result.get("category"),
                    tags=result.get("tags", []),
                )
                for result in response_json.get("results", [])
            ]

            next_cursor = response_json.get("nextPageCursor")

        return highlights
