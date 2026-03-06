from dataclasses import dataclass
from typing import NamedTuple


class HeadingSection(NamedTuple):
    heading: str
    text: str


@dataclass
class ObsidianNote:
    """A note from an Obsidian vault.

    Represents the parsed content and metadata of a single markdown file.
    """

    title: str
    path: str
    frontmatter: dict
    tags: list[str]
    content: str


@dataclass
class Chunk:
    """A text chunk ready for embedding and storage.

    The atomic unit stored in the vector database. Can originate from
    Obsidian notes (split by heading) or Readwise highlights (1:1).
    """

    text: str
    source: str
    title: str
    heading: str
    tags: list[str]
    author: str | None = None
    category: str | None = None


@dataclass
class ReadwiseHighlight:
    """A highlight fetched from the Readwise API.

    Contains the highlight text plus metadata about its source book/article.
    """

    id: str
    text: str
    title: str
    author: str
    category: str
    tags: list[str]
    readwise_url: str
