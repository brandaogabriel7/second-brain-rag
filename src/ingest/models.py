from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional


class HeadingSection(NamedTuple):
    heading: str
    text: str


@dataclass
class ObsidianNote:
    title: str
    path: str
    frontmatter: Dict
    tags: List[str]
    content: str


@dataclass
class Chunk:
    text: str
    source: str
    title: str
    heading: str
    tags: List[str]
    author: Optional[str] = None
    category: Optional[str] = None


@dataclass
class ReadwiseHighlight:
    id: str
    text: str
    title: str
    author: str
    category: str
    tags: list[str]
    readwise_url: str
