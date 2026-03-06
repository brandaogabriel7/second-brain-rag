from dataclasses import dataclass

from anthropic import Anthropic

from config import Config
from embeddings.embed import Embedder
from query.generator import Generator
from query.retriever import Retriever
from storage.vector_store import VectorStore


@dataclass
class AppContext:
    """Application context holding all wired dependencies."""

    config: Config
    embedder: Embedder
    vector_store: VectorStore
    retriever: Retriever
    generator: Generator


def create_app_context(config: Config) -> AppContext:
    """Create application context with all dependencies wired.

    This is the composition root - the single place where all
    dependencies are constructed and connected.
    """
    embedder = Embedder(model=config.embedding_model)
    vector_store = VectorStore(path=config.chroma_path)
    retriever = Retriever(embedder, vector_store)
    generator = Generator(
        client=Anthropic(),
        model=config.claude_model,
        max_tokens=config.max_tokens,
        system_prompt=config.system_prompt,
    )

    return AppContext(
        config=config,
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        generator=generator,
    )
