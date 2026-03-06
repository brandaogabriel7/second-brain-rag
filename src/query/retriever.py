import logging

from embeddings.embed import Embedder
from errors import CriticalError
from models import Chunk
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def search(self, query: str, top_k: int) -> list[dict]:
        """Find chunks semantically similar to the query.

        Embeds the query and searches the vector store for nearest neighbors.

        Raises:
            CriticalError: If embedding or search fails critically.
        """
        try:
            embedding = self._embedder.embed_query(query)
            results = self._store.search(embedding, top_k=top_k)
            return results
        except CriticalError:
            # Re-raise critical errors (auth failures, etc.)
            raise
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def ingest(self, chunks: list[Chunk]) -> None:
        """Embed chunks and add them to the vector store."""
        if len(chunks) == 0:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedder.embed_batch(texts)
        self._store.add_chunks(chunks, embeddings)
