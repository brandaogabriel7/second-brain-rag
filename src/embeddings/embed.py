import logging
from typing import List

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self):
        self._open_ai_client = OpenAI()

    def embed_query(self, text: str) -> List[float]:
        response = self._open_ai_client.embeddings.create(
            input=[text], model=EMBEDDING_MODEL
        )
        embedding = response.data[0]

        return embedding.embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if len(texts) == 0:
            return []

        logger.debug(f"Embedding batch of {len(texts)} texts")
        response = self._open_ai_client.embeddings.create(
            input=texts, model=EMBEDDING_MODEL
        )

        embeddings = [embedding.embedding for embedding in response.data]
        return embeddings
