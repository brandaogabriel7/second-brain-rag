import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError as OpenAIAuthError,
    OpenAI,
    RateLimitError,
)

from errors import AuthenticationError, EmbeddingError, ServiceUnavailableError

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model: str, client: OpenAI | None = None):
        self._client = client if client else OpenAI()
        self._model = model

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding vector for a single query string.

        Raises:
            AuthenticationError: If OpenAI API key is invalid.
            ServiceUnavailableError: If OpenAI API is unavailable.
        """
        try:
            response = self._client.embeddings.create(
                input=[text], model=self._model
            )
            return response.data[0].embedding
        except OpenAIAuthError as e:
            raise AuthenticationError(
                "OpenAI authentication failed. Check OPENAI_API_KEY."
            ) from e
        except APIConnectionError as e:
            raise ServiceUnavailableError(f"Cannot connect to OpenAI API: {e}") from e
        except RateLimitError as e:
            raise ServiceUnavailableError(f"OpenAI rate limit exceeded: {e}") from e
        except APIStatusError as e:
            raise ServiceUnavailableError(f"OpenAI API error: {e}") from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts in one API call.

        Raises:
            AuthenticationError: If OpenAI API key is invalid (critical).
            EmbeddingError: If batch embedding fails (recoverable).
        """
        if len(texts) == 0:
            return []

        logger.debug(f"Embedding batch of {len(texts)} texts")
        try:
            response = self._client.embeddings.create(
                input=texts, model=self._model
            )
            return [embedding.embedding for embedding in response.data]
        except OpenAIAuthError as e:
            # Auth errors are critical - fail fast
            raise AuthenticationError(
                "OpenAI authentication failed. Check OPENAI_API_KEY."
            ) from e
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            # Other errors are recoverable at batch level
            raise EmbeddingError(len(texts), e) from e
