"""Custom exceptions for Second Brain RAG."""


class SecondBrainError(Exception):
    """Base exception for all Second Brain errors."""

    pass


# === Critical Errors (fail fast, abort source) ===


class CriticalError(SecondBrainError):
    """Base for errors that should abort the current source."""

    pass


class AuthenticationError(CriticalError):
    """Authentication failed (invalid API key, token expired)."""

    pass


class ServiceUnavailableError(CriticalError):
    """External service is unavailable (503, connection refused, DNS failure)."""

    pass


class RateLimitExhaustedError(CriticalError):
    """Rate limit retries exhausted."""

    pass


# === Recoverable Errors (log and continue) ===


class RecoverableError(SecondBrainError):
    """Base for errors that should be logged but not abort processing."""

    pass


class FileReadError(RecoverableError):
    """Failed to read a single file (encoding, permissions)."""

    def __init__(self, path: str, original_error: Exception):
        self.path = path
        self.original_error = original_error
        super().__init__(f"Failed to read '{path}': {original_error}")


class EmbeddingError(RecoverableError):
    """Failed to embed a batch of chunks."""

    def __init__(self, batch_size: int, original_error: Exception):
        self.batch_size = batch_size
        self.original_error = original_error
        super().__init__(f"Failed to embed batch of {batch_size}: {original_error}")


class ChunkingError(RecoverableError):
    """Failed to chunk a single note."""

    def __init__(self, title: str, original_error: Exception):
        self.title = title
        self.original_error = original_error
        super().__init__(f"Failed to chunk '{title}': {original_error}")
