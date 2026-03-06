import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant. Your job is to answer questions based only on my personal notes.

If there aren't any related notes, tell the user you couldn't find relevant context in the notes.

Cite sources using [1], [2], etc. when referencing information from the notes.
"""


@dataclass
class Config:
    """Application configuration."""

    # Paths
    obsidian_vault_path: str | None
    chroma_path: str

    # API keys
    openai_api_key: str
    anthropic_api_key: str
    readwise_token: str | None

    # Model settings
    embedding_model: str
    claude_model: str
    max_tokens: int
    system_prompt: str

    # Operational settings
    request_delay: float
    batch_size: int


class ConfigError(Exception):
    """Raised when required configuration is missing."""


def _require_env(name: str) -> str:
    """Get a required environment variable or raise ConfigError."""
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def load_config() -> Config:
    """Load configuration from environment variables.

    Raises:
        ConfigError: If required API keys are missing.
    """
    from dotenv import load_dotenv

    load_dotenv()

    return Config(
        obsidian_vault_path=os.getenv("OBSIDIAN_VAULT_PATH"),
        chroma_path=os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "data" / "chroma")),
        openai_api_key=_require_env("OPENAI_API_KEY"),
        anthropic_api_key=_require_env("ANTHROPIC_API_KEY"),
        readwise_token=os.getenv("READWISE_TOKEN"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
        system_prompt=os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        request_delay=float(os.getenv("REQUEST_DELAY", "3.0")),
        batch_size=int(os.getenv("DEFAULT_BATCH_SIZE", "100")),
    )
