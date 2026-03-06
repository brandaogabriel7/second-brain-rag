# Second Brain RAG

A RAG (Retrieval-Augmented Generation) system that searches over Obsidian notes and Readwise highlights, turning your personal knowledge base into a queryable AI assistant.

## How It Works

**Ingestion:** Obsidian vault + Readwise highlights → chunked → embedded → stored in ChromaDB

**Query:** Your question → embedded → semantic search → relevant context → Claude → answer

## Tech Stack

- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector DB:** ChromaDB (local persistence)
- **LLM:** Claude (Anthropic)
- **Data sources:** Obsidian vault, Readwise API

## Cost

This project uses paid APIs:
- **OpenAI embeddings** — ~$0.02 per 1M tokens (very cheap, pennies for most vaults)
- **Claude API** — ~$3/M input tokens, $15/M output tokens (each `ask` query costs ~$0.01-0.05 depending on context size)

Ingestion is a one-time cost; queries are ongoing but inexpensive for personal use.

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
uv pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your API keys and vault path
```

### Environment Variables

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OBSIDIAN_VAULT_PATH=/path/to/your/vault
READWISE_TOKEN=...  # optional
CHROMA_PATH=data/chroma  # optional, defaults to data/chroma/
```

The vector database is stored locally at `data/chroma/` by default. This directory is created automatically on first ingest.

## Usage

```bash
# Ingest your data (run once, or when content changes)
uv run python search.py ingest

# Ask a question (uses Claude to generate an answer)
uv run python search.py ask "What are my notes about productivity?"

# Search without AI generation (returns raw chunks)
uv run python search.py query "productivity"

# Verbose mode (detailed logging)
uv run python search.py -v ingest
```

## Development

```bash
# Run tests
uv run pytest -v
```

## Project Structure

```
src/
  ingest/          # Data ingestion (Obsidian, Readwise, chunker)
  embeddings/      # Embedding wrapper (OpenAI)
  storage/         # Vector store (ChromaDB)
  query/           # Retrieval and generation (Claude)
tests/             # Automated tests
search.py          # CLI entry point
```
