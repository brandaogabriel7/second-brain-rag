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
OBSIDIAN_VAULT_PATH=/path/to/your/vault
```

## Development

```bash
# Run tests
uv run pytest -v
```

## Project Structure

```
src/
  ingest/          # Data ingestion (Obsidian reader, chunker)
  embeddings/      # Embedding wrapper (OpenAI)
  storage/         # Vector store (ChromaDB)
  query/           # Retrieval and generation
tests/             # Automated tests
```
