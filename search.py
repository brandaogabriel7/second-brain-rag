import logging
import os
from argparse import ArgumentParser

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.progress import track
from rich.table import Table

from embeddings.embed import Embedder
from ingest.chunker import Chunk, Chunker
from ingest.obsidian import ObsidianReader
from ingest.readwise import ReadwiseClient, highlight_to_chunk
from query.generator import Generator
from query.retriever import Retriever
from storage.vector_store import VectorStore

load_dotenv()

console = Console()


def configure_logging(verbose: bool = False):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


logger = logging.getLogger(__name__)

OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "")
READWISE_TOKEN = os.getenv("READWISE_TOKEN", "")
DEFAULT_TOP_K = 5
INGEST_BATCH_SIZE = 100


def get_retriever():
    embedder = Embedder()
    store = VectorStore()
    return Retriever(embedder, store)


def ingest_obsidian() -> list[Chunk]:
    logger.info("Reading Obsidian vault...")
    obsidian_reader = ObsidianReader(OBSIDIAN_VAULT_PATH)
    notes = obsidian_reader.read_all_vault_notes()
    logger.info(f"Loaded {len(notes)} notes from vault")

    logger.info("Chunking notes...")
    chunker = Chunker()
    chunks = []
    for note in notes:
        chunks.extend(chunker.chunk_note(note))
    logger.info(f"Created {len(chunks)} chunks from notes")

    return chunks


def ingest_readwise() -> list[Chunk]:
    logger.info("Fetching Readwise highlights...")
    readwise_client = ReadwiseClient(READWISE_TOKEN)
    highlights = readwise_client.fetch_highlights()
    logger.info(f"Fetched {len(highlights)} highlights from Readwise")

    chunks = [highlight_to_chunk(highlight) for highlight in highlights]
    return chunks


def ingest():
    console.print("Starting ingestion...")
    store = VectorStore()
    store.reset()

    chunks = []
    if OBSIDIAN_VAULT_PATH:
        chunks.extend(ingest_obsidian())
    else:
        logger.warning("OBSIDIAN_VAULT_PATH not set. Skipping Obsidian ingestion.")

    if READWISE_TOKEN:
        chunks.extend(ingest_readwise())
    else:
        logger.warning("READWISE_TOKEN not set. Skipping Readwise ingestion.")

    # Filter out empty chunks
    original_count = len(chunks)
    chunks = [c for c in chunks if c.text and c.text.strip()]
    if original_count != len(chunks):
        logger.info(f"Filtered out {original_count - len(chunks)} empty chunks")

    logger.info(f"Embedding and storing {len(chunks)} chunks...")

    retriever = Retriever(Embedder(), store)
    for i in track(
        range(0, len(chunks), INGEST_BATCH_SIZE), description="Embedding..."
    ):
        chunk_batch = chunks[i : i + INGEST_BATCH_SIZE]
        retriever.ingest(chunk_batch)

    console.print(f"Done! Ingested {len(chunks)} chunks.")


def query(text: str, top_k: int):
    retriever = get_retriever()
    results = retriever.search(text, top_k)

    table = Table(title="Search Results")
    table.add_column("Title", style="cyan", no_wrap=True)
    table.add_column("Heading", style="magenta")
    table.add_column("Source", style="green")
    table.add_column("Text", style="white")

    for result in results:
        table.add_row(
            result["title"], result["heading"], result["source"], result["text"]
        )

    console.print(table)


def ask(text: str, top_k: int):
    retriever = get_retriever()
    generator = Generator(client=Anthropic())

    chunks = retriever.search(text, top_k)

    if not chunks:
        console.print("[yellow]No relevant notes found.[/yellow]")
        return

    response = ""
    console.print()
    with Live(Markdown(response), console=console, refresh_per_second=10) as live:
        for token in generator.generate_stream(text, chunks):
            response += token
            live.update(Markdown(response))
    console.print()


def main():
    if not OBSIDIAN_VAULT_PATH and not READWISE_TOKEN:
        print(
            "Please set either OBSIDIAN_VAULT_PATH or READWISE_TOKEN environment variable to ingest data."
        )
        return

    parser = ArgumentParser(description="Search through your Second Brain.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed logging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("text", type=str)
    query_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("text", type=str)
    ask_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    if args.command == "ingest":
        ingest()
    elif args.command == "query":
        query(args.text, args.top_k)
    elif args.command == "ask":
        ask(args.text, args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
