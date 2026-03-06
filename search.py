import logging
import os
from argparse import ArgumentParser

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn
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
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
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
    console.print("[bold]Obsidian[/bold]")
    reader = ObsidianReader(OBSIDIAN_VAULT_PATH)
    chunker = Chunker()

    # Read all notes first (warnings may appear here)
    logger.info("Reading notes from vault...")
    notes = reader.read_all_vault_notes()
    if not notes:
        console.print("  No notes found")
        return []

    console.print(f"  Found {len(notes)} notes")
    logger.info(f"Loaded {len(notes)} notes")

    chunks = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("  Chunking...", total=len(notes))
        for note in notes:
            note_chunks = chunker.chunk_note(note)
            logger.debug(f"Chunked '{note.title}' into {len(note_chunks)} chunks")
            chunks.extend(note_chunks)
            progress.advance(task)

    console.print(f"  Created {len(chunks)} chunks")
    return chunks


def ingest_readwise() -> list[Chunk]:
    console.print("[bold]Readwise[/bold]")
    client = ReadwiseClient(READWISE_TOKEN)

    logger.info("Fetching highlights from Readwise API...")
    highlights = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("  Fetching highlights...", total=None)
        for page_num, page in enumerate(client.iter_highlight_pages(), 1):
            highlights.extend(page)
            logger.info(f"Page {page_num}: fetched {len(page)} highlights ({len(highlights)} total)")
            progress.update(task, description=f"  Fetched {len(highlights)} highlights (page {page_num})")

    console.print(f"  Fetched {len(highlights)} highlights")
    return [highlight_to_chunk(h) for h in highlights]


def ingest():
    console.print("[bold]Starting ingestion...[/bold]\n")
    store = VectorStore()
    store.reset()

    chunks = []
    if OBSIDIAN_VAULT_PATH:
        chunks.extend(ingest_obsidian())
    else:
        console.print("[dim]Obsidian: OBSIDIAN_VAULT_PATH not set, skipping[/dim]")

    if READWISE_TOKEN:
        chunks.extend(ingest_readwise())
    else:
        console.print("[dim]Readwise: READWISE_TOKEN not set, skipping[/dim]")

    original_count = len(chunks)
    chunks = [c for c in chunks if c.text and c.text.strip()]
    if original_count != len(chunks):
        console.print(f"[dim]Filtered {original_count - len(chunks)} empty chunks[/dim]")

    console.print(f"\n[bold]Embedding[/bold]")
    console.print(f"  Processing {len(chunks)} chunks")

    logger.info(f"Embedding {len(chunks)} chunks in batches of {INGEST_BATCH_SIZE}")
    retriever = Retriever(Embedder(), store)
    batch_count = (len(chunks) + INGEST_BATCH_SIZE - 1) // INGEST_BATCH_SIZE

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("  Embedding...", total=batch_count)
        for i in range(0, len(chunks), INGEST_BATCH_SIZE):
            batch_num = i // INGEST_BATCH_SIZE + 1
            chunk_batch = chunks[i : i + INGEST_BATCH_SIZE]
            logger.info(f"Embedding batch {batch_num}/{batch_count} ({len(chunk_batch)} chunks)")
            retriever.ingest(chunk_batch)
            progress.advance(task)

    console.print(f"\n[bold green]Done![/bold green] Ingested {len(chunks)} chunks.")


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
