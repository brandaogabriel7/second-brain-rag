import logging

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from embeddings.embed import Embedder
from ingest.chunker import Chunker, highlight_to_chunk
from ingest.models import Chunk
from ingest.obsidian import ObsidianReader
from ingest.readwise import ReadwiseClient
from query.retriever import Retriever
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100


def ingest_obsidian(console: Console, vault_path: str) -> list[Chunk]:
    """Read and chunk all notes from an Obsidian vault."""
    console.print("[bold]Obsidian[/bold]")
    reader = ObsidianReader(vault_path)
    chunker = Chunker()

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


def ingest_readwise(console: Console, token: str) -> list[Chunk]:
    """Fetch all highlights from Readwise and convert to chunks."""
    console.print("[bold]Readwise[/bold]")
    client = ReadwiseClient(token)

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
            logger.info(
                f"Page {page_num}: fetched {len(page)} highlights ({len(highlights)} total)"
            )
            progress.update(
                task,
                description=f"  Fetched {len(highlights)} highlights (page {page_num})",
            )

    console.print(f"  Fetched {len(highlights)} highlights")
    return [highlight_to_chunk(h) for h in highlights]


def ingest(
    console: Console,
    vault_path: str = "",
    readwise_token: str = "",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Run the full ingestion pipeline.

    Reads from configured sources (Obsidian, Readwise), chunks the content,
    embeds it, and stores in the vector database. Resets the store first.
    """
    console.print("[bold]Starting ingestion...[/bold]\n")
    store = VectorStore()
    store.reset()

    chunks = []
    if vault_path:
        chunks.extend(ingest_obsidian(console, vault_path))
    else:
        console.print("[dim]Obsidian: OBSIDIAN_VAULT_PATH not set, skipping[/dim]")

    if readwise_token:
        chunks.extend(ingest_readwise(console, readwise_token))
    else:
        console.print("[dim]Readwise: READWISE_TOKEN not set, skipping[/dim]")

    original_count = len(chunks)
    chunks = [c for c in chunks if c.text and c.text.strip()]
    if original_count != len(chunks):
        console.print(
            f"[dim]Filtered {original_count - len(chunks)} empty chunks[/dim]"
        )

    console.print("\n[bold]Embedding[/bold]")
    console.print(f"  Processing {len(chunks)} chunks")

    logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}")
    retriever = Retriever(Embedder(), store)
    batch_count = (len(chunks) + batch_size - 1) // batch_size

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("  Embedding...", total=batch_count)
        for i in range(0, len(chunks), batch_size):
            batch_num = i // batch_size + 1
            chunk_batch = chunks[i : i + batch_size]
            logger.info(
                f"Embedding batch {batch_num}/{batch_count} ({len(chunk_batch)} chunks)"
            )
            retriever.ingest(chunk_batch)
            progress.advance(task)

    console.print(f"\n[bold green]Done![/bold green] Ingested {len(chunks)} chunks.")
