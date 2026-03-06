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
from errors import ChunkingError, CriticalError, EmbeddingError
from ingest.chunker import Chunker, highlight_to_chunk
from ingest.error_collector import ErrorCollector
from models import Chunk
from ingest.obsidian import ObsidianReader
from ingest.readwise import ReadwiseClient
from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest_obsidian(
    console: Console, vault_path: str, collector: ErrorCollector
) -> list[Chunk]:
    """Read and chunk all notes from an Obsidian vault.

    File read errors are collected and processing continues.
    Chunking errors are collected and processing continues.
    """
    console.print("[bold]Obsidian[/bold]")
    reader = ObsidianReader(vault_path)
    chunker = Chunker()

    logger.info("Reading notes from vault...")
    notes = reader.read_all_vault_notes(collector)
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
            try:
                note_chunks = chunker.chunk_note(note)
                logger.debug(f"Chunked '{note.title}' into {len(note_chunks)} chunks")
                chunks.extend(note_chunks)
            except Exception as e:
                collector.add(ChunkingError(note.title, e))
            progress.advance(task)

    console.print(f"  Created {len(chunks)} chunks")
    return chunks


def ingest_readwise(
    console: Console, token: str, request_delay: float
) -> list[Chunk]:
    """Fetch all highlights from Readwise and convert to chunks."""
    console.print("[bold]Readwise[/bold]")
    client = ReadwiseClient(token, request_delay=request_delay)

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
    non_empty = [h for h in highlights if h.text and h.text.strip()]
    if len(non_empty) != len(highlights):
        logger.debug(f"Filtered {len(highlights) - len(non_empty)} empty highlights")
    return [highlight_to_chunk(h) for h in non_empty]


def ingest(
    console: Console,
    chroma_path: str,
    embedding_model: str,
    vault_path: str = "",
    readwise_token: str = "",
    request_delay: float = 3.0,
    batch_size: int = 100,
) -> bool:
    """Run the full ingestion pipeline.

    Reads from configured sources (Obsidian, Readwise), chunks the content,
    embeds it, and stores in the vector database. Resets the store first.

    Sources are independent: if one fails, others continue.
    Batch embedding errors are collected and processing continues.

    Returns:
        True if ingestion completed (possibly with warnings), False if no data.
    """
    console.print("[bold]Starting ingestion...[/bold]\n")
    collector = ErrorCollector()

    store = VectorStore(path=chroma_path)
    store.reset()

    chunks: list[Chunk] = []

    # === Obsidian (item-level errors are recoverable) ===
    if vault_path:
        try:
            obsidian_chunks = ingest_obsidian(console, vault_path, collector)
            chunks.extend(obsidian_chunks)
        except CriticalError as e:
            console.print(f"[red]Obsidian failed: {e}[/red]")
            logger.error(f"Obsidian ingestion failed: {e}")
            # Continue to try other sources
        except Exception as e:
            console.print(f"[red]Obsidian failed unexpectedly: {e}[/red]")
            logger.exception("Obsidian ingestion failed")
    else:
        console.print("[dim]Obsidian: OBSIDIAN_VAULT_PATH not set, skipping[/dim]")

    # === Readwise (API errors are critical for this source) ===
    if readwise_token:
        try:
            readwise_chunks = ingest_readwise(console, readwise_token, request_delay)
            chunks.extend(readwise_chunks)
        except CriticalError as e:
            console.print(f"[red]Readwise failed: {e}[/red]")
            logger.error(f"Readwise ingestion failed: {e}")
            # Continue to embedding with whatever we have
        except Exception as e:
            console.print(f"[red]Readwise failed unexpectedly: {e}[/red]")
            logger.exception("Readwise ingestion failed")
    else:
        console.print("[dim]Readwise: READWISE_TOKEN not set, skipping[/dim]")

    # Early exit if no chunks
    if not chunks:
        console.print("[yellow]No chunks to embed.[/yellow]")
        if collector.has_errors():
            console.print(f"\n[yellow]{collector.summarize()}[/yellow]")
        return False

    # Filter empty chunks
    original_count = len(chunks)
    chunks = [c for c in chunks if c.text and c.text.strip()]
    if original_count != len(chunks):
        console.print(
            f"[dim]Filtered {original_count - len(chunks)} empty chunks[/dim]"
        )

    # === Embedding (batch failures are recoverable) ===
    console.print("\n[bold]Embedding[/bold]")
    console.print(f"  Processing {len(chunks)} chunks")

    logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}")
    embedder = Embedder(model=embedding_model)

    embedded_count = 0
    failed_batches = 0
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

            try:
                texts = [c.text for c in chunk_batch]
                embeddings = embedder.embed_batch(texts)
                store.add_chunks(chunk_batch, embeddings)
                embedded_count += len(chunk_batch)
            except EmbeddingError as e:
                collector.add(e)
                failed_batches += 1
                # Continue to next batch
            except CriticalError as e:
                # Auth or service errors - stop embedding
                console.print(f"[red]Embedding failed: {e}[/red]")
                logger.error(f"Embedding stopped: {e}")
                break

            progress.advance(task)

    # === Summary ===
    if failed_batches > 0:
        console.print(f"[yellow]  {failed_batches} batch(es) failed[/yellow]")

    console.print(f"\n[bold green]Done![/bold green] Ingested {embedded_count} chunks.")

    if collector.has_errors():
        console.print(f"\n[yellow]{collector.summarize()}[/yellow]")

    return True
