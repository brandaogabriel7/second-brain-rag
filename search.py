import logging
import os
from argparse import ArgumentParser

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.table import Table

from embeddings.embed import Embedder
from errors import CriticalError
from ingest.pipeline import ingest
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


def get_retriever():
    embedder = Embedder()
    store = VectorStore()
    return Retriever(embedder, store)


def query(text: str, top_k: int):
    try:
        retriever = get_retriever()
    except Exception as e:
        console.print(f"[red]Failed to initialize search: {e}[/red]")
        return

    try:
        results = retriever.search(text, top_k)
    except CriticalError as e:
        console.print(f"[red]Search failed: {e}[/red]")
        return
    except Exception as e:
        console.print(f"[red]Search failed unexpectedly: {e}[/red]")
        logger.exception("Search failed")
        return

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
    try:
        retriever = get_retriever()
    except Exception as e:
        console.print(f"[red]Failed to initialize search: {e}[/red]")
        return

    try:
        generator = Generator(client=Anthropic())
    except Exception as e:
        console.print(f"[red]Failed to initialize Claude client: {e}[/red]")
        return

    try:
        chunks = retriever.search(text, top_k)
    except CriticalError as e:
        console.print(f"[red]Search failed: {e}[/red]")
        return
    except Exception as e:
        console.print(f"[red]Search failed unexpectedly: {e}[/red]")
        logger.exception("Search failed")
        return

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
        ingest(console, OBSIDIAN_VAULT_PATH, READWISE_TOKEN)
    elif args.command == "query":
        query(args.text, args.top_k)
    elif args.command == "ask":
        ask(args.text, args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
