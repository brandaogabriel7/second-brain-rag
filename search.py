import logging
import sys
from argparse import ArgumentParser

from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.table import Table

from config import ConfigError, load_config
from context import AppContext, create_app_context
from errors import CriticalError
from ingest.pipeline import ingest

console = Console()


def configure_logging(verbose: bool = False):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


def query(ctx: AppContext, text: str, top_k: int):
    try:
        results = ctx.retriever.search(text, top_k)
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


def ask(ctx: AppContext, text: str, top_k: int):
    try:
        chunks = ctx.retriever.search(text, top_k)
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
        for token in ctx.generator.generate_stream(text, chunks):
            response += token
            live.update(Markdown(response))
    console.print()


def main():
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

    try:
        config = load_config()
    except ConfigError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)

    if args.command == "ingest":
        if not config.obsidian_vault_path and not config.readwise_token:
            console.print(
                "[red]Please set either OBSIDIAN_VAULT_PATH or READWISE_TOKEN "
                "environment variable to ingest data.[/red]"
            )
            sys.exit(1)

        ingest(
            console,
            chroma_path=config.chroma_path,
            embedding_model=config.embedding_model,
            vault_path=config.obsidian_vault_path or "",
            readwise_token=config.readwise_token or "",
            request_delay=config.request_delay,
            batch_size=config.batch_size,
        )
    elif args.command == "query":
        try:
            ctx = create_app_context(config)
        except Exception as e:
            console.print(f"[red]Failed to initialize: {e}[/red]")
            sys.exit(1)
        query(ctx, args.text, args.top_k)
    elif args.command == "ask":
        try:
            ctx = create_app_context(config)
        except Exception as e:
            console.print(f"[red]Failed to initialize: {e}[/red]")
            sys.exit(1)
        ask(ctx, args.text, args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
