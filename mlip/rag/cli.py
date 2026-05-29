"""`mlip rag ...` subcommands for trying the RAG system interactively."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from mlip.rag.pipeline import PROMPT_VARIANTS, RagConfig, RagPipeline
from mlip.rag.retriever import Retriever

rag_app = typer.Typer(help="Interact with the RAG system under test.", no_args_is_help=True)
console = Console()


@rag_app.command()
def retrieve(query: str, k: int = 3) -> None:
    """Show the top-k documents the retriever returns for a query."""
    for hit in Retriever().retrieve(query, k=k):
        console.print(f"[cyan]{hit.score:.3f}[/cyan]  [bold]{hit.document.title}[/bold]")
        console.print(f"  {hit.document.text}\n")


@rag_app.command()
def ask(
    question: str,
    prompt_version: str = typer.Option("v2", help=f"One of: {', '.join(PROMPT_VARIANTS)}"),
    top_k: int = 3,
) -> None:
    """Ask the RAG system a question and print its grounded answer."""
    config = RagConfig(prompt_version=prompt_version, top_k=top_k)
    result = RagPipeline(config).answer(question)
    sources = ", ".join(d.title for d in result.retrieved)
    console.print(Panel(result.answer, title="Answer", border_style="green"))
    console.print(f"[dim]Sources: {sources}[/dim]")
