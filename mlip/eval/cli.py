"""`mlip eval ...` subcommands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from mlip.eval.runner import run_eval
from mlip.rag.pipeline import PROMPT_VARIANTS, RagConfig

eval_app = typer.Typer(help="Run the agentic eval pipeline.", no_args_is_help=True)
console = Console()


@eval_app.command()
def run(
    name: str = typer.Option("baseline", help="Run name (also the MLflow run + report name)."),
    prompt_version: str = typer.Option("v2", help=f"One of: {', '.join(PROMPT_VARIANTS)}"),
    top_k: int = typer.Option(3, help="Documents retrieved per question."),
    no_mlflow: bool = typer.Option(False, help="Skip MLflow logging (just write the report)."),
) -> None:
    """Evaluate one RAG config and print its scorecard."""
    config = RagConfig(name=name, prompt_version=prompt_version, top_k=top_k)
    console.print(f"[bold]Evaluating[/bold] config={config.to_dict()}")
    result = run_eval(config, log_to_mlflow=not no_mlflow)

    table = Table(title=f"Scorecard — {name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green", justify="right")
    for metric, value in result.scorecard.items():
        table.add_row(metric, f"{value:.3f}")
    console.print(table)
    console.print(f"[dim]Report: {result.report_path}[/dim]")
    if result.run_id:
        console.print(f"[dim]MLflow run: {result.run_id}[/dim]")
