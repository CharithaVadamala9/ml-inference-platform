"""`mlip` command-line entry point.

Subcommands are wired up as each build-slice lands (rag, eval, serve, ...).
For now it exposes `info` so the scaffold is runnable end to end.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from mlip import __version__
from mlip.config import settings
from mlip.eval.cli import eval_app
from mlip.rag.cli import rag_app

app = typer.Typer(
    name="mlip",
    help="ML Inference Platform — eval pipeline + serving control plane.",
    no_args_is_help=True,
)
app.add_typer(rag_app, name="rag")
app.add_typer(eval_app, name="eval")
console = Console()


@app.callback()
def main() -> None:
    """ML Inference Platform control plane. Run a subcommand below."""


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Run the FastAPI serving gateway (with Prometheus metrics at /metrics)."""
    import uvicorn

    console.print(f"[bold]Serving[/bold] on http://{host}:{port}  (metrics: /metrics)")
    uvicorn.run("mlip.serving.app:create_app", host=host, port=port, factory=True, reload=reload)


@app.command()
def info() -> None:
    """Show the current configuration the platform will run with."""
    table = Table(title=f"ML Inference Platform v{__version__}")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Serving backend", settings.serving_backend)
    table.add_row("Judge provider", settings.judge_provider)
    table.add_row("Judge model", settings.judge_model)
    table.add_row("MLflow URI", settings.mlflow_tracking_uri)
    console.print(table)


if __name__ == "__main__":
    app()
