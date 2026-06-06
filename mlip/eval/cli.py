"""`mlip eval ...` subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mlip.eval.ab import run_ab
from mlip.eval.champion import load_champion, promote
from mlip.eval.gate import (
    DEFAULT_TOLERANCE,
    StatGateResult,
    evaluate_gate,
    evaluate_gate_statistical,
)
from mlip.eval.runner import run_eval
from mlip.rag.pipeline import PROMPT_VARIANTS, RagConfig

eval_app = typer.Typer(help="Run the agentic eval pipeline.", no_args_is_help=True)
console = Console()


@eval_app.command()
def run(
    name: str = typer.Option("baseline", help="Run name (also the MLflow run + report name)."),
    prompt_version: str = typer.Option("v2", help=f"One of: {', '.join(PROMPT_VARIANTS)}"),
    top_k: int = typer.Option(3, help="Documents retrieved per question."),
    no_rag: bool = typer.Option(False, "--no-rag", help="Evaluate the bare model (no retrieval)."),
    no_mlflow: bool = typer.Option(False, help="Skip MLflow logging (just write the report)."),
) -> None:
    """Evaluate one config and print its scorecard (RAG by default, or --no-rag)."""
    config = RagConfig(
        name=name, prompt_version=prompt_version, top_k=top_k, use_retrieval=not no_rag
    )
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


@eval_app.command()
def ab(
    a_prompt: str = typer.Option("v1", help="Prompt version for variant A."),
    b_prompt: str = typer.Option("v2", help="Prompt version for variant B."),
    a_top_k: int = typer.Option(3, help="top_k for variant A."),
    b_top_k: int = typer.Option(3, help="top_k for variant B."),
    primary_metric: str = typer.Option("faithfulness", help="Metric that decides the winner."),
    promote_winner: bool = typer.Option(False, help="Promote the winning config to champion."),
    no_mlflow: bool = typer.Option(False, help="Skip MLflow logging."),
) -> None:
    """Run two variants through the eval pipeline and compare them head-to-head."""
    cfg_a = RagConfig(name=f"A-{a_prompt}-k{a_top_k}", prompt_version=a_prompt, top_k=a_top_k)
    cfg_b = RagConfig(name=f"B-{b_prompt}-k{b_top_k}", prompt_version=b_prompt, top_k=b_top_k)
    console.print(f"[bold]A:[/bold] {cfg_a.to_dict()}\n[bold]B:[/bold] {cfg_b.to_dict()}")
    result = run_ab(cfg_a, cfg_b, primary_metric=primary_metric, log_to_mlflow=not no_mlflow)

    table = Table(title=f"A/B — primary: {primary_metric}")
    for col in ("Metric", "A", "B", "Δ (B−A)", "Winner"):
        table.add_column(col, style="cyan" if col == "Metric" else "white")
    for c in result.comparisons:
        mark = {"A": "[yellow]A[/yellow]", "B": "[green]B[/green]", "tie": "[dim]tie[/dim]"}[
            c.winner
        ]
        table.add_row(c.metric, f"{c.a:.3f}", f"{c.b:.3f}", f"{c.delta:+.3f}", mark)
    console.print(table)
    console.print(f"[bold]Overall winner: {result.overall_winner}[/bold]")

    if promote_winner:
        winner = result.run_b if result.overall_winner == "B" else result.run_a
        promote(
            config=winner.config,
            scorecard=winner.scorecard,
            per_question=winner.per_question,
            mlflow_run_id=winner.run_id,
            tag_mlflow=not no_mlflow,
        )
        console.print(f"[green]Promoted winner ({result.overall_winner}) to champion.[/green]")


@eval_app.command(name="promote")
def promote_cmd(
    name: str = typer.Option("champion", help="Run name for the promoted config."),
    prompt_version: str = typer.Option("v2", help=f"One of: {', '.join(PROMPT_VARIANTS)}"),
    top_k: int = typer.Option(3, help="Documents retrieved per question."),
    no_mlflow: bool = typer.Option(False, help="Skip MLflow logging."),
) -> None:
    """Evaluate a config and promote it as the new champion (the quality bar)."""
    config = RagConfig(name=name, prompt_version=prompt_version, top_k=top_k)
    result = run_eval(config, log_to_mlflow=not no_mlflow)
    record = promote(
        config=result.config,
        scorecard=result.scorecard,
        per_question=result.per_question,
        mlflow_run_id=result.run_id,
        tag_mlflow=not no_mlflow,
    )
    console.print("[green]New champion promoted.[/green]")
    for metric, value in record["scorecard"].items():
        console.print(f"  {metric}: {value:.3f}")


@eval_app.command()
def champion() -> None:
    """Show the current champion (the bar the CI gate enforces)."""
    record = load_champion()
    if record is None:
        console.print("[yellow]No champion promoted yet. Run `mlip eval promote`.[/yellow]")
        raise typer.Exit(code=0)
    console.print(f"[bold]Champion config:[/bold] {record['config']}")
    for metric, value in record["scorecard"].items():
        console.print(f"  {metric}: {value:.3f}")
    console.print(
        f"[dim]Promoted at {record['promoted_at']} (run {record.get('mlflow_run_id')})[/dim]"
    )


def _candidate(report: Path | None, champion: dict, no_mlflow: bool) -> dict:
    """Return the candidate's {scorecard, per_question} from a report or a fresh run."""
    if report is not None:
        console.print(f"[bold]Gating report[/bold] {report}")
        data = json.loads(report.read_text(encoding="utf-8"))
        return {
            "scorecard": data.get("scorecard", {}),
            "per_question": data.get("per_question", []),
        }
    console.print("[bold]Gating candidate[/bold] (champion config under current code)")
    run = run_eval(RagConfig(**champion["config"]), log_to_mlflow=not no_mlflow)
    return {"scorecard": run.scorecard, "per_question": run.per_question}


def _render_stat(result: StatGateResult) -> None:
    """Print the statistical-gate verdict table to the terminal."""
    console.print(
        f"[dim]Paired on id: {result.matched} matched · "
        f"{result.dropped_candidate} candidate-only · {result.dropped_champion} champion-only · "
        f"{result.content_mismatches} content mismatch(es) · "
        f"correction={result.correction_method} @ alpha={result.alpha}[/dim]"
    )
    table = Table(title="Statistical quality gate")
    for col in ("Metric", "Cat", "n", "Champion", "Candidate", "Δ", "95% CI", "Verdict"):
        table.add_column(col, style="cyan" if col == "Metric" else "white")
    for t in result.tests:
        ci = "—" if t.kind == "mcnemar" else f"[{t.ci_lo:+.3f}, {t.ci_hi:+.3f}]"
        if not t.gated:
            verdict = "[dim]informational[/dim]"
        elif t.regression:
            verdict = "[red]significant regression[/red]"
        else:
            verdict = "[green]within noise[/green]"
        table.add_row(
            t.metric,
            t.category,
            str(t.n),
            f"{t.champion_mean:.3f}",
            f"{t.candidate_mean:.3f}",
            f"{t.delta:+.3f}",
            ci,
            verdict,
        )
    console.print(table)


@eval_app.command()
def gate(
    report: Path | None = typer.Option(
        None, help="Gate an existing report instead of running a fresh eval."
    ),
    tolerance: float = typer.Option(DEFAULT_TOLERANCE, help="Naive-mode allowed regression."),
    no_mlflow: bool = typer.Option(True, help="Skip MLflow logging (default in CI)."),
) -> None:
    """Statistically-honest gate: FAIL only on a significant regression vs the champion."""
    champion = load_champion()
    if champion is None:
        console.print("[red]No champion to gate against. Run `mlip eval promote` first.[/red]")
        raise typer.Exit(code=1)

    cand = _candidate(report, champion, no_mlflow)
    champion_pq = champion.get("per_question") or []

    # Statistical gating needs the champion's per-question scores; older champions
    # predate that. Fall back to the naive gate (loudly) and tell the user to re-promote.
    if not champion_pq:
        console.print(
            "[yellow]⚠ Champion has no per-question data — cannot run the statistical "
            "gate. Re-run `mlip eval promote` to enable it. Falling back to naive.[/yellow]"
        )
        naive = evaluate_gate(cand["scorecard"], champion, tolerance=tolerance)
        for c in naive.checks:
            status = "[green]PASS[/green]" if c.passed else "[red]FAIL[/red]"
            console.print(f"  {c.metric}: cand {c.candidate:.3f} vs floor {c.floor:.3f} → {status}")
        if not naive.passed:
            console.print("[red bold]❌ Quality gate FAILED (naive)[/red bold]")
            raise typer.Exit(code=1)
        console.print("[green bold]✅ Quality gate PASSED (naive)[/green bold]")
        return

    result = evaluate_gate_statistical(cand["per_question"], champion_pq)
    _render_stat(result)

    if result.insufficient_overlap:
        console.print(
            f"[red bold]❌ Gate FAILED — only {result.matched} paired questions "
            f"(need ≥ {result.min_paired}); won't test against a different question set.[/red bold]"
        )
        raise typer.Exit(code=1)
    if result.content_mismatches > 0:
        console.print(
            f"[red bold]❌ Gate FAILED — {result.content_mismatches} id(s) map to different "
            "questions in champion vs candidate. Re-promote the champion.[/red bold]"
        )
        raise typer.Exit(code=1)
    if not result.passed:
        regressed = ", ".join(f"{t.metric}/{t.category}" for t in result.gated_failures)
        console.print(
            f"[red bold]❌ Quality gate FAILED — significant regression: {regressed}[/red bold]"
        )
        raise typer.Exit(code=1)
    console.print("[green bold]✅ Quality gate PASSED (no significant regression)[/green bold]")


@eval_app.command(name="compare-rag")
def compare_rag(
    prompt_version: str = typer.Option(
        "v1", help=f"RAG prompt version: {', '.join(PROMPT_VARIANTS)}"
    ),
    top_k: int = typer.Option(3, help="Documents retrieved per question (RAG side)."),
    no_mlflow: bool = typer.Option(False, help="Skip MLflow logging."),
) -> None:
    """Evaluate the SAME model with RAG vs without RAG, to show what retrieval buys."""
    rag_cfg = RagConfig(name="with-rag", prompt_version=prompt_version, top_k=top_k)
    norag_cfg = RagConfig(name="no-rag", use_retrieval=False)
    console.print("[bold]Comparing[/bold] with-RAG vs no-RAG (same model)")
    result = run_ab(
        rag_cfg, norag_cfg, primary_metric="answer_correctness", log_to_mlflow=not no_mlflow
    )

    table = Table(title="RAG vs no-RAG")
    for col in ("Metric", "with-RAG", "no-RAG", "Δ (RAG−none)", "Winner"):
        table.add_column(col, style="cyan" if col == "Metric" else "white")
    for c in result.comparisons:
        # In this command A = with-RAG, B = no-RAG, so a positive delta means RAG helped.
        winner = {
            "A": "[green]RAG[/green]",
            "B": "[yellow]no-RAG[/yellow]",
            "tie": "[dim]tie[/dim]",
        }[c.winner]
        table.add_row(c.metric, f"{c.a:.3f}", f"{c.b:.3f}", f"{c.a - c.b:+.3f}", winner)
    console.print(table)
    console.print(
        "[dim]Note: faithfulness can't be measured without retrieval, so it's RAG-only.[/dim]"
    )
