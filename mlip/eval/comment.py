"""Render the gate verdict as a Markdown PR comment.

The GitHub Actions workflow posts this as a sticky (edit-last) comment, so every
push updates one comment instead of spamming new ones.
"""

from __future__ import annotations

from mlip.eval.gate import GateResult, StatGateResult


def _stat_table(result: StatGateResult) -> list[str]:
    rows = [
        "| Metric | Category | Champion | Candidate | Δ | 95% CI (per-comp) | Verdict |",
        "|---|---|--:|--:|--:|:--:|:--|",
    ]
    for t in result.tests:
        ci = "—" if t.kind == "mcnemar" else f"[{t.ci_lo:+.3f}, {t.ci_hi:+.3f}]"
        if not t.gated:
            verdict = "ℹ️ informational"
        elif t.regression:
            verdict = "❌ significant regression"
        else:
            verdict = "✅ within noise"
        rows.append(
            f"| `{t.metric}` | {t.category} | {t.champion_mean:.3f} | {t.candidate_mean:.3f} "
            f"| {t.delta:+.3f} | {ci} | {verdict} |"
        )
    return rows


def render_stat_markdown(result: StatGateResult, *, blocked_reason: str | None = None) -> str:
    header = "✅ PASS" if (result.passed and not blocked_reason) else "❌ FAIL"
    lines = [
        f"## 🧪 Eval Quality Gate — {header}",
        "",
        (
            f"Paired on `id`: **{result.matched} matched** · "
            f"{result.dropped_candidate} candidate-only · {result.dropped_champion} champion-only · "
            f"{result.content_mismatches} content-mismatch. "
            f"Correction: **{result.correction_method}** @ α={result.alpha}."
        ),
    ]
    if blocked_reason:
        lines += ["", f"> ❌ **Blocked:** {blocked_reason}"]
    if result.dropped_candidate or result.dropped_champion:
        lines += [
            "",
            f"> ⚠️ Question set differs — testing on the {result.matched} shared questions.",
        ]
    lines += ["", *_stat_table(result), ""]
    lines.append(
        f"<sub>Verdict uses the multiplicity-adjusted p-value (α={result.alpha}); "
        "the CI shown is per-comparison. The judge metric is informational.</sub>"
    )
    return "\n".join(lines)


def render_naive_markdown(naive: GateResult, *, tolerance: float) -> str:
    header = "✅ PASS" if naive.passed else "❌ FAIL"
    lines = [
        f"## 🧪 Eval Quality Gate (naive) — {header}",
        "",
        "> ⚠️ Champion has no per-question data, so the statistical gate could not run. "
        "Re-run `mlip eval promote` to enable it.",
        "",
        f"| Metric | Champion floor (−{tolerance}) | Candidate | Verdict |",
        "|---|--:|--:|:--|",
    ]
    for c in naive.checks:
        verdict = "✅ pass" if c.passed else "❌ fail"
        lines.append(f"| `{c.metric}` | {c.floor:.3f} | {c.candidate:.3f} | {verdict} |")
    return "\n".join(lines)
