"""Render the gate verdict as a Markdown PR comment.

The GitHub Actions workflow posts this as a sticky (edit-last) comment, so every
push updates one comment instead of spamming new ones.
"""

from __future__ import annotations

from mlip.eval.calibration import CalibrationResult
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


def render_stat_markdown(
    result: StatGateResult,
    *,
    blocked_reason: str | None = None,
    calibration: CalibrationResult | None = None,
) -> str:
    drifted = calibration is not None and not calibration.passed
    header = "✅ PASS" if (result.passed and not blocked_reason and not drifted) else "❌ FAIL"
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
    if calibration is not None:
        status = "✅ ok" if calibration.passed else "❌ DRIFTED — judge no longer trustworthy"
        lines += [
            "",
            f"**Judge calibration:** Cohen's κ = {calibration.kappa:.3f} "
            f"(threshold {calibration.threshold}) on {calibration.n} gold items — {status}",
        ]
    if blocked_reason:
        lines += ["", f"> ❌ **Blocked:** {blocked_reason}"]
    if result.dropped_candidate or result.dropped_champion:
        lines += [
            "",
            f"> ⚠️ Question set differs — testing on the {result.matched} shared questions.",
        ]
    lines += ["", *_stat_table(result), ""]
    footnote = (
        f"<sub>Verdict uses the multiplicity-adjusted p-value (α={result.alpha}); "
        "the CI shown is per-comparison. The judge metric is informational."
    )
    has_categories = any(t.category != "overall" for t in result.tests)
    if has_categories:
        footnote += (
            " Aggregate + per-category tests share one correction family. Per-category n "
            "is small, so only sizeable subgroup regressions are detectable."
        )
    lines.append(footnote + "</sub>")
    return "\n".join(lines)


def render_gate_markdown(
    *,
    stat: StatGateResult,
    naive: GateResult,
    calibration: CalibrationResult | None = None,
    blocked_reason: str | None = None,
    naive_binding: bool = False,
) -> str:
    """Render BOTH the naive and statistical verdicts side by side on every PR."""
    drifted = calibration is not None and not calibration.passed
    stat_ok = stat.passed and not blocked_reason and not drifted
    binding_ok = naive.passed if naive_binding else stat_ok
    binding_name = "naive (legacy mean-threshold)" if naive_binding else "statistical (adjusted)"
    header = "✅ PASS" if binding_ok else "❌ FAIL"

    lines = [
        f"## 🧪 Eval Quality Gate — {header}",
        "",
        f"**Binding decision:** {binding_name} · the other verdict is shown for comparison.",
        "",
        f"### Statistical (paired CI + McNemar + {stat.correction_method}) {'✅' if stat_ok else '❌'}",
        (
            f"Paired on `id`: {stat.matched} matched · {stat.dropped_candidate} candidate-only · "
            f"{stat.dropped_champion} champion-only · {stat.content_mismatches} mismatch."
        ),
        "",
        *_stat_table(stat),
        "",
    ]
    if calibration is not None:
        status = "✅ ok" if calibration.passed else "❌ DRIFTED — judge untrustworthy"
        lines += [
            f"**Judge calibration:** Cohen's κ = {calibration.kappa:.3f} "
            f"(threshold {calibration.threshold}) on {calibration.n} items — {status}",
            "",
        ]
    if blocked_reason:
        lines += [f"> ❌ **Blocked:** {blocked_reason}", ""]

    lines += [
        f"### Naive (legacy mean-threshold, tol {naive.checks[0].tolerance if naive.checks else '—'}) "
        f"{'✅' if naive.passed else '❌'}",
        "| Metric | Champion floor | Candidate | Verdict |",
        "|---|--:|--:|:--|",
    ]
    for c in naive.checks:
        lines.append(
            f"| `{c.metric}` | {c.floor:.3f} | {c.candidate:.3f} | "
            f"{'✅ pass' if c.passed else '❌ fail'} |"
        )
    lines += [
        "",
        "<sub>Statistical verdict uses the multiplicity-adjusted p-value (CI shown is "
        "per-comparison); judge metric is informational. The naive verdict is the old "
        "mean-threshold check, shown to make the difference visible.</sub>",
    ]
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
