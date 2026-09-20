"""A declared accuracy target, checked against measured benchmark results.

Scripture carries no author labels, so accuracy cannot be measured on it.  It can be measured on the
reference benchmark, where the author of every work is known and whole works are held out.  This module
turns that measurement into a pass/fail decision against targets that are written down in one place
(``benchmarks/accuracy_targets.json``) rather than implied by whichever number a report happens to show.

Two rules keep the gate from being satisfied by a useless system:

* Accuracy alone is not enough.  A rule that accepts every candidate scores well on accuracy among the
  authors it knows while being worthless on a text whose author is absent, so ``unknown_false_acceptance``
  is gated as well and a missing metric fails rather than being skipped.
* A target is met only when the whole 95% interval clears it.  An interval straddling the target is
  reported as inconclusive, which does not pass.
* A benchmark run whose settings were overridden is exploratory by construction.  Such a run can be
  read for evidence but never reported as meeting the target, or the target could always be met by
  changing the protocol until it was.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TARGETS_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "accuracy_targets.json"

# Metrics the gate understands, and whether a larger value is better.
DIRECTION = {
    "balanced_accuracy": "minimum",
    "known_acceptance": "minimum",
    "verification_fpr": "maximum",
    "verification_fnr": "maximum",
    "unknown_false_acceptance": "maximum",
}


@dataclass
class MetricOutcome:
    metric: str
    target: float
    direction: str
    value: float | None
    interval: tuple[float, float] | None
    status: str
    note: str = ""

    @property
    def gap(self) -> float | None:
        """How far the point estimate is from the target, signed so negative always means short."""
        if self.value is None:
            return None
        return self.value - self.target if self.direction == "minimum" else self.target - self.value


@dataclass
class GateOutcome:
    passed: bool
    targets_path: str
    report_path: str
    outcomes: list[MetricOutcome] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    scope: str = ""
    exploratory: bool = False

    def worst(self) -> list[MetricOutcome]:
        """Failing and inconclusive metrics, furthest from target first."""
        bad = [o for o in self.outcomes if o.status != "passed"]
        return sorted(bad, key=lambda o: (o.gap if o.gap is not None else -1e9))


def load_targets(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_TARGETS_PATH
    if not path.exists():
        raise SystemExit(f"{path} not found; declare the targets before gating on them")
    targets = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(targets.get("metrics", {})) - set(DIRECTION)
    if unknown:
        raise SystemExit(f"{path}: unknown metric(s) {sorted(unknown)}")
    if not targets.get("metrics"):
        raise SystemExit(f"{path}: no metrics declared")
    return targets


def evaluate_metric(metric: str, target: float, estimate: dict | None) -> MetricOutcome:
    """Decide one metric.  A metric that was never measured fails; it is not silently skipped."""
    direction = DIRECTION[metric]
    if not estimate or estimate.get("value") is None:
        return MetricOutcome(metric, target, direction, None, None, "failed", "not measured")
    value = float(estimate["value"])
    ci = estimate.get("ci95")
    if ci is None:
        return MetricOutcome(metric, target, direction, value, None, "inconclusive", "no interval")
    lower, upper = float(ci[0]), float(ci[1])
    if direction == "minimum":
        status = "passed" if lower >= target else "failed" if upper < target else "inconclusive"
    else:
        status = "passed" if upper <= target else "failed" if lower > target else "inconclusive"
    return MetricOutcome(metric, target, direction, value, (lower, upper), status)


def _best_run(report: dict, metric: str, selector: dict) -> dict | None:
    """The run the targets point at: a named language and sample length, else the strongest accuracy."""
    runs = [r for r in report.get("runs", []) if isinstance(r, dict)]
    language, length, ablation = selector.get("language"), selector.get("sample_length"), selector.get("ablation")
    if language:
        runs = [r for r in runs if r.get("language") == language]
    if length:
        runs = [r for r in runs if r.get("sample_length") == length]
    if ablation:
        runs = [r for r in runs if r.get("ablation") == ablation]
    if not runs:
        return None
    def accuracy(run: dict) -> float:
        est = run.get("metrics", {}).get("balanced_accuracy") or {}
        return est.get("value") or 0.0
    return max(runs, key=accuracy)


def evaluate(report: dict, targets: dict, targets_path: str = "", report_path: str = "") -> GateOutcome:
    selector = targets.get("select", {})
    run = _best_run(report, "balanced_accuracy", selector)
    outcomes: list[MetricOutcome] = []
    missing: list[str] = []
    metrics = (run or {}).get("metrics", {})
    for metric, target in targets["metrics"].items():
        estimate = metrics.get(metric)
        if estimate is None:
            missing.append(metric)
        outcomes.append(evaluate_metric(metric, float(target), estimate))
    exploratory = bool(report.get("exploratory_override"))
    passed = bool(run) and not exploratory and all(o.status == "passed" for o in outcomes)
    return GateOutcome(passed, str(targets_path), str(report_path), outcomes, missing,
                       scope=_describe(run, selector), exploratory=exploratory)


def _describe(run: dict | None, selector: dict) -> str:
    if not run:
        wanted = ", ".join(f"{k}={v}" for k, v in selector.items()) or "any run"
        return f"no benchmark run matched {wanted}"
    return (f"{run.get('language')} / {run.get('sample_length')}-token passages"
            f" / {run.get('ablation', 'unspecified features')}"
            f" ({run.get('authors', '?')} authors, {run.get('works', '?')} works)")


def render(outcome: GateOutcome) -> str:
    """A short, plain report: the target, what was measured, and how far short it falls."""
    head = "PASSED" if outcome.passed else "FAILED"
    lines = [f"accuracy gate: {head}", f"  measured on: {outcome.scope}"]
    if outcome.exploratory:
        lines.append("  this run overrode the protocol settings, so it is exploratory and cannot pass"
                     " however good the numbers look")
    lines.append("")
    lines.append(f"  {'metric':28s} {'target':>9s} {'measured':>10s} {'95% interval':>20s}  status")
    for o in outcome.outcomes:
        target = f"{'>=' if o.direction == 'minimum' else '<='}{o.target:.0%}"
        value = "not measured" if o.value is None else f"{o.value:.1%}"
        interval = "—" if o.interval is None else f"{o.interval[0]:.1%} to {o.interval[1]:.1%}"
        lines.append(f"  {o.metric:28s} {target:>9s} {value:>10s} {interval:>20s}  {o.status}"
                     + (f" ({o.note})" if o.note else ""))
    shortfalls = [o for o in outcome.worst() if o.gap is not None]
    if shortfalls:
        lines.append("")
        lines.append("  furthest from target:")
        for o in shortfalls[:3]:
            lines.append(f"    {o.metric}: {abs(o.gap):.1%} short of {o.target:.0%}")
    if outcome.missing:
        lines.append("")
        lines.append(f"  not present in the benchmark report: {', '.join(outcome.missing)}")
    return "\n".join(lines)


def as_dict(outcome: GateOutcome) -> dict:
    return {
        "passed": outcome.passed,
        "exploratory": outcome.exploratory,
        "scope": outcome.scope,
        "targets": outcome.targets_path,
        "report": outcome.report_path,
        "missing_metrics": outcome.missing,
        "metrics": [
            {"metric": o.metric, "target": o.target, "direction": o.direction, "measured": o.value,
             "ci95": list(o.interval) if o.interval else None, "status": o.status,
             "gap": o.gap, "note": o.note}
            for o in outcome.outcomes
        ],
    }
