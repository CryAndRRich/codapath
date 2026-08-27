"""Detect a selection that ran to completion but selected nothing meaningful.

Every failure below returns the right NUMBER of distinct in-range indices, so
neither a smoke test nor a passing run can reveal them. They have to be checked
against the mechanism.

The one that has actually happened in this project: a feature matrix whose rows
are not L2-normalized makes `1 - cos` negative, the kernel clamps to 1, every
marginal gain becomes 0, and `argmax` over a constant vector returns index
order. The run looks completely normal — a full budget of unique indices, a
plausible accuracy — and the selection is really "the first B rows of the pool".

Nothing here raises. A warning that stops a 6-hour Kaggle run at budget 200
loses more than it saves, so findings are printed loudly and stored alongside
the selection for later reading.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["check_selection", "format_report", "SEVERITY_ORDER"]

SEVERITY_ORDER = ("ok", "note", "warning", "alarm")

# An index-ordered prefix is the signature of argmax over a constant score.
# Kendall tau is 1.0 for a perfectly increasing sequence; sampled coverage
# methods legitimately reach ~0.3-0.5, so the bar is set well above that.
_MONOTONIC_TAU = 0.9
# Below this, a score vector cannot distinguish candidates in any meaningful way.
_DEGENERATE_SPREAD = 1e-9


def _kendall_tau_against_sorted(indices: Sequence[int]) -> Optional[float]:
    """Concordance of the pick order with increasing index order.

    Uses the O(n^2) pair count directly: n is the budget (<= a few hundred), so
    this costs nothing and avoids depending on scipy.
    """
    values = np.asarray(indices, dtype=np.int64)
    count = len(values)
    if count < 3:
        return None
    concordant = discordant = 0
    for i in range(count - 1):
        difference = values[i + 1:] - values[i]
        concordant += int(np.count_nonzero(difference > 0))
        discordant += int(np.count_nonzero(difference < 0))
    total = concordant + discordant
    if total == 0:
        return None
    return (concordant - discordant) / total


def check_selection(
    selected_indices: Sequence[int],
    pool_size: int,
    labels: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return `{"severity": ..., "findings": [...], "stats": {...}}`."""
    findings: List[Dict[str, str]] = []
    indices = np.asarray(list(selected_indices), dtype=np.int64)
    stats: Dict[str, Any] = {
        "num_selected": int(indices.size),
        "num_unique": int(np.unique(indices).size) if indices.size else 0,
        "pool_size": int(pool_size),
    }

    def add(severity: str, code: str, message: str) -> None:
        findings.append({"severity": severity, "code": code, "message": message})

    if indices.size == 0:
        add("alarm", "empty", "The sampler returned no indices at all.")
        return {"severity": "alarm", "findings": findings, "stats": stats}

    if stats["num_unique"] != stats["num_selected"]:
        add(
            "alarm", "duplicates",
            f"{stats['num_selected'] - stats['num_unique']} duplicate indices; "
            "a selected point was offered to the greedy again.",
        )
    if int(indices.min()) < 0 or int(indices.max()) >= pool_size:
        add(
            "alarm", "out_of_range",
            f"Indices span [{int(indices.min())}, {int(indices.max())}] outside "
            f"the pool of {pool_size}.",
        )

    in_range = indices[(indices >= 0) & (indices < pool_size)]

    tau = _kendall_tau_against_sorted(indices)
    stats["order_tau"] = tau
    if tau is not None and tau >= _MONOTONIC_TAU:
        add(
            "alarm", "index_ordered",
            f"Picks are {tau:.3f}-concordant with plain increasing index order. "
            "This is what argmax returns when every score is equal, so the "
            "acquisition score was probably constant (check row normalization "
            "and sigma) rather than genuinely preferring these points.",
        )

    # Index only with the in-range subset: an out-of-range index is already
    # reported above, and indexing with it here would raise, turning a
    # diagnostic into the crash it exists to describe.
    if labels is not None and in_range.size:
        selected_labels = np.asarray(labels)[in_range]
        present = int(np.unique(selected_labels).size)
        stats["classes_present"] = present
        counts = np.bincount(
            selected_labels.astype(np.int64),
            minlength=num_classes or (int(selected_labels.max()) + 1),
        ).astype(np.float64)
        observed = counts[counts > 0] / counts.sum()
        stats["label_entropy"] = float(-(observed * np.log(observed)).sum())
        if num_classes:
            stats["label_entropy_max"] = float(np.log(num_classes))
            if present == 1 and in_range.size >= 2 * num_classes:
                add(
                    "alarm", "single_class",
                    f"All {in_range.size} points share one class out of "
                    f"{num_classes}; a probe cannot be trained from this.",
                )
            elif present < num_classes and in_range.size >= 2 * num_classes:
                add(
                    "note", "classes_missing",
                    f"{num_classes - present} of {num_classes} classes are "
                    f"absent from {in_range.size} selected points.",
                )

    if trace:
        _check_trace(trace, add, stats)

    severity = "ok"
    for finding in findings:
        if SEVERITY_ORDER.index(finding["severity"]) > SEVERITY_ORDER.index(severity):
            severity = finding["severity"]
    return {"severity": severity, "findings": findings, "stats": stats}


def _check_trace(trace: Dict[str, Any], add, stats: Dict[str, Any]) -> None:
    """Check the per-step scores and per-round state a traced sampler recorded."""
    steps = [step for step in (trace.get("steps") or []) if step.get("score") is not None]
    scores = np.asarray([step["score"] for step in steps], dtype=np.float64)
    if scores.size >= 2:
        spread = float(scores.max() - scores.min())
        stats["step_score_spread"] = spread
        if spread <= _DEGENERATE_SPREAD:
            add(
                "alarm", "constant_step_score",
                f"Every pick scored the same value ({scores[0]:.3e}); the "
                "objective could not tell candidates apart.",
            )

    # Diminishing returns hold WITHIN a round only. Between rounds the
    # bandwidth is re-adapted and the running-max coverage is rebuilt at the
    # new sigma, so the gain legitimately jumps; comparing across that boundary
    # would report a violation on a correct run.
    rises = total = 0
    for round_index in {step["round_index"] for step in steps}:
        within = np.asarray(
            [step["score"] for step in steps if step["round_index"] == round_index],
            dtype=np.float64,
        )
        if within.size >= 2:
            differences = np.diff(within)
            rises += int(np.count_nonzero(differences > 1e-9))
            total += int(differences.size)
    if total and rises:
        stats["nonmonotone_steps"] = rises
        add(
            "warning", "nonmonotone_gain",
            f"The marginal gain rose at {rises} of {total} within-round steps. "
            "A submodular objective's gains must not increase as coverage grows, "
            "so the running-max coverage may not be updating after a pick.",
        )

    for record in trace.get("rounds") or []:
        label = f"round {record.get('round_index')}"
        sigma = record.get("sigma")
        if sigma is not None and not np.isfinite(sigma):
            add("alarm", "sigma_nonfinite", f"{label}: sigma is {sigma}.")
        elif sigma is not None and sigma <= 0.0:
            add(
                "alarm", "sigma_zero",
                f"{label}: sigma={sigma:.3e}. The kernel degenerates into an "
                "indicator of exact duplicates.",
            )
        for name in ("weight_summary", "score_summary"):
            summary = record.get(name)
            if not summary or summary.get("count", 0) < 2:
                continue
            kind = name.split("_")[0]
            # A round can declare that its weights are uniform on purpose:
            # round 1 of a coverage method has no labels yet, so U=1 reduces the
            # objective to plain MaxHerding. That is the documented design, not
            # a degenerate weight vector.
            intentional = bool(record.get(f"{kind}_uniform_by_design"))
            if summary.get("spread", 1.0) <= _DEGENERATE_SPREAD and not intentional:
                add(
                    "alarm", f"constant_{kind}",
                    f"{label}: the {kind} vector is constant "
                    f"({summary.get('mean', float('nan')):.3e}) over "
                    f"{int(summary['count'])} pool points, so it contributed "
                    "no preference. Note minmax of a constant vector is all "
                    "zeros, which then zeroes the whole score.",
                )
            if summary.get("nonfinite", 0.0) > 0:
                add(
                    "alarm", f"nonfinite_{kind}",
                    f"{label}: {int(summary['nonfinite'])} non-finite values.",
                )


def format_report(result: Dict[str, Any], run_name: str, budget: int) -> str:
    """One block per check, loud enough to notice in a long Kaggle log."""
    severity = result["severity"]
    stats = result["stats"]
    lines: List[str] = []
    marker = {"ok": "OK", "note": "NOTE", "warning": "WARNING", "alarm": "ALARM"}[severity]
    lines.append(f"[sanity {marker}] {run_name} budget={budget}")
    detail = [
        f"selected={stats['num_selected']}/{stats['pool_size']}",
        f"unique={stats['num_unique']}",
    ]
    if stats.get("order_tau") is not None:
        detail.append(f"order_tau={stats['order_tau']:+.3f}")
    if stats.get("classes_present") is not None:
        detail.append(f"classes={stats['classes_present']}")
    if stats.get("label_entropy") is not None:
        maximum = stats.get("label_entropy_max")
        detail.append(
            f"label_entropy={stats['label_entropy']:.3f}"
            + (f"/{maximum:.3f}" if maximum else "")
        )
    if stats.get("step_score_spread") is not None:
        detail.append(f"score_spread={stats['step_score_spread']:.3e}")
    lines.append("  " + "  ".join(detail))
    for finding in result["findings"]:
        lines.append(f"  [{finding['severity'].upper()}] {finding['code']}: {finding['message']}")
    return "\n".join(lines)
