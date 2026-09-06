"""ALDA: risk-aware deployment advice on top of a fitted PALM curve.

PALM answers "how does this sampler's curve behave"; ALDA answers the question
a clinical team actually asks before committing an annotation budget:

    given a short pilot, which sampler should get the remaining labels, and
    how many expert annotations will it take to reach accuracy `target`?

Three quantities, all read off the fitted curve (paper Sec. 3.2-3.4):

* **feasibility** -- a method is feasible only if `Amax >= target`. A method
  predicted never to reach the target is screened out before any cost is
  computed, rather than being ranked by a cost it cannot achieve.
* **`B_abs(target)`** -- the smallest budget at which the curve reaches the
  target, rounded UP to a whole acquisition episode (Eq. 2). Rounding up is
  not cosmetic: labels are bought one episode at a time, so a prediction of
  "137 labels" on a 25-per-episode protocol means buying 150.
* **`W`** -- the deployment window `B_abs(target + dt) - B_abs(target - dt)`.
  The target is rarely a precise number, and a method whose curve has already
  flattened can need many extra labels for a small upward revision. `W` is
  that sensitivity, in labels.

The recommendation is deliberately NOT `argmin B_abs`. Costs within a few
percent of each other are not meaningfully different, so ALDA takes the
cost-competitive set `C_eta = {m : (B_abs - B_min)/B_min <= eta}` and inside
it picks the SMALLEST WINDOW (Eq. 4-6) -- cheapest among equals, then most
robust among the cheap.

Ported from the official implementation (`repos/PALM/deep-al/tools/alda/
advisor.py`, arXiv 2608.03511) and verified against its own unit tests. Two
deliberate differences, both about fitting this project into the project's
existing code rather than about the method:

* The official script owns its own PALM fit (L-BFGS-B, 18 random restarts).
  That fit is reproduced here as `fit_palm_restarts` instead of reusing
  `palm.py::palm_evaluate`, because ALDA's B_abs and W are inversions of the
  fitted parameters and a less stable fit moves them directly. `palm.py` keeps
  its single `curve_fit` call so previously reported PALM tables stay
  reproducible.
* Scores are normalized to percentage units internally (the official code does
  the same), so a project that reports accuracy in [0,1] and a paper that
  reports it in [0,100] get identical advice.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "PalmFit",
    "fit_palm_restarts",
    "palm_curve",
    "budget_at_target",
    "deployment_window",
    "alda_advise",
    "format_alda_report",
]

EPS = 1e-8
# Paper Sec. 3.4: never evaluate B_abs arbitrarily close to the fitted
# asymptote, where a small error in Amax produces an unbounded cost. Scores are
# in percentage units at this point, so this is 0.5 accuracy points.
WINDOW_CEILING_MARGIN_PP = 0.5
# Paper Sec. 3.1: L-BFGS-B restarts per curve fit.
N_RESTARTS = 18


@dataclass(frozen=True)
class PalmFit:
    """A fitted PALM curve, in percentage score units."""

    a_max: float
    delta: float
    alpha: float
    beta: float
    rmse: float
    budget_size: float


def palm_curve(
    budgets: np.ndarray,
    a_max: float,
    delta: float,
    alpha: float,
    beta: float,
    budget_size: float = 1.0,
) -> np.ndarray:
    """`A(B) = Amax * [1 - (1 - delta) ** ((B/b + alpha) ** beta)]`."""
    episodes = np.asarray(budgets, dtype=float) / float(budget_size)
    inner = np.clip(episodes + alpha, EPS, None) ** beta
    return a_max * (1.0 - (1.0 - np.clip(delta, EPS, 1.0 - EPS)) ** inner)


def fit_palm_restarts(
    budgets: Sequence[float],
    scores: Sequence[float],
    budget_size: float,
    rng: np.random.Generator,
    n_restarts: int = N_RESTARTS,
    maxiter: int = 2000,
) -> Optional[PalmFit]:
    """Least-squares PALM fit via L-BFGS-B with seeded random restarts.

    `rng` drives the restart initial guesses, so a fixed seed makes the fit --
    and therefore every B_abs and W derived from it -- reproducible. Returns
    None when the curve has fewer than four distinct budgets or every restart
    fails, which the caller reports rather than treating as a fit of zero.
    """
    budgets = np.asarray(budgets, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(budgets) < 4 or len(np.unique(budgets)) < 4:
        return None

    observed_max = float(np.max(scores))
    max_episode = float(np.max(budgets) / budget_size)
    a_max_hi = max(100.0, observed_max * 1.25 + 1.0)
    bounds = [
        (observed_max, a_max_hi),
        (EPS, 1.0 - EPS),
        (-max_episode, max_episode * 10.0 + 1.0),
        (0.05, 10.0),
    ]
    # Draw restarts from a moderate sub-range of the (deliberately wide)
    # bounds, so they explore plausible curve shapes rather than mostly landing
    # on numerically unstable corners.
    init_ranges = [
        (observed_max, min(a_max_hi, observed_max * 1.5 + 1.0)),
        (0.005, 0.4),
        (-1.0, max(2.0, max_episode * 0.5)),
        (0.2, 3.0),
    ]

    def objective(params: np.ndarray) -> float:
        a_max, delta, alpha, beta = params
        residual = palm_curve(budgets, a_max, delta, alpha, beta, budget_size) - scores
        return float(np.dot(residual, residual))

    best: Optional[tuple] = None
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_restarts):
            x0 = np.clip(
                [rng.uniform(lo, hi) for lo, hi in init_ranges],
                [bound[0] for bound in bounds],
                [bound[1] for bound in bounds],
            )
            try:
                result = minimize(
                    objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter}
                )
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            if not np.isfinite(result.fun):
                continue
            rmse = math.sqrt(max(result.fun, 0.0) / len(scores))
            if best is None or rmse < best[1]:
                best = (result.x, rmse)

    if best is None:
        return None
    params, rmse = best
    return PalmFit(*(float(value) for value in params), rmse=rmse, budget_size=float(budget_size))


def budget_at_target(fit: PalmFit, target: float) -> Optional[float]:
    """Labels needed to reach `target`, rounded up to a whole episode (Eq. 2).

    Returns None when the target is unreachable for this fit -- at or above the
    asymptote, or with a degenerate delta/beta -- so an infeasible method is
    never reported with a finite cost.
    """
    if target <= 0.0:
        return 0.0
    if not 0.0 < fit.delta < 1.0 or fit.beta <= 0.0:
        return None
    safe_target = min(float(target), fit.a_max - EPS)
    if safe_target <= 0.0:
        return None
    ratio = 1.0 - safe_target / fit.a_max
    if ratio <= 0.0:
        return None
    exponent = math.log(ratio) / math.log(1.0 - fit.delta)
    if exponent <= 0.0:
        return None
    raw_episodes = exponent ** (1.0 / fit.beta) - fit.alpha
    episodes = max(0, math.ceil(raw_episodes - EPS))
    return float(episodes * fit.budget_size)


def deployment_window(fit: PalmFit, target: float, delta_target: float) -> Optional[float]:
    """`W = B_abs(target + dt) - B_abs(target - dt)`, capped below the ceiling.

    The upper target is capped `WINDOW_CEILING_MARGIN_PP` below the fitted
    `Amax` (paper Sec. 3.4). That cap is a deliberate safeguard and is distinct
    from `EPS`, which only keeps the inversion off its singularity.
    """
    target_lo = max(0.0, target - delta_target)
    target_hi = min(target + delta_target, fit.a_max - WINDOW_CEILING_MARGIN_PP)
    if target_hi < target_lo:
        return None
    budget_lo = budget_at_target(fit, target_lo)
    budget_hi = budget_at_target(fit, target_hi)
    if budget_lo is None or budget_hi is None:
        return None
    return max(0.0, budget_hi - budget_lo)


def _select_recommendation(candidates: List[Dict[str, Any]], eta: float) -> Dict[str, Any]:
    """Mark `C_eta`, pick the minimum-window member, flag the risky rest.

    Two independent judgements, easy to conflate: `cost_competitive` is about
    `B_abs` relative to the cheapest method, `risky` is about `W` relative to
    the RECOMMENDATION. So a cheap method with a wide window is risky, while an
    expensive method with a narrow one is merely not cost-competitive.
    """
    b_min = min(float(row["b_abs"]) for row in candidates)
    denominator = max(b_min, EPS)
    competitive = []
    for row in candidates:
        relative_cost = (float(row["b_abs"]) - b_min) / denominator
        row["cost_competitive"] = bool(relative_cost <= eta + EPS)
        row["risky"] = False
        if row["cost_competitive"]:
            competitive.append(row)

    selected = min(
        competitive, key=lambda row: (float(row["deployment_window"]), float(row["b_abs"]))
    )
    selected_window = float(selected["deployment_window"])
    for row in candidates:
        if row is not selected and float(row["deployment_window"]) > selected_window:
            row["risky"] = True
    return selected


def alda_advise(
    curves: Dict[str, Dict[str, Sequence[float]]],
    target: float,
    delta_target: Optional[float] = None,
    eta: float = 0.05,
    max_points: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Risk-aware deployment advice across candidate samplers on one dataset.

    `curves` maps a method name to `{"budgets": [...], "accuracies": [...]}`.
    Accuracies may be fractions (0-1) or percentages (0-100); `target` and
    `delta_target` must use the SAME scale as the accuracies, and everything is
    converted to percentage units internally.

    `max_points` truncates every curve to its first N budgets, which is how the
    paper simulates a decision made from a short pilot: fit on 3-4 budgets and
    ask whether the recommendation already matches the full-curve answer.
    """
    if eta < 0.0:
        raise ValueError(f"eta must be non-negative, got {eta}")
    if not curves:
        raise ValueError("alda_advise needs at least one method curve")

    largest = max(float(np.max(curve["accuracies"])) for curve in curves.values())
    fractional = largest <= 1.0
    scale = 100.0 if fractional else 1.0
    target_pct = float(target) * scale
    if delta_target is None:
        delta_target = 0.05 if fractional else 5.0
    delta_pct = float(delta_target) * scale
    if delta_pct < 0.0:
        raise ValueError(f"delta_target must be non-negative, got {delta_target}")

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    for method in sorted(curves):
        budgets = np.asarray(curves[method]["budgets"], dtype=float)
        scores = np.asarray(curves[method]["accuracies"], dtype=float) * scale
        order = np.argsort(budgets)
        budgets, scores = budgets[order], scores[order]
        if max_points is not None:
            budgets, scores = budgets[:max_points], scores[:max_points]

        # One episode is the smallest positive gap between budgets: B_abs is
        # rounded up to a multiple of it, so it must be the real acquisition
        # step rather than the mean gap.
        differences = np.diff(np.unique(budgets))
        positive = differences[differences > 0]
        if not len(positive):
            raise ValueError(f"{method}: need at least two distinct budgets to infer the episode size")
        budget_size = float(np.min(positive))

        row: Dict[str, Any] = {
            "method": method,
            "points_used": int(len(budgets)),
            "budget_size": budget_size,
            "observed_final": float(scores[-1]) / scale,
            "fit_status": "ok",
            "a_max": None, "delta": None, "alpha": None, "beta": None, "rmse": None,
            "feasible": False, "b_abs": None, "deployment_window": None,
            "window_over_b_abs": None, "cost_competitive": False, "risky": False,
        }

        fit = fit_palm_restarts(budgets, scores, budget_size, rng)
        if fit is None:
            row["fit_status"] = "too_few_points_or_fit_failed"
            rows.append(row)
            continue

        feasible = fit.a_max >= target_pct
        row.update({
            "a_max": fit.a_max / scale,
            "delta": fit.delta,
            "alpha": fit.alpha,
            "beta": fit.beta,
            "rmse": fit.rmse / scale,
            "feasible": bool(feasible),
        })
        if feasible:
            b_abs = budget_at_target(fit, target_pct)
            window = deployment_window(fit, target_pct, delta_pct)
            if b_abs is not None and window is not None:
                row.update({
                    "b_abs": b_abs,
                    "deployment_window": window,
                    "window_over_b_abs": window / max(b_abs, EPS),
                })
        rows.append(row)

    candidates = [
        row for row in rows
        if row["feasible"] and row["b_abs"] is not None and row["deployment_window"] is not None
    ]
    if candidates:
        selected = _select_recommendation(candidates, eta)
        advice = {
            "selected_method": selected["method"],
            "selected_b_abs": selected["b_abs"],
            "selected_window": selected["deployment_window"],
            "b_min": min(float(row["b_abs"]) for row in candidates),
            "reason": "risk_aware_minimum_window_within_cost_competitive_set",
        }
    else:
        fitted = [row for row in rows if row["fit_status"] == "ok"]
        if fitted:
            # Nothing reaches the target: name the highest predicted ceiling so
            # the report says which method came closest, not that no method ran.
            selected = max(fitted, key=lambda row: float(row["a_max"]))
            advice = {
                "selected_method": selected["method"],
                "selected_b_abs": None,
                "selected_window": None,
                "b_min": None,
                "reason": "infeasible_fallback_highest_estimated_ceiling",
            }
        else:
            advice = {
                "selected_method": None,
                "selected_b_abs": None,
                "selected_window": None,
                "b_min": None,
                "reason": "no_successful_palm_fits",
            }

    advice.update({
        "target": float(target),
        "delta_target": float(delta_target),
        "eta": float(eta),
        "score_scale": "fraction" if fractional else "percent",
        "max_points": max_points,
        "seed": int(seed),
    })
    return {"advice": advice, "fits": rows}


def format_alda_report(result: Dict[str, Any], dataset: str) -> str:
    """Render `alda_advise`'s output as an aligned table plus its decision."""
    advice = result["advice"]
    scale = 100.0 if advice["score_scale"] == "fraction" else 1.0
    unit = "%" if advice["score_scale"] == "fraction" else ""

    separator = "=" * 92
    lines = [
        separator,
        f"ALDA Report - {dataset.upper()}  "
        f"(target={advice['target'] * scale:.1f}{unit} "
        f"+/-{advice['delta_target'] * scale:.1f}, eta={advice['eta']:.2f})",
        separator,
    ]
    if advice["max_points"] is not None:
        lines.append(f"  pilot: first {advice['max_points']} budgets only")

    header = (
        f"  {'method':<28}{'Amax':>8}{'B_abs':>9}{'W':>9}{'W/B_abs':>9}"
        f"{'RMSE':>8}  {'flags':<24}"
    )
    lines += [header, "  " + "-" * (len(header) - 2)]

    def _number(value: Any, fmt: str) -> str:
        """Format a number, or a placeholder padded to the SAME column width.

        A bare "-" for a missing value collapses the column and shifts every
        field after it, which is exactly what happens on the rows that matter
        most: an infeasible method has no B_abs, no W and no ratio.
        """
        if value is not None:
            return format(value, fmt)
        match = re.search(r"(\d+)(?:\.\d+)?[a-z]$", fmt)
        return "-".rjust(int(match.group(1)) if match else 1)

    for row in sorted(
        result["fits"],
        key=lambda item: (item["b_abs"] is None, item["b_abs"] if item["b_abs"] is not None else 0.0),
    ):
        flags = []
        if row["fit_status"] != "ok":
            flags.append(row["fit_status"])
        elif not row["feasible"]:
            flags.append("INFEASIBLE")
        else:
            if row["method"] == advice["selected_method"]:
                flags.append("<- SELECTED")
            if row["cost_competitive"]:
                flags.append("in C_eta")
            if row["risky"]:
                flags.append("risky")
        lines.append(
            f"  {row['method']:<28}"
            f"{_number(None if row['a_max'] is None else row['a_max'] * scale, '>8.2f')}"
            f"{_number(row['b_abs'], '>9.0f')}"
            f"{_number(row['deployment_window'], '>9.0f')}"
            f"{_number(row['window_over_b_abs'], '>9.2f')}"
            f"{_number(None if row['rmse'] is None else row['rmse'] * scale, '>8.3f')}"
            f"  {' '.join(flags):<24}"
        )

    lines.append("")
    if advice["selected_method"] is None:
        lines.append("  No method could be fitted.")
    elif advice["reason"] == "infeasible_fallback_highest_estimated_ceiling":
        lines += [
            "  No method is predicted to reach the target.",
            f"  Highest estimated ceiling: {advice['selected_method']}",
        ]
    else:
        lines += [
            f"  Recommended: {advice['selected_method']}",
            f"    labels to reach target (B_abs) : {advice['selected_b_abs']:.0f}",
            f"    deployment window (W)          : {advice['selected_window']:.0f}",
            f"    cheapest feasible cost (B_min) : {advice['b_min']:.0f}",
        ]
    lines.append(separator)
    return "\n".join(lines)
