"""PALM label-efficiency curve fitting.

Accuracy at a handful of budgets is a noisy way to rank samplers: a method can
win at one budget and lose at the next. PALM fits the whole learning curve and
reports interpretable parameters instead — ceiling, cold-start offset, and how
fast the curve climbs — so two samplers are compared as curves, not points.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import OptimizeWarning, curve_fit

__all__ = ["palm_evaluate", "format_palm_report"]


def _palm_model(
    B: np.ndarray,
    Amax: float,
    delta: float,
    alpha: float,
    beta: float,
    *,
    b: float,
) -> np.ndarray:
    base     = np.maximum(B / b + alpha, 1e-9)
    exponent = np.minimum(base ** beta, 700.0)
    return Amax * (1.0 - (1.0 - delta) ** exponent)


def _compute_auc(
    Amax: float, delta: float, alpha: float, beta: float,
    b: float, B_min: float, B_max: float, n_grid: int = 1000,
) -> float:
    B_grid = np.linspace(B_min, B_max, n_grid)
    A_grid = _palm_model(B_grid, Amax, delta, alpha, beta, b=b)
    return float(trapezoid(A_grid, B_grid) / (B_max - B_min))


def _budget_to_target(
    Amax: float, delta: float, alpha: float, beta: float,
    b: float, B_max: float, target_fraction: float = 0.90,
) -> Optional[float]:
    if delta >= 1.0 - 1e-9:
        return None
    log_1m_delta    = math.log(1.0 - delta)
    log_remainder   = math.log(1.0 - target_fraction)
    required_exp    = log_remainder / log_1m_delta
    if required_exp <= 0.0:
        return None
    try:
        base = required_exp ** (1.0 / beta)
    except (ValueError, ZeroDivisionError):
        return None
    B_star = b * (base - alpha)
    if B_star > B_max or not math.isfinite(B_star):
        return None
    return float(B_star)


def palm_evaluate(
    budgets: List[int],
    accuracies: List[float],
    b: Optional[float] = None,
    target_fraction: float = 0.90,
) -> Dict[str, Any]:
    budgets_arr = np.array(budgets, dtype=float)
    accs_arr    = np.array(accuracies, dtype=float)
    n_points    = len(budgets_arr)

    if n_points < 4:
        raise ValueError(
            f"PALM requires ≥ 4 budget points, got {n_points}. "
            "Collect more budget evaluations before fitting."
        )

    if b is None:
        b = float(np.mean(np.diff(budgets_arr)))

    B_min, B_max = float(budgets_arr[0]), float(budgets_arr[-1])
    p0           = [float(np.max(accs_arr)), 0.1, 1.0, 1.0]
    lower_bounds = [0.0,  0.0, -100.0,  0.0]
    upper_bounds = [1.0,  1.0,  100.0, 10.0]

    def _model_for_fit(B, Amax, delta, alpha, beta):
        return _palm_model(B, Amax, delta, alpha, beta, b=b)

    fit_success = True
    popt: Tuple[float, ...] = (float("nan"),) * 4

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(
                _model_for_fit, budgets_arr, accs_arr,
                p0=p0, bounds=(lower_bounds, upper_bounds),
                method="trf", maxfev=10000,
            )
    except (RuntimeError, OptimizeWarning, ValueError):
        fit_success = False

    Amax_fit, delta_fit, alpha_fit, beta_fit = popt

    auc_normalized  = float("nan")
    budget_to_tgt   = None
    fit_rmse        = float("nan")

    if fit_success:
        auc_normalized = _compute_auc(Amax_fit, delta_fit, alpha_fit, beta_fit, b, B_min, B_max)
        budget_to_tgt  = _budget_to_target(Amax_fit, delta_fit, alpha_fit, beta_fit, b, B_max, target_fraction)
        fitted_vals    = _model_for_fit(budgets_arr, Amax_fit, delta_fit, alpha_fit, beta_fit)
        fit_rmse       = float(np.sqrt(np.mean((fitted_vals - accs_arr) ** 2)))

    target_pct = int(round(target_fraction * 100))
    return {
        "Amax":  float(Amax_fit),
        "delta": float(delta_fit),
        "alpha": float(alpha_fit),
        "beta":  float(beta_fit),
        "b":     float(b),
        "auc_normalized": auc_normalized,
        f"budget_to_{target_pct}pct_amax": budget_to_tgt,
        "fit_rmse":    fit_rmse,
        "fit_success": fit_success,
        "n_points":    n_points,
    }


def format_palm_report(params: Dict[str, Any], sampler_name: str, dataset: str) -> str:
    """Return a human-readable PALM report string."""
    sep   = "=" * 56
    lines = [sep, f"PALM Report — {sampler_name.upper()} on {dataset.upper()}", sep]

    if not params.get("fit_success", False):
        lines += ["  [WARNING] Curve fitting did not converge.", "  PALM parameters are unreliable (NaN).", sep]
        return "\n".join(lines)

    def _fmt(v: Any, fmt: str = ".4f") -> str:
        if v is None:
            return "None (unreachable within observed range)"
        if isinstance(v, float) and math.isnan(v):
            return "NaN"
        return format(v, fmt)

    lines += [
        f"  n_points   : {params['n_points']}",
        f"  b (fixed)  : {_fmt(params['b'], '.1f')}",
        "",
        "  Fitted parameters:",
        f"    Amax  = {_fmt(params['Amax'])}   (asymptotic accuracy ceiling)",
        f"    delta = {_fmt(params['delta'])}   (coverage efficiency — primary metric)",
        f"    alpha = {_fmt(params['alpha'])}   (early-stage shift; lower = better cold-start)",
        f"    beta  = {_fmt(params['beta'])}   (scalability exponent; higher = steeper growth)",
        "",
        "  Derived metrics:",
        f"    AUC (normalized)     : {_fmt(params['auc_normalized'])}",
    ]

    target_key = next((k for k in params if k.startswith("budget_to_") and k.endswith("pct_amax")), None)
    if target_key:
        pct_label = target_key.replace("budget_to_", "").replace("_amax", "")
        lines.append(f"    Budget to {pct_label} Amax : {_fmt(params[target_key], '.1f')}")

    lines += [f"    Fit RMSE             : {_fmt(params['fit_rmse'])}", sep]
