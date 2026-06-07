"""
palm_eval.py — PALM (Parametric Active Learning Metric) evaluation module.

Reference: PALM paper (Definition 7).

PALM fits a parametric learning curve to observed (budget, accuracy) data:

    A(B) = Amax * (1 - (1 - delta)^((B/b + alpha)^beta))

Parameters:
    Amax  ∈ (0, 1]    — asymptotic accuracy ceiling
    delta ∈ (0, 1]    — coverage efficiency per labeled sample (primary ranking metric)
    alpha ∈ (-inf, inf) — early-stage shift (lower = better cold-start)
    beta  > 0          — scalability / steepness of growth curve
    b     > 0          — mean budget step size (fixed, not fitted)

Note on deviation from the official PALM repo:
    The official repo fits b as a 5th parameter. With only 6 budget points
    (e.g. [50,100,150,200,250,300]) fitting 5 parameters leaves 1 degree of
    freedom — effectively underdetermined. Fixing b = mean(np.diff(budgets))
    reduces to 4 parameters, 2 dof, which is more stable at this scale.

Bounds used here (matching PALM paper, NOT the wider defaults in the design spec):
    lower = [0.0,  0.0,  -100.0, 0.0]
    upper = [1.0,  1.0,   100.0, 5.0]

Overflow guard:
    base^beta is clipped to max 700.0 before use as a power-of-(1-delta) exponent
    to prevent float64 overflow when alpha or beta are large.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import OptimizeWarning, curve_fit

__all__ = ["palm_evaluate", "format_palm_report"]


# ---------------------------------------------------------------------------
# Internal model
# ---------------------------------------------------------------------------

def _palm_model(B: np.ndarray,
                Amax: float,
                delta: float,
                alpha: float,
                beta: float,
                *,
                b: float) -> np.ndarray:
    """PALM parametric learning curve model.

    A(B) = Amax * (1 - (1 - delta)^((B/b + alpha)^beta))

    Safety guards:
      - base clipped to [1e-9, inf) to avoid log(0) when B/b + alpha -> 0
      - exponent clipped to 700.0 to prevent float64 overflow
    """
    base = np.maximum(B / b + alpha, 1e-9)
    exponent = np.minimum(base ** beta, 700.0)
    return Amax * (1.0 - (1.0 - delta) ** exponent)


# ---------------------------------------------------------------------------
# AUC (normalized)
# ---------------------------------------------------------------------------

def _compute_auc(Amax: float,
                 delta: float,
                 alpha: float,
                 beta: float,
                 b: float,
                 B_min: float,
                 B_max: float,
                 n_grid: int = 1000) -> float:
    """Normalized AUC of the fitted learning curve over [B_min, B_max].

    Approximated with scipy.integrate.trapezoid on an n_grid-point grid.
    Normalized by dividing by (B_max - B_min) so the result is in [0, 1].
    """
    B_grid = np.linspace(B_min, B_max, n_grid)
    A_grid = _palm_model(B_grid, Amax, delta, alpha, beta, b=b)
    auc = trapezoid(A_grid, B_grid) / (B_max - B_min)
    return float(auc)


# ---------------------------------------------------------------------------
# Analytical inversion for budget_to_X_pct_amax
# ---------------------------------------------------------------------------

def _budget_to_target(Amax: float,
                      delta: float,
                      alpha: float,
                      beta: float,
                      b: float,
                      B_max: float,
                      target_fraction: float = 0.90) -> Optional[float]:
    """Return the smallest budget B* where A(B*) >= target_fraction * Amax.

    Derived by inverting the PALM formula:
        target_fraction = 1 - (1 - delta)^(base^beta)
        => (1 - delta)^(base^beta) = 1 - target_fraction
        => base^beta = log(1 - target_fraction) / log(1 - delta)
        => base = (log(1 - target_fraction) / log(1 - delta))^(1/beta)
        => B* = b * (base - alpha)

    Returns None if:
      - delta is effectively 1 (log(1-delta) -> -inf, inversion unstable)
      - B* is not real (negative base raised to fractional power)
      - B* > B_max (target unreachable within observed range)
    """
    # Guard: delta too close to 1
    if delta >= 1.0 - 1e-9:
        return None

    log_1_minus_delta = math.log(1.0 - delta)   # negative
    log_remainder = math.log(1.0 - target_fraction)  # negative (for fraction < 1)

    # required_exponent = log_remainder / log_1_minus_delta > 0
    required_exponent = log_remainder / log_1_minus_delta
    if required_exponent <= 0.0:
        return None

    # base = required_exponent^(1/beta)
    try:
        base = required_exponent ** (1.0 / beta)
    except (ValueError, ZeroDivisionError):
        return None

    B_star = b * (base - alpha)

    if B_star > B_max or not math.isfinite(B_star):
        return None

    return float(B_star)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def palm_evaluate(
    budgets: List[int],
    accuracies: List[float],
    b: Optional[float] = None,
    target_fraction: float = 0.90,
) -> Dict[str, Any]:
    """Fit the PALM learning curve and return evaluation metrics.

    Args:
        budgets:         List of cumulative labeled budgets, e.g. [50,100,150,200,250,300].
        accuracies:      Corresponding test accuracy values in [0, 1].
        b:               Budget normalization constant (mean step size). If None,
                         computed as mean(np.diff(budgets)).
        target_fraction: Fraction of Amax used for budget_to_X_pct_amax (default 0.90).

    Returns:
        Dict with keys:
            Amax, delta, alpha, beta, b         — fitted / fixed parameters
            auc_normalized                       — normalized AUC over [B_min, B_max]
            budget_to_90pct_amax                 — smallest B* with A(B*) >= target*Amax, or None
            fit_rmse                             — RMSE between fitted curve and observed data
            fit_success                          — True if curve_fit converged
            n_points                             — number of (budget, accuracy) pairs used

    Raises:
        ValueError: if fewer than 4 budget points are provided.
    """
    budgets_arr = np.array(budgets, dtype=float)
    accs_arr = np.array(accuracies, dtype=float)

    n_points = len(budgets_arr)
    if n_points < 4:
        raise ValueError(
            f"PALM requires at least 4 budget points, got {n_points}. "
            "Collect more budget evaluations before fitting."
        )

    # Fixed normalization constant
    if b is None:
        b = float(np.mean(np.diff(budgets_arr)))

    B_min = float(budgets_arr[0])
    B_max = float(budgets_arr[-1])

    # Initial guesses
    Amax0 = float(np.max(accs_arr))
    delta0 = 0.1
    alpha0 = 1.0
    beta0 = 1.0
    p0 = [Amax0, delta0, alpha0, beta0]

    # Bounds: paper-aligned (not the wider -1000/100 from the design spec)
    lower_bounds = [0.0,  0.0,  -100.0, 0.0]
    upper_bounds = [1.0,  1.0,   100.0, 5.0]

    # Wrap _palm_model to inject fixed b for curve_fit
    def _model_for_fit(B: np.ndarray,
                       Amax: float,
                       delta: float,
                       alpha: float,
                       beta: float) -> np.ndarray:
        return _palm_model(B, Amax, delta, alpha, beta, b=b)

    fit_success = True
    popt: Tuple[float, float, float, float] = (float("nan"),) * 4

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            popt, _ = curve_fit(
                _model_for_fit,
                budgets_arr,
                accs_arr,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                method="trf",
                maxfev=10000,
            )
    except (RuntimeError, OptimizeWarning, ValueError):
        fit_success = False

    Amax_fit, delta_fit, alpha_fit, beta_fit = popt

    # Derived metrics (only meaningful when fit succeeded)
    auc_normalized = float("nan")
    budget_to_target = None
    fit_rmse = float("nan")

    if fit_success:
        auc_normalized = _compute_auc(
            Amax_fit, delta_fit, alpha_fit, beta_fit, b, B_min, B_max
        )

        budget_to_target = _budget_to_target(
            Amax_fit, delta_fit, alpha_fit, beta_fit, b, B_max, target_fraction
        )

        fitted_vals = _model_for_fit(budgets_arr, Amax_fit, delta_fit, alpha_fit, beta_fit)
        fit_rmse = float(np.sqrt(np.mean((fitted_vals - accs_arr) ** 2)))

    target_pct = int(round(target_fraction * 100))
    return {
        "Amax": float(Amax_fit),
        "delta": float(delta_fit),
        "alpha": float(alpha_fit),
        "beta": float(beta_fit),
        "b": float(b),
        "auc_normalized": auc_normalized,
        f"budget_to_{target_pct}pct_amax": budget_to_target,
        "fit_rmse": fit_rmse,
        "fit_success": fit_success,
        "n_points": n_points,
    }


def format_palm_report(params: Dict[str, Any],
                       sampler_name: str,
                       dataset: str) -> str:
    """Return a human-readable PALM report string.

    Args:
        params:       Dict returned by palm_evaluate().
        sampler_name: Name of the active learning sampler.
        dataset:      Dataset name.

    Returns:
        Formatted multi-line string suitable for print().
    """
    sep = "=" * 56
    lines = [
        sep,
        f"PALM Report — {sampler_name.upper()} on {dataset.upper()}",
        sep,
    ]

    if not params.get("fit_success", False):
        lines.append("  [WARNING] Curve fitting did not converge.")
        lines.append("  PALM parameters are unreliable (NaN).")
        lines.append(sep)
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

    # Find the budget_to_X_pct key dynamically
    target_key = next((k for k in params if k.startswith("budget_to_") and k.endswith("pct_amax")), None)
    if target_key:
        pct_label = target_key.replace("budget_to_", "").replace("_amax", "")
        lines.append(f"    Budget to {pct_label} Amax : {_fmt(params[target_key], '.1f')}")

    lines += [
        f"    Fit RMSE             : {_fmt(params['fit_rmse'])}",
        sep,
    ]
    return "\n".join(lines)
