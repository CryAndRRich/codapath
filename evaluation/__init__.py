"""Test-set metrics, PALM/ALDA curve analysis, and plots."""

from .alda import alda_advise, format_alda_report
from .metrics import evaluate_probe
from .palm import format_palm_report, palm_evaluate
from .results_io import (
    average_seeds,
    discover_runs,
    format_metric_table,
    load_curves,
    load_run_result,
)

__all__ = [
    "evaluate_probe",
    "format_palm_report",
    "palm_evaluate",
    "alda_advise",
    "format_alda_report",
    "average_seeds",
    "discover_runs",
    "format_metric_table",
    "load_curves",
    "load_run_result",
]
