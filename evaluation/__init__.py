"""Test-set metrics and re-scoring finished runs from their SAVED WEIGHTS.

There is deliberately no reader for `<run>_results.pt` here. Those files are a
run's own report of itself: they load and print whatever was written, so a
table built from them is identical whether the weights beside them are usable
or corrupt. `rescore` loads the probe (and the LoRA adapter when the run
trained one) and computes the metrics against real test features, which is the
only thing that exercises what the archive actually ships.

Plotting lives in `evaluation/visualize/` -- standalone scripts, not importable
modules, so nothing here pulls in matplotlib.
"""

from .metrics import evaluate_probe
from .rescore import (
    compare_to_recorded,
    find_run_dir,
    load_test_features,
    make_lora_encoder,
    probe_feature_paths,
    read_run_metadata,
    rescore_run,
)

__all__ = [
    "evaluate_probe",
    "compare_to_recorded",
    "find_run_dir",
    "load_test_features",
    "make_lora_encoder",
    "probe_feature_paths",
    "read_run_metadata",
    "rescore_run",
]
