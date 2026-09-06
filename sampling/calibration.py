"""Temperature scaling used by Uncertainty Herding and by `pact`.

Raw probe logits from a few dozen labels are badly over-confident, so a margin
read off them is not comparable across rounds. Uncertainty Herding's Sec. 3.2
(Proposition 3) fixes this by rescaling logits with a temperature chosen to
minimise Expected Calibration Error on a held-out slice of the labeled set.

Shared here rather than duplicated: `baselines.uncertainty_herding`,
`baselines.refine` (its stage-2 head is Uncertainty Herding) and `pact` all
need the same procedure, and `pact` calibrates each of its two probes.
"""

from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

TEMPERATURE_GRID = np.arange(1.0, 20.0, 0.1)
VALIDATION_FRACTION = 0.3


def expected_calibration_error(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
) -> float:
    """Equal-width confidence bins over [0, 1], weighted mean |confidence - accuracy|.

    Matches the official ECELoss in `repos/uherding/deep-al/pycls/calibration/metrics.py`
    (Naeini et al., AAAI 2015).
    """
    probabilities = F.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1)
    confidences, predictions = torch.max(probabilities, dim=1)
    correct = predictions.eq(torch.as_tensor(labels, dtype=torch.long))

    boundaries = torch.linspace(0, 1, n_bins + 1)
    error = 0.0
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        in_bin = confidences.gt(low.item()) & confidences.le(high.item())
        share = in_bin.float().mean().item()
        if share > 0:
            error += share * abs(
                confidences[in_bin].mean().item() - correct[in_bin].float().mean().item()
            )
    return error


def calibrate_temperature(
    features: np.ndarray,
    labels: np.ndarray,
    selected: Sequence[int],
    num_classes: int,
    probe_epochs: int,
    probe_lr: float,
    device: torch.device,
    max_temperature: float = None,
) -> float:
    """Split the labeled set, fit a throwaway probe on the train part, and
    return the grid temperature minimising ECE on the held-out part.

    Matches the official `obtain_temperature()`: 70/30 split, grid
    `arange(1.0, 20, 0.1)`, `n_bins = num_classes`. The probe fitted here is
    discarded — only the temperature is reused, to rescale the logits of the
    real probe that the caller trains on the FULL labeled set.

    Returns 1.0 (no scaling) whenever the split leaves too little to calibrate
    on, which is the normal case for the first one or two rounds.

    `max_temperature` truncates the grid. The default is None -- the full
    paper grid, which the Uncertainty Herding and REFINE baselines must keep
    to stay faithful. `pact` passes a cap, because a temperature at the
    grid's ceiling is destructive there in a way it is not for a single-probe
    method: dividing BOTH probes' logits by ~20 flattens each softmax toward
    uniform, and two near-uniform distributions cannot disagree. Measured on
    14-class logits, mean Jensen-Shannon divergence between two genuinely
    different probes falls 0.349 -> 0.0023 (150x) as T goes 1.0 -> 19.9,
    which is exactly the `tau=19.9/19.9 js=0.0001` seen in a real histoset
    run: the disagreement signal -- the whole of what PACT adds over
    Uncertainty Herding -- was annihilated by its own calibration step.

    A ceiling temperature is also weak evidence to begin with at these
    budgets: it is chosen by minimising ECE over a validation split that can
    be ~7 points across 14 classes, where the binned estimate is mostly
    noise.
    """
    from training.probe import train_probe

    selected = list(selected)
    n_validation = max(1, int(len(selected) * VALIDATION_FRACTION))
    if len(selected) - n_validation < 2:
        return 1.0

    train_index: List[int] = selected[:-n_validation]
    validation_index: List[int] = selected[-n_validation:]
    train_labels = labels[train_index]
    validation_labels = labels[validation_index]
    if len(np.unique(train_labels)) < 2 or len(validation_index) < 2:
        return 1.0

    probe = train_probe(
        features[train_index], train_labels, num_classes, probe_epochs, probe_lr, device
    )
    validation_logits = probe.predict_logits(features[validation_index], device)
    del probe

    grid = TEMPERATURE_GRID
    if max_temperature is not None:
        grid = grid[grid <= float(max_temperature)]
        if len(grid) == 0:
            return 1.0

    best_temperature, best_error = 1.0, float("inf")
    for temperature in grid:
        error = expected_calibration_error(
            validation_logits / temperature, validation_labels, n_bins=num_classes
        )
        if error < best_error:
            best_error, best_temperature = error, float(temperature)
    return best_temperature
