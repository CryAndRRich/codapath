import numpy as np
import pytest
import torch

from trainer import train_dual_linear


def test_dual_probe_supports_gated_consistency():
    image = np.asarray(
        [[-1.0, 0.0], [-0.8, 0.1], [0.8, 0.0], [1.0, 0.1]],
        dtype=np.float32,
    )
    cell = np.asarray(
        [[-0.9, 0.0], [0.0, 0.0], [0.9, 0.0], [1.1, 0.1]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1])
    valid = np.asarray([True, False, True, True])
    image_probe, cell_probe = train_dual_linear(
        image,
        cell,
        labels,
        num_classes=2,
        num_epochs=2,
        lr=1e-2,
        device=torch.device("cpu"),
        cell_valid=valid,
        cell_reliability=valid.astype(np.float32),
        consistency_weight=0.1,
    )
    assert image_probe.predict_logits(image, torch.device("cpu")).shape == (4, 2)
    assert cell_probe.predict_logits(cell, torch.device("cpu")).shape == (4, 2)


def test_dual_probe_rejects_missing_cell_view():
    with pytest.raises(ValueError, match="valid cell view"):
        train_dual_linear(
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            np.asarray([0, 1]),
            num_classes=2,
            num_epochs=1,
            lr=1e-3,
            device=torch.device("cpu"),
            cell_valid=np.zeros(2, dtype=bool),
        )


@pytest.mark.parametrize(
    "mode", ["symmetric_js", "visual_teacher", "cell_teacher"]
)
def test_all_consistency_directions_train(mode):
    image = np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    cell = np.asarray([[-0.8, 0.0], [0.8, 0.0]], dtype=np.float32)
    probes = train_dual_linear(
        image,
        cell,
        np.asarray([0, 1]),
        num_classes=2,
        num_epochs=1,
        lr=1e-2,
        device=torch.device("cpu"),
        cell_valid=np.ones(2, dtype=bool),
        consistency_weight=0.1,
        consistency_mode=mode,
    )
    assert len(probes) == 2
