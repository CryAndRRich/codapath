import numpy as np
import pytest

from nucleus.alignment import (
    concat_blocks,
    fit_pca_block,
    normalized_auc,
    residualize_cell_block,
    standardize_l2,
)


def test_alignment_transforms_are_finite_and_keep_empty_rows_zero():
    rng = np.random.default_rng(7)
    dino = rng.normal(size=(40, 12)).astype(np.float32)
    cells = rng.normal(size=(40, 10)).astype(np.float32)
    valid = np.ones(40, dtype=bool)
    valid[[3, 9]] = False
    cells[~valid] = 0.0

    cell_pca = fit_pca_block(cells, valid, 5, 30, 42)
    residual = residualize_cell_block(dino, cell_pca, valid, 30, 1.0, 42)
    joined = concat_blocks(standardize_l2(dino), residual)

    assert cell_pca.shape == (40, 5)
    assert residual.shape == (40, 5)
    assert joined.shape == (40, 17)
    assert np.all(residual[~valid] == 0.0)
    assert np.isfinite(joined).all()


def test_normalized_auc_and_validation():
    assert normalized_auc([25, 50], [0.7, 0.9]) == pytest.approx(0.8)
    with pytest.raises(ValueError, match="strictly increasing"):
        normalized_auc([50, 25], [0.7, 0.9])
    with pytest.raises(ValueError, match="same shape"):
        normalized_auc([25, 50], [0.7])
