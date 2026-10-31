from __future__ import annotations

import numpy as np
import pytest

from lnn_ukf.data import make_synthetic_underwater_data
from lnn_ukf.filters import run_classical_filter


@pytest.mark.parametrize("kind", ["ekf", "ukf", "ckf"])
def test_classical_filters_produce_finite_spd_trajectories(kind: str) -> None:
    data = make_synthetic_underwater_data(n_trajectories=3, n_steps=14, seed=6, dropout_probability=0.2)
    result = run_classical_filter(
        kind,
        data.controls[0],
        data.measurements[0],
        data.beacons,
        data.states[0, 0],
    )
    assert result.states.shape == (14, 2)
    assert result.covariances.shape == (14, 2, 2)
    assert np.isfinite(result.states).all()
    assert np.all(np.linalg.eigvalsh(result.covariances) > 0.0)
