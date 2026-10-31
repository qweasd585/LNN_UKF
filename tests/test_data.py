from __future__ import annotations

import numpy as np

from lnn_ukf.data import (
    load_npz_dataset,
    make_synthetic_underwater_data,
    save_npz_dataset,
    split_by_trajectory,
)


def test_synthetic_data_contract_and_round_trip(tmp_path) -> None:
    data = make_synthetic_underwater_data(n_trajectories=10, n_steps=12, seed=3)
    assert data.controls.shape == (10, 11, 2)
    assert data.measurements.shape[:2] == (10, 12)
    assert data.states.shape == (10, 12, 2)
    assert data.beacons.shape[1] == 2
    target = save_npz_dataset(data, tmp_path / "underwater.npz")
    restored = load_npz_dataset(target)
    assert np.allclose(restored.controls, data.controls)
    assert np.allclose(restored.states, data.states)
    assert np.allclose(restored.measurements, data.measurements, equal_nan=True)


def test_split_is_trajectory_level() -> None:
    data = make_synthetic_underwater_data(n_trajectories=15, n_steps=9, seed=4)
    splits = split_by_trajectory(data, seed=9)
    assert sum(item.n_trajectories for item in splits.values()) == data.n_trajectories
    assert all(item.states.shape[1:] == data.states.shape[1:] for item in splits.values())
