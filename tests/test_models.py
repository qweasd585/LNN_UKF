from __future__ import annotations

import numpy as np
import pytest
import torch

from lnn_ukf.data import make_synthetic_underwater_data
from lnn_ukf.lnn_ukf import LNNUKF, LNNUKFConfig
from lnn_ukf.networks import RecurrentModelConfig, RecurrentPositionRegressor


def test_lnn_ukf_forward_is_finite_and_covariances_are_spd() -> None:
    data = make_synthetic_underwater_data(n_trajectories=3, n_steps=10, seed=7)
    model = LNNUKF(
        torch.tensor(data.beacons, dtype=torch.float32),
        LNNUKFConfig(hidden_dim=4, innovation_window=3),
    )
    output = model(
        torch.tensor(data.controls, dtype=torch.float32),
        torch.tensor(data.measurements, dtype=torch.float32),
        torch.tensor(data.states[:, 0], dtype=torch.float32),
        data.dt,
    )
    assert output["positions"].shape == (3, 10, 2)
    assert output["tau"].shape == (3, 9, 4)
    assert torch.isfinite(output["positions"]).all()
    eigenvalues = torch.linalg.eigvalsh(output["position_covariances"])
    assert torch.all(eigenvalues > 0.0)
    loss = torch.mean((output["positions"] - torch.tensor(data.states, dtype=torch.float32)) ** 2)
    loss.backward()
    assert model.readout.raw_weight.grad is not None


@pytest.mark.parametrize("kind", ["lnn", "lstm", "ctgru", "ctrnn"])
def test_all_direct_recurrent_baselines_share_output_contract(kind: str) -> None:
    controls = torch.randn(2, 7, 2)
    initial_position = torch.zeros(2, 2)
    model = RecurrentPositionRegressor(kind, RecurrentModelConfig(input_dim=4, hidden_dim=4))
    output = model(controls, initial_position, dt=0.25)
    assert output["positions"].shape == (2, 8, 2)
    assert output["deltas"].shape == (2, 7, 2)
    assert torch.isfinite(output["positions"]).all()
