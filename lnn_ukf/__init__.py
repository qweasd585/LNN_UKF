"""Reproducible LNN+UKF underwater-localisation research implementation."""

from .data import TrajectoryData, load_npz_dataset, make_synthetic_underwater_data, save_npz_dataset, split_by_trajectory
from .lnn_ukf import LNNUKF, LNNUKFConfig
from .networks import RecurrentModelConfig, RecurrentPositionRegressor

__all__ = [
    "LNNUKF",
    "LNNUKFConfig",
    "RecurrentModelConfig",
    "RecurrentPositionRegressor",
    "TrajectoryData",
    "load_npz_dataset",
    "make_synthetic_underwater_data",
    "save_npz_dataset",
    "split_by_trajectory",
]
