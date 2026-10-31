"""Reproducible training helpers for direct recurrent and LNN+UKF models."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import TrajectoryData
from .lnn_ukf import LNNUKF
from .networks import RecurrentPositionRegressor


@dataclass(frozen=True)
class TrainingConfig:
    """Shared optimizer and early-stopping settings."""

    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-5
    patience: int = 8
    gradient_clip_norm: float = 5.0
    innovation_nll_weight: float = 2.0e-3
    covariance_trace_weight: float = 1.0e-5
    device: str = "cpu"


@dataclass
class TrainingHistory:
    """Loss history retained for plotting or audit logs."""

    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch RNGs for a repeatable experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensors_from_data(data: TrajectoryData, device: str | torch.device = "cpu") -> dict[str, Tensor]:
    """Convert one validated trajectory split to float32 PyTorch tensors."""
    target = torch.device(device)
    return {
        "controls": torch.as_tensor(data.controls, dtype=torch.float32, device=target),
        "measurements": torch.as_tensor(data.measurements, dtype=torch.float32, device=target),
        "states": torch.as_tensor(data.states, dtype=torch.float32, device=target),
        "beacons": torch.as_tensor(data.beacons, dtype=torch.float32, device=target),
    }


def _loader(*tensors: Tensor, batch_size: int, shuffle: bool) -> DataLoader[tuple[Tensor, ...]]:
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _restore_best(model: nn.Module, state: dict[str, Tensor] | None) -> None:
    if state is not None:
        model.load_state_dict(state)


@torch.no_grad()
def direct_validation_loss(model: RecurrentPositionRegressor, data: TrajectoryData, device: str) -> float:
    """Return position MSE for the direct open-loop prediction task."""
    tensor = tensors_from_data(data, device)
    output = model(tensor["controls"], tensor["states"][:, 0], data.dt)
    return float(F.mse_loss(output["positions"][:, 1:], tensor["states"][:, 1:]).cpu())


def train_direct_model(
    model: RecurrentPositionRegressor,
    train_data: TrajectoryData,
    validation_data: TrajectoryData,
    config: TrainingConfig,
) -> TrainingHistory:
    """Train a direct LNN/LSTM/CTGRU/CTRNN on residual-position targets."""
    device = torch.device(config.device)
    model.to(device)
    train = tensors_from_data(train_data, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = TrainingHistory()
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    stalled = 0
    batches = _loader(train["controls"], train["states"], batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for controls, states in batches:
            optimizer.zero_grad(set_to_none=True)
            output = model(controls, states[:, 0], train_data.dt)
            loss = F.mse_loss(output["positions"][:, 1:], states[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach()) * controls.shape[0]
            total_examples += controls.shape[0]
        history.train_loss.append(total_loss / max(total_examples, 1))
        model.eval()
        validation_loss = direct_validation_loss(model, validation_data, config.device)
        history.validation_loss.append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
            stalled = 0
        else:
            stalled += 1
            if stalled >= config.patience:
                break
    _restore_best(model, best_state)
    return history


def _innovation_nll(output: dict[str, Tensor], measurements: Tensor) -> Tensor:
    """Gaussian NLL on fully observed innovations, avoiding dropout bias."""
    valid_rows = torch.isfinite(measurements).all(dim=-1)
    if not bool(valid_rows.any()):
        return output["positions"].new_zeros(())
    innovation = output["innovations"][valid_rows]
    covariance = output["innovation_covariances"][valid_rows]
    chol = torch.linalg.cholesky(covariance)
    solved = torch.cholesky_solve(innovation.unsqueeze(-1), chol).squeeze(-1)
    quadratic = (innovation * solved).sum(dim=-1)
    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    return 0.5 * (quadratic + log_det).mean()


@torch.no_grad()
def lnn_ukf_validation_loss(model: LNNUKF, data: TrajectoryData, device: str) -> float:
    """Return posterior position MSE for the online localization task."""
    tensor = tensors_from_data(data, device)
    output = model(tensor["controls"], tensor["measurements"], tensor["states"][:, 0], data.dt)
    return float(F.mse_loss(output["positions"], tensor["states"]).cpu())


def train_lnn_ukf(
    model: LNNUKF,
    train_data: TrajectoryData,
    validation_data: TrajectoryData,
    config: TrainingConfig,
) -> TrainingHistory:
    """Train the differentiable LNN+UKF end-to-end from trajectory positions.

    The loss combines posterior position MSE with a small innovation likelihood
    term.  The latter discourages degenerate adaptive covariance estimates but
    only scores times whose full acoustic measurement vector is available.
    """
    device = torch.device(config.device)
    model.to(device)
    train = tensors_from_data(train_data, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = TrainingHistory()
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    stalled = 0
    batches = _loader(
        train["controls"], train["measurements"], train["states"], batch_size=config.batch_size, shuffle=True
    )

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for controls, measurements, states in batches:
            optimizer.zero_grad(set_to_none=True)
            output = model(controls, measurements, states[:, 0], train_data.dt)
            position_loss = F.mse_loss(output["positions"], states)
            nll = _innovation_nll(output, measurements)
            covariance_trace = torch.diagonal(output["position_covariances"], dim1=-2, dim2=-1).sum(dim=-1).mean()
            loss = position_loss + config.innovation_nll_weight * nll + config.covariance_trace_weight * covariance_trace
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach()) * controls.shape[0]
            total_examples += controls.shape[0]
        history.train_loss.append(total_loss / max(total_examples, 1))
        model.eval()
        validation_loss = lnn_ukf_validation_loss(model, validation_data, config.device)
        history.validation_loss.append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
            stalled = 0
        else:
            stalled += 1
            if stalled >= config.patience:
                break
    _restore_best(model, best_state)
    return history


@torch.no_grad()
def predict_direct(model: RecurrentPositionRegressor, data: TrajectoryData, device: str) -> dict[str, np.ndarray]:
    """Run a trained direct model and return CPU NumPy diagnostics."""
    model.eval()
    tensor = tensors_from_data(data, device)
    output = model(tensor["controls"], tensor["states"][:, 0], data.dt)
    return {key: value.detach().cpu().numpy() for key, value in output.items()}


@torch.no_grad()
def predict_lnn_ukf(model: LNNUKF, data: TrajectoryData, device: str, tau_multiplier: float = 1.0) -> dict[str, np.ndarray]:
    """Run a trained LNN+UKF and return CPU NumPy diagnostics."""
    model.eval()
    tensor = tensors_from_data(data, device)
    output = model(
        tensor["controls"],
        tensor["measurements"],
        tensor["states"][:, 0],
        data.dt,
        tau_multiplier=tau_multiplier,
    )
    return {key: value.detach().cpu().numpy() for key, value in output.items()}
