"""Interpretable liquid-neural-network unscented Kalman filter.

This module implements the document's three stages in one differentiable model:

1. the LNN hidden state is expanded into UKF sigma points;
2. each point is propagated by a liquid-time-constant ODE and decoded as a
   displacement residual;
3. nonlinear acoustic ranges update both physical position and the observable
   hidden-state covariance through UKF cross-covariances.

The code intentionally does *not* invert a rectangular hidden-to-position
readout. With a row-orthogonal readout, a measurement update is transported
into the hidden state through the statistically correct sigma-point
cross-covariance. This gives a valid covariance feedback loop when the hidden
state is higher-dimensional than physical position.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .networks import LiquidTimeConstantCell, SemiOrthogonalReadout


def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def positive_definite(matrix: Tensor, floor: float = 1.0e-6) -> Tensor:
    """Symmetrise and make covariances differentiably strictly positive definite.

    A direct eigendecomposition is attractive for eigenvalue clipping but its
    backward pass is undefined at repeated eigenvalues (common for initial
    diagonal covariances).  Instead we add the minimum diagonal correction
    required by the Gershgorin criterion for strict diagonal dominance.  A
    symmetric strictly diagonally dominant matrix with positive diagonal is
    positive definite, and this construction has stable gradients.
    """
    symmetric = _symmetrize(matrix)
    diagonal = torch.diagonal(symmetric, dim1=-2, dim2=-1)
    off_diagonal_absolute_sum = symmetric.abs().sum(dim=-1) - diagonal.abs()
    gershgorin_lower_bound = (diagonal - off_diagonal_absolute_sum).amin(dim=-1)
    correction = F.relu(torch.as_tensor(floor, dtype=matrix.dtype, device=matrix.device) - gershgorin_lower_bound)
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    return symmetric + (correction + floor).unsqueeze(-1).unsqueeze(-1) * identity


def _weighted_covariance(left: Tensor, right: Tensor, weights: Tensor) -> Tensor:
    """Return a weighted covariance for arrays shaped ``[B, S, D]``."""
    return torch.einsum("s,bsi,bsj->bij", weights, left, right)


class CholeskyCovariancePredictor(nn.Module):
    """Innovation-history LSTM that emits a strictly positive-definite matrix."""

    def __init__(
        self,
        innovation_dim: int,
        covariance_dim: int,
        history_hidden_dim: int = 24,
        floor: float = 1.0e-5,
        adaptive: bool = True,
    ) -> None:
        super().__init__()
        self.innovation_dim = innovation_dim
        self.covariance_dim = covariance_dim
        self.floor = float(floor)
        self.adaptive = adaptive
        number_of_lower_entries = covariance_dim * (covariance_dim + 1) // 2
        if adaptive:
            self.encoder = nn.LSTM(innovation_dim, history_hidden_dim, batch_first=True)
            self.head = nn.Linear(history_hidden_dim, number_of_lower_entries)
        else:
            self.raw_fixed_diagonal = nn.Parameter(torch.full((covariance_dim,), -2.2))

    def forward(self, innovation_history: Tensor) -> Tensor:
        """Return ``[batch, covariance_dim, covariance_dim]`` SPD matrices.

        Only the provided history is used. The current measurement innovation
        is appended after this call, so adaptive Q/R never uses future data.
        """
        if innovation_history.ndim != 3 or innovation_history.shape[-1] != self.innovation_dim:
            raise ValueError("innovation_history must have shape [batch, window, innovation_dim]")
        batch = innovation_history.shape[0]
        if not self.adaptive:
            diagonal = F.softplus(self.raw_fixed_diagonal) + self.floor
            return torch.diag_embed(diagonal.expand(batch, -1))
        _, (hidden, _) = self.encoder(innovation_history)
        raw = self.head(hidden[-1])
        row, column = torch.tril_indices(
            self.covariance_dim,
            self.covariance_dim,
            device=raw.device,
        )
        lower = raw.new_zeros((batch, self.covariance_dim, self.covariance_dim))
        lower[:, row, column] = raw
        diagonal = torch.diagonal(lower, dim1=-2, dim2=-1)
        lower = lower - torch.diag_embed(diagonal) + torch.diag_embed(F.softplus(diagonal) + self.floor)
        identity = torch.eye(self.covariance_dim, device=raw.device, dtype=raw.dtype)
        return lower @ lower.transpose(-1, -2) + self.floor * identity


@dataclass(frozen=True)
class LNNUKFConfig:
    """Architecture and numerically relevant UKF settings."""

    input_dim: int = 4
    state_dim: int = 2
    hidden_dim: int = 8
    innovation_window: int = 8
    covariance_lstm_hidden_dim: int = 24
    min_tau: float = 0.05
    liquid: bool = True
    orthogonal_readout: bool = True
    adaptive_process_noise: bool = True
    adaptive_measurement_noise: bool = True
    covariance_feedback: bool = True
    position_scale: float = 50.0
    alpha: float = 0.75
    beta: float = 2.0
    kappa: float = 0.0
    initial_hidden_covariance: float = 0.20
    covariance_floor: float = 1.0e-5
    missing_measurement_variance: float = 1.0e6


class LNNUKF(nn.Module):
    """Batch-capable LNN+UKF model for range-based 2-D localization."""

    def __init__(self, beacons: Tensor, config: LNNUKFConfig | None = None) -> None:
        super().__init__()
        self.config = config or LNNUKFConfig()
        beacons = torch.as_tensor(beacons, dtype=torch.float32)
        if beacons.ndim != 2 or beacons.shape[1] != self.config.state_dim:
            raise ValueError("beacons must have shape [measurement_dim, state_dim]")
        if self.config.hidden_dim < self.config.state_dim and self.config.orthogonal_readout:
            raise ValueError("row-orthogonal readout requires hidden_dim >= state_dim")
        if self.config.alpha <= 0.0:
            raise ValueError("UKF alpha must be positive")
        self.register_buffer("beacons", beacons)
        self.measurement_dim = int(beacons.shape[0])
        self.cell = LiquidTimeConstantCell(
            self.config.input_dim,
            self.config.hidden_dim,
            min_tau=self.config.min_tau,
            liquid=self.config.liquid,
        )
        self.readout = SemiOrthogonalReadout(
            self.config.hidden_dim,
            self.config.state_dim,
            orthogonal=self.config.orthogonal_readout,
        )
        self.initial_hidden = nn.Parameter(torch.zeros(self.config.hidden_dim))
        self.process_noise = CholeskyCovariancePredictor(
            self.measurement_dim,
            self.config.hidden_dim,
            self.config.covariance_lstm_hidden_dim,
            self.config.covariance_floor,
            adaptive=self.config.adaptive_process_noise,
        )
        self.measurement_noise = CholeskyCovariancePredictor(
            self.measurement_dim,
            self.measurement_dim,
            self.config.covariance_lstm_hidden_dim,
            self.config.covariance_floor,
            adaptive=self.config.adaptive_measurement_noise,
        )

        dimension = self.config.hidden_dim
        lam = self.config.alpha**2 * (dimension + self.config.kappa) - dimension
        spread = dimension + lam
        if spread <= 0.0:
            raise ValueError("alpha and kappa produce a non-positive UKF spread")
        mean_weights = torch.full((2 * dimension + 1,), 0.5 / spread)
        covariance_weights = mean_weights.clone()
        mean_weights[0] = lam / spread
        covariance_weights[0] = mean_weights[0] + (1.0 - self.config.alpha**2 + self.config.beta)
        self.register_buffer("sigma_spread", torch.tensor(spread, dtype=torch.float32))
        self.register_buffer("mean_weights", mean_weights)
        self.register_buffer("covariance_weights", covariance_weights)

    def _sigma_points(self, mean: Tensor, covariance: Tensor) -> Tensor:
        covariance = positive_definite(covariance, self.config.covariance_floor)
        root = torch.linalg.cholesky(self.sigma_spread.to(mean.dtype) * covariance)
        # Cholesky columns are the correctly oriented sigma displacements.
        offsets = root.transpose(-1, -2)
        return torch.cat((mean.unsqueeze(1), mean.unsqueeze(1) + offsets, mean.unsqueeze(1) - offsets), dim=1)

    def measurement_function(self, positions: Tensor) -> Tensor:
        """Nonlinear acoustic-range observation function ``h(position)``."""
        difference = positions.unsqueeze(-2) - self.beacons.to(dtype=positions.dtype)
        return torch.linalg.vector_norm(difference, dim=-1).clamp_min(1.0e-7)

    def _cell_inputs(self, controls: Tensor, positions: Tensor) -> Tensor:
        """Build documented external inputs from command and current position."""
        if self.config.input_dim == controls.shape[-1]:
            return controls
        expected = controls.shape[-1] + self.config.state_dim
        if self.config.input_dim == expected:
            return torch.cat((controls, positions / self.config.position_scale), dim=-1)
        raise ValueError(
            f"input_dim={self.config.input_dim} is incompatible with controls of dimension "
            f"{controls.shape[-1]}; use control_dim or control_dim + state_dim"
        )

    def _measurement_update(
        self,
        hidden_sigma: Tensor,
        hidden_mean: Tensor,
        hidden_covariance: Tensor,
        position_sigma: Tensor,
        position_mean: Tensor,
        position_covariance: Tensor,
        measurement: Tensor,
        measurement_covariance: Tensor,
    ) -> dict[str, Tensor]:
        """UKF measurement update, including hidden-state covariance feedback."""
        observation_sigma = self.measurement_function(position_sigma)
        observation_mean = torch.einsum("s,bsm->bm", self.mean_weights.to(position_sigma.dtype), observation_sigma)
        hidden_delta = hidden_sigma - hidden_mean.unsqueeze(1)
        position_delta = position_sigma - position_mean.unsqueeze(1)
        observation_delta = observation_sigma - observation_mean.unsqueeze(1)
        hidden_observation_covariance = _weighted_covariance(
            hidden_delta, observation_delta, self.covariance_weights.to(position_sigma.dtype)
        )
        position_observation_covariance = _weighted_covariance(
            position_delta, observation_delta, self.covariance_weights.to(position_sigma.dtype)
        )
        observation_covariance = _weighted_covariance(
            observation_delta, observation_delta, self.covariance_weights.to(position_sigma.dtype)
        )

        valid = torch.isfinite(measurement)
        observed = torch.where(valid, measurement, observation_mean)
        missing_noise = (~valid).to(position_sigma.dtype) * self.config.missing_measurement_variance
        effective_measurement_covariance = measurement_covariance + torch.diag_embed(missing_noise)
        innovation_covariance = positive_definite(
            observation_covariance + effective_measurement_covariance,
            self.config.covariance_floor,
        )
        innovation = observed - observation_mean
        hidden_gain = torch.linalg.solve(
            innovation_covariance,
            hidden_observation_covariance.transpose(-1, -2),
        ).transpose(-1, -2)
        position_gain = torch.linalg.solve(
            innovation_covariance,
            position_observation_covariance.transpose(-1, -2),
        ).transpose(-1, -2)
        posterior_hidden = hidden_mean + (hidden_gain @ innovation.unsqueeze(-1)).squeeze(-1)
        posterior_position = position_mean + (position_gain @ innovation.unsqueeze(-1)).squeeze(-1)
        posterior_position_covariance = positive_definite(
            position_covariance - position_gain @ innovation_covariance @ position_gain.transpose(-1, -2),
            self.config.covariance_floor,
        )
        if self.config.covariance_feedback:
            posterior_hidden_covariance = positive_definite(
                hidden_covariance - hidden_gain @ innovation_covariance @ hidden_gain.transpose(-1, -2),
                self.config.covariance_floor,
            )
        else:
            posterior_hidden_covariance = positive_definite(hidden_covariance, self.config.covariance_floor)
        nis = (innovation.unsqueeze(-2) @ torch.linalg.solve(innovation_covariance, innovation.unsqueeze(-1))).squeeze(-1).squeeze(-1)
        return {
            "hidden": posterior_hidden,
            "hidden_covariance": posterior_hidden_covariance,
            "position": posterior_position,
            "position_covariance": posterior_position_covariance,
            "innovation": innovation,
            "innovation_covariance": innovation_covariance,
            "nis": nis,
            "observation_mean": observation_mean,
            "valid_measurement": valid,
        }

    def forward(
        self,
        controls: Tensor,
        measurements: Tensor,
        initial_position: Tensor,
        dt: float | Tensor = 1.0,
        tau_multiplier: float | Tensor = 1.0,
    ) -> dict[str, Tensor]:
        """Run the complete LNN+UKF over a batch of trajectories.

        ``controls`` is ``[B, T-1, C]``, ``measurements`` is ``[B, T, M]``, and
        missing measurements are represented by ``NaN``. The returned tensors
        include all state, covariance, innovation, Q/R and tau diagnostics.
        """
        if controls.ndim != 3 or measurements.ndim != 3:
            raise ValueError("controls and measurements must be rank-3 tensors")
        batch, transitions, _ = controls.shape
        if measurements.shape != (batch, transitions + 1, self.measurement_dim):
            raise ValueError("measurements must have shape [batch, controls_time + 1, measurement_dim]")
        if initial_position.shape != (batch, self.config.state_dim):
            raise ValueError("initial_position has an incompatible shape")

        dtype = controls.dtype
        device = controls.device
        hidden = self.initial_hidden.to(dtype).unsqueeze(0).expand(batch, -1)
        identity_hidden = torch.eye(self.config.hidden_dim, device=device, dtype=dtype)
        hidden_covariance = identity_hidden.unsqueeze(0).expand(batch, -1, -1) * self.config.initial_hidden_covariance
        position = initial_position
        innovation_history = torch.zeros(
            batch, self.config.innovation_window, self.measurement_dim, device=device, dtype=dtype
        )

        # Assimilate the initial acoustic observation. Sigma deviations are
        # mapped around the known physical state without a displacement bias.
        initial_sigma = self._sigma_points(hidden, hidden_covariance)
        initial_delta = initial_sigma - hidden.unsqueeze(1)
        readout_weight = self.readout.weight().to(dtype)
        initial_position_sigma = position.unsqueeze(1) + initial_delta @ readout_weight.transpose(-1, -2)
        initial_position_covariance = _weighted_covariance(
            initial_position_sigma - position.unsqueeze(1),
            initial_position_sigma - position.unsqueeze(1),
            self.covariance_weights.to(dtype),
        )
        initial_r = self.measurement_noise(innovation_history)
        initial_update = self._measurement_update(
            initial_sigma,
            hidden,
            hidden_covariance,
            initial_position_sigma,
            position,
            initial_position_covariance,
            measurements[:, 0],
            initial_r,
        )
        hidden = initial_update["hidden"]
        hidden_covariance = initial_update["hidden_covariance"]
        position = initial_update["position"]
        position_covariance = initial_update["position_covariance"]
        innovation_history = torch.cat((innovation_history[:, 1:], initial_update["innovation"].unsqueeze(1)), dim=1)

        positions = [position]
        position_covariances = [position_covariance]
        hidden_states = [hidden]
        hidden_covariances = [hidden_covariance]
        innovations = [initial_update["innovation"]]
        innovation_covariances = [initial_update["innovation_covariance"]]
        nis_values = [initial_update["nis"]]
        tau_values: list[Tensor] = []
        process_covariances: list[Tensor] = []
        measurement_covariances: list[Tensor] = [initial_r]

        sigma_count = 2 * self.config.hidden_dim + 1
        for time_index in range(transitions):
            process_covariance = self.process_noise(innovation_history)
            measurement_covariance = self.measurement_noise(innovation_history)
            sigma = self._sigma_points(hidden, hidden_covariance)
            repeated_control = controls[:, time_index].unsqueeze(1).expand(-1, sigma_count, -1)
            repeated_position = position.unsqueeze(1).expand(-1, sigma_count, -1)
            sigma_input = self._cell_inputs(
                repeated_control.reshape(batch * sigma_count, -1),
                repeated_position.reshape(batch * sigma_count, -1),
            )
            propagated_sigma, propagated_tau = self.cell(
                sigma.reshape(batch * sigma_count, -1),
                sigma_input,
                dt=dt,
                tau_multiplier=tau_multiplier,
            )
            propagated_sigma = propagated_sigma.reshape(batch, sigma_count, self.config.hidden_dim)
            propagated_tau = propagated_tau.reshape(batch, sigma_count, self.config.hidden_dim)
            predicted_hidden = torch.einsum("s,bsn->bn", self.mean_weights.to(dtype), propagated_sigma)
            hidden_delta = propagated_sigma - predicted_hidden.unsqueeze(1)
            predicted_hidden_covariance = positive_definite(
                _weighted_covariance(hidden_delta, hidden_delta, self.covariance_weights.to(dtype)) + process_covariance,
                self.config.covariance_floor,
            )

            residual_sigma = self.readout(propagated_sigma)
            nominal_displacement = controls[:, time_index, : self.config.state_dim].unsqueeze(1)
            position_sigma = position.unsqueeze(1) + nominal_displacement + residual_sigma
            predicted_position = torch.einsum("s,bsd->bd", self.mean_weights.to(dtype), position_sigma)
            position_delta = position_sigma - predicted_position.unsqueeze(1)
            projected_process_covariance = readout_weight @ process_covariance @ readout_weight.transpose(-1, -2)
            predicted_position_covariance = positive_definite(
                _weighted_covariance(position_delta, position_delta, self.covariance_weights.to(dtype))
                + projected_process_covariance,
                self.config.covariance_floor,
            )

            update = self._measurement_update(
                propagated_sigma,
                predicted_hidden,
                predicted_hidden_covariance,
                position_sigma,
                predicted_position,
                predicted_position_covariance,
                measurements[:, time_index + 1],
                measurement_covariance,
            )
            hidden = update["hidden"]
            hidden_covariance = update["hidden_covariance"]
            position = update["position"]
            position_covariance = update["position_covariance"]
            innovation_history = torch.cat((innovation_history[:, 1:], update["innovation"].unsqueeze(1)), dim=1)
            positions.append(position)
            position_covariances.append(position_covariance)
            hidden_states.append(hidden)
            hidden_covariances.append(hidden_covariance)
            innovations.append(update["innovation"])
            innovation_covariances.append(update["innovation_covariance"])
            nis_values.append(update["nis"])
            tau_values.append(propagated_tau[:, 0])
            process_covariances.append(process_covariance)
            measurement_covariances.append(measurement_covariance)

        return {
            "positions": torch.stack(positions, dim=1),
            "position_covariances": torch.stack(position_covariances, dim=1),
            "hidden": torch.stack(hidden_states, dim=1),
            "hidden_covariances": torch.stack(hidden_covariances, dim=1),
            "innovations": torch.stack(innovations, dim=1),
            "innovation_covariances": torch.stack(innovation_covariances, dim=1),
            "nis": torch.stack(nis_values, dim=1),
            "tau": torch.stack(tau_values, dim=1),
            "process_covariances": torch.stack(process_covariances, dim=1),
            "measurement_covariances": torch.stack(measurement_covariances, dim=1),
            "orthogonality_error": self.readout.orthogonality_error(),
        }

    @torch.no_grad()
    def explainable_readout_contributions(self, hidden: Tensor) -> Tensor:
        """Return per-unit physical contributions ``A[:, i] h_i``.

        Output shape is ``[..., hidden_dim, state_dim]``. Summing over hidden
        units and adding the readout bias reproduces the residual exactly.
        """
        weight = self.readout.weight().to(dtype=hidden.dtype)
        return hidden.unsqueeze(-1) * weight.transpose(0, 1)
