"""Continuous-time recurrent models used by the localization experiments.

The LNN implementation deliberately exposes its liquid time constants.  This is
needed both for the LNN+UKF state transition and for the mathematical
interpretability experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _as_step(dt: float | Tensor, reference: Tensor) -> Tensor:
    """Return a positive step size broadcastable to a hidden-state tensor."""
    if isinstance(dt, Tensor):
        value = dt.to(device=reference.device, dtype=reference.dtype)
    else:
        value = torch.tensor(float(dt), device=reference.device, dtype=reference.dtype)
    return value.clamp_min(torch.finfo(reference.dtype).eps)


class SemiOrthogonalReadout(nn.Module):
    """Affine hidden-to-displacement map with optional row-orthogonal weights.

    For a readout ``A`` of shape ``[state_dim, hidden_dim]``, row orthogonality
    gives ``A A^T = I``.  Unlike an attempted inverse of a rectangular matrix,
    this remains mathematically valid when the hidden state has more dimensions
    than the physical displacement.
    """

    def __init__(self, hidden_dim: int, state_dim: int, orthogonal: bool = True) -> None:
        super().__init__()
        if hidden_dim < state_dim:
            raise ValueError("hidden_dim must be at least state_dim for a row-orthogonal readout")
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.orthogonal = orthogonal
        self.raw_weight = nn.Parameter(torch.empty(state_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(state_dim))
        if orthogonal:
            nn.init.orthogonal_(self.raw_weight)
        else:
            # Direct residual predictors should begin at the known odometry
            # model, then learn only the unmodelled displacement correction.
            nn.init.zeros_(self.raw_weight)

    def weight(self) -> Tensor:
        if not self.orthogonal:
            return self.raw_weight
        # QR on A^T gives columns with orthonormal directions; transposing makes
        # the rows of the returned readout orthonormal.
        q, _ = torch.linalg.qr(self.raw_weight.transpose(0, 1), mode="reduced")
        return q.transpose(0, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.weight(), self.bias)

    def orthogonality_error(self) -> Tensor:
        weight = self.weight()
        identity = torch.eye(self.state_dim, device=weight.device, dtype=weight.dtype)
        return torch.linalg.matrix_norm(weight @ weight.transpose(0, 1) - identity, ord="fro")


class LiquidTimeConstantCell(nn.Module):
    """Euler-exact discretization of a liquid-time-constant recurrent ODE.

    The cell uses ``dh/dt = (tanh(W_h h + W_u u + b) - h) / tau(h, u)``.
    The exponential update is stable for variable time steps and makes the
    meaning of ``tau`` explicit: small values follow the instantaneous drive;
    large values retain memory.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        min_tau: float = 0.05,
        liquid: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.min_tau = float(min_tau)
        self.liquid = liquid
        self.input_drive = nn.Linear(input_dim, hidden_dim, bias=False)
        self.hidden_drive = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.tau_network = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.log_fixed_tau = nn.Parameter(torch.full((hidden_dim,), 0.55))
        nn.init.xavier_uniform_(self.input_drive.weight, gain=0.25)
        nn.init.orthogonal_(self.hidden_drive.weight, gain=0.25)
        nn.init.zeros_(self.hidden_drive.bias)
        for layer in self.tau_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.2)
                nn.init.zeros_(layer.bias)

    def time_constant(self, hidden: Tensor, inputs: Tensor) -> Tensor:
        if self.liquid:
            raw_tau = self.tau_network(torch.cat([hidden, inputs], dim=-1))
        else:
            raw_tau = self.log_fixed_tau.expand_as(hidden)
        return F.softplus(raw_tau) + self.min_tau

    def forward(
        self,
        hidden: Tensor,
        inputs: Tensor,
        dt: float | Tensor = 1.0,
        tau_multiplier: float | Tensor = 1.0,
    ) -> tuple[Tensor, Tensor]:
        if hidden.shape[:-1] != inputs.shape[:-1]:
            raise ValueError("hidden and inputs must have matching leading dimensions")
        tau = self.time_constant(hidden, inputs)
        multiplier = torch.as_tensor(tau_multiplier, dtype=tau.dtype, device=tau.device)
        tau = (tau * multiplier).clamp_min(self.min_tau * 0.1)
        drive = torch.tanh(self.input_drive(inputs) + self.hidden_drive(hidden))
        step = _as_step(dt, hidden)
        mixing = -torch.expm1(-step / tau)
        return hidden + mixing * (drive - hidden), tau


class CTGRUCell(nn.Module):
    """A gated continuous-time recurrent cell used as the CTGRU baseline."""

    def __init__(self, input_dim: int, hidden_dim: int, min_tau: float = 0.05) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.min_tau = min_tau
        self.gates = nn.Linear(input_dim + hidden_dim, 2 * hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.tau_net = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, hidden: Tensor, inputs: Tensor, dt: float | Tensor = 1.0) -> tuple[Tensor, Tensor]:
        both = torch.cat([inputs, hidden], dim=-1)
        update_gate, reset_gate = self.gates(both).chunk(2, dim=-1)
        update_gate = torch.sigmoid(update_gate)
        reset_gate = torch.sigmoid(reset_gate)
        candidate = torch.tanh(self.candidate(torch.cat([inputs, reset_gate * hidden], dim=-1)))
        target = (1.0 - update_gate) * hidden + update_gate * candidate
        tau = F.softplus(self.tau_net(both)) + self.min_tau
        mixing = -torch.expm1(-_as_step(dt, hidden) / tau)
        return hidden + mixing * (target - hidden), tau


class CTRNNCell(nn.Module):
    """Leaky continuous-time RNN baseline with learnable per-unit constants."""

    def __init__(self, input_dim: int, hidden_dim: int, min_tau: float = 0.05) -> None:
        super().__init__()
        self.drive = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.log_tau = nn.Parameter(torch.full((hidden_dim,), 0.55))
        self.min_tau = min_tau

    def forward(self, hidden: Tensor, inputs: Tensor, dt: float | Tensor = 1.0) -> tuple[Tensor, Tensor]:
        tau = F.softplus(self.log_tau).expand_as(hidden) + self.min_tau
        target = torch.tanh(self.drive(torch.cat([inputs, hidden], dim=-1)))
        mixing = -torch.expm1(-_as_step(dt, hidden) / tau)
        return hidden + mixing * (target - hidden), tau


ModelKind = Literal["lnn", "lstm", "ctgru", "ctrnn"]


@dataclass(frozen=True)
class RecurrentModelConfig:
    input_dim: int = 2
    state_dim: int = 2
    hidden_dim: int = 8
    min_tau: float = 0.05
    liquid: bool = True
    orthogonal_readout: bool = False
    position_scale: float = 50.0


class RecurrentPositionRegressor(nn.Module):
    """Direct residual-position predictor for LNN/LSTM/CTGRU/CTRNN comparisons.

    All models receive exactly the same command/displacement input and start at
    the same initial position. Their output is an accumulated sequence of
    predicted displacement residuals.
    """

    def __init__(self, kind: ModelKind, config: RecurrentModelConfig) -> None:
        super().__init__()
        self.kind = kind
        self.config = config
        self.initial_hidden = nn.Parameter(torch.zeros(config.hidden_dim))
        if kind == "lnn":
            self.cell: nn.Module = LiquidTimeConstantCell(
                config.input_dim, config.hidden_dim, config.min_tau, liquid=config.liquid
            )
        elif kind == "lstm":
            self.cell = nn.LSTMCell(config.input_dim, config.hidden_dim)
        elif kind == "ctgru":
            self.cell = CTGRUCell(config.input_dim, config.hidden_dim, config.min_tau)
        elif kind == "ctrnn":
            self.cell = CTRNNCell(config.input_dim, config.hidden_dim, config.min_tau)
        else:
            raise ValueError(f"unknown recurrent model kind: {kind}")
        self.readout = SemiOrthogonalReadout(
            config.hidden_dim, config.state_dim, orthogonal=config.orthogonal_readout
        )

    def forward(
        self,
        controls: Tensor,
        initial_position: Tensor,
        dt: float | Tensor = 1.0,
        tau_multiplier: float | Tensor = 1.0,
    ) -> dict[str, Tensor]:
        if controls.ndim != 3:
            raise ValueError("controls must have shape [batch, time, input_dim]")
        batch, steps, _ = controls.shape
        if initial_position.shape != (batch, self.config.state_dim):
            raise ValueError("initial_position has an incompatible shape")
        hidden = self.initial_hidden.unsqueeze(0).expand(batch, -1)
        cell_state = torch.zeros_like(hidden) if self.kind == "lstm" else None
        positions = [initial_position]
        deltas: list[Tensor] = []
        hidden_history: list[Tensor] = []
        tau_history: list[Tensor] = []
        position = initial_position
        for step_index in range(steps):
            control = controls[:, step_index]
            # Position is an admissible external state in the documented LNN
            # formulation.  Supplying the model's own previous estimate avoids
            # teacher-forcing leakage while allowing spatially varying currents.
            if self.config.input_dim == control.shape[-1]:
                inputs = control
            elif self.config.input_dim == control.shape[-1] + self.config.state_dim:
                inputs = torch.cat((control, position / self.config.position_scale), dim=-1)
            else:
                raise ValueError(
                    "input_dim must equal control dimension or control dimension plus state dimension"
                )
            if self.kind == "lstm":
                hidden, cell_state = self.cell(inputs, (hidden, cell_state))  # type: ignore[arg-type]
                tau = torch.full_like(hidden, float("nan"))
            elif self.kind == "lnn":
                hidden, tau = self.cell(hidden, inputs, dt=dt, tau_multiplier=tau_multiplier)  # type: ignore[operator]
            else:
                hidden, tau = self.cell(hidden, inputs, dt=dt)  # type: ignore[operator]
            # The neural readout estimates the unmodelled displacement
            # residual; known commanded odometry is always added explicitly.
            delta = control[:, : self.config.state_dim] + self.readout(hidden)
            position = position + delta
            positions.append(position)
            deltas.append(delta)
            hidden_history.append(hidden)
            tau_history.append(tau)
        return {
            "positions": torch.stack(positions, dim=1),
            "deltas": torch.stack(deltas, dim=1),
            "hidden": torch.stack(hidden_history, dim=1),
            "tau": torch.stack(tau_history, dim=1),
            "orthogonality_error": self.readout.orthogonality_error(),
        }


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the parameter count used in the comparison table."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
