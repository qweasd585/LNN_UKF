"""Data contracts and reproducible 2-D underwater-localisation data.

The project deliberately uses a small, explicit on-disk contract so that a
legacy data set can be connected without silently changing the experimental
protocol.  A data set stores nominal displacement controls, acoustic range
measurements and the reference 2-D trajectory in one ``.npz`` file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_float_array(value: ArrayLike, name: str) -> FloatArray:
    """Convert *value* to a float64 array with a useful error message."""

    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # pragma: no cover - NumPy dependent
        raise ValueError(f"{name} must be numeric") from exc


@dataclass(frozen=True)
class TrajectoryData:
    """A batched localisation data set.

    Attributes
    ----------
    controls:
        Nominal Cartesian displacements with shape ``[N, T - 1, 2]``.  They
        are the input available to the dynamics model before an acoustic
        measurement correction.
    measurements:
        Acoustic ranges with shape ``[N, T, M]``.  A missing range is encoded
        as ``NaN``; all non-missing values must be finite.
    states:
        Reference Cartesian positions with shape ``[N, T, 2]``.  These are
        used for supervised training and evaluation, never by an online
        filter after its initialisation.
    beacons:
        Fixed beacon positions with shape ``[M, 2]``.
    dt:
        Sampling period in seconds.
    """

    controls: FloatArray
    measurements: FloatArray
    states: FloatArray
    beacons: FloatArray
    dt: float

    def __post_init__(self) -> None:
        controls = _as_float_array(self.controls, "controls")
        measurements = _as_float_array(self.measurements, "measurements")
        states = _as_float_array(self.states, "states")
        beacons = _as_float_array(self.beacons, "beacons")
        dt = float(self.dt)

        if controls.ndim != 3 or controls.shape[-1] != 2:
            raise ValueError("controls must have shape [N, T - 1, 2]")
        if states.ndim != 3 or states.shape[-1] != 2:
            raise ValueError("states must have shape [N, T, 2]")
        if measurements.ndim != 3:
            raise ValueError("measurements must have shape [N, T, M]")
        if beacons.ndim != 2 or beacons.shape[-1] != 2:
            raise ValueError("beacons must have shape [M, 2]")
        if beacons.shape[0] < 3:
            raise ValueError("at least three beacons are required for 2-D ranging")
        if controls.shape[0] != states.shape[0] or controls.shape[0] != measurements.shape[0]:
            raise ValueError("controls, measurements and states must have the same N")
        if states.shape[1] < 2:
            raise ValueError("at least two time points are required")
        if controls.shape[1] != states.shape[1] - 1:
            raise ValueError("controls must have one fewer time point than states")
        if measurements.shape[1] != states.shape[1]:
            raise ValueError("measurements and states must have the same T")
        if measurements.shape[2] != beacons.shape[0]:
            raise ValueError("measurement dimension must equal number of beacons")
        if not np.isfinite(controls).all() or not np.isfinite(states).all() or not np.isfinite(beacons).all():
            raise ValueError("controls, states and beacons must be finite")
        if not np.isfinite(measurements[~np.isnan(measurements)]).all():
            raise ValueError("measurements may contain NaN for dropout, but no infinities")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite scalar")

        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "beacons", beacons)
        object.__setattr__(self, "dt", dt)

    @property
    def n_trajectories(self) -> int:
        """Number of independent trajectories."""

        return int(self.states.shape[0])

    @property
    def n_steps(self) -> int:
        """Number of state time points per trajectory."""

        return int(self.states.shape[1])

    @property
    def measurement_dim(self) -> int:
        """Number of acoustic beacon ranges per time point."""

        return int(self.measurements.shape[2])

    def take(self, indices: ArrayLike) -> "TrajectoryData":
        """Return a trajectory-level subset without mixing time points."""

        index = np.asarray(indices)
        return TrajectoryData(
            controls=self.controls[index],
            measurements=self.measurements[index],
            states=self.states[index],
            beacons=self.beacons,
            dt=self.dt,
        )


def _beacon_layout(n_beacons: int, radius: float = 42.0) -> FloatArray:
    """Create a well-conditioned circular beacon layout for a demo task."""

    if n_beacons < 3:
        raise ValueError("n_beacons must be at least 3")
    angles = np.linspace(0.0, 2.0 * np.pi, n_beacons, endpoint=False)
    # A small angular offset prevents the first beacon aligning with the x axis
    # for every generated trajectory.
    angles = angles + np.pi / 8.0
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def make_synthetic_underwater_data(
    n_trajectories: int = 120,
    n_steps: int = 96,
    dt: float = 0.25,
    n_beacons: int = 4,
    seed: int = 0,
    measurement_noise: float = 0.35,
    current_scale: float = 0.42,
    process_noise: float = 0.035,
    dropout_probability: float = 0.04,
    outlier_probability: float = 0.015,
    outlier_scale: float = 3.5,
) -> TrajectoryData:
    """Generate a nonlinear AUV localisation benchmark.

    Nominal controls describe a smoothly turning commanded displacement.  The
    reference vehicle experiences a spatially and temporally varying current,
    so simple integration drifts during long trajectories.  Measurements are
    nonlinear ranges to fixed acoustic beacons and may include independent
    dropout and outliers.  The generator is deterministic for a given seed.

    The synthetic task is intentionally a *demo* and is never presented as a
    replacement for the user's real AUV data.  It allows all model paths and
    interpretation plots to be executed before a ``LNN_Lack`` data adapter is
    supplied.
    """

    if n_trajectories < 3:
        raise ValueError("n_trajectories must be at least 3")
    if n_steps < 3:
        raise ValueError("n_steps must be at least 3")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    for name, probability in {
        "dropout_probability": dropout_probability,
        "outlier_probability": outlier_probability,
    }.items():
        if not 0.0 <= probability < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    if min(measurement_noise, current_scale, process_noise, outlier_scale) < 0.0:
        raise ValueError("noise and scale arguments must be non-negative")

    rng = np.random.default_rng(seed)
    beacons = _beacon_layout(n_beacons)
    states = np.empty((n_trajectories, n_steps, 2), dtype=np.float64)
    controls = np.empty((n_trajectories, n_steps - 1, 2), dtype=np.float64)

    time = np.arange(n_steps - 1, dtype=np.float64) * dt
    initial_heading = rng.uniform(-np.pi, np.pi, size=n_trajectories)
    heading_phase = rng.uniform(-np.pi, np.pi, size=n_trajectories)
    speed_phase = rng.uniform(-np.pi, np.pi, size=n_trajectories)
    current_phase = rng.uniform(-np.pi, np.pi, size=n_trajectories)
    states[:, 0] = rng.uniform(-11.0, 11.0, size=(n_trajectories, 2))

    for trajectory in range(n_trajectories):
        heading = (
            initial_heading[trajectory]
            + 0.55 * np.sin(0.055 * time + heading_phase[trajectory])
            + 0.18 * np.sin(0.13 * time + 0.5 * heading_phase[trajectory])
        )
        speed = (
            1.35
            + 0.32 * np.sin(0.075 * time + speed_phase[trajectory])
            + 0.12 * np.sin(0.19 * time + heading_phase[trajectory])
        )
        nominal = dt * speed[:, None] * np.column_stack((np.cos(heading), np.sin(heading)))
        controls[trajectory] = nominal

        for step in range(n_steps - 1):
            x, y = states[trajectory, step]
            # A rotational/sheared flow field plus a time-varying tide gives
            # a smooth but nonstationary unmodelled displacement.
            tide = 0.16 * np.sin(0.10 * time[step] + current_phase[trajectory])
            current_velocity = current_scale * np.array(
                [
                    0.66 * np.sin(0.105 * y + 0.071 * time[step]) + tide,
                    0.58 * np.cos(0.095 * x - 0.061 * time[step]) - 0.55 * tide,
                ]
            )
            perturbation = dt * current_velocity + rng.normal(0.0, process_noise, size=2)
            states[trajectory, step + 1] = states[trajectory, step] + nominal[step] + perturbation

    residual = states[:, :, None, :] - beacons[None, None, :, :]
    true_ranges = np.linalg.norm(residual, axis=-1)
    measurements = true_ranges + rng.normal(0.0, measurement_noise, size=true_ranges.shape)

    outlier_mask = rng.random(measurements.shape) < outlier_probability
    if np.any(outlier_mask):
        measurements[outlier_mask] += rng.normal(
            0.0,
            outlier_scale * max(measurement_noise, 1e-8),
            size=int(outlier_mask.sum()),
        )

    dropout_mask = rng.random(measurements.shape) < dropout_probability
    # A filter can handle partial dropout, but every synthetic time point keeps
    # at least one range so a demo does not devolve into an all-missing update.
    all_missing = dropout_mask.all(axis=-1)
    if np.any(all_missing):
        missing_trajectories, missing_steps = np.where(all_missing)
        retained = rng.integers(0, n_beacons, size=len(missing_trajectories))
        dropout_mask[missing_trajectories, missing_steps, retained] = False
    measurements[dropout_mask] = np.nan

    return TrajectoryData(
        controls=controls,
        measurements=measurements,
        states=states,
        beacons=beacons,
        dt=dt,
    )


def split_by_trajectory(
    data: TrajectoryData,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    seed: int = 0,
) -> Dict[str, TrajectoryData]:
    """Split complete trajectories into train/validation/test partitions.

    Splitting at trajectory level prevents temporal leakage: no later point of
    a test path can appear as a training sample.
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be below 1")
    n = data.n_trajectories
    n_train = int(np.floor(n * train_fraction))
    n_validation = int(np.floor(n * validation_fraction))
    n_test = n - n_train - n_validation
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError("split fractions leave an empty partition")

    permutation = np.random.default_rng(seed).permutation(n)
    train_index = permutation[:n_train]
    validation_index = permutation[n_train : n_train + n_validation]
    test_index = permutation[n_train + n_validation :]
    return {
        "train": data.take(train_index),
        "validation": data.take(validation_index),
        "test": data.take(test_index),
    }


def _read_required_array(archive: np.lib.npyio.NpzFile, key: str) -> FloatArray:
    if key not in archive.files:
        raise ValueError(f"dataset is missing required key {key!r}")
    return _as_float_array(archive[key], key)


def load_npz_dataset(path: str | Path) -> TrajectoryData:
    """Load a real or synthetic data set using the documented ``.npz`` schema.

    Required keys are ``controls``, ``measurements``, ``states`` and
    ``beacons``.  ``dt`` is optional and defaults to ``1.0`` seconds.  The
    function never guesses column order or splits; mapping legacy data to this
    explicit schema is intentionally left to a visible preprocessing step.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"dataset not found: {source}")
    with np.load(source, allow_pickle=False) as archive:
        controls = _read_required_array(archive, "controls")
        measurements = _read_required_array(archive, "measurements")
        states = _read_required_array(archive, "states")
        beacons = _read_required_array(archive, "beacons")
        dt = float(np.asarray(archive["dt"]).reshape(())) if "dt" in archive.files else 1.0
    return TrajectoryData(
        controls=controls,
        measurements=measurements,
        states=states,
        beacons=beacons,
        dt=dt,
    )


def save_npz_dataset(data: TrajectoryData, path: str | Path) -> Path:
    """Persist a :class:`TrajectoryData` object using the public NPZ schema."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        controls=data.controls,
        measurements=data.measurements,
        states=data.states,
        beacons=data.beacons,
        dt=np.asarray(data.dt, dtype=np.float64),
    )
    return target


__all__ = [
    "TrajectoryData",
    "load_npz_dataset",
    "make_synthetic_underwater_data",
    "save_npz_dataset",
    "split_by_trajectory",
]
