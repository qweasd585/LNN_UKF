"""Classical nonlinear filters used as fair 2-D range-localisation baselines.

The implementations in this module deliberately keep the physical model small
and explicit.  The state is planar position ``x = [east, north]``; a known
odometry/control increment propagates it via ``x[k+1] = x[k] + u[k]``.  The
measurement model contains ranges to fixed acoustic beacons.  This makes the
measurement relation nonlinear while retaining an auditable reference model
for EKF, UKF, and CKF comparisons.

All covariance operations are symmetrised and projected to a small positive
definite cone before a Cholesky factorisation.  This is a numerical safeguard,
not a substitute for reporting consistency metrics such as NIS or NEES.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Array = np.ndarray
FilterKind = Literal["ekf", "ukf", "ckf"]
_EPS = 1.0e-9


@dataclass(frozen=True)
class FilterResult:
    """Posterior trajectory and innovation diagnostics from one filter run.

    Attributes:
        states: Posterior positions, shape ``[T, 2]``.
        covariances: Posterior position covariances, shape ``[T, 2, 2]``.
        innovations: Full range innovations, shape ``[T, M]``.  Entries for a
            missing measurement are ``NaN``.
        nis: Normalised innovation squared at each time.  It is ``NaN`` when
            all measurements at that time are unavailable.
    """

    states: Array
    covariances: Array
    innovations: Array
    nis: Array


def range_observation(state: Array, beacons: Array, *, eps: float = _EPS) -> Array:
    """Return ranges from a planar state to ``beacons``.

    Args:
        state: Position with shape ``[2]``.
        beacons: Beacon coordinates with shape ``[M, 2]``.
        eps: Lower bound used only to avoid a zero derivative at a beacon.
    """

    delta = np.asarray(state, dtype=float)[None, :2] - np.asarray(beacons, dtype=float)
    return np.maximum(np.linalg.norm(delta, axis=1), eps)


def range_jacobian(state: Array, beacons: Array, *, eps: float = _EPS) -> Array:
    """Analytic Jacobian of :func:`range_observation`, shape ``[M, 2]``."""

    delta = np.asarray(state, dtype=float)[None, :2] - np.asarray(beacons, dtype=float)
    denominator = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), eps)
    return delta / denominator


def _spd(matrix: Array, *, floor: float = 1.0e-9) -> Array:
    """Symmetrise ``matrix`` and clip its eigenvalues to a positive floor."""

    matrix = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, floor)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _chol(matrix: Array) -> Array:
    """Robust lower Cholesky factor for a covariance matrix."""

    matrix = _spd(matrix)
    jitter = 0.0
    identity = np.eye(matrix.shape[0])
    for _ in range(7):
        try:
            return np.linalg.cholesky(matrix + jitter * identity)
        except np.linalg.LinAlgError:
            jitter = 1.0e-9 if jitter == 0.0 else jitter * 10.0
    # The eigendecomposition in _spd should make this path exceptionally rare.
    return np.linalg.cholesky(_spd(matrix, floor=max(jitter, 1.0e-6)))


def _covariance(value: float | Array, dimension: int, *, name: str) -> Array:
    """Convert a variance scalar, diagonal vector, or covariance matrix."""

    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        if array < 0:
            raise ValueError(f"{name} must be non-negative")
        return np.eye(dimension) * float(array)
    if array.ndim == 1:
        if array.shape != (dimension,):
            raise ValueError(f"{name} vector must have shape ({dimension},)")
        if np.any(array < 0):
            raise ValueError(f"{name} diagonal entries must be non-negative")
        return np.diag(array)
    if array.shape != (dimension, dimension):
        raise ValueError(f"{name} matrix must have shape ({dimension}, {dimension})")
    return _spd(array)


def _validate_inputs(
    controls: Array,
    measurements: Array,
    beacons: Array,
    initial_state: Array,
) -> tuple[Array, Array, Array, Array]:
    """Validate inputs and normalise controls to one increment per transition."""

    measurements = np.asarray(measurements, dtype=float)
    if measurements.ndim == 1:
        measurements = measurements[:, None]
    if measurements.ndim != 2:
        raise ValueError("measurements must have shape [T, M]")
    time_steps, measurement_dim = measurements.shape
    if time_steps < 1:
        raise ValueError("measurements must contain at least one time step")

    beacons = np.asarray(beacons, dtype=float)
    if beacons.ndim != 2 or beacons.shape != (measurement_dim, 2):
        raise ValueError("beacons must have shape [M, 2] matching measurements")

    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        if controls.size == 0 and time_steps == 1:
            controls = controls.reshape(0, 2)
        elif controls.shape[0] == 2:
            controls = controls.reshape(1, 2)
    if controls.ndim != 2 or controls.shape[1] != 2:
        raise ValueError("controls must have shape [T-1, 2] or [T, 2]")
    required_transitions = time_steps - 1
    if controls.shape[0] < required_transitions:
        raise ValueError("controls has fewer than T-1 increments")
    controls = controls[:required_transitions]

    initial_state = np.asarray(initial_state, dtype=float).reshape(-1)
    if initial_state.shape != (2,):
        raise ValueError("initial_state must contain two coordinates")
    return controls, measurements, beacons, initial_state


def _measurement_update_linearised(
    mean: Array,
    covariance: Array,
    measurement: Array,
    beacons: Array,
    measurement_covariance: Array,
) -> tuple[Array, Array, Array, float]:
    """EKF measurement update with a Joseph covariance form."""

    valid = np.isfinite(measurement)
    innovation_full = np.full(measurement.shape, np.nan, dtype=float)
    if not np.any(valid):
        return mean, _spd(covariance), innovation_full, float("nan")

    expected = range_observation(mean, beacons)
    innovation = measurement[valid] - expected[valid]
    innovation_full[valid] = innovation
    jacobian = range_jacobian(mean, beacons)[valid]
    r_valid = measurement_covariance[np.ix_(valid, valid)]
    innovation_covariance = _spd(jacobian @ covariance @ jacobian.T + r_valid)
    gain = np.linalg.solve(innovation_covariance, jacobian @ covariance).T
    posterior_mean = mean + gain @ innovation
    identity = np.eye(mean.size)
    posterior_covariance = (
        (identity - gain @ jacobian) @ covariance @ (identity - gain @ jacobian).T
        + gain @ r_valid @ gain.T
    )
    nis = float(innovation.T @ np.linalg.solve(innovation_covariance, innovation))
    return posterior_mean, _spd(posterior_covariance), innovation_full, nis


def _ukf_weights(dimension: int, alpha: float, beta: float, kappa: float) -> tuple[float, Array, Array]:
    """Return UKF scale and mean/covariance weights for ``2n+1`` sigma points."""

    if not (alpha > 0.0):
        raise ValueError("alpha must be positive")
    scale = alpha * alpha * (dimension + kappa) - dimension
    spread = dimension + scale
    if spread <= 0.0:
        raise ValueError("alpha and kappa produce non-positive sigma-point spread")
    mean_weights = np.full(2 * dimension + 1, 0.5 / spread)
    covariance_weights = mean_weights.copy()
    mean_weights[0] = scale / spread
    covariance_weights[0] = mean_weights[0] + (1.0 - alpha * alpha + beta)
    return spread, mean_weights, covariance_weights


def _ukf_sigma_points(mean: Array, covariance: Array, spread: float) -> Array:
    """Construct standard symmetric UKF sigma points, shape ``[2n+1, n]``."""

    dimension = mean.size
    root = _chol(spread * covariance)
    points = np.empty((2 * dimension + 1, dimension), dtype=float)
    points[0] = mean
    # Columns of the lower Cholesky factor are covariance principal directions.
    points[1 : dimension + 1] = mean[None, :] + root.T
    points[dimension + 1 :] = mean[None, :] - root.T
    return points


def _cubature_points(mean: Array, covariance: Array) -> Array:
    """Construct spherical-radial cubature points, shape ``[2n, n]``."""

    dimension = mean.size
    root = _chol(covariance)
    offsets = np.sqrt(dimension) * root.T
    return np.concatenate((mean[None, :] + offsets, mean[None, :] - offsets), axis=0)


def _measurement_update_sigma(
    points: Array,
    point_weights: Array,
    covariance_weights: Array,
    mean: Array,
    covariance: Array,
    measurement: Array,
    beacons: Array,
    measurement_covariance: Array,
) -> tuple[Array, Array, Array, float]:
    """Apply a UKF/CKF update from state sigma/cubature points.

    ``point_weights`` and ``covariance_weights`` are separated because UKF
    assigns a special covariance weight to its central sigma point.  For CKF
    both arrays contain the same uniform weights.
    """

    valid = np.isfinite(measurement)
    innovation_full = np.full(measurement.shape, np.nan, dtype=float)
    if not np.any(valid):
        return mean, _spd(covariance), innovation_full, float("nan")

    predicted_measurements = np.stack(
        [range_observation(point, beacons)[valid] for point in points], axis=0
    )
    predicted_measurement_mean = point_weights @ predicted_measurements
    state_deviations = points - mean[None, :]
    measurement_deviations = predicted_measurements - predicted_measurement_mean[None, :]
    r_valid = measurement_covariance[np.ix_(valid, valid)]
    innovation_covariance = np.einsum(
        "i,ij,ik->jk", covariance_weights, measurement_deviations, measurement_deviations
    ) + r_valid
    innovation_covariance = _spd(innovation_covariance)
    cross_covariance = np.einsum(
        "i,ij,ik->jk", covariance_weights, state_deviations, measurement_deviations
    )
    gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
    innovation = measurement[valid] - predicted_measurement_mean
    innovation_full[valid] = innovation
    posterior_mean = mean + gain @ innovation
    posterior_covariance = covariance - gain @ innovation_covariance @ gain.T
    nis = float(innovation.T @ np.linalg.solve(innovation_covariance, innovation))
    return posterior_mean, _spd(posterior_covariance), innovation_full, nis


def _run_ekf(
    controls: Array,
    measurements: Array,
    beacons: Array,
    initial_state: Array,
    q: Array,
    r: Array,
    initial_covariance: Array,
) -> FilterResult:
    """Run the planar-range extended Kalman filter."""

    time_steps, measurement_dim = measurements.shape
    states = np.empty((time_steps, 2), dtype=float)
    covariances = np.empty((time_steps, 2, 2), dtype=float)
    innovations = np.full((time_steps, measurement_dim), np.nan, dtype=float)
    nis = np.full(time_steps, np.nan, dtype=float)
    mean = initial_state.copy()
    covariance = initial_covariance.copy()

    for time_index in range(time_steps):
        if time_index > 0:
            mean = mean + controls[time_index - 1]
            covariance = _spd(covariance + q)
        mean, covariance, innovations[time_index], nis[time_index] = _measurement_update_linearised(
            mean, covariance, measurements[time_index], beacons, r
        )
        states[time_index] = mean
        covariances[time_index] = covariance
    return FilterResult(states, covariances, innovations, nis)


def _run_ukf(
    controls: Array,
    measurements: Array,
    beacons: Array,
    initial_state: Array,
    q: Array,
    r: Array,
    initial_covariance: Array,
    alpha: float,
    beta: float,
    kappa: float,
) -> FilterResult:
    """Run an unscented Kalman filter for the same physical baseline model."""

    state_dimension = initial_state.size
    spread, mean_weights, covariance_weights = _ukf_weights(state_dimension, alpha, beta, kappa)
    time_steps, measurement_dim = measurements.shape
    states = np.empty((time_steps, state_dimension), dtype=float)
    covariances = np.empty((time_steps, state_dimension, state_dimension), dtype=float)
    innovations = np.full((time_steps, measurement_dim), np.nan, dtype=float)
    nis = np.full(time_steps, np.nan, dtype=float)
    mean = initial_state.copy()
    covariance = initial_covariance.copy()

    for time_index in range(time_steps):
        points = _ukf_sigma_points(mean, covariance, spread)
        if time_index > 0:
            propagated = points + controls[time_index - 1][None, :]
            mean = mean_weights @ propagated
            deviations = propagated - mean[None, :]
            covariance = _spd(
                np.einsum("i,ij,ik->jk", covariance_weights, deviations, deviations) + q
            )
            # Reconstruct points around the predicted covariance.  This keeps
            # the measurement update consistent with Q even for nonlinear h.
            points = _ukf_sigma_points(mean, covariance, spread)
        mean, covariance, innovations[time_index], nis[time_index] = _measurement_update_sigma(
            points,
            mean_weights,
            covariance_weights,
            mean,
            covariance,
            measurements[time_index],
            beacons,
            r,
        )
        states[time_index] = mean
        covariances[time_index] = covariance
    return FilterResult(states, covariances, innovations, nis)


def _run_ckf(
    controls: Array,
    measurements: Array,
    beacons: Array,
    initial_state: Array,
    q: Array,
    r: Array,
    initial_covariance: Array,
) -> FilterResult:
    """Run a third-degree spherical-radial cubature Kalman filter."""

    state_dimension = initial_state.size
    point_count = 2 * state_dimension
    weights = np.full(point_count, 1.0 / point_count)
    time_steps, measurement_dim = measurements.shape
    states = np.empty((time_steps, state_dimension), dtype=float)
    covariances = np.empty((time_steps, state_dimension, state_dimension), dtype=float)
    innovations = np.full((time_steps, measurement_dim), np.nan, dtype=float)
    nis = np.full(time_steps, np.nan, dtype=float)
    mean = initial_state.copy()
    covariance = initial_covariance.copy()

    for time_index in range(time_steps):
        points = _cubature_points(mean, covariance)
        if time_index > 0:
            propagated = points + controls[time_index - 1][None, :]
            mean = weights @ propagated
            deviations = propagated - mean[None, :]
            covariance = _spd(np.einsum("i,ij,ik->jk", weights, deviations, deviations) + q)
            points = _cubature_points(mean, covariance)
        mean, covariance, innovations[time_index], nis[time_index] = _measurement_update_sigma(
            points,
            weights,
            weights,
            mean,
            covariance,
            measurements[time_index],
            beacons,
            r,
        )
        states[time_index] = mean
        covariances[time_index] = covariance
    return FilterResult(states, covariances, innovations, nis)


def run_classical_filter(
    kind: FilterKind | str,
    controls: Array,
    measurements: Array,
    beacons: Array,
    initial_state: Array,
    q_scale: float | Array = 0.02,
    r_scale: float | Array = 0.25,
    *,
    initial_covariance: float | Array = 1.0,
    alpha: float = 0.35,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> FilterResult:
    """Run a classical EKF, UKF, or CKF localisation baseline.

    Args:
        kind: One of ``"ekf"``, ``"ukf"``, or ``"ckf"`` (case-insensitive).
        controls: Known displacement increments, shape ``[T-1, 2]``.  A
            ``[T, 2]`` array is also accepted and its final unused row ignored.
        measurements: Beacon-range observations, shape ``[T, M]``; use ``NaN``
            for a dropped acoustic measurement.
        beacons: Fixed beacon locations, shape ``[M, 2]``.
        initial_state: Initial position at time zero, shape ``[2]``.
        q_scale: Process-noise covariance (scalar variance, diagonal, or 2x2).
        r_scale: Range-noise covariance (scalar variance, diagonal, or MxM).
        initial_covariance: Initial position covariance in the same forms as
            ``q_scale``.
        alpha, beta, kappa: Standard scaled-UKF sigma-point parameters.  They
            are ignored by EKF and CKF.

    Returns:
        A :class:`FilterResult` containing posterior estimates and diagnostics.
    """

    controls, measurements, beacons, initial_state = _validate_inputs(
        controls, measurements, beacons, initial_state
    )
    q = _covariance(q_scale, 2, name="q_scale")
    r = _covariance(r_scale, measurements.shape[1], name="r_scale")
    p0 = _covariance(initial_covariance, 2, name="initial_covariance")
    normalised_kind = str(kind).lower()
    if normalised_kind == "ekf":
        return _run_ekf(controls, measurements, beacons, initial_state, q, r, p0)
    if normalised_kind == "ukf":
        return _run_ukf(
            controls,
            measurements,
            beacons,
            initial_state,
            q,
            r,
            p0,
            alpha,
            beta,
            kappa,
        )
    if normalised_kind == "ckf":
        return _run_ckf(controls, measurements, beacons, initial_state, q, r, p0)
    raise ValueError("kind must be one of: ekf, ukf, ckf")
