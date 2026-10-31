"""Metrics and visual diagnostics for two-dimensional trajectory experiments.

The functions in this module deliberately depend only on arrays.  This keeps the
experimental claims auditable: metrics, uncertainty checks and interpretability
plots can be regenerated from saved ``.npz`` results without importing a model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

# Managed runners commonly expose a read-only home directory.  Keep Matplotlib's
# cache out of the checkout while retaining a usable non-interactive backend.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lnn_ukf_matplotlib")

import matplotlib

# The cloud runner has no display.  Selecting the backend before importing
# pyplot makes every plotting function usable in scripts and CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr


Array = np.ndarray


def _batched_trajectory(values: Array, name: str) -> Array:
    """Return a trajectory as ``[N, T, D]`` and validate its rank."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [T, D] or [N, T, D], got {array.shape}")
    return array


def _batched_covariance(values: Array, name: str) -> Array:
    """Return covariance matrices as ``[N, T, D, D]``."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[-1] != array.shape[-2]:
        raise ValueError(
            f"{name} must have shape [T, D, D] or [N, T, D, D], got {array.shape}"
        )
    return array


def _ensure_matching_trajectories(reference: Array, estimate: Array) -> tuple[Array, Array]:
    reference_b = _batched_trajectory(reference, "reference")
    estimate_b = _batched_trajectory(estimate, "estimate")
    if reference_b.shape != estimate_b.shape:
        raise ValueError(
            "reference and estimate must have identical [N, T, D] shapes; "
            f"got {reference_b.shape} and {estimate_b.shape}"
        )
    return reference_b, estimate_b


def trajectory_metrics(estimate: Array, truth: Array) -> dict[str, float]:
    """Compute standard trajectory-position errors.

    ``rmse`` and ``mae`` aggregate coordinate errors.  ``ade`` (average
    displacement error) and ``fde`` (final displacement error) aggregate
    Euclidean position errors, which is usually more interpretable for AUV
    navigation.
    """
    truth_b, estimate_b = _ensure_matching_trajectories(truth, estimate)
    error = estimate_b - truth_b
    point_error = np.linalg.norm(error, axis=-1)
    finite = np.isfinite(error).all(axis=-1)
    if not np.any(finite):
        return {"rmse": np.nan, "mae": np.nan, "ade": np.nan, "fde": np.nan}
    coordinate_error = error[finite]
    final_error = point_error[:, -1]
    return {
        "rmse": float(np.sqrt(np.mean(coordinate_error**2))),
        "mae": float(np.mean(np.abs(coordinate_error))),
        "ade": float(np.mean(point_error[finite])),
        "fde": float(np.nanmean(final_error)),
    }


def per_trajectory_metrics(
    predictions: Mapping[str, Array], truth: Array
) -> pd.DataFrame:
    """Return a tidy table with one row per model and full trajectory.

    Parameters
    ----------
    predictions:
        Mapping from model name to an estimate shaped ``[N, T, 2]`` or
        ``[T, 2]``.
    truth:
        Reference position with the corresponding shape.
    """
    truth_b = _batched_trajectory(truth, "truth")
    rows: list[dict[str, float | int | str]] = []
    for model, prediction in predictions.items():
        prediction_b = _batched_trajectory(prediction, f"predictions[{model!r}]")
        if prediction_b.shape != truth_b.shape:
            raise ValueError(
                f"predictions[{model!r}] shape {prediction_b.shape} does not match "
                f"truth shape {truth_b.shape}"
            )
        for trajectory_id in range(truth_b.shape[0]):
            row: dict[str, float | int | str] = {
                "model": str(model),
                "trajectory_id": trajectory_id,
            }
            row.update(trajectory_metrics(prediction_b[trajectory_id], truth_b[trajectory_id]))
            rows.append(row)
    return pd.DataFrame(rows, columns=["model", "trajectory_id", "rmse", "mae", "ade", "fde"])


def mahalanobis_squared(errors: Array, covariances: Array) -> Array:
    """Return per-step squared Mahalanobis distances with PSD-safe inverses."""
    error_b = _batched_trajectory(errors, "errors")
    covariance_b = _batched_covariance(covariances, "covariances")
    if error_b.shape[:2] != covariance_b.shape[:2] or error_b.shape[-1] != covariance_b.shape[-1]:
        raise ValueError(
            "errors and covariances disagree in trajectory, time, or state dimensions: "
            f"{error_b.shape} versus {covariance_b.shape}"
        )
    output = np.full(error_b.shape[:2], np.nan, dtype=float)
    for batch_index in range(error_b.shape[0]):
        for time_index in range(error_b.shape[1]):
            error = error_b[batch_index, time_index]
            covariance = covariance_b[batch_index, time_index]
            if not (np.isfinite(error).all() and np.isfinite(covariance).all()):
                continue
            covariance = 0.5 * (covariance + covariance.T)
            output[batch_index, time_index] = float(error @ np.linalg.pinv(covariance) @ error)
    return output[0] if np.asarray(errors).ndim == 2 else output


def nees(errors: Array, covariances: Array) -> Array:
    """Compute Normalized Estimation Error Squared (NEES) at each time step."""
    return mahalanobis_squared(errors, covariances)


def nis(innovations: Array, innovation_covariances: Array) -> Array:
    """Compute Normalized Innovation Squared (NIS) at each time step."""
    return mahalanobis_squared(innovations, innovation_covariances)


def elliptical_coverage(errors: Array, covariances: Array, level: float = 0.95) -> float:
    """Fraction of errors inside the stated Gaussian covariance ellipse."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    error_b = _batched_trajectory(errors, "errors")
    distances = np.asarray(mahalanobis_squared(errors, covariances), dtype=float)
    threshold = chi2.ppf(level, df=error_b.shape[-1])
    valid = np.isfinite(distances)
    return float(np.mean(distances[valid] <= threshold)) if np.any(valid) else np.nan


def consistency_summary(
    errors: Array,
    covariances: Array,
    innovations: Array | None = None,
    innovation_covariances: Array | None = None,
    level: float = 0.95,
) -> dict[str, float]:
    """Summarize NEES/NIS and empirical confidence coverage."""
    error_b = _batched_trajectory(errors, "errors")
    nees_values = np.asarray(nees(errors, covariances), dtype=float)
    summary: dict[str, float] = {
        "mean_nees": float(np.nanmean(nees_values)),
        "expected_nees": float(error_b.shape[-1]),
        f"coverage_{int(100 * level)}": elliptical_coverage(errors, covariances, level),
    }
    if innovations is not None or innovation_covariances is not None:
        if innovations is None or innovation_covariances is None:
            raise ValueError("innovations and innovation_covariances must be supplied together")
        innovation_b = _batched_trajectory(innovations, "innovations")
        nis_values = np.asarray(nis(innovations, innovation_covariances), dtype=float)
        summary["mean_nis"] = float(np.nanmean(nis_values))
        summary["expected_nis"] = float(innovation_b.shape[-1])
    return summary


def time_aligned_overlap(reference: Array, counterfactual: Array, epsilon: float) -> float:
    """Return the fraction of aligned positions closer than ``epsilon`` metres."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    reference_b, counterfactual_b = _ensure_matching_trajectories(reference, counterfactual)
    distance = np.linalg.norm(reference_b - counterfactual_b, axis=-1)
    valid = np.isfinite(distance)
    return float(np.mean(distance[valid] <= epsilon)) if np.any(valid) else np.nan


def occupancy_iou(reference: Array, counterfactual: Array, grid_size: float = 1.0) -> float:
    """Compare spatial paths by intersection-over-union of visited grid cells."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    reference_b, counterfactual_b = _ensure_matching_trajectories(reference, counterfactual)
    ref_points = reference_b.reshape(-1, reference_b.shape[-1])
    cf_points = counterfactual_b.reshape(-1, counterfactual_b.shape[-1])
    ref_points = ref_points[np.isfinite(ref_points).all(axis=1)]
    cf_points = cf_points[np.isfinite(cf_points).all(axis=1)]
    if len(ref_points) == 0 or len(cf_points) == 0:
        return np.nan
    ref_cells = {tuple(cell) for cell in np.floor(ref_points / grid_size).astype(int)}
    cf_cells = {tuple(cell) for cell in np.floor(cf_points / grid_size).astype(int)}
    return len(ref_cells & cf_cells) / len(ref_cells | cf_cells)


def _finite_spearman(x: Array, y: Array) -> tuple[float, float]:
    """Compute Spearman rho without issuing warnings for constant series."""
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid, y_valid = x[valid], y[valid]
    if x_valid.size < 3 or np.ptp(x_valid) <= 1.0e-12 or np.ptp(y_valid) <= 1.0e-12:
        return np.nan, np.nan
    correlation, p_value = spearmanr(x_valid, y_valid)
    return float(correlation), float(p_value)


def vector_feature_spearman_correlation(
    vector: Array,
    features: Array,
    feature_names: Sequence[str],
    vector_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the pairwise Spearman matrix between vector components and inputs.

    Spearman correlation is defined for two scalar series, not directly for two
    multivariate vectors.  This function therefore retains every coordinate of
    ``vector`` and every input feature, and returns their full pairwise matrix
    instead of first averaging the liquid-time-constant vector.
    """
    vector_array = np.asarray(vector, dtype=float)
    feature_array = np.asarray(features, dtype=float)
    if vector_array.ndim == 2:
        vector_array = vector_array[None, ...]
    if feature_array.ndim == 2:
        feature_array = feature_array[None, ...]
    if vector_array.ndim != 3 or feature_array.ndim != 3:
        raise ValueError("vector and features must have shape [T, D] or [N, T, D]")
    if vector_array.shape[:2] != feature_array.shape[:2]:
        raise ValueError("vector and features must have matching [N, T] prefixes")
    if len(feature_names) != feature_array.shape[-1]:
        raise ValueError("feature_names length must match the final features dimension")
    names = list(vector_names) if vector_names is not None else [f"component_{i}" for i in range(vector_array.shape[-1])]
    if len(names) != vector_array.shape[-1]:
        raise ValueError("vector_names length must match the final vector dimension")

    rows: list[dict[str, float | int | str]] = []
    for component_index, component_name in enumerate(names):
        x = vector_array[..., component_index].reshape(-1)
        for feature_index, feature_name in enumerate(feature_names):
            y = feature_array[..., feature_index].reshape(-1)
            valid = np.isfinite(x) & np.isfinite(y)
            correlation, p_value = _finite_spearman(x, y)
            rows.append(
                {
                    "vector_component": str(component_name),
                    "input_feature": str(feature_name),
                    "spearman_rho": float(correlation),
                    "p_value": float(p_value),
                    "n": int(valid.sum()),
                }
            )
    return pd.DataFrame(rows)


def vector_change_comparison(vector: Array, features: Array) -> pd.DataFrame:
    """Measure aligned changes of a state vector and standardized input vector.

    The input coordinates are standardized over all supplied trajectories before
    their Euclidean change is formed, so mixed-unit features such as position,
    speed, and heading do not let the largest numerical scale dominate.
    """
    vector_array = np.asarray(vector, dtype=float)
    feature_array = np.asarray(features, dtype=float)
    if vector_array.ndim == 2:
        vector_array = vector_array[None, ...]
    if feature_array.ndim == 2:
        feature_array = feature_array[None, ...]
    if vector_array.ndim != 3 or feature_array.ndim != 3:
        raise ValueError("vector and features must have shape [T, D] or [N, T, D]")
    if vector_array.shape[:2] != feature_array.shape[:2]:
        raise ValueError("vector and features must have matching [N, T] prefixes")

    feature_flat = feature_array.reshape(-1, feature_array.shape[-1])
    means = np.nanmean(feature_flat, axis=0)
    scales = np.nanstd(feature_flat, axis=0)
    scales = np.where(scales > 1.0e-12, scales, 1.0)
    standardized_features = (feature_array - means) / scales
    tau_change = np.linalg.norm(np.diff(vector_array, axis=1), axis=-1)
    input_change = np.linalg.norm(np.diff(standardized_features, axis=1), axis=-1)

    finite_tau = tau_change[np.isfinite(tau_change)]
    finite_input = input_change[np.isfinite(input_change)]
    tau_mean, tau_scale = (
        (float(np.mean(finite_tau)), max(float(np.std(finite_tau)), 1.0e-12))
        if finite_tau.size
        else (0.0, 1.0)
    )
    input_mean, input_scale = (
        (float(np.mean(finite_input)), max(float(np.std(finite_input)), 1.0e-12))
        if finite_input.size
        else (0.0, 1.0)
    )
    trajectories, time_steps = tau_change.shape
    return pd.DataFrame(
        {
            "trajectory_index": np.repeat(np.arange(trajectories), time_steps),
            "time_index": np.tile(np.arange(1, time_steps + 1), trajectories),
            "tau_vector_change_l2": tau_change.reshape(-1),
            "input_vector_change_standardized_l2": input_change.reshape(-1),
            "tau_vector_change_zscore": ((tau_change - tau_mean) / tau_scale).reshape(-1),
            "input_vector_change_zscore": ((input_change - input_mean) / input_scale).reshape(-1),
        }
    )


def tau_intervention_summary(
    baseline: Array,
    counterfactuals: Mapping[float, Array],
    truth: Array | None = None,
    epsilon: float = 1.0,
    grid_size: float = 1.0,
) -> pd.DataFrame:
    """Quantify trajectory changes caused by prescribed liquid-time interventions.

    The key ``scale`` convention is intentionally left to the caller (for
    example 0.5, 1.0, 1.5 for multiplicative interventions).  The returned
    table separates two questions: how an intervention changes the full
    dynamic-tau prediction, and (when ``truth`` is given) how accurately each
    intervened prediction follows the true trajectory.
    """
    baseline_b = _batched_trajectory(baseline, "baseline")
    truth_b = None
    if truth is not None:
        truth_b = _batched_trajectory(truth, "truth")
        if truth_b.shape != baseline_b.shape:
            raise ValueError(
                f"truth shape {truth_b.shape} does not match baseline shape {baseline_b.shape}"
            )
    rows: list[dict[str, float]] = []
    for scale, trajectory in sorted(counterfactuals.items(), key=lambda item: item[0]):
        trajectory_b = _batched_trajectory(trajectory, f"counterfactuals[{scale}]")
        if trajectory_b.shape != baseline_b.shape:
            raise ValueError(
                f"counterfactuals[{scale}] shape {trajectory_b.shape} does not match {baseline_b.shape}"
            )
        displacement = np.linalg.norm(trajectory_b - baseline_b, axis=-1)
        row: dict[str, float] = {
            "tau_scale": float(scale),
            "mean_displacement_from_dynamic_tau": float(np.nanmean(displacement)),
            "final_displacement_from_dynamic_tau": float(np.nanmean(displacement[:, -1])),
            "time_aligned_overlap_with_dynamic_tau": time_aligned_overlap(
                baseline_b, trajectory_b, epsilon
            ),
            "occupancy_iou_with_dynamic_tau": occupancy_iou(
                baseline_b, trajectory_b, grid_size
            ),
        }
        if truth_b is not None:
            for metric, value in trajectory_metrics(trajectory_b, truth_b).items():
                row[f"{metric}_to_truth"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_path(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_trajectory_comparison(
    truth: Array,
    predictions: Mapping[str, Array],
    path: str | Path,
    beacons: Array | None = None,
    title: str = "Trajectory comparison",
) -> Path:
    """Save a two-dimensional trajectory overlay, using the first trajectory."""
    truth_b = _batched_trajectory(truth, "truth")
    output = _prepare_path(path)
    figure, axis = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    axis.plot(truth_b[0, :, 0], truth_b[0, :, 1], color="black", linewidth=2.8, label="Truth")
    for model, prediction in predictions.items():
        prediction_b = _batched_trajectory(prediction, f"predictions[{model!r}]")
        if prediction_b.shape != truth_b.shape:
            raise ValueError(f"prediction shape for {model!r} does not match truth")
        axis.plot(prediction_b[0, :, 0], prediction_b[0, :, 1], linewidth=1.6, label=str(model))
    if beacons is not None:
        beacon_array = np.asarray(beacons, dtype=float)
        if beacon_array.ndim != 2 or beacon_array.shape[1] != 2:
            raise ValueError("beacons must have shape [M, 2]")
        axis.scatter(beacon_array[:, 0], beacon_array[:, 1], marker="^", s=80, color="tab:orange", label="Beacon")
    axis.set(title=title, xlabel="East / m", ylabel="North / m")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_trajectory_comparison_grid(
    truth: Array,
    predictions: Mapping[str, Array],
    path: str | Path,
    trajectory_indices: Sequence[int] | None = None,
    title: str = "Six-trajectory model comparison",
) -> Path:
    """Plot all models for six locally cropped trajectories without beacons.

    Every panel computes its own bounds from the truth and all model outputs.
    This avoids an otherwise distant beacon layout making the path comparison
    unnecessarily small.
    """
    truth_b = _batched_trajectory(truth, "truth")
    prepared_predictions: dict[str, Array] = {}
    for model, prediction in predictions.items():
        prediction_b = _batched_trajectory(prediction, f"predictions[{model!r}]")
        if prediction_b.shape != truth_b.shape:
            raise ValueError(f"prediction shape for {model!r} does not match truth")
        prepared_predictions[str(model)] = prediction_b
    if truth_b.shape[0] < 1:
        raise ValueError("at least one trajectory is required")
    if trajectory_indices is None:
        count = min(6, truth_b.shape[0])
        trajectory_indices = np.linspace(0, truth_b.shape[0] - 1, count, dtype=int).tolist()
    indices = list(trajectory_indices)
    if not indices or len(indices) > 6:
        raise ValueError("trajectory_indices must contain between one and six trajectories")
    if min(indices) < 0 or max(indices) >= truth_b.shape[0]:
        raise ValueError("trajectory index is out of bounds")

    styles = {
        "EKF": "#1f77b4",
        "UKF": "#ff7f0e",
        "CKF": "#2ca02c",
        "LNN": "#9467bd",
        "LSTM": "#e377c2",
        "CTGRU": "#8c564b",
        "CTRNN": "#17becf",
        "LNN+UKF": "#d62728",
    }
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 9.3))
    axes_flat = axes.ravel()
    handles: list[plt.Line2D] = []
    labels: list[str] = []
    for panel, trajectory_index in enumerate(indices):
        axis = axes_flat[panel]
        all_points = [truth_b[trajectory_index]] + [
            prediction[trajectory_index] for prediction in prepared_predictions.values()
        ]
        finite_points = np.concatenate(all_points, axis=0)
        finite_points = finite_points[np.isfinite(finite_points).all(axis=1)]
        lower = finite_points.min(axis=0)
        upper = finite_points.max(axis=0)
        span = float(np.max(upper - lower))
        padding = max(0.45, 0.07 * span)
        truth_line, = axis.plot(
            truth_b[trajectory_index, :, 0],
            truth_b[trajectory_index, :, 1],
            color="#111111",
            linewidth=2.4,
            zorder=4,
            label="Truth",
        )
        if panel == 0:
            handles.append(truth_line)
            labels.append("Truth")
        for model, prediction in prepared_predictions.items():
            line, = axis.plot(
                prediction[trajectory_index, :, 0],
                prediction[trajectory_index, :, 1],
                color=styles.get(model, "#555555"),
                linewidth=2.0 if model == "LNN+UKF" else 1.15,
                alpha=0.96,
                zorder=3 if model == "LNN+UKF" else 2,
                label=model,
            )
            if panel == 0:
                handles.append(line)
                labels.append(model)
        centre = (lower + upper) / 2.0
        half_width = max(0.5 * span + padding, 1.0)
        axis.set_xlim(centre[0] - half_width, centre[0] + half_width)
        axis.set_ylim(centre[1] - half_width, centre[1] + half_width)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"Trajectory {trajectory_index + 1}", fontsize=10)
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.tick_params(labelsize=8)
        if panel % 3 == 0:
            axis.set_ylabel("North / m", fontsize=9)
        if panel >= 3:
            axis.set_xlabel("East / m", fontsize=9)
    for axis in axes_flat[len(indices) :]:
        axis.set_visible(False)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    figure.suptitle(title, fontsize=13, y=0.99)
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.87, wspace=0.20, hspace=0.28)
    figure.savefig(_prepare_path(path), dpi=200, bbox_inches="tight")
    plt.close(figure)
    return Path(path)


def plot_tau_input_diagnostics(
    correlations: pd.DataFrame,
    change_comparison: pd.DataFrame,
    path: str | Path,
    title: str = "Liquid-time vector and input diagnostics",
) -> Path:
    """Plot the full tau-input Spearman matrix and aligned vector changes."""
    required_correlation = {"vector_component", "input_feature", "spearman_rho"}
    required_change = {
        "time_index",
        "tau_vector_change_zscore",
        "input_vector_change_zscore",
    }
    if missing := required_correlation - set(correlations.columns):
        raise ValueError(f"correlations is missing required columns: {sorted(missing)}")
    if missing := required_change - set(change_comparison.columns):
        raise ValueError(f"change_comparison is missing required columns: {sorted(missing)}")
    output = _prepare_path(path)
    components = list(dict.fromkeys(correlations["vector_component"].astype(str)))
    features = list(dict.fromkeys(correlations["input_feature"].astype(str)))
    matrix = (
        correlations.pivot(index="vector_component", columns="input_feature", values="spearman_rho")
        .reindex(index=components, columns=features)
        .to_numpy(dtype=float)
    )
    figure, (heatmap_axis, change_axis, scatter_axis) = plt.subplots(
        1, 3, figsize=(16.0, 4.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]}
    )
    image = heatmap_axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    heatmap_axis.set_xticks(np.arange(len(features)), features, rotation=35, ha="right", fontsize=8)
    heatmap_axis.set_yticks(np.arange(len(components)), components, fontsize=8)
    heatmap_axis.set(title="Pairwise Spearman: tau vector × inputs")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                heatmap_axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > 0.52 else "black",
                )
    figure.colorbar(image, ax=heatmap_axis, fraction=0.047, pad=0.04, label="Spearman rho")

    change_by_time = change_comparison.groupby("time_index", sort=True)[
        ["tau_vector_change_zscore", "input_vector_change_zscore"]
    ].mean()
    change_axis.plot(
        change_by_time.index,
        change_by_time["tau_vector_change_zscore"],
        color="#d62728",
        linewidth=2.0,
        label=r"$\|\Delta\tau\|_2$",
    )
    change_axis.plot(
        change_by_time.index,
        change_by_time["input_vector_change_zscore"],
        color="#1f77b4",
        linewidth=2.0,
        label=r"$\|\Delta z(u)\|_2$",
    )
    change_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.45)
    change_axis.set(title="Mean aligned changes", xlabel="Time step", ylabel="z-score")
    change_axis.grid(alpha=0.22)
    change_axis.legend(fontsize=8)

    tau_change = change_comparison["tau_vector_change_zscore"].to_numpy(dtype=float)
    input_change = change_comparison["input_vector_change_zscore"].to_numpy(dtype=float)
    valid = np.isfinite(tau_change) & np.isfinite(input_change)
    rho, _ = _finite_spearman(tau_change, input_change)
    scatter_axis.scatter(input_change[valid], tau_change[valid], s=13, alpha=0.42, color="#5b5b5b", edgecolors="none")
    scatter_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
    scatter_axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.35)
    scatter_axis.set(
        title=f"Change association (rho={rho:.2f})",
        xlabel=r"$\|\Delta z(u)\|_2$ z-score",
        ylabel=r"$\|\Delta\tau\|_2$ z-score",
    )
    scatter_axis.grid(alpha=0.18)
    figure.suptitle(title, fontsize=13)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_metrics_table(metrics: pd.DataFrame, path: str | Path, title: str = "Trajectory metrics") -> Path:
    """Save a compact model-comparison bar chart with mean ± standard error."""
    required = {"model", "rmse", "ade", "fde"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"metrics is missing required columns: {sorted(missing)}")
    output = _prepare_path(path)
    grouped = metrics.groupby("model", sort=False)[["rmse", "ade", "fde"]].agg(["mean", "sem"])
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, metric in zip(axes, ("rmse", "ade", "fde"), strict=True):
        means = grouped[(metric, "mean")]
        errors = grouped[(metric, "sem")].fillna(0.0)
        axis.bar(np.arange(len(means)), means, yerr=errors, capsize=3, color="tab:blue", alpha=0.8)
        axis.set_xticks(np.arange(len(means)), means.index, rotation=35, ha="right")
        axis.set_ylabel("metres")
        axis.set_title(metric.upper())
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(title)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_tau_trajectory_panel(
    tau: Array,
    baseline: Array,
    path: str | Path,
    counterfactuals: Mapping[float, Array] | None = None,
    truth: Array | None = None,
    max_units: int = 8,
    title: str = "Liquid time constants and counterfactual trajectories",
) -> Path:
    """Save a tau trace and corresponding reference/counterfactual paths."""
    tau_array = np.asarray(tau, dtype=float)
    if tau_array.ndim == 3:
        tau_array = tau_array[0]
    if tau_array.ndim != 2:
        raise ValueError("tau must have shape [T, H] or [N, T, H]")
    baseline_b = _batched_trajectory(baseline, "baseline")
    if tau_array.shape[0] != baseline_b.shape[1]:
        raise ValueError("tau time axis must equal trajectory length")
    output = _prepare_path(path)
    figure, (tau_axis, trajectory_axis) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for hidden_unit in range(min(max_units, tau_array.shape[1])):
        tau_axis.plot(tau_array[:, hidden_unit], linewidth=1.0, label=fr"$\tau_{{{hidden_unit}}}$")
    tau_axis.set(title="Effective liquid time constants", xlabel="Time step", ylabel=r"$\tau$")
    tau_axis.grid(alpha=0.25)
    if tau_array.shape[1] <= max_units:
        tau_axis.legend(ncol=2, fontsize=7)
    if truth is not None:
        truth_b, _ = _ensure_matching_trajectories(truth, baseline_b)
        trajectory_axis.plot(truth_b[0, :, 0], truth_b[0, :, 1], "k-", lw=2.5, label="Truth")
    trajectory_axis.plot(baseline_b[0, :, 0], baseline_b[0, :, 1], lw=2.0, label="Dynamic tau")
    for scale, trajectory in (counterfactuals or {}).items():
        trajectory_b = _batched_trajectory(trajectory, f"counterfactuals[{scale}]")
        if trajectory_b.shape != baseline_b.shape:
            raise ValueError(f"counterfactual shape for {scale} does not match baseline")
        trajectory_axis.plot(trajectory_b[0, :, 0], trajectory_b[0, :, 1], lw=1.2, label=fr"$\tau\times {scale:g}$")
    trajectory_axis.set(title="Tau intervention trajectories", xlabel="East / m", ylabel="North / m")
    trajectory_axis.axis("equal")
    trajectory_axis.grid(alpha=0.25)
    trajectory_axis.legend(fontsize=8)
    figure.suptitle(title)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_covariance_calibration(
    nees_values: Array,
    nis_values: Array | None,
    path: str | Path,
    state_dimension: int,
    measurement_dimension: int | None = None,
    title: str = "Uncertainty calibration",
) -> Path:
    """Save NEES/NIS histograms with their chi-square reference means."""
    if state_dimension < 1:
        raise ValueError("state_dimension must be positive")
    output = _prepare_path(path)
    panels = 2 if nis_values is not None else 1
    figure, axes = plt.subplots(1, panels, figsize=(5.6 * panels, 3.8), constrained_layout=True)
    axes_array = np.atleast_1d(axes)

    def _panel(axis: plt.Axes, values: Array, dimension: int, label: str) -> None:
        finite = np.asarray(values, dtype=float).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            axis.hist(finite, bins=min(30, max(8, int(np.sqrt(finite.size)))), density=True, alpha=0.7, label=label)
            upper = max(float(np.max(finite)), float(chi2.ppf(0.995, dimension)))
        else:
            upper = float(chi2.ppf(0.995, dimension))
        grid = np.linspace(0.0, upper, 300)
        axis.plot(grid, chi2.pdf(grid, dimension), "k--", label=fr"$\chi^2({dimension})$")
        axis.axvline(dimension, color="tab:red", linestyle=":", label=f"expected mean = {dimension}")
        axis.set(title=label, xlabel=label, ylabel="Density")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)

    _panel(axes_array[0], nees_values, state_dimension, "NEES")
    if nis_values is not None:
        if measurement_dimension is None or measurement_dimension < 1:
            raise ValueError("measurement_dimension must be supplied for NIS")
        _panel(axes_array[1], nis_values, measurement_dimension, "NIS")
    figure.suptitle(title)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_noise_adaptation(
    process_covariances: Array,
    measurement_covariances: Array,
    innovations: Array,
    path: str | Path,
    title: str = "Causal adaptive Q/R diagnostics",
) -> Path:
    """Plot predicted covariance traces beside the realised innovation norm.

    The curves are averaged over trajectories.  They are a diagnostic for the
    *meaning* of dynamic Q/R, rather than evidence that a learned covariance is
    calibrated by itself; NIS/NEES must still be reported separately.
    """
    process = np.asarray(process_covariances, dtype=float)
    measurement = np.asarray(measurement_covariances, dtype=float)
    innovation = np.asarray(innovations, dtype=float)
    if process.ndim == 3:
        process = process[None, ...]
    if measurement.ndim == 3:
        measurement = measurement[None, ...]
    if innovation.ndim == 2:
        innovation = innovation[None, ...]
    if process.ndim != 4 or measurement.ndim != 4 or innovation.ndim != 3:
        raise ValueError("process/R covariances and innovations must be batched matrices/vectors")
    if process.shape[0] != measurement.shape[0] or measurement.shape[:2] != innovation.shape[:2]:
        raise ValueError("noise diagnostics must agree in batch and compatible time axes")
    output = _prepare_path(path)
    process_trace = np.trace(process, axis1=-2, axis2=-1).mean(axis=0)
    measurement_trace = np.trace(measurement, axis1=-2, axis2=-1).mean(axis=0)
    innovation_norm = np.linalg.norm(innovation, axis=-1)
    innovation_norm = np.nanmean(innovation_norm, axis=0)
    figure, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(np.arange(1, len(process_trace) + 1), process_trace, color="tab:blue")
    axes[0].set(ylabel="tr($Q_t$)", title="Process-noise trace")
    axes[1].plot(np.arange(len(measurement_trace)), measurement_trace, color="tab:orange")
    axes[1].set(ylabel="tr($R_t$)", title="Measurement-noise trace")
    axes[2].plot(np.arange(len(innovation_norm)), innovation_norm, color="tab:red")
    axes[2].set(xlabel="Time step", ylabel=r"mean $\|\nu_t\|_2$", title="Observed innovation magnitude")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(title)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_readout_contributions(
    contributions: Array,
    path: str | Path,
    title: str = "LNN internal-unit displacement contributions",
) -> Path:
    """Save a heatmap of each hidden unit's physical displacement contribution.

    ``contributions`` has shape ``[N, T, H, D]`` or ``[T, H, D]`` and is
    expected to contain ``A[:, i] h_i(t)``.  The plotted magnitude is an
    interpretable aggregate across physical coordinate components.
    """
    values = np.asarray(contributions, dtype=float)
    if values.ndim == 3:
        values = values[None, ...]
    if values.ndim != 4:
        raise ValueError("contributions must have shape [T,H,D] or [N,T,H,D]")
    magnitude = np.linalg.norm(values[0], axis=-1).T
    output = _prepare_path(path)
    figure, axis = plt.subplots(figsize=(10, max(3.0, 0.5 * magnitude.shape[0] + 1)), constrained_layout=True)
    image = axis.imshow(magnitude, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
    axis.set(xlabel="Time step", ylabel="Hidden unit", title=title)
    figure.colorbar(image, ax=axis, label=r"$\|A_{:,i}h_i(t)\|_2$")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output

