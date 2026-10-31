"""End-to-end benchmark, ablation, and interpretability experiment runner.

The runner uses the same trajectory-level split for every model.  It produces
numerical tables and figures from saved arrays; it does not manufacture claims
about an unavailable real data set.  With no ``--data`` argument it creates a
clearly labelled synthetic underwater benchmark so the full implementation can
be executed and audited immediately.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .analysis import (
    consistency_summary,
    lagged_vector_feature_spearman,
    nees,
    per_trajectory_metrics,
    plot_covariance_calibration,
    plot_metrics_table,
    plot_noise_adaptation,
    plot_readout_contributions,
    plot_readout_role_consistency,
    plot_tau_input_diagnostics,
    plot_tau_input_lag_diagnostics,
    plot_tau_trajectory_panel,
    plot_trajectory_comparison,
    plot_trajectory_comparison_grid,
    tau_intervention_summary,
    trajectory_metrics,
    vector_change_comparison,
    vector_feature_spearman_correlation,
    summarize_lagged_spearman,
    summarize_readout_roles,
)
from .data import TrajectoryData, load_npz_dataset, make_synthetic_underwater_data, save_npz_dataset, split_by_trajectory
from .filters import FilterResult, run_classical_filter
from .lnn_ukf import LNNUKF, LNNUKFConfig
from .networks import RecurrentModelConfig, RecurrentPositionRegressor, count_trainable_parameters
from .training import (
    TrainingConfig,
    predict_direct,
    predict_lnn_ukf,
    set_reproducible_seed,
    train_direct_model,
    train_lnn_ukf,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """All choices that affect an experimental result."""

    output_dir: str = "results"
    data_path: str | None = None
    seeds: tuple[int, ...] = (7,)
    data_seed: int = 101
    synthetic_trajectories: int = 72
    synthetic_steps: int = 72
    hidden_dim: int = 8
    epochs: int = 24
    ablation_epochs: int = 14
    batch_size: int = 12
    learning_rate: float = 2.0e-3
    device: str = "cpu"
    run_ablations: bool = True
    save_models: bool = True
    tau_overlap_epsilon: float = 0.10
    tau_grid_size: float = 0.50
    tau_lag_max_steps: int = 8
    filter_q_candidates: tuple[float, ...] = (0.0025, 0.01, 0.025, 0.05, 0.10)
    filter_r_candidates: tuple[float, ...] = (0.04, 0.09, 0.16, 0.25, 0.49)

    @classmethod
    def fast(cls, output_dir: str = "results") -> "ExperimentConfig":
        """Return a short smoke/illustration configuration for CPU execution."""
        return cls(
            output_dir=output_dir,
            synthetic_trajectories=30,
            synthetic_steps=40,
            hidden_dim=4,
            epochs=5,
            ablation_epochs=3,
            batch_size=6,
            run_ablations=True,
        )


@dataclass
class SeedRun:
    """In-memory outputs needed to aggregate one random-seed run."""

    metrics: pd.DataFrame
    ablation_metrics: pd.DataFrame
    calibration: pd.DataFrame
    filter_tuning: pd.DataFrame
    full_model: LNNUKF
    full_prediction: dict[str, np.ndarray]
    test_data: TrajectoryData
    main_predictions: dict[str, np.ndarray]
    parameter_counts: dict[str, int]


def _source_data(config: ExperimentConfig) -> tuple[TrajectoryData, str]:
    if config.data_path is not None:
        return load_npz_dataset(config.data_path), f"NPZ data: {Path(config.data_path).resolve()}"
    data = make_synthetic_underwater_data(
        n_trajectories=config.synthetic_trajectories,
        n_steps=config.synthetic_steps,
        seed=config.data_seed,
    )
    return data, "synthetic 2-D underwater range-localisation benchmark"


def _filter_mean_rmse(kind: str, data: TrajectoryData, q: float, r: float) -> float:
    errors: list[float] = []
    for index in range(data.n_trajectories):
        result = run_classical_filter(
            kind,
            data.controls[index],
            data.measurements[index],
            data.beacons,
            data.states[index, 0],
            q_scale=q,
            r_scale=r,
        )
        errors.append(trajectory_metrics(result.states, data.states[index])["rmse"])
    return float(np.nanmean(errors))


def tune_classical_filters(data: TrajectoryData, config: ExperimentConfig) -> tuple[dict[str, tuple[float, float]], pd.DataFrame]:
    """Tune fixed Q/R only on the validation trajectories."""
    selected: dict[str, tuple[float, float]] = {}
    rows: list[dict[str, float | str]] = []
    for kind in ("ekf", "ukf", "ckf"):
        best: tuple[float, float, float] | None = None
        for q in config.filter_q_candidates:
            for r in config.filter_r_candidates:
                rmse = _filter_mean_rmse(kind, data, q, r)
                rows.append({"filter": kind.upper(), "q_variance": q, "r_variance": r, "validation_rmse": rmse})
                if best is None or rmse < best[0]:
                    best = (rmse, q, r)
        assert best is not None
        selected[kind] = (best[1], best[2])
    return selected, pd.DataFrame(rows)


def _run_classical_predictions(
    data: TrajectoryData, choices: Mapping[str, tuple[float, float]]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    predictions: dict[str, np.ndarray] = {}
    covariances: dict[str, np.ndarray] = {}
    innovations: dict[str, np.ndarray] = {}
    nis_values: dict[str, np.ndarray] = {}
    for kind, display_name in (("ekf", "EKF"), ("ukf", "UKF"), ("ckf", "CKF")):
        q, r = choices[kind]
        results: list[FilterResult] = [
            run_classical_filter(
                kind,
                data.controls[index],
                data.measurements[index],
                data.beacons,
                data.states[index, 0],
                q_scale=q,
                r_scale=r,
            )
            for index in range(data.n_trajectories)
        ]
        predictions[display_name] = np.stack([result.states for result in results])
        covariances[display_name] = np.stack([result.covariances for result in results])
        innovations[display_name] = np.stack([result.innovations for result in results])
        nis_values[display_name] = np.stack([result.nis for result in results])
    return predictions, covariances, innovations, nis_values


def _training_config(config: ExperimentConfig, epochs: int) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        patience=max(3, min(8, epochs // 2)),
        device=config.device,
    )


def _direct_models(
    train_data: TrajectoryData,
    validation_data: TrajectoryData,
    test_data: TrajectoryData,
    config: ExperimentConfig,
    model_directory: Path,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, dict[str, list[float] | int]]]:
    predictions: dict[str, np.ndarray] = {}
    parameters: dict[str, int] = {}
    histories: dict[str, dict[str, list[float] | int]] = {}
    cell_input_dim = train_data.controls.shape[-1] + train_data.states.shape[-1]
    for kind, display_name in (("lnn", "LNN"), ("lstm", "LSTM"), ("ctgru", "CTGRU"), ("ctrnn", "CTRNN")):
        model = RecurrentPositionRegressor(
            kind, RecurrentModelConfig(input_dim=cell_input_dim, hidden_dim=config.hidden_dim)
        )
        history = train_direct_model(model, train_data, validation_data, _training_config(config, config.epochs))
        predictions[display_name] = predict_direct(model, test_data, config.device)["positions"]
        parameters[display_name] = count_trainable_parameters(model)
        histories[display_name] = {
            "train_loss": history.train_loss,
            "validation_loss": history.validation_loss,
            "best_epoch": history.best_epoch,
        }
        if config.save_models:
            model_directory.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_directory / f"{display_name.lower()}.pt")
    return predictions, parameters, histories


def _full_lnn_ukf(
    train_data: TrajectoryData,
    validation_data: TrajectoryData,
    test_data: TrajectoryData,
    config: ExperimentConfig,
    model_directory: Path,
    model_config: LNNUKFConfig | None = None,
    epochs: int | None = None,
) -> tuple[LNNUKF, dict[str, np.ndarray], dict[str, list[float] | int]]:
    input_dim = train_data.controls.shape[-1] + train_data.states.shape[-1]
    resolved_config = model_config or LNNUKFConfig(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        innovation_window=8,
    )
    model = LNNUKF(torch.as_tensor(train_data.beacons, dtype=torch.float32), resolved_config)
    history = train_lnn_ukf(model, train_data, validation_data, _training_config(config, epochs or config.epochs))
    prediction = predict_lnn_ukf(model, test_data, config.device)
    history_dict: dict[str, list[float] | int] = {
        "train_loss": history.train_loss,
        "validation_loss": history.validation_loss,
        "best_epoch": history.best_epoch,
    }
    if config.save_models:
        model_directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "model_config": asdict(resolved_config)},
            model_directory / "lnn_ukf.pt",
        )
    return model, prediction, history_dict


def _ablation_variants(base: LNNUKFConfig) -> list[tuple[str, LNNUKFConfig]]:
    """Return documented ablations that retain the same data protocol."""
    return [
        ("Fixed Q/R", replace(base, adaptive_process_noise=False, adaptive_measurement_noise=False)),
        ("No covariance feedback", replace(base, covariance_feedback=False)),
        ("Fixed tau", replace(base, liquid=False)),
        ("Unconstrained readout", replace(base, orthogonal_readout=False)),
    ]


def _run_ablations(
    full_prediction: dict[str, np.ndarray],
    base_config: LNNUKFConfig,
    train_data: TrajectoryData,
    validation_data: TrajectoryData,
    test_data: TrajectoryData,
    config: ExperimentConfig,
    model_directory: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    outputs: dict[str, np.ndarray] = {"Full LNN+UKF": full_prediction["positions"]}
    if config.run_ablations:
        for label, variant in _ablation_variants(base_config):
            model, prediction, _ = _full_lnn_ukf(
                train_data,
                validation_data,
                test_data,
                config,
                model_directory / "ablations" / label.replace(" ", "_"),
                model_config=variant,
                epochs=config.ablation_epochs,
            )
            del model
            outputs[label] = prediction["positions"]
    metrics = per_trajectory_metrics(outputs, test_data.states)
    return metrics, outputs


def _bootstrap_summary(metrics: pd.DataFrame, seed: int = 0, draws: int = 2000) -> pd.DataFrame:
    """Aggregate per-trajectory results with non-parametric 95% mean CIs."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for model, group in metrics.groupby("model", sort=False):
        for metric in ("rmse", "mae", "ade", "fde"):
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                lower = upper = mean = np.nan
            else:
                mean = float(values.mean())
                samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
                lower, upper = np.quantile(samples, [0.025, 0.975])
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_trajectory_runs": int(len(values)),
                    "mean": mean,
                    "ci95_low": float(lower),
                    "ci95_high": float(upper),
                }
            )
    return pd.DataFrame(rows)


def _tau_input_features(
    prediction: Mapping[str, np.ndarray],
    test_data: TrajectoryData,
    position_scale: float,
) -> tuple[np.ndarray, list[str]]:
    """Build causal LNN inputs plus interpretable kinematic descriptors.

    The liquid cell directly receives control east/north increments and the
    previous filtered position.  Speed and continuous heading are deterministic
    descriptors of those controls, included so the resulting correlation matrix
    covers the navigation quantities used in scientific interpretation.
    """
    controls = test_data.controls
    heading = np.unwrap(np.arctan2(controls[..., 1], controls[..., 0]), axis=1)
    speed = np.linalg.norm(controls, axis=-1) / test_data.dt
    previous_position = prediction["positions"][:, :-1] / position_scale
    features = np.concatenate(
        (
            controls,
            speed[..., None],
            heading[..., None],
            previous_position,
        ),
        axis=-1,
    )
    names = [
        "control_east_m",
        "control_north_m",
        "speed_m_per_s",
        "heading_rad",
        "filtered_east_scaled",
        "filtered_north_scaled",
    ]
    return features, names


def _interpretability_outputs(
    model: LNNUKF,
    prediction: Mapping[str, np.ndarray],
    test_data: TrajectoryData,
    output_directory: Path,
    device: str,
    tau_overlap_epsilon: float,
    tau_grid_size: float,
    tau_lag_max_steps: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Run tau interventions and record auditable internal/external diagnostics."""
    output_directory.mkdir(parents=True, exist_ok=True)
    baseline = prediction["positions"]
    counterfactuals = {
        scale: predict_lnn_ukf(model, test_data, device, tau_multiplier=scale)["positions"]
        for scale in (0.5, 0.75, 1.0, 1.25, 1.5)
    }
    intervention = tau_intervention_summary(
        baseline,
        counterfactuals,
        truth=test_data.states,
        epsilon=tau_overlap_epsilon,
        grid_size=tau_grid_size,
    )

    tau = prediction["tau"]
    tau_inputs, input_names = _tau_input_features(
        prediction,
        test_data,
        position_scale=model.config.position_scale,
    )
    correlation = vector_feature_spearman_correlation(
        tau,
        tau_inputs,
        feature_names=input_names,
        vector_names=[f"tau_{index}" for index in range(tau.shape[-1])],
    )
    change_comparison = vector_change_comparison(tau, tau_inputs)
    selected_count = min(6, tau.shape[0])
    selected_indices = np.linspace(0, tau.shape[0] - 1, selected_count, dtype=int)
    selected_trajectories = pd.DataFrame(
        {
            "trajectory_slot": np.arange(1, selected_count + 1),
            "trajectory_index": selected_indices,
        }
    )
    per_trajectory_correlations: list[pd.DataFrame] = []
    for trajectory_index in selected_indices:
        trajectory_correlation = vector_feature_spearman_correlation(
            tau[trajectory_index],
            tau_inputs[trajectory_index],
            feature_names=input_names,
            vector_names=[f"tau_{index}" for index in range(tau.shape[-1])],
        )
        trajectory_correlation.insert(0, "trajectory_index", int(trajectory_index))
        per_trajectory_correlations.append(trajectory_correlation)
        trajectory_changes = vector_change_comparison(tau[trajectory_index], tau_inputs[trajectory_index])
        trajectory_changes["trajectory_index"] = int(trajectory_index)
        plot_tau_input_diagnostics(
            trajectory_correlation,
            trajectory_changes,
            output_directory / f"tau_input_diagnostics_trajectory_{trajectory_index + 1:02d}.png",
            title=f"Trajectory {trajectory_index + 1}: tau vector and inputs",
        )
    correlation_by_trajectory = pd.concat(per_trajectory_correlations, ignore_index=True)
    lag_limit = min(max(int(tau_lag_max_steps), 0), max(tau.shape[1] - 4, 0))
    lagged_correlation = lagged_vector_feature_spearman(
        tau[selected_indices],
        tau_inputs[selected_indices],
        feature_names=input_names,
        vector_names=[f"tau_{index}" for index in range(tau.shape[-1])],
        max_lag_steps=lag_limit,
    )
    lagged_correlation["trajectory_index"] = lagged_correlation["trajectory_index"].map(
        dict(enumerate(selected_indices))
    ).astype(int)
    lag_summary, delay_summary = summarize_lagged_spearman(lagged_correlation, test_data.dt)

    process_trace = np.trace(prediction["process_covariances"], axis1=-2, axis2=-1)
    measurement_trace = np.trace(prediction["measurement_covariances"][:, 1:], axis1=-2, axis2=-1)
    innovation_norm = np.linalg.norm(np.nan_to_num(prediction["innovations"][:, 1:], nan=0.0), axis=-1)
    noise_correlation = vector_feature_spearman_correlation(
        np.stack((process_trace, measurement_trace), axis=-1),
        innovation_norm[..., None],
        feature_names=["innovation_norm"],
        vector_names=["trace_Q", "trace_R"],
    ).rename(columns={"vector_component": "quantity"})

    # Exact readout closure: sum_i A[:,i] h_i + b equals the affine residual.
    hidden = prediction["hidden"][:, 1:]
    weight = model.readout.weight().detach().cpu().numpy()
    bias = model.readout.bias.detach().cpu().numpy()
    contribution_sum = np.einsum("bth,dh->btd", hidden, weight) + bias
    model_device = next(model.parameters()).device
    with torch.no_grad():
        residual_from_module = model.readout(
            torch.as_tensor(hidden, dtype=torch.float32, device=model_device)
        ).cpu().numpy()
    closure_error = float(np.max(np.abs(contribution_sum - residual_from_module)))
    contributions = hidden[..., :, None] * weight.T[None, None, :, :]

    hidden_covariance = prediction["hidden_covariances"]
    projected_covariance = np.matmul(np.matmul(weight[None, None, :, :], hidden_covariance), weight.T[None, None, :, :])
    position_covariance = prediction["position_covariances"]
    numerator = np.linalg.norm(projected_covariance - position_covariance, axis=(-2, -1))
    denominator = np.maximum(np.linalg.norm(position_covariance, axis=(-2, -1)), 1.0e-12)
    covariance_projection_error = float(np.mean(numerator / denominator))
    smallest_eigenvalue = float(np.min(np.linalg.eigvalsh(position_covariance)))

    tau_for_plot = np.concatenate((tau[:, :1], tau), axis=1)
    plot_tau_trajectory_panel(
        tau_for_plot,
        baseline,
        output_directory / "tau_trajectory_interventions.png",
        counterfactuals={scale: value for scale, value in counterfactuals.items() if scale != 1.0},
        truth=test_data.states,
    )
    plot_tau_input_diagnostics(
        correlation,
        change_comparison,
        output_directory / "tau_input_diagnostics.png",
    )
    plot_readout_contributions(contributions, output_directory / "readout_contributions.png")
    for trajectory_index in selected_indices:
        plot_readout_contributions(
            contributions[trajectory_index],
            output_directory / f"readout_contributions_trajectory_{trajectory_index + 1:02d}.png",
            title=f"Trajectory {trajectory_index + 1}: internal-unit readout contributions",
        )
    readout_by_trajectory, readout_role_summary = summarize_readout_roles(contributions[selected_indices])
    readout_by_trajectory["trajectory_index"] = readout_by_trajectory["trajectory_index"].map(
        dict(enumerate(selected_indices))
    ).astype(int)
    plot_readout_role_consistency(
        readout_role_summary,
        output_directory / "readout_role_consistency.png",
    )
    plot_tau_input_lag_diagnostics(
        lag_summary,
        delay_summary,
        output_directory / "tau_input_lag_diagnostics.png",
    )
    plot_noise_adaptation(
        prediction["process_covariances"],
        prediction["measurement_covariances"],
        prediction["innovations"],
        output_directory / "adaptive_noise_diagnostics.png",
    )
    diagnostics = {
        "readout_closure_max_abs_error": closure_error,
        "mean_covariance_projection_relative_error": covariance_projection_error,
        "minimum_position_covariance_eigenvalue": smallest_eigenvalue,
        "orthogonality_frobenius_error": float(prediction["orthogonality_error"]),
        "lag_max_steps": float(lag_limit),
        "lagged_pairs_with_input_leading_peak": float(
            (delay_summary["delay_direction"] == "input_leads_tau").sum()
        ),
    }
    tables = {
        "intervention": intervention,
        "correlation": correlation,
        "change_comparison": change_comparison,
        "correlation_by_trajectory": correlation_by_trajectory,
        "lagged_correlation": lagged_correlation,
        "lag_summary": lag_summary,
        "delay_summary": delay_summary,
        "selected_trajectories": selected_trajectories,
        "readout_by_trajectory": readout_by_trajectory,
        "readout_role_summary": readout_role_summary,
        "noise_correlation": noise_correlation,
    }
    return tables, diagnostics


def _run_single_seed(
    source_data: TrajectoryData,
    config: ExperimentConfig,
    seed: int,
    output_directory: Path,
) -> SeedRun:
    set_reproducible_seed(seed)
    splits = split_by_trajectory(source_data, seed=config.data_seed)
    train_data, validation_data, test_data = splits["train"], splits["validation"], splits["test"]
    seed_directory = output_directory / f"seed_{seed}"
    model_directory = seed_directory / "models"
    choices, filter_tuning = tune_classical_filters(validation_data, config)
    filter_predictions, filter_covariances, filter_innovations, filter_nis = _run_classical_predictions(test_data, choices)

    direct_predictions, direct_parameter_counts, _ = _direct_models(
        train_data, validation_data, test_data, config, model_directory
    )
    base_config = LNNUKFConfig(
        input_dim=train_data.controls.shape[-1] + train_data.states.shape[-1],
        hidden_dim=config.hidden_dim,
        innovation_window=8,
        orthogonal_readout=True,
    )
    full_model, full_prediction, _ = _full_lnn_ukf(
        train_data, validation_data, test_data, config, model_directory, model_config=base_config
    )
    main_predictions = {**filter_predictions, **direct_predictions, "LNN+UKF": full_prediction["positions"]}
    metrics = per_trajectory_metrics(main_predictions, test_data.states)
    metrics.insert(0, "seed", seed)
    parameter_counts = {"EKF": 0, "UKF": 0, "CKF": 0, **direct_parameter_counts, "LNN+UKF": count_trainable_parameters(full_model)}
    metrics["parameters"] = metrics["model"].map(parameter_counts).astype(int)

    ablation_metrics, _ = _run_ablations(
        full_prediction, base_config, train_data, validation_data, test_data, config, model_directory
    )
    ablation_metrics.insert(0, "seed", seed)

    calibration_rows: list[dict[str, float | str | int]] = []
    for name in ("EKF", "UKF", "CKF"):
        summary = consistency_summary(
            filter_predictions[name] - test_data.states,
            filter_covariances[name],
        )
        summary.update({"model": name, "seed": seed})
        calibration_rows.append(summary)
    full_summary = consistency_summary(
        full_prediction["positions"] - test_data.states,
        full_prediction["position_covariances"],
    )
    full_summary.update({"model": "LNN+UKF", "seed": seed})
    calibration_rows.append(full_summary)
    calibration = pd.DataFrame(calibration_rows)

    # Only the first seed creates the detailed plots; all seeds contribute to
    # numerical tables and confidence intervals.
    if seed == config.seeds[0]:
        figures = output_directory / "figures"
        plot_trajectory_comparison(test_data.states, main_predictions, figures / "trajectory_comparison.png", test_data.beacons)
        plot_trajectory_comparison_grid(
            test_data.states,
            main_predictions,
            figures / "trajectory_comparison_six.png",
        )
        plot_metrics_table(metrics, figures / "metrics_comparison.png")
        full_nees = nees(full_prediction["positions"] - test_data.states, full_prediction["position_covariances"])
        plot_covariance_calibration(
            full_nees,
            full_prediction["nis"],
            figures / "lnn_ukf_calibration.png",
            state_dimension=2,
            measurement_dimension=test_data.measurement_dim,
            title="LNN+UKF uncertainty calibration",
        )
        interpretability_tables, diagnostics = _interpretability_outputs(
            full_model,
            full_prediction,
            test_data,
            figures,
            config.device,
            config.tau_overlap_epsilon,
            config.tau_grid_size,
            config.tau_lag_max_steps,
        )
        output_tables = {
            "tau_interventions.csv": "intervention",
            "tau_input_spearman.csv": "correlation",
            "tau_input_change_comparison.csv": "change_comparison",
            "tau_input_spearman_by_trajectory.csv": "correlation_by_trajectory",
            "tau_input_lagged_spearman_by_trajectory.csv": "lagged_correlation",
            "tau_input_lagged_spearman_summary.csv": "lag_summary",
            "tau_input_delay_summary.csv": "delay_summary",
            "selected_interpretability_trajectories.csv": "selected_trajectories",
            "readout_contributions_by_trajectory.csv": "readout_by_trajectory",
            "readout_role_consistency.csv": "readout_role_summary",
            "noise_adaptation_correlation.csv": "noise_correlation",
        }
        for filename, table_name in output_tables.items():
            interpretability_tables[table_name].to_csv(output_directory / filename, index=False)
        (output_directory / "interpretability_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    filter_tuning.insert(0, "seed", seed)
    return SeedRun(
        metrics=metrics,
        ablation_metrics=ablation_metrics,
        calibration=calibration,
        filter_tuning=filter_tuning,
        full_model=full_model,
        full_prediction=full_prediction,
        test_data=test_data,
        main_predictions=main_predictions,
        parameter_counts=parameter_counts,
    )


def _write_report(
    path: Path,
    source_description: str,
    config: ExperimentConfig,
    summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
) -> None:
    """Write a concise provenance-first report for the generated artifacts."""
    def markdown_table(frame: pd.DataFrame) -> str:
        """Render a small frame without Pandas' optional ``tabulate`` package."""
        table = frame.reset_index()
        columns = [str(column) for column in table.columns]
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in table.itertuples(index=False, name=None):
            formatted = [f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
            lines.append("| " + " | ".join(formatted) + " |")
        return "\n".join(lines)

    summary_table = markdown_table(summary.pivot(index="model", columns="metric", values="mean").round(4))
    ablation_table = markdown_table(ablation_summary.pivot(index="model", columns="metric", values="mean").round(4))
    text = f"""# LNN+UKF experiment report

Data source: **{source_description}**.

The figures and tables in this directory are computed from the arrays produced
by this run.  If the source is synthetic, they validate code paths and the
experimental protocol only; they must not be reported as AUV field-data
results.  A real run requires an NPZ file with `controls`, `measurements`,
`states`, `beacons`, and optional `dt`, created from the user's LNN_Lack data
through a documented preprocessing step.

## Main comparison

{summary_table}

## Ablation comparison

{ablation_table}

## Protocol

- Split by complete trajectory, using only training trajectories for learned
  weights and validation trajectories for classical-filter Q/R tuning.
- All classical filters receive the same initial position, odometry controls,
  beacon geometry, and acoustic ranges.  Direct LNN/LSTM/CTGRU/CTRNN models
  receive the same odometry and initial position in the open-loop task.
- LNN+UKF uses hidden-state sigma points, causal innovation-history Q/R LSTMs,
  nonlinear range updates, and hidden covariance feedback via cross-covariance.
- Tau interventions are in-model counterfactuals, not claims of a physical
  intervention on an AUV.  The complete tau-vector/input-feature Spearman
  matrix and aligned vector-change data are in `tau_input_spearman.csv` and
  `tau_input_change_comparison.csv`.  Lagged change correlations use positive
  lag to mean that an input change precedes a tau change; see
  `tau_input_delay_summary.csv` and `tau_input_lagged_spearman_summary.csv`.

Configuration:

```json
{json.dumps(asdict(config), indent=2)}
```
"""
    path.write_text(text, encoding="utf-8")


def run_experiment(config: ExperimentConfig) -> dict[str, Path]:
    """Run full comparison, ablations, calibration, and interpretation output."""
    if not config.seeds:
        raise ValueError("at least one random seed is required")
    output_directory = Path(config.output_dir).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    source_data, source_description = _source_data(config)
    if config.data_path is None:
        save_npz_dataset(source_data, output_directory / "synthetic_dataset.npz")

    runs = [_run_single_seed(source_data, config, seed, output_directory) for seed in config.seeds]
    metrics = pd.concat([run.metrics for run in runs], ignore_index=True)
    ablation_metrics = pd.concat([run.ablation_metrics for run in runs], ignore_index=True)
    calibration = pd.concat([run.calibration for run in runs], ignore_index=True)
    filter_tuning = pd.concat([run.filter_tuning for run in runs], ignore_index=True)
    summary = _bootstrap_summary(metrics, seed=config.data_seed)
    ablation_summary = _bootstrap_summary(ablation_metrics, seed=config.data_seed + 1)

    metrics.to_csv(output_directory / "metrics_per_trajectory.csv", index=False)
    summary.to_csv(output_directory / "metrics_summary.csv", index=False)
    ablation_metrics.to_csv(output_directory / "ablation_metrics_per_trajectory.csv", index=False)
    ablation_summary.to_csv(output_directory / "ablation_metrics_summary.csv", index=False)
    calibration.to_csv(output_directory / "calibration_summary.csv", index=False)
    filter_tuning.to_csv(output_directory / "classical_filter_validation_tuning.csv", index=False)
    (output_directory / "run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    _write_report(output_directory / "REPORT.md", source_description, config, summary, ablation_summary)
    return {
        "output_dir": output_directory,
        "report": output_directory / "REPORT.md",
        "metrics": output_directory / "metrics_summary.csv",
        "ablations": output_directory / "ablation_metrics_summary.csv",
        "figures": output_directory / "figures",
    }


__all__ = ["ExperimentConfig", "run_experiment", "tune_classical_filters"]
