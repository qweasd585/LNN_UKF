from __future__ import annotations

import numpy as np

from lnn_ukf.analysis import (
    lagged_vector_feature_spearman,
    occupancy_iou,
    plot_readout_contributions,
    plot_readout_input_spearman_grid,
    plot_tau_input_diagnostics,
    plot_tau_input_lag_diagnostics,
    plot_tau_input_vector_norms_grid,
    plot_trajectory_comparison_grid,
    readout_heatmap_combination_similarity,
    summarize_lagged_spearman,
    readout_input_spearman_correlation,
    tau_intervention_summary,
    time_aligned_overlap,
    trajectory_metrics,
    vector_change_comparison,
    vector_feature_spearman_correlation,
    vector_norm_comparison,
)


def test_interpretability_metrics_and_plot(tmp_path) -> None:
    baseline = np.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    changed = baseline + np.array([[[0.0, 0.0], [0.2, 0.0], [0.2, 0.0]]])
    assert time_aligned_overlap(baseline, changed, epsilon=0.25) == 1.0
    assert 0.0 < occupancy_iou(baseline, changed, grid_size=1.0) <= 1.0
    summary = tau_intervention_summary(
        baseline, {0.5: changed, 1.0: baseline}, truth=baseline
    )
    assert set(summary.columns) >= {
        "tau_scale",
        "mean_displacement_from_dynamic_tau",
        "occupancy_iou_with_dynamic_tau",
        "rmse_to_truth",
        "fde_to_truth",
    }
    assert trajectory_metrics(baseline, baseline)["rmse"] == 0.0
    contributions = np.ones((1, 3, 4, 2))
    image = plot_readout_contributions(contributions, tmp_path / "contributions.png")
    assert image.is_file() and image.stat().st_size > 0


def test_vector_input_correlation_and_compact_trajectory_grid(tmp_path) -> None:
    tau = np.array([[[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]]])
    inputs = np.array([[[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]])
    correlation = vector_feature_spearman_correlation(
        tau,
        inputs,
        feature_names=["speed", "heading"],
        vector_names=["tau_0", "tau_1"],
    )
    assert correlation.shape[0] == 4
    assert set(correlation.columns) == {
        "vector_component",
        "input_feature",
        "spearman_rho",
        "p_value",
        "n",
    }
    changes = vector_change_comparison(tau, inputs)
    assert len(changes) == 3
    diagnostic = plot_tau_input_diagnostics(correlation, changes, tmp_path / "tau_input.png")
    assert diagnostic.is_file() and diagnostic.stat().st_size > 0
    norms = vector_norm_comparison(tau, inputs)
    assert set(norms.columns) >= {"tau_vector_l2", "input_vector_standardized_l2"}
    norm_image = plot_tau_input_vector_norms_grid(norms, tmp_path / "norms.png")
    assert norm_image.is_file() and norm_image.stat().st_size > 0

    truth = np.stack((np.c_[np.arange(4), np.zeros(4)], np.c_[np.arange(4), np.ones(4)]))
    image = plot_trajectory_comparison_grid(
        truth,
        {"UKF": truth + 0.1, "LNN+UKF": truth - 0.1},
        tmp_path / "grid.png",
    )
    assert image.is_file() and image.stat().st_size > 0


def test_lagged_tau_input_and_readout_input_diagnostics(tmp_path) -> None:
    feature = np.array([0.0, 0.4, -0.3, 0.9, 0.2, 1.1, -0.5, 0.6, 0.1, 0.8])
    feature_change = np.diff(feature)
    tau_change = np.concatenate(([0.2, -0.1], feature_change[:-2]))
    tau = np.concatenate(([0.0], np.cumsum(tau_change)))[None, :, None]
    lagged = lagged_vector_feature_spearman(
        tau,
        feature[None, :, None],
        feature_names=["speed"],
        vector_names=["tau_0"],
        max_lag_steps=3,
    )
    lag_summary, peaks = summarize_lagged_spearman(lagged, dt=0.25)
    assert int(peaks.iloc[0]["peak_lag_steps"]) == 2
    assert peaks.iloc[0]["delay_direction"] == "input_leads_tau"
    image = plot_tau_input_lag_diagnostics(lag_summary, peaks, tmp_path / "lag.png")
    assert image.is_file() and image.stat().st_size > 0

    contributions = np.zeros((3, 4, 2, 2))
    inputs = np.tile(np.array([0.0, 1.0, -1.0, 2.0])[None, :, None], (3, 1, 1))
    feature_values = inputs[:, :, 0]
    contributions[:, :, 0, 0] = feature_values
    contributions[:, :, 0, 1] = -feature_values
    contributions[0, :, 1, 0] = feature_values[0] ** 2
    contributions[1, :, 1, 0] = -feature_values[1] ** 2
    contributions[2, :, 1, 0] = 0.5 * feature_values[2] + feature_values[2] ** 2
    contributions[:, :, 1, 1] = feature_values
    readout_correlation = readout_input_spearman_correlation(
        contributions,
        inputs,
        feature_names=["speed"],
    )
    assert readout_correlation.shape[0] == 12
    assert set(readout_correlation["output_component"]) == {"east", "north"}
    heatmap_similarity = readout_heatmap_combination_similarity(readout_correlation)
    assert heatmap_similarity.shape[0] == 4  # C(3,2) + C(3,3)
    assert set(heatmap_similarity["trajectory_count"]) == {2, 3}
    assert (heatmap_similarity["heatmap_cell_count"] == 4).all()
    correlation_image = plot_readout_input_spearman_grid(
        readout_correlation, tmp_path / "readout_input.png"
    )
    assert correlation_image.is_file() and correlation_image.stat().st_size > 0
