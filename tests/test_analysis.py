from __future__ import annotations

import numpy as np

from lnn_ukf.analysis import (
    occupancy_iou,
    plot_readout_contributions,
    plot_tau_input_diagnostics,
    plot_trajectory_comparison_grid,
    tau_intervention_summary,
    time_aligned_overlap,
    trajectory_metrics,
    vector_change_comparison,
    vector_feature_spearman_correlation,
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

    truth = np.stack((np.c_[np.arange(4), np.zeros(4)], np.c_[np.arange(4), np.ones(4)]))
    image = plot_trajectory_comparison_grid(
        truth,
        {"UKF": truth + 0.1, "LNN+UKF": truth - 0.1},
        tmp_path / "grid.png",
    )
    assert image.is_file() and image.stat().st_size > 0
