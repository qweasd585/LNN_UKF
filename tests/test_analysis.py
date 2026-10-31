from __future__ import annotations

import numpy as np

from lnn_ukf.analysis import (
    occupancy_iou,
    plot_readout_contributions,
    tau_intervention_summary,
    time_aligned_overlap,
    trajectory_metrics,
)


def test_interpretability_metrics_and_plot(tmp_path) -> None:
    baseline = np.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    changed = baseline + np.array([[[0.0, 0.0], [0.2, 0.0], [0.2, 0.0]]])
    assert time_aligned_overlap(baseline, changed, epsilon=0.25) == 1.0
    assert 0.0 < occupancy_iou(baseline, changed, grid_size=1.0) <= 1.0
    summary = tau_intervention_summary(baseline, {0.5: changed, 1.0: baseline})
    assert set(summary.columns) >= {"tau_scale", "mean_displacement", "occupancy_iou"}
    assert trajectory_metrics(baseline, baseline)["rmse"] == 0.0
    contributions = np.ones((1, 3, 4, 2))
    image = plot_readout_contributions(contributions, tmp_path / "contributions.png")
    assert image.is_file() and image.stat().st_size > 0
