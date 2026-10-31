from __future__ import annotations

from lnn_ukf.experiment import ExperimentConfig, run_experiment


def test_tiny_end_to_end_experiment_writes_auditable_artifacts(tmp_path) -> None:
    config = ExperimentConfig(
        output_dir=str(tmp_path / "results"),
        synthetic_trajectories=12,
        synthetic_steps=12,
        hidden_dim=4,
        epochs=1,
        ablation_epochs=1,
        batch_size=3,
        run_ablations=False,
        save_models=False,
        filter_q_candidates=(0.01,),
        filter_r_candidates=(0.16,),
    )
    artifacts = run_experiment(config)
    assert artifacts["report"].is_file()
    assert artifacts["metrics"].is_file()
    assert (artifacts["figures"] / "tau_trajectory_interventions.png").is_file()
    assert (artifacts["figures"] / "tau_input_vector_norms_six.png").is_file()
    assert (artifacts["figures"] / "tau_input_lag_diagnostics.png").is_file()
    assert (artifacts["figures"] / "readout_role_consistency.png").is_file()
    assert (artifacts["figures"] / "trajectory_comparison_six.png").is_file()
    assert (tmp_path / "results" / "tau_input_spearman.csv").is_file()
    assert (tmp_path / "results" / "tau_input_change_comparison.csv").is_file()
    assert (tmp_path / "results" / "tau_input_vector_norms_by_trajectory.csv").is_file()
    assert (tmp_path / "results" / "tau_input_delay_summary.csv").is_file()
    assert (tmp_path / "results" / "readout_role_consistency.csv").is_file()
