#!/usr/bin/env python3
"""Run the LNN+UKF comparison, ablation, and interpretability experiments.

Examples
--------
Quick synthetic smoke experiment::

    .venv/bin/python run_experiment.py --fast --output results/demo

Run a real data set converted to the documented NPZ schema::

    .venv/bin/python run_experiment.py --data /path/to/auv.npz --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from lnn_ukf.experiment import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LNN+UKF underwater-localisation benchmark")
    parser.add_argument("--data", type=str, default=None, help="NPZ data with controls, measurements, states, beacons, optional dt")
    parser.add_argument("--output", type=str, default="results", help="directory for tables, figures, checkpoints, and report")
    parser.add_argument("--fast", action="store_true", help="small synthetic configuration for a quick CPU run")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="training random seeds, e.g. --seeds 1 2 3 4 5")
    parser.add_argument("--epochs", type=int, default=None, help="epochs for each main learned model")
    parser.add_argument("--ablation-epochs", type=int, default=None, help="epochs for each LNN+UKF ablation")
    parser.add_argument("--hidden-dim", type=int, default=None, help="LNN and baseline hidden width")
    parser.add_argument("--trajectories", type=int, default=None, help="synthetic trajectory count when --data is absent")
    parser.add_argument("--steps", type=int, default=None, help="synthetic time points per trajectory when --data is absent")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device, such as cpu or cuda")
    parser.add_argument("--threads", type=int, default=None, help="PyTorch CPU threads")
    parser.add_argument("--no-ablations", action="store_true", help="skip extra ablation training (main comparison still runs)")
    parser.add_argument("--no-save-models", action="store_true", help="do not save trained checkpoints")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = ExperimentConfig.fast(arguments.output) if arguments.fast else ExperimentConfig(output_dir=arguments.output)
    changes: dict[str, object] = {"data_path": arguments.data, "device": arguments.device}
    if arguments.seeds is not None:
        changes["seeds"] = tuple(arguments.seeds)
    for argument_name, field_name in (
        ("epochs", "epochs"),
        ("ablation_epochs", "ablation_epochs"),
        ("hidden_dim", "hidden_dim"),
        ("trajectories", "synthetic_trajectories"),
        ("steps", "synthetic_steps"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
    ):
        value = getattr(arguments, argument_name)
        if value is not None:
            changes[field_name] = value
    if arguments.no_ablations:
        changes["run_ablations"] = False
    if arguments.no_save_models:
        changes["save_models"] = False
    config = replace(config, **changes)
    if arguments.threads is not None:
        if arguments.threads < 1:
            raise ValueError("--threads must be positive")
        torch.set_num_threads(arguments.threads)
    artifacts = run_experiment(config)
    for name, path in artifacts.items():
        print(f"{name}: {Path(path)}")


if __name__ == "__main__":
    main()
