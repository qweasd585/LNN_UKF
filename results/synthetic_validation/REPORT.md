# LNN+UKF experiment report

Data source: **synthetic 2-D underwater range-localisation benchmark**.

The figures and tables in this directory are computed from the arrays produced
by this run.  If the source is synthetic, they validate code paths and the
experimental protocol only; they must not be reported as AUV field-data
results.  A real run requires an NPZ file with `controls`, `measurements`,
`states`, `beacons`, and optional `dt`, created from the user's LNN_Lack data
through a documented preprocessing step.

## Main comparison

| model | ade | fde | mae | rmse |
| --- | --- | --- | --- | --- |
| CKF | 0.2064 | 0.2271 | 0.1319 | 0.1638 |
| CTGRU | 0.9520 | 1.6168 | 0.6205 | 0.7649 |
| CTRNN | 0.9753 | 1.6344 | 0.6359 | 0.7802 |
| EKF | 0.2064 | 0.2271 | 0.1318 | 0.1638 |
| LNN | 0.7874 | 1.4658 | 0.5042 | 0.6351 |
| LNN+UKF | 0.1873 | 0.2510 | 0.1185 | 0.1485 |
| LSTM | 0.9032 | 1.7114 | 0.5705 | 0.7352 |
| UKF | 0.2064 | 0.2271 | 0.1319 | 0.1638 |

## Ablation comparison

| model | ade | fde | mae | rmse |
| --- | --- | --- | --- | --- |
| Fixed Q/R | 0.2359 | 0.2994 | 0.1474 | 0.1865 |
| Fixed tau | 0.2125 | 0.1858 | 0.1349 | 0.1693 |
| Full LNN+UKF | 0.1873 | 0.2510 | 0.1185 | 0.1485 |
| No covariance feedback | 0.2192 | 0.1845 | 0.1393 | 0.1733 |
| Unconstrained readout | 0.6134 | 1.3047 | 0.4104 | 0.5117 |

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
  `tau_input_change_comparison.csv`.

Configuration:

```json
{
  "output_dir": "/workspace/LNN_UKF/results/synthetic_validation",
  "data_path": null,
  "seeds": [
    7
  ],
  "data_seed": 101,
  "synthetic_trajectories": 30,
  "synthetic_steps": 40,
  "hidden_dim": 4,
  "epochs": 20,
  "ablation_epochs": 8,
  "batch_size": 6,
  "learning_rate": 0.002,
  "device": "cpu",
  "run_ablations": true,
  "save_models": false,
  "tau_overlap_epsilon": 0.1,
  "tau_grid_size": 0.5,
  "filter_q_candidates": [
    0.0025,
    0.01,
    0.025,
    0.05,
    0.1
  ],
  "filter_r_candidates": [
    0.04,
    0.09,
    0.16,
    0.25,
    0.49
  ]
}
```
