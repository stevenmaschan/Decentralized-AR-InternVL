# Results — 150k training set

Evaluated on the held-out set `sweep_datasets/heldout_20k` (n=20000), an independently
generated dataset (seed 7) with zero byte-identical images against training data.
All systems: 1.35M active parameters, 58,593 optimizer updates, identical val/test.

Dataset: 160,000 images, 150k train / 5k val / 5k test, `--alpha-beta 0.197`,
`--exclude-tasks 3`, angle thresholds {3,4,6}, one question sampled per image
with P(OCR) = alpha.

## Overall exact match

| system | exact match | vs dense mean |
|---|---|---|
| dense (seed 0) | 67.78 | — |
| dense (seed 1) | 67.79 | — |
| dense (seed 2) | 67.66 | — |
| context-routed experts | 68.04 | +0.30 |
| alpha-routed experts (oracle) | 65.71 | -2.03 |
| MoE layerwise (top-1) | 67.88 | +0.14 |
| MoE full-model (hard top-1) | 65.96 | -1.78 |
| random experts, 0.5/0.5 mixture | 66.99 | -0.75 |
| **dense mean of 3 seeds** | **67.74** | sd 0.07, range 0.12 |

Standard error of an unpaired gap at n=20000: ~0.47 points.
Measured seed spread (sd 0.07) is the tighter and more relevant noise floor.

## Per task

| system | ocr | count_by_type | name_types | count_by_angles |
|---|---|---|---|---|
| *n* | 9870 | 3331 | 3366 | 3433 |
| dense (seed 0) | 58.74 | 78.23 | 73.44 | 78.07 |
| dense (seed 1) | 58.37 | 79.26 | 73.77 | 77.89 |
| dense (seed 2) | 58.06 | 79.02 | 73.89 | 78.15 |
| context-routed experts | 58.46 | 79.77 | 73.38 | 78.97 |
| alpha-routed experts (oracle) | 57.24 | 75.83 | 70.17 | 75.85 |
| MoE layerwise (top-1) | 58.61 | 78.72 | 73.47 | 78.56 |
| MoE full-model (hard top-1) | 57.76 | 75.83 | 69.25 | 76.76 |
| random experts, 0.5/0.5 mixture | 56.69 | 79.50 | 73.98 | 77.63 |

### Per task, difference from dense seed 0

| system | ocr | count_by_type | name_types | count_by_angles |
|---|---|---|---|---|
| context-routed experts | -0.28 | +1.53 | -0.06 | +0.90 |
| alpha-routed experts (oracle) | -1.50 | -2.40 | -3.27 | -2.21 |
| MoE layerwise (top-1) | -0.13 | +0.48 | +0.03 | +0.50 |
| MoE full-model (hard top-1) | -0.98 | -2.40 | -4.19 | -1.31 |
| random experts, 0.5/0.5 mixture | -2.06 | +1.26 | +0.53 | -0.44 |

## Secondary metrics

| system | OCR CER | name_types set-F1 | count MAE (type / angles) |
|---|---|---|---|
| dense (seed 0) | 0.173 | 0.952 | 0.230 / 0.237 |
| dense (seed 1) | 0.176 | 0.954 | 0.220 / 0.245 |
| dense (seed 2) | 0.176 | 0.953 | 0.224 / 0.240 |
| context-routed experts | 0.175 | 0.954 | 0.216 / 0.227 |
| alpha-routed experts (oracle) | 0.187 | 0.939 | 0.264 / 0.279 |
| MoE layerwise (top-1) | 0.177 | 0.952 | 0.225 / 0.232 |
| MoE full-model (hard top-1) | 0.186 | 0.941 | 0.264 / 0.260 |
| random experts, 0.5/0.5 mixture | 0.180 | 0.953 | 0.220 / 0.244 |

## Comparison with the 100k suite

| system | gap @100k | gap @150k |
|---|---|---|
| context-routed experts | +0.07 | +0.30 |
| alpha-routed experts (oracle) | -2.38 | -2.03 |
| MoE layerwise (top-1) | +0.09 | +0.14 |
| MoE full-model (hard top-1) | -1.64 | -1.78 |
| random experts, 0.5/0.5 mixture | -0.79 | -0.75 |
| *dense mean* | *65.65* | *67.74* |

Dense gains +2.09 points from 100k to 150k, so the task is not saturated.
Every gap reproduces within 0.23 points across the two scales.

## CLIP feature clustering (k=2, balanced spherical, mean-subtracted)

| reference partition | agreement | ARI |
|---|---|---|
| alpha split (>=0.5) | 83.9% | 0.460 |
| task split (OCR vs shape) | 79.7% | 0.352 |
| *alpha vs task, for reference* | *89.9%* | *0.636* |

A learned image-based router approximates the **alpha** split, which is the
partition that loses ~2 points. Its ceiling is therefore below the oracle-alpha
row above, not above it.
