### Headline results, test split, stateful rollout

| arm | n seeds | straight [0,5)<br><sub>n=6711, 43 events</sub> | gentle [5,15)<br><sub>n=3812, 64 events</sub> | curve [15,inf)<br><sub>n=1774, 19 events</sub> | (diagnostic) >=40<br><sub>n=146, 2 events</sub> | macro MAE | macro skill | Pearson r |
|---|---|---|---|---|---|---|---|---|
| **predict-0** | -- | 2.30 (1.00x) | 9.04 (1.00x) | 28.86 (1.00x) | 127.03 (1.00x) | 13.40 | 0.000 | -- |
| **persistence**<sub>*</sub> | -- | 0.27 (0.12x) | 0.46 (0.05x) | 1.20 (0.04x) | 6.18 (0.05x) | -- | -- | -- |
| | | | | | | | | |
| cfc | 45 | 7.26 (3.15x) | 10.22 (1.13x) | 25.89 (0.90x) | 127.63 (1.00x) | 14.28 | -0.066 <sub>[-0.443, +0.043]</sub> | +0.094 |
| cnn_2frame | 3 | 5.23 (2.27x) | 8.48 (0.94x) | 27.06 (0.94x) | 128.32 (1.01x) | 13.25 | +0.011 <sub>[-0.014, +0.041]</sub> | +0.099 |
| cnn_avg | 10 | 2.61 (1.14x) | 8.23 (0.91x) | 28.19 (0.98x) | 127.26 (1.00x) | 12.96 | +0.032 <sub>[-0.051, +0.060]</sub> | -0.039 |
| cnn_linear | 3 | 3.54 (1.54x) | 8.50 (0.94x) | 27.69 (0.96x) | 126.51 (1.00x) | 13.40 | -0.000 <sub>[-0.020, +0.034]</sub> | +0.106 |
| cnn_mlp | 2 | 3.80 (1.65x) | 8.32 (0.92x) | 27.46 (0.95x) | 127.29 (1.00x) | 13.19 | +0.015 <sub>[-0.002, +0.032]</sub> | +0.060 |
| lstm | 31 | 2.37 (1.03x) | 8.35 (0.92x) | 28.20 (0.98x) | 127.07 (1.00x) | 12.99 | +0.030 <sub>[-0.230, +0.044]</sub> | +0.004 |

<sub>*</sub> Persistence uses the previous TRUE angle. It is reported because it is the naive baseline a reader will think of, and it is not a solution to the posed task: the model receives images only and no past ground-truth angles, so persistence is unavailable the moment labels are absent -- which is always, at deployment.

Cells are MAE in degrees with the ratio to predict-0 in the same bin. Bins are assigned by GROUND TRUTH; metrics are per frame; invalid frames (a logging default of 0.0 in place of a real angle) are excluded. Bands over multiple seeds are min-max, not confidence intervals. The `>=40` column is a SUBSET of the curve bin, reported for diagnosis -- with 2 independent turn events in the test split it is not a measurement on its own.

### Per-position MAE (windowed, state reset at each window start)

| arm | t=1 | t=T | delta |
|---|---:|---:|---:|
| cfc | 13.02 | 15.23 | -2.20 |
| cnn_2frame | 9.45 | 9.39 | +0.06 |
| cnn_avg | 14.58 | 12.85 | +1.73 |
| cnn_linear | 8.71 | 8.71 | -0.01 |
| cnn_mlp | 9.33 | 9.33 | -0.00 |
| lstm | 7.92 | 7.94 | -0.02 |

A non-recurrent arm must be FLAT here: that is the figure's own self-test.

### Runs

| run | arm | seed | best epoch | params (recurrent) |
|---|---|---:|---:|---:|
| `cfc_s0_T16_lr0.0003_k0.1_aug-full` | cfc | 0 | 16 | 24,832 |
| `cfc_s0_T16_lr0.0005_k0.1_aug-full` | cfc | 0 | 20 | 24,832 |
| `cfc_s0_T16_lr0.0007_k0.1_aug-full` | cfc | 0 | 4 | 24,832 |
| `cfc_s0_T16_lr0.0015_k0.1_aug-full` | cfc | 0 | 9 | 24,832 |
| `cfc_s0_T16_lr0.001_k0.1_aug-basic` | cfc | 0 | 11 | 24,832 |
| `cfc_s0_T16_lr0.001_k0.1_aug-full` | cfc | 0 | 18 | 24,832 |
| `cfc_s0_T16_lr0.001_k0.1_aug-full_fixeddt` | cfc | 0 | 4 | 24,832 |
| `cfc_s0_T16_lr0.001_k0.1_aug-full_shuf` | cfc | 0 | 4 | 24,832 |
| `cfc_s0_T16_lr0.001_k0.1_aug-none` | cfc | 0 | 2 | 24,832 |
| `cfc_s0_T16_lr0.002_k0.1_aug-full` | cfc | 0 | 13 | 24,832 |
| `cfc_s0_T16_lr0.003_k0.1_aug-full` | cfc | 0 | 10 | 24,832 |
| `cfc_s1_T16_lr0.0003_k0.1_aug-full` | cfc | 1 | 2 | 24,832 |
| `cfc_s1_T16_lr0.0005_k0.1_aug-full` | cfc | 1 | 7 | 24,832 |
| `cfc_s1_T16_lr0.0007_k0.1_aug-full` | cfc | 1 | 5 | 24,832 |
| `cfc_s1_T16_lr0.0015_k0.1_aug-full` | cfc | 1 | 14 | 24,832 |
| `cfc_s1_T16_lr0.001_k0.1_aug-basic` | cfc | 1 | 20 | 24,832 |
| `cfc_s1_T16_lr0.001_k0.1_aug-full` | cfc | 1 | 2 | 24,832 |
| `cfc_s1_T16_lr0.001_k0.1_aug-full_fixeddt` | cfc | 1 | 2 | 24,832 |
| `cfc_s1_T16_lr0.001_k0.1_aug-full_shuf` | cfc | 1 | 1 | 24,832 |
| `cfc_s1_T16_lr0.001_k0.1_aug-none` | cfc | 1 | 4 | 24,832 |
| `cfc_s1_T16_lr0.002_k0.1_aug-full` | cfc | 1 | 8 | 24,832 |
| `cfc_s1_T16_lr0.003_k0.1_aug-full` | cfc | 1 | 8 | 24,832 |
| `cfc_s2_T16_lr0.0003_k0.1_aug-full` | cfc | 2 | 2 | 24,832 |
| `cfc_s2_T16_lr0.0005_k0.1_aug-full` | cfc | 2 | 1 | 24,832 |
| `cfc_s2_T16_lr0.0007_k0.1_aug-full` | cfc | 2 | 4 | 24,832 |
| `cfc_s2_T16_lr0.0015_k0.1_aug-full` | cfc | 2 | 14 | 24,832 |
| `cfc_s2_T16_lr0.001_k0.1_aug-basic` | cfc | 2 | 6 | 24,832 |
| `cfc_s2_T16_lr0.001_k0.1_aug-full_fixeddt` | cfc | 2 | 4 | 24,832 |
| `cfc_s2_T16_lr0.001_k0.1_aug-full_shuf` | cfc | 2 | 2 | 24,832 |
| `cfc_s2_T16_lr0.001_k0.1_aug-none` | cfc | 2 | 0 | 24,832 |
| `cfc_s2_T16_lr0.002_k0.1_aug-full` | cfc | 2 | 12 | 24,832 |
| `cfc_s2_T16_lr0.003_k0.1_aug-full` | cfc | 2 | 20 | 24,832 |
| `cfc_s3_T16_lr0.001_k0.1_aug-full` | cfc | 3 | 0 | 24,832 |
| `cfc_s3_T16_lr0.001_k0.1_aug-full_fixeddt` | cfc | 3 | 7 | 24,832 |
| `cfc_s3_T16_lr0.001_k0.1_aug-full_shuf` | cfc | 3 | 23 | 24,832 |
| `cfc_s4_T16_lr0.001_k0.1_aug-full` | cfc | 4 | 14 | 24,832 |
| `cfc_s4_T16_lr0.001_k0.1_aug-full_fixeddt` | cfc | 4 | 14 | 24,832 |
| `cfc_s4_T16_lr0.001_k0.1_aug-full_shuf` | cfc | 4 | 4 | 24,832 |
| `cfc_s5_T16_lr0.001_k0.1_aug-full` | cfc | 5 | 4 | 24,832 |
| `cfc_s5_T16_lr0.001_k0.1_aug-full_fixeddt` | cfc | 5 | 6 | 24,832 |
| `cfc_s5_T16_lr0.001_k0.1_aug-full_shuf` | cfc | 5 | 17 | 24,832 |
| `cfc_s6_T16_lr0.001_k0.1_aug-full` | cfc | 6 | 6 | 24,832 |
| `cfc_s7_T16_lr0.001_k0.1_aug-full` | cfc | 7 | 0 | 24,832 |
| `cfc_s8_T16_lr0.001_k0.1_aug-full` | cfc | 8 | 10 | 24,832 |
| `cfc_s9_T16_lr0.001_k0.1_aug-full` | cfc | 9 | 2 | 24,832 |
| `cnn_2frame_s0_T16_lr0.001_k0.1_aug-full` | cnn_2frame | 0 | 2 | 0 |
| `cnn_2frame_s1_T16_lr0.001_k0.1_aug-full` | cnn_2frame | 1 | 2 | 0 |
| `cnn_2frame_s2_T16_lr0.001_k0.1_aug-full` | cnn_2frame | 2 | 0 | 0 |
| `cnn_avg_s0_T16_lr0.001_k0.1_aug-full` | cnn_avg | 0 | 19 | 0 |
| `cnn_avg_s1_T16_lr0.001_k0.1_aug-full` | cnn_avg | 1 | 20 | 0 |
| `cnn_avg_s2_T16_lr0.001_k0.1_aug-full` | cnn_avg | 2 | 25 | 0 |
| `cnn_avg_s3_T16_lr0.001_k0.1_aug-full` | cnn_avg | 3 | 17 | 0 |
| `cnn_avg_s4_T16_lr0.001_k0.1_aug-full` | cnn_avg | 4 | 23 | 0 |
| `cnn_avg_s5_T16_lr0.001_k0.1_aug-full` | cnn_avg | 5 | 16 | 0 |
| `cnn_avg_s6_T16_lr0.001_k0.1_aug-full` | cnn_avg | 6 | 6 | 0 |
| `cnn_avg_s7_T16_lr0.001_k0.1_aug-full` | cnn_avg | 7 | 28 | 0 |
| `cnn_avg_s8_T16_lr0.001_k0.1_aug-full` | cnn_avg | 8 | 25 | 0 |
| `cnn_avg_s9_T16_lr0.001_k0.1_aug-full` | cnn_avg | 9 | 2 | 0 |
| `cnn_linear_s0_T16_lr0.001_k0.1_aug-full` | cnn_linear | 0 | 0 | 0 |
| `cnn_linear_s1_T16_lr0.001_k0.1_aug-full` | cnn_linear | 1 | 2 | 0 |
| `cnn_linear_s2_T16_lr0.001_k0.1_aug-full` | cnn_linear | 2 | 0 | 0 |
| `cnn_mlp_s1_T16_lr0.001_k0.1_aug-full` | cnn_mlp | 1 | 2 | 0 |
| `cnn_mlp_s2_T16_lr0.001_k0.1_aug-full` | cnn_mlp | 2 | 0 | 0 |
| `lstm_s0_T16_lr0.001_k0.1_aug-basic` | lstm | 0 | 11 | 25,088 |
| `lstm_s0_T16_lr0.001_k0.1_aug-full` | lstm | 0 | 3 | 25,088 |
| `lstm_s0_T16_lr0.001_k0.1_aug-full_shuf` | lstm | 0 | 2 | 25,088 |
| `lstm_s0_T16_lr0.001_k0.1_aug-none` | lstm | 0 | 0 | 25,088 |
| `lstm_s0_T16_lr0.003_k0.1_aug-full` | lstm | 0 | 4 | 25,088 |
| `lstm_s0_T16_lr0.01_k0.1_aug-full` | lstm | 0 | 0 | 25,088 |
| `lstm_s1_T16_lr0.001_k0.1_aug-full` | lstm | 1 | 7 | 25,088 |
| `lstm_s1_T16_lr0.001_k0.1_aug-full_shuf` | lstm | 1 | 7 | 25,088 |
| `lstm_s1_T16_lr0.001_k0.1_aug-none` | lstm | 1 | 7 | 25,088 |
| `lstm_s1_T16_lr0.003_k0.1_aug-full` | lstm | 1 | 4 | 25,088 |
| `lstm_s1_T16_lr0.01_k0.1_aug-full` | lstm | 1 | 22 | 25,088 |
| `lstm_s2_T16_lr0.001_k0.1_aug-basic` | lstm | 2 | 0 | 25,088 |
| `lstm_s2_T16_lr0.001_k0.1_aug-full` | lstm | 2 | 2 | 25,088 |
| `lstm_s2_T16_lr0.001_k0.1_aug-full_shuf` | lstm | 2 | 4 | 25,088 |
| `lstm_s2_T16_lr0.001_k0.1_aug-none` | lstm | 2 | 6 | 25,088 |
| `lstm_s2_T16_lr0.003_k0.1_aug-full` | lstm | 2 | 0 | 25,088 |
| `lstm_s2_T16_lr0.01_k0.1_aug-full` | lstm | 2 | 14 | 25,088 |
| `lstm_s3_T16_lr0.001_k0.1_aug-full` | lstm | 3 | 11 | 25,088 |
| `lstm_s3_T16_lr0.001_k0.1_aug-full_shuf` | lstm | 3 | 0 | 25,088 |
| `lstm_s4_T16_lr0.001_k0.1_aug-full` | lstm | 4 | 6 | 25,088 |
| `lstm_s4_T16_lr0.001_k0.1_aug-full_shuf` | lstm | 4 | 9 | 25,088 |
| `lstm_s5_T16_lr0.001_k0.1_aug-full` | lstm | 5 | 7 | 25,088 |
| `lstm_s5_T16_lr0.001_k0.1_aug-full_shuf` | lstm | 5 | 5 | 25,088 |
| `lstm_s6_T16_lr0.001_k0.1_aug-full` | lstm | 6 | 9 | 25,088 |
| `lstm_s6_T16_lr0.001_k0.1_aug-full_shuf` | lstm | 6 | 5 | 25,088 |
| `lstm_s7_T16_lr0.001_k0.1_aug-full` | lstm | 7 | 11 | 25,088 |
| `lstm_s7_T16_lr0.001_k0.1_aug-full_shuf` | lstm | 7 | 1 | 25,088 |
| `lstm_s8_T16_lr0.001_k0.1_aug-full` | lstm | 8 | 3 | 25,088 |
| `lstm_s8_T16_lr0.001_k0.1_aug-full_shuf` | lstm | 8 | 1 | 25,088 |
| `lstm_s9_T16_lr0.001_k0.1_aug-full` | lstm | 9 | 2 | 25,088 |
| `lstm_s9_T16_lr0.001_k0.1_aug-full_shuf` | lstm | 9 | 1 | 25,088 |

### Escape rate and controls (test split, lr=1e-3, aug=full)

Collapse to a constant predictor is this task's dominant failure mode, so the escape
RATE is reported and metrics are conditioned on escaping. A median over a mixture of
escaped and collapsed runs describes neither.

| arm | variant | n | escaped | curve MAE | straight MAE | macro MAE | Pearson r |
|---|---|---:|---:|---:|---:|---:|---:|
| cfc | baseline | 12 | 7/12 | 25.55 | 5.76 | 14.28 | +0.143 |
| cfc | fixed-dt | 6 | 5/6 | 25.99 | 7.69 | 15.59 | +0.064 |
| cfc | shuffled | 6 | 6/6 | 25.43 | 9.19 | 16.31 | +0.116 |
| cnn_2frame | baseline | 3 | 3/3 | 27.06 | 5.23 | 13.25 | +0.099 |
| cnn_avg | baseline | 10 | 5/10 | 28.62 | 2.63 | 13.26 | -0.090 |
| cnn_linear | baseline | 3 | 2/3 | 26.67 | 4.88 | 13.30 | +0.122 |
| cnn_mlp | baseline | 2 | 1/2 | 26.61 | 5.31 | 13.42 | +0.087 |
| lstm | baseline | 10 | 4/10 | 25.09 | 6.78 | 14.06 | +0.079 |
| lstm | shuffled | 10 | 6/10 | 25.92 | 6.65 | 14.53 | +0.040 |
| **predict-0** | -- | -- | -- | 28.86 | 2.30 | 13.40 | -- |
