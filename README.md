# Steering-angle regression with a hand-written closed-form continuous-time RNN

> **The authors of this dataset ask that it never be used to drive a car.** Sully Chen's
> repository states, under a heading titled `# IMPORTANT`, that these datasets are for
> research and statistics and explicitly not for application or testing of any sort. The
> data itself carries a permissive MIT licence that would legally allow more than that; we
> honour the request rather than the licence. Every result below is **open-loop, offline
> prediction**. Nothing here was or should be put in a control loop. We redistribute no
> frames and publish no trained weights.

<!-- SKELETON NOTICE -- delete before submission.
     One heading per scored rubric item, in rubric order, with its point value. Sections
     whose inputs already exist are written; the rest carry a TODO naming the input that
     unblocks them and the workstream that owns them. `python scripts/readme_status.py`
     prints point-weighted completion. Fill each section the day its input lands -- do NOT
     save the writing for the final week. -->

---

## Introduction
<!-- RUBRIC: Introduction | 4 | readme | owner: writing | unblocked-by: nothing -->

We train a **many-to-many sequence regressor** that maps a variable-length sequence of
forward-facing RGB dashcam frames to a **per-frame steering-wheel angle in degrees**.
Input is `(B, T, 3, 66, 200)`; output is `(B, T, 1)`. `T` varies at both training and
evaluation time, and final test numbers come from a stateful rollout over contiguous
recording segments of arbitrary length. The target at step `t` is the angle recorded at
the **same** timestamp as frame `t`, so the task is estimating what the driver is doing
now, not planning what to do next.

The model is a PilotNet-style CNN encoder, weight-shared across timesteps, feeding a
**hand-written closed-form continuous-time (CfC) recurrent cell** with a shared linear
readout. The controls are a parameter-matched `nn.LSTM` at the same hidden width, and two
non-recurrent heads that bracket the recurrent arms from below and at matched capacity.

The question the project is built to answer is not "how accurate can we get". Accuracy alone
is uninformative here: about half the frames sit within ±5° of centre, so a constant-zero
predictor already scores 2.25° MAE on the test straight bin, and the three splits differ
sharply in how much cornering they contain — a single global number cannot be read.

The question is:

> **Under an identical visual encoder and a near-identical parameter budget, do past frames
> and the real inter-frame interval improve steering-angle estimation — and does a CfC differ
> from an LSTM?**

which decomposes into three nested ones, each with its own control:

1. **Does video history help at all?** A single-frame CNN and a two-frame CNN against the
   recurrent arms. If they tie, nothing downstream is measuring an architecture.
2. **Does the real `Δt` help?** True intervals against a fixed interval, on the same weights
   where possible.
3. **Does the CfC's update mechanism differ?** CfC against a parameter-matched LSTM — and
   against an LSTM that also receives `Δt`, because if only the CfC sees the time input then
   a gap cannot be attributed to the update equation.

We do not assume the CfC wins. The cited flight-robustness work reports parity in
distribution, so overlapping bands are the expected outcome; a tie or a loss with a fair
control and a clear explanation is a result, and the write-up is built to carry one.

## Model

### Model figure
<!-- RUBRIC: Model Figure | 4 | readme | owner: model | unblocked-by: nothing -->

> **TODO** — four panels, drawable today because none depends on results. Shapes come from
> the live forward hooks asserted in `tests/test_encoder_shapes.py`, so the figure cannot
> drift from the code. Produce as `figures/model.png` via `scripts/make_figures.py`.
>
> 1. **Encoder**, tensor shape on every arrow (verified):
>    `(3,66,200) → (24,31,98) → (36,14,47) → (48,5,22) → (64,3,20) → (64,1,18) → flatten 1152 → FC 32`
> 2. **One CfC step**: the 96-d concatenation, four labelled `Linear(96→64)` boxes of 6,208
>    parameters each, two `tanh`, `×Δt`, `+`, `sigmoid`, and the interpolation node. Every
>    arrow shaped, every box counted.
> 3. **Three-step unrolled band** with the state arrow and the shared `Linear(64,1)` drawn
>    once and dashed into each step.
> 4. **Predicted-vs-true trace** for one window.
>
> Anti-pattern, from `PROJECT.md §10`: ❌ do not draw a box labelled "CfC".

The forward pass, per step, with `x_t = concat([z_t, h_{t-1}])`:

```
cand_h = tanh(W_h x_t + b_h)                          equilibrium candidate
cand_g = tanh(W_g x_t + b_g)                          fast candidate
γ_t    = sigmoid((W_f x_t + b_f) ⊙ Δt_t               liquid time constant
                + (W_o x_t + b_o))                    time-independent offset
h_t    = (1 − γ_t) ⊙ cand_h + γ_t ⊙ cand_g            per-unit convex interpolation
y_t    = W_out h_t + b_out                            shared readout
```

`γ_t ∈ (0,1)` elementwise, so `h_t ∈ (−1,1)` **strictly** — which is why the readout is
mandatory rather than decorative against labels that reach ±100°. `h_0 = 0`; there is no
learnable initial state. `Δt_t` is the measured inter-frame interval in units of the
train-split median gap.

**What we implement versus what the paper prints.** We implement the reference form of
Hasani et al. (2022), not Eq. (4) literally, in four respects, and the differences matter
enough to state: our gate carries an additional learned time-independent offset `o(x_t)`
that the published equation does not contain; we use the no-backbone variant, so the three
heads are separate linear maps of `[z_t ; h_{t-1}]` rather than heads on a shared backbone;
the head the paper calls `f` is unconstrained in code, so the learned inverse time constant
may be negative and the paper's Theorem 1 does not cover the trained model.

The fourth difference is one we expected and the data contradicted. We had assumed this
footage would be near-uniformly sampled, which would leave `Δt` effectively constant and
reduce the gate to `sigmoid(affine(x_t))` — a different gating algebra at an identical
parameter budget, but not "continuous time" in any exercised sense. Measured, the frame
intervals run **28 / 57 / 101 ms at p1 / p50 / p99**, and they are quantised to integer
multiples of a 33.00 ms base: 47% of gaps are one interval, 47% are two, 6% are three.
**`Δt` here is a dropped-frame count, not jitter and not a constant**, so the CfC's
elapsed-time input carries a physically meaningful quantity and the mechanism is genuinely
exercised. Because only the CfC would otherwise receive that information, the matched LSTM
is also run with `Δt` appended as an input feature, so a performance gap cannot be
attributed to the update equation when it could come from the extra input. `tests/test_cfc_anchors.py` asserts that
reduction as a closed form, and the Results section reports a frame-dropped variant in
which `Δt` genuinely varies.

### Model parameters
<!-- RUBRIC: Model Parameters | 4 | readme | owner: model | unblocked-by: nothing -->

Every table in this subsection is **generated output** — `python scripts/param_table.py`
emits it from the same formula functions that `tests/test_param_counts.py` asserts against
`named_parameters()`. `make test && make params` is the answer to "how do I check this?".

```
CfC cell = 4 · [H·(I+H) + H] = 4·H·(I+H+1)        readout = H + 1
```

At the headline configuration `I = 32`, `H = 64`, `D = 96`:

| block | arithmetic | params |
|---|---|---:|
| `h_head` `Linear(96→64)` | 64·96 + 64 | 6,208 |
| `g_head` `Linear(96→64)` | 64·96 + 64 | 6,208 |
| `f_head` `Linear(96→64)` | 64·96 + 64 | 6,208 |
| `o_head` `Linear(96→64)` | 64·96 + 64 | 6,208 |
| **CfC cell** | `4·64·(32+64+1)` | **24,832** |
| readout `Linear(64→1)` | 64 + 1 | 65 |
| conv1 … conv5 + FC (shared encoder) | 1,824 + 21,636 + 43,248 + 27,712 + 36,928 + 36,896 | **168,244** |
| **total** | | **193,141** |

The recurrent layer is **12.9%** of the model and the shared perception encoder is 87%.
Stating that split is the point: the literature's "19 neurons can drive a car" refers to
control *neurons* in a sparse wiring fed by a separate convolutional head, never to a total
parameter count, and we make no such claim.

| arm | recurrent formula | H | recurrent params | vs CfC |
|---|---|---:|---:|---:|
| **CfC (ours)** | `4H(I+H+1)` | 64 | **24,832** | — |
| **`nn.LSTM` (baseline, unmodified)** | `4H(I+H) + 8H` | 64 | **25,088** | +256 = exactly `4H`, +1.03% |
| CNN + MLP head, capacity-matched | `32D + 2D + 1`, D=730 | — | 24,821 | −0.04% |
| CNN + Linear head (lower bracket) | `32 + 1` | — | 33 | — |

The `nn.LSTM − CfC = 4H` identity is exact at every width — 16: 3,136/3,200; 32:
8,320/8,448; 64: 24,832/25,088; 128: 82,432/82,944 — and is asserted at all four. It is
nothing but the redundant second bias vector torch carries for cuDNN's RNN convention, and
it is **in the baseline's favour**: we did not shrink our opponent.

**Parameter matching is necessary, not sufficient.** After matching, the remaining
differences are the update equation (intended), the absence of a separate additive cell
state in the CfC (a principled a-priori reason to expect the LSTM to hold its own), and the
explicit `Δt` term in the CfC's gate. Any gap is attributable to "this family at this
scale", not to continuous time as such.

Every arm reports trainable == total, with **zero frozen parameters**. One contrast worth a
footnote: the reference library reports 6,898 parameters for a CfC whose trainable count is
5,544 — a 24.4% overcount — because it registers its sparsity mask as an `nn.Parameter`
with `requires_grad=False`. We use `register_buffer`, asserted by identity in
`tests/test_param_counts.py`.

### Model examples
<!-- RUBRIC: Model Examples | 4 | readme | owner: writing | unblocked-by: P7 checkpoints + P13 aggregation -->

> **TODO** — one successful and one unsuccessful **16-frame window** from the test set, each
> as two panels: four sampled frames with predicted/true angle overlaid, plus a line plot of
> predicted versus true across all `T` steps annotated with that window's MAE *and* the
> predict-0 MAE for the same window.
>
> Selection rules, fixed in advance so the choice is not post-hoc:
> - The **success** case must come from a curve bin (`|y| ≥ 15°`), so it shows something
>   predict-0 cannot. A straight-road window where the model predicts ≈0 demonstrates
>   nothing beyond the baseline.
> - The **failure** case must come from a named, diagnosable mode, and the write-up must say
>   **when** the prediction diverges: early (perception failure) versus late (state/memory
>   failure). That distinction is direct evidence about whether the recurrent layer
>   contributes, so it belongs here and is cited again under Justification.
> - Publish both at model-input resolution and blur any legible face or licence plate.

## Data

### Data source
<!-- RUBRIC: Data Source | 1 | readme | owner: data | unblocked-by: download.sh (archive SHA-256) -->

SullyChen driving dataset, 2018 release: <https://github.com/SullyChen/driving-datasets>.
Frames were collected with a consumer camera mounted on a windshield on public roads on the
Palos Verdes Peninsula, California; labels are steering-**wheel** angles in degrees read
from the vehicle bus. Labels are one line per frame in `data.txt`:
`filename.jpg angle,YYYY-MM-DD HH:MM:SS:mmm` (note the millisecond separator is a colon).

> **TODO** — record the commit SHA, the access date, and the SHA-256 of the archive, all
> emitted by `data/download.sh`. Owner: data.

Licensing: the repository carries a formal `LICENSE` file — **MIT, Copyright (c) 2018 Sully
Chen** — which is permissive and contains no research-only restriction. The prohibition on
real-vehicle use is **not** a licence term; it is a request the author makes in the
repository's README. There is no data licence, no DOI, and no canonical citation. We state
the gap explicitly because it is the honest description: the licence permits this use, the
author separately asks that it never drive a car, and we honour the request.

Two dataset facts that are widely repeated and that we do **not** assert: the frame rate
(no primary source states one — we measure it from the timestamps) and the route length
(the "6 km" figure comes from a third-party paper that used only the first 40,000 frames).

### Data summary
<!-- RUBRIC: Data Summary | 4 | readme | owner: data | unblocked-by: stats.txt (P1) -->

> **TODO** — fill from `stats.txt`, produced by `python data/eda_sullychen.py --data-root
> data/raw --images`. Every number in this section must be measured by us, not quoted.
> Required, because each one is load-bearing for a later interpretation:
> - frame count from `data.txt` line count; archive size; verified-uniform image dimensions
> - **measured** duration and effective fps from the timestamps; Δt median/p1/p99; the count
>   and locations of gaps > 0.5 s; the number of contiguous segments
> - `|angle|` fraction below each of {1, 2, 5, 15, 40, 100}°, and the true min/max/p1/p50/p99
> - global predict-0 **and** persistence MAE, and the per-bin predict-0 MAE
> - lag-1 and lag-300 angle autocorrelation; median adjacent-frame |Δangle| and the fraction
>   below 1° — these are the **irreducible-error** ingredients the Justification section uses
> - the **(split × bin) table**: frames, percentage, wall-clock minutes,
>   **independent turn events**, mean `|angle|`, and per-split per-bin predict-0 and
>   persistence MAE
> - the likely-stationary-frame fraction and run-length distribution, per split
> - the revisit audit: max off-diagonal descriptor similarity, and the test-to-train
>   nearest-neighbour distance histogram
>
> The turn-event count is not decoration. 200 correlated frames from one corner are one
> sample, not 200, so every results table carries the event count in its column header.

### Data transformation
<!-- RUBRIC: Data Transformation | 3 | readme | owner: data | unblocked-by: nothing -->

Frames are decoded exactly once, by `data/preprocess.py`, into a `uint8` memmap; the
training loop never touches a JPEG. The steps, in order:

1. **`data.txt` is parsed in order and line *i* becomes memmap row *i*.** We never glob the
   image directory. ~161 images in the archive have no label line, so any positional join over
   the directory is silently misaligned — and a uniform frame/label off-by-one passes *every*
   summary statistic in the section above. `--verify-gif` renders a ten-second strip with the
   true angle drawn as a needle; watching it lean the same way the road curves is the only
   check in the project that catches that failure. Label lines whose image is absent are
   recorded in the manifest as `missing_rows` and the window sampler refuses any window
   containing one, rather than training on a black frame with a real label.
2. **Crop the bottom 150 rows** of the 455×256 source, which discards sky and most of the
   hood. This matches the reference loader's `[-150:]`; we cite that provenance rather than
   inventing a crop.
3. **Resize to 66×240 and store as `uint8`** (2.99 GB). The model input is 66×**200**: the
   extra 40 px of width exists so the horizontal-shift augmentation has real pixels to shift
   into. Storing at the model width would force every shift to edge-pad, and the padding
   band's width is then a perfect linear function of the label correction — the model can
   regress the label off a border artifact instead of off the road, which makes the
   augmentation ablation meaningless. `uint8` is about eliminating per-epoch JPEG decode, a
   CPU cost; it is not a VRAM decision, since float32 would also fit.
4. **Cut the frame index into segments at timestamp gaps** larger than 0.5 s. The split buffer
   guards split boundaries only, so without this a 16-frame window could straddle a
   multi-second recording break and be a legal sample with a discontinuous label.
5. **Normalisation statistics are computed on the train split only**, written into
   `manifest.json` once, and read identically by training, evaluation and the perturbation
   grid. Targets are standardised and inverted at report time so every reported MAE is in
   degrees. Standardisation is not cosmetic: the recurrent state is bounded to (−1,1), and
   fitting raw degrees against a ±100° tail reaches only ~14° of error on a sharp-turn frame
   after 400 steps where standardised targets reach 0.0° — a *scale* problem that reads
   exactly like "the recurrent cell cannot handle sharp turns".

Augmentation runs on the GPU, batched, inside the training step — there is no `DataLoader`.
**Every augmentation parameter is drawn once per window and applied identically to all `T`
frames.** Parameters are sampled at shape `(B,1,1,1,1)` against a `(B,T,C,H,W)` batch, so the
invariant is what broadcasting does by default and the per-frame variant is the one that takes
extra effort to write. Written the usual way — a transform inside `Dataset.__getitem__` — each
frame draws its own shift, its own flip decision and its own brightness, so the camera
teleports laterally every ~34 ms and a flip negates the labels of only some frames. That
failure is silent, it is asymmetric (it corrupts every recurrent arm and leaves the
single-frame CNN untouched), and it manufactures exactly the conclusion "temporal information
does not help here". `tests/test_augmentation_windows.py` asserts the invariant and asserts
that the `per_frame_bug=True` flag breaks it, so the deliberate control run reported under
Results is a measured instrument rather than an accident.

The three transforms: **horizontal translation** with geometric steering compensation at `k`
deg/px (random 200-wide crop from the stored 240, centre crop at validation and test);
**horizontal flip** with the entire label sequence negated; and **brightness and shadow
jitter**, which are photometric and do not touch the label. The last two are why brightness
appears in the *seen-in-training* column of the perturbation table.

**On the translation coefficient `k`.** The NVIDIA precedent derived its label correction
from a calibrated three-camera rig plus vehicle speed. This dataset publishes neither
camera calibration nor a speed channel, and the steering ratio is undisclosed, so a
degrees-per-pixel constant cannot be derived here — asserting one would be a claim a reader
can falsify. We therefore state the assumptions (flat ground, fixed lookahead, constant
speed), state the coefficient with its units, and treat `k` as one of our three searched
hyperparameters, reporting validation curve-bin MAE against `k ∈ {0, 0.05, 0.1, 0.2, 0.4}`.
An empirically identified constant with a sensitivity curve is a stronger claim than an
asserted derivation that does not hold. We also note that horizontal flip produces a
left-hand-traffic world that does not exist in the deployment domain.

### Data split
<!-- RUBRIC: Data Split | 2 | readme | owner: data | unblocked-by: stats.txt (P1) -->

The split is **strictly chronological**, 60/20/20, with a discarded buffer band of 300
frames at each boundary, and window sampling refuses to cross a split boundary, a buffer
band, or a timestamp-gap segment boundary.

The ratio is not the conventional one, and the reason is measured. Sharp turns are
structurally concentrated in the middle of this recording and the final 30% contains almost
none, so at 70/15/15 the validation split held only **6 independent curve events** —
checkpoint and hyperparameter selection would have rested on a handful of corners. 60/20/20
gives validation 17 and test 19, at the cost of 14% of the training frames.

This is the one place where following the community would be a mistake. The widely-copied
loader for this dataset shuffles before splitting 80/20, with no seed and no test set. At
~30 fps adjacent frames are nearly identical, so every validation frame has a near-duplicate
neighbour in training and the reported metrics are badly inflated. Most published numbers on
this dataset inherit that, which is also why our numbers should look worse than theirs.

> **TODO** — from `stats.txt`: the measured decorrelation lag and therefore the buffer width
> (`max(first lag with |r| < 0.1, 300)`); the frame ranges and durations of each split; and
> the result of gate **G1**. If the sharp-turn test cell holds fewer than ~10 independent
> turn events, we merge `[15,40)` with `[40,∞)` and say so here; if a split contains almost
> no hill section we re-cut the boundaries **by hand, still chronologically**. We do not
> block-rotate and we do not hold out geographically: on a single short route that places
> train and test blocks on the same road minutes apart, which is a worse leakage violation
> than the one this split exists to fix. Owner: data.

## Training

### Training curve
<!-- RUBRIC: Training Curve | 4 | readme | owner: model | unblocked-by: first P7 run -->

> **TODO** — `figures/training_curves.png`. Plot **validation binned MAE**, which is the
> same quantity early stopping and checkpoint selection use; plot every arm on shared axes.
> Two sentences must accompany it or it reads as a bug:
> - With augmentation active, training loss sits **above** validation loss for much of
>   training.
> - Per-epoch validation uses batched `T=16` windows and therefore includes cold-start
>   frames, so it reads slightly **worse** than the final stateful-rollout number.
>
> Selection is on validation binned MAE, **never** global MSE — global MSE selects the most
> shrunken checkpoint of every run and silently undoes the entire point of per-bin metrics.
> Both are logged every epoch, so the selection counterfactual is free.

### Hyperparameter tuning
<!-- RUBRIC: Hyperparameter Tuning | 4 | readme | owner: model | unblocked-by: P5 + P6 grids -->

Exactly **three** hyperparameters are searched, on the CfC only: learning rate
`∈ {3e-4, 1e-3, 3e-3}`, window length `T ∈ {8, 16, 32, 64}`, and the translation
coefficient `k ∈ {0, 0.05, 0.1, 0.2, 0.4}` deg/px. `T` doubles as the variable-length
evidence and as a direct measurement of how much temporal context is worth; `k` doubles as
the augmentation calibration above.

Everything else is **fixed by rule and declared as fixed**, not implied to have been swept:
AdamW, weight decay 1e-4, `clip_grad_norm_` 1.0, batch 64 sequences, cosine decay, 30
epochs, patience 10, hidden width 64. Batch size was chosen for optimisation quality, not
hardware utilisation — the optimizer-step count is a confound in every comparison here, so
we deliberately did not spend available VRAM on it.

The winning learning rate is transferred to every arm. That is a declared trade-off, and
because it invites the objection that we tuned for the CfC, we measured the objection
instead of arguing about it.

> **TODO** — the run table (one row per run: config hash, hyperparameters, seed, git SHA,
> torch/ncps versions, validation global MAE, validation per-bin MAE), pasted from
> `runs/grid.csv`; the three one-dimensional marginal slices; the `k` curve with its interior
> minimum; and the shared-versus-own learning-rate comparison. State that selection used
> validation only. Owner: model.

## Results

### Quantitative measures
<!-- RUBRIC: Quantitative Measures | 2 | readme | owner: writing | unblocked-by: stats.txt (P1) -->

The primary view is **MAE in degrees, binned by the magnitude of the ground-truth angle**,
with a constant-zero baseline in every table. The headline scalar is the **macro skill
score**, `1 − mean over bins of (MAE_model / MAE_predict0)`, which pins predict-0 at exactly
0 and cannot be won by shrinking predictions toward the mean. We also report global MAE,
Pearson *r*, per-bin p95 absolute error, and the false-alarm rate `P(|ŷ| > 5 | |y| < 5)`.

Three protocol decisions, stated because each silently changes the headline number:
bins are assigned by **ground-truth** angle; metrics are computed **per frame** over all
timesteps; and final numbers come from a **stateful contiguous rollout** over each test
segment, which covers every test frame exactly once with no double counting.

**Predict-0 is a demanding target in the straight bin, and a small margin there is expected
rather than a defect.** It is not a floor, and we do not claim it is: the constant-zero
predictor scores 2.25° MAE on the test straight bin, and a model that reads the road can beat
that — small angles are still predictable. What we expect is that the achievable margin in
that bin is narrow, because the bin is defined by the target already being near zero. We say
this before the table rather than after a reader asks. Persistence
(`ŷ_t = y_{t−1}`) beats us too; it is reported, and then disposed of in one paragraph,
because the model receives images only and no past ground-truth angles — persistence is not
a solution to the posed task and is unavailable the moment labels are absent, which is
always, at deployment.

> **TODO** — the bin boundaries, derived from the angle histogram's own structure and from
> the requirement that every bin have adequate **test** support, plus the commit SHA in which
> these measures and the decision rule were frozen. Owner: writing, on the day `stats.txt`
> exists, **before** the first architecture comparison run.

### Quantitative and qualitative results
<!-- RUBRIC: Quantitative and Qualitative Results | 8 | readme | owner: writing | unblocked-by: P7 + P13 -->

> **TODO** — this item is graded on **clear presentation**, so the budget is one headline
> table, a small number of figures, and an appendix. Everything else goes in `runs/`.
>
> **Table 1.** Rows: predict-0, persistence, a rule, then the models. Columns: the bins plus
> overall Pearson *r*, each column header carrying that bin's test-frame count **and**
> independent-turn-event count. Each cell gives MAE in degrees with the same-bin ratio to
> predict-0 in parentheses — `3.9° (0.62×)` — so the "must not lose to a naive baseline"
> question is answered without the reader dividing. Best per column in bold; seed *n* and
> band width in a footnote.
>
> **Figure 1 (headline).** Per-frame MAE versus position within the evaluation window,
> `t = 1…16`, on the frozen `|y| ≥ 5°` subset. One line per arm (median over seeds, min-max
> band, individual runs as dots), plus each recurrent arm's shuffled-frame-order twin — the
> vertical gap between a line and its twin is the shuffle control drawn as a curve. Predict-0
> and persistence as horizontal references; `t = 1…3` shaded "cold start". The three
> non-recurrent lines must come out **flat**: that is the figure's own self-test, and a
> non-flat one means the position accounting is broken.
>
> **Figure 2.** Effect ladder: every intervention's ΔMAE against the seed band, with rules
> at 1× and 3× the band width. Marker shape separates retrained from same-weights-evaluated-
> differently; colour separates temporal structure / augmentation / architecture / protocol.
>
> **Figure 3.** OOD degradation, 2×2 shared-y grid, with perturbations partitioned into
> seen-in-training and unseen. The *y*-axis quantity is defined in prose **before** any curve
> is drawn.
>
> Demote GRU, LTC, RMSE, seed variance and the augmentation ablation to an appendix or a
> collapsed block. Everything regenerates with `python scripts/make_figures.py && python
> scripts/make_tables.py`.

### Justification of results
<!-- RUBRIC: Justification of Results | 20 | readme | owner: TBD (must be a NAMED person) | unblocked-by: P7 + P13 -->

> **TODO — the single largest item in the marking scheme, 20 points, and it needs a named
> owner and three dedicated days. Nobody is "the one who writes things".** It is graded on
> interpretation anchored to two named things: the data summary and the hyperparameter
> choices. Cite both sections **by name** in the prose; a reader is scanning for them.
>
> Five subsections, in this order, with robustness **last**:
>
> 1. **What counts as reasonable here.** Derive a noise floor from the measured
>    adjacent-frame angle-difference distribution and the autocorrelations. Name persistence
>    as the naive baseline a reader will actually think of, report it, and dispose of it.
>    **State the target range before showing our number.**
> 2. **Is our number reasonable?** Against predict-0, persistence and the noise floor, per
>    bin, citing the (split × bin) table by name.
> 3. **Where does it fail, and why?** Per bin and per named failure mode, each tied to a
>    specific data-summary fact — the stationary-frame mass inside the straight bin, the
>    turn-event count in the sharp bin, the terrain composition of each split.
> 4. **What did the hyperparameters buy?** Citing the tuning table by name.
> 5. **Robustness as additional analysis**, explicitly not the spine.
>
> **Plan for a null.** The cited flight-robustness work reports a CfC advantage mainly
> *out* of distribution and parity *in* distribution, so overlapping CfC and LSTM bands are
> the expected outcome, not a failure. Frame it that way in advance. Four things still carry
> a first-order result if the two lines coincide: recurrence versus none, the predict-0
> floor, the train-time frame-shuffle gap, and the two-frame CNN — which supplies a
> *mechanism* for a null, since a ~25k-parameter recurrent layer of either family can
> represent a first difference, and on 30 fps single-route video there may be little beyond
> a first difference to find.

### Advanced concept
<!-- RUBRIC: Advanced Concept | 10 | advanced | owner: data | unblocked-by: P8 ablation -->

Domain-adapted data augmentation: horizontal translation with geometric steering
compensation, plus flip-with-label-negation and photometric jitter. See **Data
transformation** for the coefficient calibration, which is the substance of the claim.

> **TODO** — the ablation: `{none, flip+brightness only, full at k*}` × `{CfC, LSTM}`, with
> per-bin MAE and special attention to the curve bins. Report the deliberate per-frame-
> augmentation control alongside it. Get written confirmation from the instructor in week 1
> that domain-adapted augmentation satisfies this item, because the cheap alternative — a
> causal-attention arm — cannot perform the stateful rollout every other arm shares and
> would force a second evaluation protocol. Owner: data.

## Ethical consideration
<!-- RUBRIC: Ethical Consideration | 4 | readme | owner: writing | unblocked-by: nothing -->

The rubric asks three things, so this section answers them under three headings.

### A use that could give rise to ethical issues

Different groups are exposed differently, and the asymmetry is the point. **The in-car
driver** retains takeover authority and chose to be there. **Pedestrians and cyclists**
bear the physical risk, appear in no training signal whatsoever — the only supervision is a
scalar steering angle, with no annotation of vulnerable road users — and cannot opt out.
**Bystanders filmed on public roads** never consented; a fixed, timestamped, short route
also reveals one identifiable person's daily routine. **The single demonstrator** whose
driving is cloned transmits their habits, including the unsafe ones, to every copy of the
model. **Professional drivers** face the labour-market consequences of systems like this at
scale, and bear none of the upside. And **anyone who does not know the technology's limits**
is exposed to automation bias: good behaviour on a familiar route invites the overtrust that
makes the rare failure worse.

One benefit worth stating, with its tension intact: a recurrent controller small enough to
enumerate its state variables is more auditable after an incident than a
hundred-million-parameter backbone — and that same auditability argument is routinely used
to market systems that are not safe. A small model is not a safe model. The claim also
scopes only to the 24,832-parameter recurrent controller; the 168,244-parameter CNN frontend
in front of it remains opaque.

Finally, a misreading of our own result to pre-empt: **robustness under controlled synthetic
corruption is not evidence of safe deployment.** Our perturbations are photometric and
spatial, not semantic — they model sensor degradation, not a change in who is on the road —
and we do not claim the latter.

### Limitations of the model

Behaviour cloning never observes recovery from its own mistakes, so open-loop test MAE says
little about closed-loop behaviour. The model is a point regression with **no uncertainty
estimate**: it cannot signal "I don't know", which is the one output that would matter most.
`T` frames at the measured frame rate is under a second of context, so intent, right of way,
and anything requiring longer-horizon reasoning are not representable. And the model
predicts the *current* angle rather than a future command, so it is a perception model
wearing a controller's clothes.

### Limitations of the training data

A single short route, one driver, one vehicle, one camera, daytime, clear weather. The
generalisation boundary is therefore extremely narrow, and it is narrow in a specific
direction: an affluent Californian suburb with good lane markings, paved roads and
consistent signage. There is no speed channel, so stationary frames cannot be filtered and
sit inside the most common bin, inflating both the near-zero mass and the apparent strength
of the constant-zero baseline. The degree scale itself is a hand-fitted two-point linear
calibration the author describes as a rough approximation, and these are steering-**wheel**
degrees, not road-wheel degrees.

### What we did about it

- Offline prediction only; nothing is placed in a control loop.
- No data committed to this repository, including frames inside `figures/`.
- No trained weights published — the handout permits this — so the model cannot be picked up
  and run by someone who has not read this section.
- The author's no-real-vehicle request is reproduced at the **top** of this README, so the
  restriction is passed downstream rather than treated as discharged by our own compliance.
- The two required example frames are published at model-input resolution with any legible
  face or plate blurred.
- A constant-zero baseline appears in every results table, and negative results are reported
  as measured, with curves unretouched.

## Authors
<!-- RUBRIC: Authors | 2 | readme | owner: writing | unblocked-by: team names -->

> **TODO** — four workstreams, each with a **named owner and a named backup**, written once
> here and reused verbatim in the proposal. Every row needs a deliverable and a done-when
> condition, not a topic: the bar is "if a team member is replaced, they know exactly what
> their responsibilities are".
>
> | workstream | deliverables | owner | backup |
> |---|---|---|---|
> | data | download + SHA check, EDA and `stats.txt`, preprocess to memmap, split audit, augmenter + its unit tests, README Data sections | TBD | TBD |
> | model | encoder, CfC cell + anchors + oracle, arm factory, training loop, run tracking, README Model sections | TBD | TBD |
> | harness | `evaluate.py`, `ood.py`, `make_tables.py`, `make_figures.py` — built and unit-tested against random tensors in week 1, before any real model exists, so it never blocks on the model track | TBD | TBD |
> | writing | README skeleton, Ethics, Introduction, **Justification (20 pts — named owner, 3 days)**, code documentation, clean-clone replication read-through by whoever did not write it | TBD | TBD |
>
> Also record: which steps were pair-coded and by whom. The handout recommends pair coding;
> the single-batch overfit gate is the natural place for it.

## Reproducing our results
<!-- RUBRIC: Code/Documentation (repo-wide) | 20 | code | owner: harness | unblocked-by: train.py + make_figures.py -->

```bash
pip install -r requirements-dev.txt
python scripts/smoke_test.py          # day-zero gate: asserts, does not print
make test                             # 55 unit tests, a few seconds
make test-all                         # + the single-batch overfit gate (~45 s)
bash data/download.sh                 # gdown + SHA-256 + file/row count audit
python data/eda_sullychen.py --data-root data/raw --images
python data/preprocess.py --data-root data/raw --out data/processed --verify-gif 1200
python train.py --arm cfc --seed 0    # TODO
python scripts/make_tables.py && python scripts/make_figures.py
```

`scripts/smoke_test.py` comes first for a reason. On Blackwell-class hardware with a driver
older than the torch build expects, torch installs cleanly, `torch.cuda.is_available()`
returns `True`, the card is named correctly — and then every kernel fails. The smoke test
asserts `sm_120` is in the build's arch list and runs a **real on-device backward pass**,
checking that every trainable parameter receives a finite, non-zero gradient. It also
asserts the parameter breakdown, that no parameter is frozen, and that the `Δt` shape
contract rejects the wrong shapes.

**The recurrent cell is written from scratch.** `models/` imports no liquid-network library,
and `tests/test_hygiene.py` enforces that by grep rather than by promise.
`tests/test_cfc_anchors.py` asserts the seven claims the model figure makes — among them
that the gate drives the state toward the fast candidate as `Δt` grows, and that with `Δt`
held constant the cell reduces exactly to an ordinary input-and-state-dependent gated RNN —
against closed forms, with no reference implementation involved. `tests/test_cfc_oracle.py`
then loads identical weights into our cell and into the authors' reference implementation
(`ncps==1.0.1`, a test-only dependency) and asserts agreement of the forward pass, all eight
named parameter gradients, the backbone path, the masked path and the full `T`-step wrapper:
**bit-exact in float64, to 1e-6 in float32.**

The ordering is deliberate and the anchors come first, because the oracle is the test a wrong
implementation can be made to pass. Since `sigmoid(−u) = 1 − sigmoid(u)`, inverting the
interpolation and swapping two head names reproduces the reference **exactly** — a wrong cell
plus a compensating bijection is a fixed point of the whole weight-transfer procedure. We
keep that counterexample as a live test
(`test_oracle_alone_cannot_catch_the_inverted_cell`) paired with the anchor that does catch
it (`test_inverted_cell_is_caught_by_gate_limit`). The margin between identical algebra
(0.0) and any real bug (≥0.1) is eleven orders of magnitude; a tolerance that has to be
tuned is a tolerance that hides failures.

This proves our cell **is** the reference implementation. It does **not** prove the reference
is Eq. (4) of the paper — it is not; see **Model figure** for the four differences.

## References

- Bojarski et al., *End to End Learning for Self-Driving Cars*, 2016 (PilotNet).
- Lechner et al., *Neural circuit policies enabling auditable autonomy*, Nature Machine
  Intelligence, 2020.
- Hasani et al., *Liquid time-constant networks*, AAAI 2021.
- Hasani et al., *Closed-form continuous-time neural networks*, Nature Machine Intelligence
  4, 992–1003, 2022 (arXiv:2106.13898). The cell we implement.
- Chahine et al., *Robust flight navigation out of distribution with liquid neural networks*,
  Science Robotics, 2023. Cited for the perturbation-sweep design; their metric is output
  self-consistency rather than error against ground truth, and their occlusion test was
  physical. We credit the digital occlusion axis as our own extension.
