# Datasets

Load when generating, reading, or reasoning about a dataset.

## Where they live

```
DATA_ROOT = /common/users/shared/pracsys/genMoPlan/data_trajectories
```

Shared with other people. Read freely; never write by hand — a `PreToolUse` hook
denies it. To change a dataset, change the generator and re-run.

Pendulum layout (`generate_inverted_pendulum_trajectories.py::default_output_dir`):

```
$DATA_ROOT/deterministic/pendulum/<controller>/
$DATA_ROOT/noisy/pendulum/<controller>/<level>/
```

`<controller>` is `lqr` or `<variant>_<strength>` for the RL policies —
variants `v1..v4`, strengths `strong`/`weak`. `<level>` is the noise preset's
trailing level (`control_proportional_med` → `med`).

**What the noisy pendulum success rates do and do not mean.** These datasets use
`pendulum_noise.py`'s state-additive families, which write `eps` into
`(theta, theta_dot)` directly. Measured 2026-08-05 over the whole state space at
161x161, LQR: success *rises* with noise (0.386 deterministic → 0.431 at
`velocity_proportional_high`), where a torque disturbance of comparable strength
*lowers* it (→ 0.256). One draw moves theta by ~0.083 against a goal radius of
0.075, so the noise can place the state in the goal set instead of the controller
driving it there.

Consequences for anyone reading these numbers:

- A rising success rate at a higher noise level is **not** evidence the controller
  is more robust. Part of the rate is the noise hitting the target.
- It cannot be fixed by re-scoring. Re-running with a per-channel box and a
  10-step dwell requirement still gives 0.426 vs 0.386, because
  `velocity_proportional`'s sigma scales with `|theta_dot|` and collapses to 0.008
  at the goal — the noise carries the state in, then switches off. The inflation
  is in the trajectories, so only re-collection changes it.
- They remain usable as conditional distributions to learn from; the caveat is on
  interpretation, not on validity as training data.
- **They are not comparable to the cartpole noisy datasets.** Different mechanism
  (state write vs force), different units (rad and rad/s vs Newtons), different
  sampling rate (per integrator substep vs per control step, a ~7x difference in
  per-control-step sigma). A `med` in one says nothing about a `med` in the other.

See `.claude/docs/architecture.md` for the mechanism and the three candidate
fixes.

The older systems predate that scheme and take an explicit `--output_dir`:
cartpole requires one; quadrotor-2D RL defaults to `$DATA_ROOT/quadrotor2D_rl`,
quadrotor-3D LQR to `$DATA_ROOT/quadrotor3D_lqr`.

## Generators

| Script | System | Controller |
| --- | --- | --- |
| `generate_inverted_pendulum_trajectories.py` | inverted pendulum | LQR + RL, deterministic or noisy |
| `generate_cartpole_trajectories.py` | cartpole | LQR |
| `generate_quadrotor_2d_trajectories.py` | quadrotor 2D | LQR |
| `generate_quadrotor_2d_trajectories_rl.py` | quadrotor 2D | trained RL policy (`--algo`, `--model_path`) |
| `generate_quadrotor_3d_trajectories.py` | quadrotor 3D | LQR |

The pendulum generator is the most developed and the one that carries the
current design; the others still follow the older single-pass scheme. When the
two disagree, the pendulum generator and the spec that produced it are the
reference.

There is no shared output contract. All five write `roa_labels.txt` and
`trajectories/sequence_<i>.txt`; beyond that they diverge. Only
`generate_quadrotor_2d_trajectories_rl.py` writes `eval_states.txt` and
`trajectory_labels.txt` (`generate_eval_states_and_labels`, line 817). Only it
and the pendulum generator write `dataset_description.json` — both added on
2026-07-16 as *independent* implementations, never a shared helper, and never
backported to the three collectors written in December 2025. The pendulum alone
writes `success_probabilities.txt` and split-scoped descriptions.

Datasets that carry files their own collector cannot produce were backfilled
afterwards by scripts living outside this repo, under `DATA_ROOT`'s parent:
`generate_eval_states.py` (KD-tree matches an `roa_labels.txt` row to a
trajectory at a `1e-4` threshold to recover its final state),
`generate_roa_labels.py` (stratified sampling for large state spaces such as
quad3d's 12-D), and `dataset_split_randomizer.py` (`train_test_splits/`, seed
42, train = 20,000 when total >= 25,000 else 80%). They are not version
controlled here. `cal_set.txt` and `test_set.txt` have **no** producer anywhere.

### Every shipped quadrotor dataset ran at the wrong damping

`base_aviary.py`'s `changeDynamics` call omitted `physicsClientId`, so it
targeted PyBullet client 0. The quadrotor collectors hold two envs at once — LQR
builds one from `env_func`, then the collector builds the one it rolls out — so
the *rollout* env never received `linearDamping=0, angularDamping=0` and ran at
PyBullet's default instead.

Measured against a single-env reference, five steps, seed 7: the rollout env
deviates `0.069001` without the fix, `0.000000` with it. Only the quadrotors are
affected; `base_aviary.py` is theirs alone, and the cartpole slice still
reproduces.

Fixed in the library, because SB3's `EvalCallback` holds a second env open and
would otherwise train against corrupted dynamics. The fixtures are deliberately
**not** regenerated — they record what the shipped datasets contain — and
`test_slice_reproduces` is `xfail(strict=True)` for `quad2d`, `quad2d_rl` and
`quad3d` so that regenerating them fails the suite instead of quietly blessing
the change. Whether to re-collect is an open decision.

## Success labelling — the thing to get right

The downstream model predicts terminal states, so **the success label must be a
function of the terminal state.** Two regimes:

**Invariant terminal sets (`--invariant_terminal_sets`, default off).**
Success = the terminal state lies inside a strictly invariant ellipsoid
`{(s-s0)' P (s-s0) <= c}` around the closed-loop attractor, and successful
rollouts run to a fixed horizon so the terminal state settles deep inside it.
Euclidean goal balls are *not* invariant here: every one of these closed loops is
non-normal, so a state entering a radius-R ball transiently excurses to a
multiple of R before converging (pendulum 2.6x, quad2D 3.2x, quad3D 4.9x,
cartpole 5.4x, roughly constant in R). Enlarging the radius does not fix it.
The ellipsoids come from finite-differencing the true closed-loop step map at
the attractor and solving the discrete Lyapunov equation. Rationale and
measurements: `plans/invariant-terminal-sets-recollection.md`.

Fixed horizons live in `DEFAULT_HORIZON = {'lqr': 600, 'rl': 1100}`.

**Noisy collection (entry-cut).** Under the state-additive presets the state
never settles, and at `high`/`xhigh` the stationary noise floor exceeds the 0.075
goal radius — so no invariant success set exists for *those* and the invariant
scheme does not apply. That is a fact about the mechanism: under torque noise the
loop is confined (see `glossary.md`, noise floor). Success =
the rollout *ever entered* the 0.075 L2 ball; a stored trajectory is cut at (and
includes) that entry state. Non-successes run the full 1000-step horizon and are
labelled 0. Rationale: `docs/superpowers/specs/2026-07-25-noisy-pendulum-collection-design.md`.

## Splits (pendulum, `--split`)

**`train`** — 300,000 rollouts from random starts, `theta ~ U(-pi, pi)`,
`theta_dot ~ U(-2pi, 2pi)`. Full trajectories stored. Random starts because
training wants coverage of the continuous state space.

**`eval`** — repeated batches over a fixed grid, storing only a per-cell success
probability. One batch = one rollout from every grid state. Grid is half-open,
`lo + resolution * arange(ceil((hi - lo) / resolution))`, resolution 0.04 →
158 x 315 = **49,770** states. Half-open is correct, not merely convenient:
theta is periodic, so -pi and +pi are the same physical state.

Stopping rule: after every `--check_every` batches compute
`mean_se = mean_i sqrt(p_i (1 - p_i) / B)`; stop when `mean_se < --se_tol`
(default 0.01), bounded by `--min_batches` (10) and `--max_batches` (500).
It is near-monotone in B, so it cannot trip early by chance. Drift is logged as a
diagnostic but does **not** gate stopping.

Storing the probability instead of every rollout is ~2,600x smaller than the
scheme it replaced (a 4.26 GB `eval.npz` whose only use was a success rate per
cell) and is a better estimator, because batch count is no longer pinned.

The two splits are independent processes, meant to run concurrently.

## Formats

`train.npz`:

| key | dtype | shape | notes |
| --- | --- | --- | --- |
| `states` | float32 | (M, 2) | all trajectories concatenated |
| `offsets` | int64 | (N+1,) | trajectory `i` is `states[offsets[i]:offsets[i+1]]` |
| `starts` | float64 | (N, 2) | sampled initial states |
| `labels` | uint8 | (N,) | 1 = reached goal, 0 = timeout |
| `seeds` | int64 | (N,) | per-rollout seed; enables exact regeneration |

float32 for `states` is measured, not a default: its 2.4e-7 max error is three
orders of magnitude below the smallest real per-step state change. int16
fixed-point was rejected because it quantises near-equilibrium motion — exactly
what the terminal-state model cares about — into noise. DEFLATE was rejected for
8% gain at decompression cost on every read.

`eval_success_prob.npz`: `starts` (49770, 2) float64, `successes`/`trials`
int32, `p_success` float64, `grid_theta` (158,), `grid_theta_dot` (315,),
`grid_shape`, `n_batches`. Plus a `success_probabilities.txt` mirror
(`theta,theta_dot,p`, 6 decimals).

Older systems write `trajectories/sequence_<i>.txt` plus a `roa_labels.txt` of
`(init_state, label)` in the parent directory.

Every dataset directory carries a `dataset_description.json`. Split
descriptions are named per split so concurrent train and eval runs cannot
clobber each other.

## Seeding

`rollout_seed(base_seed, split_id, index, batch)` returns
`np.random.SeedSequence([...]).generate_state(1, uint32)[0]` — a pure function of
the rollout's coordinates. This is what makes a resumed run draw exactly what an
uninterrupted run would have drawn. Do not replace it with a sequentially
advanced RNG.

## Incremental publication (eval)

The published dataset *is* the checkpoint; there is no finalize step. After each
batch the generator folds in 49,770 outcomes, recomputes `p_success` and
`mean_se`, and writes each file to `<name>.tmp` followed by `os.replace()` —
atomic within a filesystem, so a SIGKILL mid-write cannot tear a file. Only
whole batches are published, so every cell always has equal `trials`.
`dataset_description.json` records `converged: true|false`, so a killed run is
labelled honestly rather than silently looking finished.

## Invariant set artifacts

`invariant_sets/{pendulum,cartpole,quad2d,quad3d}.npz` hold `P`, `center`, `c`.
Committed, loaded by the generators at startup, regenerated with:

```bash
python compute_invariant_sets.py --systems pendulum cartpole quad2d quad3d
```

`--skip_validation` skips the boundary-sampling check that verifies `V` never
exceeds `c` and all samples converge. Do not skip it when publishing a new
artifact.

The four `.npz` are mode `-r--r-----` on disk on purpose. The script rewrites
them in place, so a test or stray run that recomputes them would silently
clobber committed artifacts; read-only makes that fail loudly instead.

### quad2d's ellipsoid is not reproducible, and that is expected

Recomputing with current code reproduces `pendulum` and `cartpole` bit-exactly
at `atol=1e-12`. It does **not** reproduce `quad2d` (max |dP| 3.22, ~0.39% of
the matrix scale) or `quad3d` (5.76e-11, noise-level). Two runs in one session
are bit-identical to each other, so the computation is deterministic — the
committed artifacts are simply not what today's code produces.

Cause: `quad2d` is the only system whose closed loop contains a **trained neural
network** (`safe_explorer_ppo`); the others use analytic controllers. ReLU
networks are piecewise-linear, so the step map has creases. `fd_linearize`
finite-differences it with `FD_EPS = 1e-4`, and a crease inside that stencil
means `A_d` is not a well-defined Jacobian — measured at the attractor, the
forward and backward differences disagree by ~5% (-0.438 vs -0.462). `P`
inherits that indeterminacy.

**This does not invalidate the labels.** `compute_system` runs `validate()`,
which samples the ellipsoid boundary and checks empirically that trajectories
never leave it. Invariance is established by that check, not by `P` being
canonical, so a slightly different `P` that still validates still gives sound
success labels.

Consequence for tooling: do not gate `quad2d` or `quad3d` at a tight tolerance
against the committed artifacts. `tests/test_envs/test_invariant_sets.py` was
deliberately dropped for this reason — a test that fails for a reason nobody
intends to fix teaches people to ignore the suite.

---

Related: [architecture.md](architecture.md) for the library the generators call into, [workflows.md](workflows.md) for launching a run, [compute.md](compute.md) for where to launch it, [glossary.md](glossary.md) for every term used above.


## Torque-noise pendulum datasets (`stochastic/`)

```
DATA_ROOT/stochastic/pendulum/noisy_torque/lqr/tau_{0.00,0.10,0.15,0.30,0.50,1.00,2.00,5.00}/
```

Collected 2026-08-06 at commit `2e0b9ddc`; each records that commit and its
environment in `dataset_description.json['provenance']`. A **third** noise family
alongside `deterministic/` and `noisy/`, and deliberately not a level inside
`noisy/` — the mechanisms share no units and are not comparable.

| tau | % of `u_sat` | train p | eval p | K | mean length |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0% | 0.3845 | 0.3860 | 1 | 555.3 |
| 0.10 | 15.7% | 0.3509 | 0.3513 | 100 | 571.1 |
| 0.15 | 23.5% | 0.3356 | 0.3358 | 100 | 578.5 |
| 0.30 | 47.1% | 0.2987 | 0.2991 | 100 | 597.8 |
| 0.50 | 78.5% | 0.2579 | 0.2587 | 100 | 623.6 |
| 1.00 | 156.9% | 0.1683 | 0.1694 | 100 | 684.4 |
| 2.00 | 313.9% | 0.0595 | 0.0597 | 100 | 760.1 |
| 5.00 | 784.7% | 0.0146 | 0.0150 | 100 | 790.2 |

The last three were collected 2026-08-13 and published 2026-08-15; they had sat
in Amarel scratch, complete and reduced, waiting on the publication step the
collection script deliberately leaves separate. They take the sweep well past
saturation — `tau = 5.0` is 785% of `u_sat`, and at that level the mean
trajectory length is 790.2 against an 800-step horizon, i.e. almost nothing
reaches the goal and almost everything runs the full horizon.

**The published `lqr/` tree is a re-collection at horizon 800**, not the
2026-08-06 run this section originally described. `lqr_legacy_20260806/` holds
that earlier one at horizon 1000 (`tau = 0.30`: 0.29916 there against 0.29869
here). Numbers quoted from the legacy tree will not match the published one.

**Mechanism.** `disturbances: {'action': [uniform(-tau, tau)]}` — added to the
commanded torque in `_preprocess_control`, i.e. *before* the `u_sat` clip, so a
saturated actuator cannot be pushed further. Unlike the state-additive presets it
cannot raise the success rate: measured across all 49,770 cells and all four
noisy levels, the largest gain over the noiseless field is **+0.000**, and
3,047,000 rollouts from failing cells produced no success at all.

That last property belongs to the **placement**, not to the channel. It holds for
every family applied before the clip and fails immediately for the same noise
applied after it — see the external-torque family below. Earlier wording here
called this "the physically admissible channel", which reads as though matched
torque noise is inherently unable to help; it is not.

**Success rule.** First state with `|theta| < 0.05` AND `|theta_dot| < 0.05`; the
rollout stops there. No dwell — see `glossary.md`, recurrence vs invariance. The
consequence is that `terminal state in the box` and `label 1` are the same
statement in both directions, which the earlier 10-step dwell broke (9,863 of
100,000 trajectories at `tau = 0.5` ended inside the box carrying label 0).

**What these numbers are.** Backward-reachable-tube probabilities — "reached the
box within 8 s" (800 steps at 100 Hz) — not stability and not an asymptotic ROA. Two caveats travel
with them: the `0.05` velocity tolerance was inherited from the angle tolerance
rather than derived, and at `tau = 0.30/0.50` the settled state satisfies it only
62.6%/39.5% of the time, so those two levels partly measure the tolerance.

**Layout** differs from every older dataset because the consuming loader has a
separate npz path (`pool_format: npz`): a flat `train.npz` plus
`train_test_splits/shuffled_indices_0.txt` holding **integer row ids**, not
`sequence_<i>.txt` filenames, and a 3-column `eval_states.txt`
(`theta, theta_dot, p_success`) rather than start/end/label. No `trajectories/`
directory is written. `dataset_description.json` is required — `PendulumSystem`
raises `FileNotFoundError` without its `achieved_bounds`.

Train starts are 100k uniform-random rolled once each; eval is the full
49,770-cell grid rolled 100 times per cell. The two are disjoint by construction,
so there is no start-state leakage.

Produced by `prepare_stochastic_layout.py` from the collector's npz output.
Audited 2026-08-06: 120 checks across the five levels, including 15 rollouts per
level replayed bit-for-bit from their recorded seeds.


## Signal-dependent pendulum datasets (`stochastic/`)

```
DATA_ROOT/signal_dependent/pendulum/lqr/beta_{0.000,0.200,0.400,0.800,1.600,3.200}/
```

Collected 2026-08-15. Same LQR, grid, horizon (800), `ctrl_freq` 100, `pyb_freq`
300, seed 42 and entry-cut box rule as `noisy_torque/`, so the two are comparable
cell-for-cell. The law is Gaussian with a command-dependent scale, still applied
**before** the clip:

```
xdot = f(x, sat(u + w)),   w ~ Normal(0, alpha + beta*|u|),   alpha = 0.008
```

| beta | sigma at `u_sat` | train p | eval p | interior | K |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.0080 | 0.3800 | 0.3812 | 0.3% | 100 |
| 0.2 | 0.1354 | 0.3180 | 0.3180 | 4.1% | 100 |
| 0.4 | 0.2629 | 0.2742 | 0.2752 | 6.9% | 100 |
| 0.8 | 0.5177 | 0.1977 | 0.1994 | 12.0% | 100 |
| 1.6 | 1.0275 | 0.1082 | 0.1099 | 16.0% | 100 |
| 3.2 | 2.0470 | 0.0427 | 0.0437 | 13.2% | 100 |

`beta = 0` is kept at `alpha = 0.008` as the control the sweep needs — a constant
sigma floor with no signal dependence — which separates what `beta` adds from
what the floor does. The deterministic reference is the published `tau_0.00`.
Interior fraction is not monotone: it peaks at `beta = 1.6` and falls once enough
cells have been driven to a flat p = 0.

Gains are **zero at every level**, for the reason in `glossary.md` under
saturation placement.

## `gaussian_signal` — the standard stochastic family (`stochastic/`)

**This is the canonical noise family for both pendulum and cartpole**
[user, 2026-08-17]. `noisy_torque` and the state-additive presets are historical;
prefer this one for new work and for anything a downstream model consumes.

```
DATA_ROOT/stochastic/pendulum/gaussian_signal/lqr/{low,med,high}/
DATA_ROOT/stochastic/cartpole/noisy_torque/lqr/{low,med,high}/   (uniform, being replaced)
```

| system | level | alpha | beta | mean p | interior | rescued | broken |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pendulum | `low` | 0.05 | 0.16 | 0.3869 | 11.2% | 3,027 | 2,548 |
| pendulum | `med` | 0.10 | 0.64 | 0.4067 | 64.6% | 25,191 | 6,955 |
| pendulum | `high` | 0.20 | 1.00 | 0.5457 | 82.4% | 30,561 | 10,443 |

Deterministic reference `noisy_torque/lqr/tau_0.00`, mean p 0.3860. Archived
beside the published tree: `gaussian_signal/archive_alpha_0.008/` holds an
earlier `alpha = 0.008` beta sweep.

**Level names carry no parameters.** `low`/`med`/`high` [user, 2026-08-16],
with the constants recorded in `README.md` beside the levels and in each
description's `level_name` and `noise_model.parameters`. The earlier
`a<alpha>_b<beta>` convention is gone from the published tree; before it, an
even earlier run used `beta_<b>` with `alpha` implicit, which is the trap that
motivated making both explicit in the first place.

Collected 2026-08-15/16. Identical to the signal-dependent family in every respect
**except where `w` is applied**:

```
xdot = f(x, sat(u) + w),   w ~ Normal(0, alpha + beta*|u|)
```

`w` models an external torque on the shaft rather than noise inside the actuator,
so `u_sat` does not bound it and the applied torque can exceed the motor's limit.
It is still matched — same `B` — which is exactly why this family refutes the
claim that matchedness biases an ROA toward the nominal.

| alpha | beta | train p | eval p | interior | rescued | broken |
| --- | --- | --- | --- | --- | --- | --- |
| 0.050 | 0.16 | 0.3862 | 0.3869 | 11.2% | 3,027 | 2,548 |
| 0.008 | 0.64 | 0.3957 | 0.3961 | 40.8% | 14,441 | 5,849 |
| 0.100 | 0.64 | 0.4060 | 0.4067 | 64.6% | 25,191 | 6,955 |

`rescued` = deterministic label 0 and `p > 0`; `broken` = label 1 and `p < 1`.
Both counts are K-dependent — more trials find more cells that are not perfectly
deterministic — so they must be quoted at the K they were measured at. The K = 20
sweep gives 2,200 / 8,596 / 14,430 rescued for these same settings, and an
earlier version of this table mixed those into rows whose other columns were
K = 100.

**Both classes are boundary cells, not a moved boundary.** Rescued cells sit at
mean `p` 0.09-0.15 and **not one of them exceeds 0.9** — a rescued state now
succeeds sometimes, never reliably. Broken cells sit at mean `p` ~0.83 and none
fall below 0.1. So the effect is the deterministic boundary being *blurred* in
both directions, not displaced. At `alpha = 0.1, beta = 0.64` that blur covers
32,146 of 49,770 cells; only 12,254 remain a hard 1 and 5,370 a hard 0.

**This is the first family here whose ROA is not a subset of the deterministic
one.** It gains cells — start states the noise-free controller fails from that
noise rescues — and at the low end gains and losses nearly cancel (824 against
772 at `beta = 0.08, alpha = 0.008`), so the mean is preserved to four decimals
while thousands of cells change status. The field is a *reshaping* of the
deterministic ROA rather than an erosion of it. For a terminal-state consumer
that removes a property the other families give for free: `p_success > 0` no
longer implies the deterministic label was 1.

The family is named for the noise **law**, not the placement; the placement is
recorded per level in `noise_model.placement`. Not published, still on cluster
scratch: the six pre-saturation `signal_dependent` levels and the alpha x beta
sweep.

**`high` behaves differently from the other two.** Every one of the 30,561
deterministically-failing cells has `p > 0` and no cell on the grid reads a hard
0, with rescued cells reaching p = 0.76. At `low` and `med` both rescued and
broken are boundary cells — no rescued cell exceeds 0.9, no broken cell falls
below 0.1 — so the boundary is blurred rather than moved. At `high` sigma also
exceeds the motor's own authority (131% of `u_sat`), so it is informative as a
probability field but is not a statement about the controller. Its description
carries a `regime_note` saying so.

**The published descriptions are enriched beyond what the collector writes.**
Each level carries `noise_model` (style, equation, what alpha and beta each mean,
placement and why it is load-bearing, plus the *realised* sigma and saturation
statistics measured over 32,000 real control steps), `plant` (the ODE and its
constants, the 0.866 authority ratio, the 1.4715 J swing-up barrier), and
`statistics` including the rescued/broken comparison against `tau_0.00`. The
comparison cannot be written at collection time because the cluster cannot see
the reference set, so it is a publication-step addition.

**Levels do not transfer between placements.** The same `beta` is far more potent
outside the clip, because none of it is discarded, so the external sweep starts an
order of magnitude lower. Roughly, `beta` ~ sigma at saturation as a fraction of
`u_sat`. Levels past `beta ~ 1.0` put the disturbance above the motor's own
authority: real, but the pendulum is then substantially driven by the noise
rather than controlled, and the descriptions carry the sigma-to-`u_sat` ratio so
a reader can tell which side of that crossover a level is on.



## `gaussian_signal` on cartpole

Same law, same standard family, but the system differs from the pendulum in two
ways that change what the parameters mean. Design and Amarel runbook:
`docs/superpowers/specs/2026-08-17-cartpole-gaussian-signal-collection.md`.

**Placement is inert here.** The cartpole LQR demands a median **0.27 N** against
an `action_scale` of **2000 N** — p99 28.9, max 57.8, and it never saturates in
16,494 measured steps. So `sat(u + w)` and `sat(u) + w` are the same function and
the collector does not offer the switch. That absence is also *why* the published
uniform cartpole family gains 743-911 cells under pre-saturation noise where the
pendulum's gains nothing: the pendulum is saturated 70-98% of steps, and a
saturated clip discards every positive draw. `action_scale = 2000` is inherited
from the deterministic set and is not physically motivated; the clip would only
begin binding near 10 N (3% of steps).

**The action is on the cart and reaches the pole through `cos(theta)`.**
Finite-differenced through the simulator: 1.46 rad/s^2 per N upright, 0.69 at 60
degrees, 0.02 at 89 — tracking `cos(theta)` to a few percent, and vanishing with
the pole horizontal. So constant force noise is *already* a state-dependent
disturbance on the pole. Sign: positive force gives negative `theta_ddot`, the
pole being driven by the cart accelerating out from under it.

**`beta` needs a different scale.** On the pendulum `|u|` sits pinned at `u_sat`,
so `beta ~ 1` is meaningful. On cartpole `|u|` is heavily skewed — median 0.27 N,
p99 28.9 — so the same `beta` contributes ~0.02 N and does nothing, while
`beta ~ 1` leaves the median untouched and multiplies the tail sevenfold. Levels
are reduced to one knob by fixing the share of noise *variance* from the signal
term at 50%, which against the measured `E|u| = 1.581`, `E[u^2] = 26.436` gives

```
beta = k,   alpha = 3.80 * k
```

`k = 0.635 / 0.873 / 1.429` deliver the same standard deviation as the published
uniform `low`/`med`/`high` (sigma 8/11/18 = 4.62/6.35/10.39 N).

**Matched variance is not matched difficulty.** Measured on 240 stratified cells
at K = 10, the gaussian family is substantially gentler at every matched pairing:

| k | gaussian p | uniform counterpart | uniform p |
| --- | --- | --- | --- |
| 0.635 | 0.4567 | `low` (sigma 8) | 0.3692 |
| 0.873 | 0.3804 | `med` (sigma 11) | 0.2592 |
| 1.429 | 0.1825 | `high` (sigma 18) | 0.1029 |

24%, 47% and 77% more success, widening with strength, and it breaks fewer cells
(63 against 109 at the `low` pairing). What kills a cartpole run is noise *at*
the goal preventing entry into the 0.05 ball, and this family goes quiet exactly
there. So **timing matters, not just variance** — which is the reason the family
is worth having, and the reason a level set has to declare which of the two it
matches.


## Unmatched-force datasets: quad3d, quad2d, and the cartpole re-collection

```
DATA_ROOT/stochastic/quadrotor3D/noisy_dynamics/lqr/f_{0.000,0.032,0.048,0.060,0.072}/
DATA_ROOT/stochastic/quadrotor2D/noisy_dynamics/rl/ f_{0.000,0.070,0.100,0.150,0.200}/
DATA_ROOT/stochastic/cartpole/noisy_torque/lqr/     {low,med,high}/  + archive/
```

Collected 2026-08-14/15. Two mechanisms across the three:

| | mechanism | matched? | units |
| --- | --- | --- | --- |
| quad3d, quad2d | `disturbances: {'dynamics': uniform}` | **no** | N (world force) |
| cartpole | `disturbances: {'action': uniform}` | yes | N (cart force) |

The quadrotor sets are the first use of the `dynamics` mode anywhere in this
repo. It is a world-frame force applied at the COM every PyBullet substep, so it
produces **no torque** and reaches attitude only through the controller's
reaction:

```
xdot = f(x, u) + B_d w,   B_d nonzero only in the linear-acceleration rows, 1/m
```

`range(B_d)` is not contained in `range(G(x))` — the vehicle has no way to
produce a lateral force except by tilting first — which is what makes it
unmatched, and is the substantive difference from the pendulum and cartpole
families. Note the cartpole path says `noisy_torque` but the mechanism is
`action`: the noise is on the commanded cart **force**, `f(x, sat(u + w))`. The
directory name is a misnomer chosen at placement time; the description JSON is
authoritative.

**Eval starts are the deterministic set's own states**, read from its
`eval_states.txt`, so row *i* indexes the same physical state in both datasets.
Train uses random starts within the same sampling bounds. The `f = 0` / `sigma =
0` level exists in every family and is the baseline the noisy levels are compared
against — the shipped deterministic set is *not*, because it was produced by
different code.

| | level-0 vs shipped labels | eval states | train | K |
| --- | --- | --- | --- | --- |
| cartpole | **1.0000** | 116,242 | 116,242 random | 100 |
| quad2d | 0.9949 | 489,789 | 500,000 random | 50 |
| quad3d | 0.9702 | 1,000,000 | 800,000 sampler | 100 |

quad3d's 3% gap is chaos amplification over ~500-step tumbling trajectories, not
a config error: the residual splits into one boundary tie at 0.0499 against a
0.05 threshold and six genuine divergences out of 400. It cannot be closed from
this repo — the collector that produced the shipped set is not here in runnable
form (its `env_func` omits `task_info`, which the 3D branch indexes, so it raises
`IndexError` on construction).

### The cartpole re-collection, and why the old one is superseded

`stochastic/cartpole/noisy_action/lqr/sigma_{015.0..040.0}` is left in place but
should not be used. Six defects, measured against `deterministic/cartpole_pybullet`:

| | old | correct |
| --- | --- | --- |
| control bound | 100 N | **2000 N** (`action_scale`) |
| success | uniform 0.1 per-channel box | **L2 ball, radius 0.05** |
| termination `x_dot`/`theta_dot` | 20.0 | 5.0 |
| eval states | 131,859, aligned with nothing | the 116,242 deterministic states |
| baseline level | absent | `sigma = 0` present |
| state order | env order, reordered post-hoc | file order written directly |

The control bound is the consequential one: 20x too little authority, and it also
breaks the noise scale, since `sigma = 20` is 20% of a 100 N bound but 1% of a
2000 N one.

### The deterministic cartpole description states its own success rule wrongly

`generation_parameters.termination_conditions.success` claims per-channel
tolerances (`x < 0.01`, others `< 0.05`) held for **10 consecutive steps**. That
was never implemented. Every shipped success ends with `||state||` in
`[0.0497, 0.0500]` and **not one** satisfies `|x| < 0.01` — the signature of first
entry into an L2 ball of radius 0.05 with no dwell.

**Labels cannot detect this.** A gate scored **300/300 against the wrong rule**,
because converging and diverging trajectories are separated by a wide gap and many
rules partition them identically — the same insensitivity that makes quad3d's
labels invariant across four orders of magnitude of tolerance. Only the stored
**final states** discriminate: under the real rule they match at median 4.97e-07,
the 6-decimal storage floor. When validating a reproduction, compare final states, not
labels.

`cost='quadratic'` is required for all four systems or `goal_reached` never
reaches `info`; the env still terminates in the right place, so trajectories look
perfect while every label reads 0.

### Noise levels are coupled to the success rule and the horizon

They do not transfer when either changes. Measured on quad3d: retention at
`f = 0.14` is **0.618** under a 0.1 per-channel box and **0.015** under the 0.05
L2 ball — the settled noise cloud fits inside the box but not the ball. Levels
calibrated under one rule were ~3x too aggressive under the other.

`p_success` is a **bounded-time** reach probability at each system's horizon
(1000 steps for cartpole and quad3d, 1200 for quad2d, inherited from its
deterministic set). Under noise the controllers largely still reach the goal, just
later: given quad3d's own 100,000-step allowance, success at `f = 0.072` is ~0.24
against ~0.25 at `f = 0`, while at H=1000 it reads 0.058. About 15% of `f = 0.072`
rollouts would succeed with unlimited time. The per-level `hit_horizon` count is
recorded in each description rather than left implicit.

### Rate injection has two directions, and they are not interchangeable

quad3d only. PyBullet's `resetBaseVelocity` takes a **world** angular velocity,
but the env stores `Rbo @ ang_v`, a **body** rate. So:

- **sampler starts** must be passed body-as-world, repeating the original
  collector's conflation — that is how the shipped data was generated. Converting
  instead drops label agreement 393/400 -> 367/400.
- **states read back from a file** were written as body rates and need
  `world = R @ body`. Skipping it drops 148/150 -> 138/150 and inflates
  final-state error 29x.

Both are correct; they differ because one is a sampler output and the other an env
output. quad2d has no such asymmetry — for `TWO_D` the env stores `ang_v[1]`
directly in the world frame.
