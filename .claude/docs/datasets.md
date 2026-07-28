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

**Noisy collection (entry-cut).** Under noise the state never settles, and at
`high`/`xhigh` the stationary noise floor exceeds the 0.075 goal radius — so no
invariant success set exists and the invariant scheme does not apply. Success =
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

---

Related: [architecture.md](architecture.md) for the library the generators call into, [workflows.md](workflows.md) for launching a run, [compute.md](compute.md) for where to launch it, [glossary.md](glossary.md) for every term used above.
