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
