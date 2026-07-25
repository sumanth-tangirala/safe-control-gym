# Spec: Split Train/Eval Collection for the Noisy Inverted Pendulum

Date: 2026-07-25
Scope: `generate_inverted_pendulum_trajectories.py`, control-proportional dynamics noise.

## Goal

Replace the single-pass noisy-pendulum collection with two purpose-built modes:

- **train** — N rollouts from *random* start states, full trajectories stored.
- **eval** — repeated batches over the *grid*, storing only the per-start-state
  probability of success, run until those probabilities stabilize.

The two are independent processes and are meant to run concurrently.

## Motivation

The previous noisy datasets stored every rollout of both splits. For `med`,
`eval.npz` alone was 4.26 GB of trajectories whose only downstream use was a
success rate per grid cell. Storing the probability directly is ~2,600x smaller
and is a better estimator, because the batch count is no longer pinned at the 10
rollouts/start the old collection happened to use — it runs until the estimate
stops moving.

Splitting train from eval also fixes a conflation: training wants coverage of the
continuous state space (random starts), while evaluation wants a fixed,
reproducible grid to compare methods on. The old scheme sampled both from the
same grid rollouts.

## Prior state

- `noisy/pendulum_lqr_stoch-dyn-ctrl-{low,med,high,xhigh}/` (raw, per-cell npz)
  and `noisy/pendulum/lqr/{level}/{train,eval}.npz` — **all deleted** by the user
  before this work. `noisy/pendulum/lqr/` is an empty skeleton; nothing is
  overwritten.
- Those datasets were produced by the external pendulum repo, not this one. The
  noise models were ported here in d591b448 (`pendulum_noise.py`).
- `plans/invariant-terminal-sets-recollection.md` ruled noisy datasets out of the
  invariant-terminal-set scheme: under noise the state never settles, and at
  `high`/`xhigh` the stationary noise floor (p50 distance 0.086 / 0.139) exceeds
  the 0.075 goal radius, so no invariant success set exists.

## Decisions

### Success rule (both splits)

Success = the rollout **ever entered** the 0.075 L2 ball around upright within
the horizon. A stored training trajectory is **cut at (and includes)** that entry
state; non-successes run the full 1000-step horizon and are labelled 0.

Rationale: under noise a rollout can enter the ball and drift back out. Cutting
at entry keeps the label a *function of the terminal state*, which is what the
downstream terminal-state model needs. The alternatives — storing the full
horizon, or labelling by terminal-state membership — both break that property at
`high`/`xhigh`, where the noise floor exceeds the goal radius and genuine
successes would end outside the ball.

This matches the deleted datasets' semantics exactly, so labels remain comparable.

### Train split

- N = 300,000 (`--num_trajs`, default 300000).
- Starts sampled uniformly: theta ~ U(-pi, pi), theta_dot ~ U(-2pi, 2pi) — the
  existing `sample_initial_states(random_init=True)`.
- One rollout per start, each with its own seed.
- Horizon 1000 steps at 100 Hz (dt = 0.01), matching the deleted datasets.

### Eval split

- Grid: half-open, `lo + resolution * arange(n)` with
  `n = ceil((hi - lo) / resolution)`, over theta in [-pi, pi) and theta_dot in
  [-2pi, 2pi) at `--resolution 0.04` = 158 x 315 = **49,770** states.

  This requires **fixing a porting bug** in `sample_initial_states`; see below.
- One **batch** = one rollout from every grid state (49,770 rollouts).
- Same 1000-step horizon and same success rule as train.
- Only counts are kept. No trajectories are stored.

### Grid bug fix (`sample_initial_states`)

The grid branch currently reads

    np.arange(-math.pi, math.pi + resolution, resolution)

The `+ resolution` makes it **overshoot the domain**: at `resolution=0.04` the
last theta is 3.178407 > pi, and the last theta_dot is 6.316815 > 2pi, giving
159 x 316 = 50,244 states. Those overshoot points are not new states — theta
3.178407 wraps to -3.104778 and theta_dot 6.316815 is clipped to 2pi — so the
grid silently contains duplicated cells, and both bounds are sampled twice.

Fix: drop the `+ resolution` and compute the axis explicitly as
`lo + resolution * np.arange(ceil((hi - lo) / resolution))`, rather than relying
on `arange`'s float-endpoint behaviour. This yields 158 x 315 = 49,770.

Half-open is the *correct* convention here, not merely a convenient one: theta is
periodic, so -pi and +pi are the same physical state and including both would
duplicate a column. (theta_dot is not periodic, so [-2pi, 2pi) drops the +2pi
boundary state; that is a single-cell asymmetry, it matches the existing
datasets, and theta_dot is clipped to that bound anyway.)

**This makes the repo reproduce the shipped datasets.** All three
`deterministic/pendulum/{lqr,rl,rl-weak}` datasets have exactly 49,770 states on
a step-0.04 half-open grid — the current code cannot produce them. The residual
difference is 2.7e-6 per coordinate, because the external repo started from
-3.14159 (pi truncated to 5 dp) rather than -pi; shape and ordering are
identical, so index-based comparison against those datasets is exact and value
comparison is well below the smallest noise scale (sigma0 = 0.002).

The fix changes the legacy (no-`--split`) grid path from 50,244 to 49,770 states.
That is intended: the old count was the bug. The random-init path is untouched.

### Stopping rule

After every `--check_every` (default 10) batches, compute

    mean_se = mean_i sqrt( p_i * (1 - p_i) / B )

over all 49,770 cells, where `p_i = successes_i / B`. Stop when
`mean_se < --se_tol` (default **0.01**), subject to `--min_batches` (default 10)
and `--max_batches` (default **500**).

Rationale: this measures how much the estimate could still move, and is
near-monotone in B, so unlike a drift statistic it cannot trip early by chance.
It self-adapts to noise level — at `low` most cells are a hard 0/1 and contribute
~0, so it stops in few batches; at `xhigh` the wide uncertain band keeps it
running.

Drift, `mean_i |p_i(B) - p_i(B - check_every)|`, is computed and logged as a
diagnostic but does **not** gate stopping.

### Formats

`train.npz` — key layout identical to the deleted `train.npz` so downstream needs
no new reader; only the states dtype changes.

| key | dtype | shape | notes |
|---|---|---|---|
| `states` | float32 | (M, 2) | all trajectories concatenated |
| `offsets` | int64 | (N+1,) | trajectory i is `states[offsets[i]:offsets[i+1]]` |
| `starts` | float64 | (N, 2) | sampled initial states |
| `labels` | uint8 | (N,) | 1 = reached goal, 0 = timeout |
| `seeds` | int64 | (N,) | per-rollout seed; enables exact regeneration |

`start_ids` is dropped — it indexed grid cells, and train starts are continuous.

**float32 is a measured decision, not a default.** On 351k real states from the
deleted `med` dataset:

| representation | ratio | max error |
|---|---|---|
| float64 raw | 1.00x | — |
| float32 | 2.00x | 2.4e-7 |
| int16 fixed-point | 4.00x | 9.6e-5 |
| int16 modular-delta + DEFLATE | 5.56x | 9.6e-5 |

int16 is rejected: per-step state changes have p1 = 1.7e-4 (theta) and 2.0e-4
(theta_dot) against int16 steps of 9.6e-5 / 1.9e-4, so the slowest ~1% of
steps — the near-equilibrium motion the terminal-state model cares about — would
quantize into noise or into zero. float32's 2.4e-7 is three orders of magnitude
below the smallest real state change, and is *more* precise than the 6-decimal
txt format the deterministic datasets ship. DEFLATE is rejected: only 8% further
gain on float data, at decompression cost on every hot read.

At the observed mean of 671 states/trajectory, 300k trajectories = **~1.6 GB**.

`eval_success_prob.npz` (~1.6 MB):

| key | dtype | shape |
|---|---|---|
| `starts` | float64 | (49770, 2) |
| `successes` | int32 | (49770,) |
| `trials` | int32 | (49770,) |
| `p_success` | float64 | (49770,) |
| `grid_theta` | float64 | (158,) |
| `grid_theta_dot` | float64 | (315,) |
| `grid_shape` | int64 | (2,) = (158, 315) |
| `n_batches` | int64 | scalar |

Plus a `success_probabilities.txt` mirror (`theta,theta_dot,p`, 6 decimals) for
human inspection, matching the `roa_labels.txt` convention.

### Incremental publication (eval)

The published dataset **is** the checkpoint. There is no separate counts file and
no finalize step. After every batch:

1. Fold the batch's 49,770 outcomes into `successes` / `trials`.
2. Recompute `p_success`, `mean_se`, and the drift series.
3. Write `eval_success_prob.npz`, `success_probabilities.txt`, and
   `dataset_description.json`, each to `<name>.tmp` in the same directory
   followed by `os.replace()`.

`os.replace()` is atomic within a filesystem, so a `SIGKILL` mid-write cannot
leave a torn file. At any instant the directory holds a complete, self-consistent
dataset for however many batches have finished.

Only **whole** batches are published, so every cell always has the same `trials`;
killing mid-batch discards at most ~2 min of work. `dataset_description.json`
records `n_batches`, `mean_se`, and `converged: true|false`, so a terminated run
is labelled honestly rather than silently looking finished.

Resume reads `successes`/`trials` back out of `eval_success_prob.npz` and
continues at batch `n_batches + 1`. A converged dataset can therefore be extended
later by re-running with a higher `--max_batches`.

### Reproducibility

Every rollout gets

    seed = SeedSequence([base_seed, split_id, index, batch]).generate_state(1)[0]

where `split_id` is 0 for train and 1 for eval; `index` is the trajectory index
for train and the grid-cell index for eval; and `batch` is 0 for train and the
0-based batch number for eval. Seeds are therefore independent across splits,
cells, and batches, and are a pure function of `--seed` — so a resumed eval run
draws exactly the noise an uninterrupted run would have.

The seed is passed to `env.reset(seed=...)`, which reseeds `self.np_random` — the RNG the
noise model draws from (verified in `benchmark_env.py`: `before_reset` calls
`self.seed(seed)`).

This closes a real gap: `run_trajectory` currently calls `env.reset()` with **no**
seed, so today's noisy rollouts are not reproducible at all. `run_trajectory`
gains a `seed` parameter.

### Train resume

Workers write `_shards/shard_<k>.npz`; completed shards are skipped on re-run and
merged into `train.npz` at the end. A kill loses no computation, but `train.npz`
does not exist until the merge — an interrupted train run leaves shards, and
re-running merges them without recomputation. Accepted for a ~19 min job.

### CLI

`--split {train,eval}` is new. Omitting it preserves today's behavior — txt
sequence files, `--invariant_terminal_sets`, and the random-init path are all
unchanged — with the single intended exception of the grid bug fix above.

```
--split train --noise control_proportional_med --num_trajs 300000 --num_workers 24
--split eval  --noise control_proportional_med --num_workers 48
```

New flags: `--split`, `--se_tol` (0.01), `--min_batches` (10), `--max_batches`
(500), `--check_every` (10). Eval uses the existing `--resolution`, defaulting to
**0.04** in that mode.

### Output layout

`.../genMoPlan/data_trajectories/noisy/pendulum/<controller>/<level>/`, following
the `<family>/pendulum/<controller>/` convention already used by
`deterministic/pendulum/`. `<level>` is the preset suffix (`low`, `med`, `high`,
`xhigh`). Each split writes its own `dataset_description.json` mirroring the
deleted datasets' schema (noise model, controller gains, physics, manifold
structure, data format) plus the new fields above.

### Concurrency

Train and eval are independent invocations sharing no state, launched
concurrently with the core split 24 / 48 of 72. Train exits after ~19 min; eval's
worker count is fixed at start and will **not** expand to reclaim those cores — a
deliberate simplicity trade costing ~10% of the first hour.

## Measured costs

89.7 ms per LQR rollout at `med` (mean length 594, 40-rollout sample), 72 cores:

| job | core-hours | wall clock |
|---|---|---|
| train 300k | 7.5 | ~6 min at 72 workers / ~19 min at 24 |
| eval, 1 batch | 1.2 | ~1 min at 72 / ~1.9 min at 48 |
| eval, B=100 | 124 | ~1.7 h/level |
| eval, B=500 (cap) | 622 | ~8.6 h/level |

No vectorized rollout is needed; the env-based path is faithful and cheap enough.

## Code structure

One script, three clearly-bounded paths sharing `make_env_func`,
`make_controller`, and `run_trajectory`:

- `generate()` — legacy, untouched.
- `collect_train()` + `write_train_npz()`.
- `collect_eval()` + `publish_eval()` (the atomic writer, called per batch).

The file grows to roughly 650 lines. Splitting into a module was considered and
rejected: all three paths share env/controller/rollout construction, and
separating them would scatter tightly-coupled code across files.

## Validation

1. **Seeding**: same seed ⇒ bit-identical states; different seeds ⇒ different.
2. **float32 round-trip**: max error < 1e-6 vs the float64 rollout.
3. **Grid**: the fixed `sample_initial_states` yields exactly 158 x 315 = 49,770
   states at `resolution=0.04`, no coordinate exceeds pi / 2pi, no duplicate
   cells, and it agrees with `deterministic/pendulum/lqr/roa_labels.txt`
   elementwise to < 3e-6 in the same row order.
4. **Atomicity**: kill the eval process mid-write in a loop; the npz always loads
   and `successes.sum()` is consistent with `n_batches`.
5. **Resume**: interrupt at batch k, re-run, confirm it continues from k+1 and the
   final counts equal an uninterrupted run with the same seed.
6. **Estimator**: on a small sub-grid, probabilities from a short run agree with a
   long reference run within the reported standard error.
7. **Sanity**: `p_success` at `low` noise is near-binary; success rate is in the
   neighbourhood of the deleted datasets' 40% at `high`.
8. **Legacy**: existing tests in `tests/test_inverted_pendulum/` still pass, and
   no-`--split` behavior is byte-identical to before **except** for the grid
   branch, which now returns 49,770 states instead of 50,244 (the bug fix above).
   A regression test pins the new grid.

## Collection plan

LQR x {low, med, high, xhigh}, i.e. `control_proportional_{low,med,high,xhigh}`.
RL controllers are out of scope for this pass.

## Out of scope

- RL (SAC) controllers.
- Deterministic datasets and the invariant-terminal-set scheme.
- Downstream `partial_deterministic` derivatives.
- Expanding eval workers when the train process exits.
