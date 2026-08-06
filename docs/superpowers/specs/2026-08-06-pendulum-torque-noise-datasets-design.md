# Pendulum torque-noise datasets

**Date:** 2026-08-06
**Status:** design
**Scope:** `generate_inverted_pendulum_trajectories.py`, `.claude/docs/{datasets,architecture,glossary}.md`

Five pendulum datasets, one per torque-noise level `tau` in
`{0.0, 0.1, 0.15, 0.30, 0.5}`, each with a `train` split (random starts, full
trajectories) and an `eval` split (grid, per-cell `p(success | start)`).

## Why not the existing `--noise` presets

`--noise` selects from `NOISE_PRESETS` in `pendulum_noise.py`, whose dynamics
families write the drawn value straight into `(theta, theta_dot)`. Measured over
the full state space at 161x161 on 2026-08-05, LQR, box+dwell criterion:

| mechanism | mean p |
| --- | --- |
| deterministic | 0.386 |
| `velocity_proportional_high` (state-additive) | 0.426 |
| uniform torque, tau = 0.05 | 0.368 |
| uniform torque, tau = 0.15 | 0.337 |
| uniform torque, tau = 0.30 | 0.299 |

State-additive noise *raises* the success rate because a draw can place the state
inside the goal set; a force can only ever take success away, and the torque
column is monotone in `tau` as it must be. Generalised forces enter the
acceleration row only — `theta_dot = d(theta)/dt` is a definition, not a law to
perturb — so the state-additive families cannot represent a disturbance. These
datasets therefore use the `disturbances` mechanism, not `--noise`.

Both mechanisms stay in the tree. The presets remain the only way to reproduce
the shipped noisy pendulum datasets, which is a reason to keep them and not a
reason to use them here.

## The noise

```python
disturbances = {'action': [{'disturbance_func': 'uniform', 'low': -tau, 'high': tau}]}
```

Applied in `inverted_pendulum.py::_preprocess_control`, i.e. to the commanded
torque *before* the `u_sat` clip, so at saturation part of the kick is discarded.
That is physically right — a saturated actuator cannot be pushed further — and it
biases `p` slightly up relative to a disturbance acting on the shaft. Recorded
here so nobody reads it as a bug later.

`u_sat = 0.6371781908344007 = m*g*l*sin(60 deg)` exactly (verified bit-exact), so
the levels are 0%, 15.7%, 23.5%, 47.1% and 78.5% of saturation. `tau = 0.5` is
expected to be near-zero success over most of the space; it is a corner of the
sweep, not a mistake.

**`tau` is not a spectral density.** It parameterises a zero-order hold over one
control step, so the same `tau` at a different `ctrl_freq` is a different physical
disturbance. The generator pins `ctrl_freq = pyb_freq = 100`, which makes
`PYB_STEPS_PER_CTRL = 1` and the hold trivial. Numbers here will therefore *not*
reproduce the 2026-08-05 sweep, which ran 50/150. Rate-independent
parameterisation (`sigma_step = sqrt(q/dt)`) is deferred, not rejected.

## Success rule

Per-channel box, no dwell-free L2 ball:

```
|theta| < 0.05  and  |theta_dot| < 0.05,  held for 10 consecutive control steps
```

Replaces the shipped `||[theta, theta_dot]||_2 < 0.075`, which adds radians to
rad/s with equal weight. The 0.05 tolerances are the cartpole's own angular
tolerances, so the two systems' angular criteria now agree.

**Trajectories are cut at the entry state of the window that achieved the dwell**,
not at the end of the window. The label must remain a function of the terminal
state — that is the whole reason the entry-cut exists — and cutting at entry keeps
the stored terminal state inside the box while the dwell serves as confirmation
that the controller held it rather than passed through.

Known edge case, accepted: a *failing* rollout can be inside the box at the
horizon without ever having held it, giving a terminal state in the box with
label 0. It requires entering the box within the last 10 steps of a 1000-step
horizon and is left unhandled rather than papered over.

This makes these datasets **not comparable** to any existing pendulum dataset,
all of which use the L2 ball. That is a deliberate trade: comparability with the
cartpole and a criterion that means what it says, against comparability with
shipped data collected under a mechanism we now know is wrong anyway.

## Layout

```
$DATA_ROOT/noisy_torque/pendulum/<controller>/tau_<tau>/
```

A sibling family to `noisy/`, not a new level inside it. `noise_level()` maps
preset suffixes to `{low, med, high, ...}` directories; a torque level shares no
vocabulary with those, and putting `tau_0.10` beside `high` would invite exactly
the comparison the previous section forbids. `tau = 0.0` is regenerated into
`tau_0.00` through the identical code path rather than reusing
`deterministic/pendulum/lqr`, so all five come from one pipeline.

## Seeding

`rollout_seed(base_seed, split_id, index, batch)` does **not** include `tau`. With
one `--seed` across all five runs, cell *i* batch *b* draws the same stream at
every level, so differences between datasets are the noise level rather than
sampling luck. This is common random numbers and it is intentional; do not add
`tau` to the seed tuple.

## Changes to the generator

1. `--torque_noise TAU` (float, default `None`), mutually exclusive with
   `--noise`. Rejecting the combination is better than defining a precedence
   nobody will remember.
2. `env_config['torque_noise']`, consumed in `make_env_func` to build the
   `disturbances` kwarg. Threading it through `env_config` rather than a module
   global keeps `_eval_worker`/`_train_worker` picklable and the config
   self-describing.
3. `make_env_func` sets `goal_threshold=0.0` whenever the box rule is active, so
   the env's own L2 termination cannot fire first; `run_trajectory` owns the
   criterion.
4. `run_trajectory` gains the box+dwell test and the entry-cut truncation.
5. `default_output_dir` learns the `noisy_torque` family.
6. `label_semantics` and a new `success_rule` block in both description JSONs.

Deterministic runs (`tau = 0.0`) need one eval batch, not ten: every batch is
identical. `--min_batches 1` is the caller's job, not a special case in the code.

## Rejected

- **Reusing `--noise` with a new preset family.** `pendulum_noise.py`'s models
  take `(rng, state, u)` and return a state; a torque disturbance is not
  expressible in that signature without lying about what it returns.
- **A separate collector script.** The train/eval machinery — sharded resume,
  atomic publication, the Jeffreys stopping rule — is the valuable part and
  already exists. A second copy would drift.
- **Adding `tau` to `rollout_seed`.** Destroys the coupling across levels for no
  benefit; the levels are separate runs writing separate directories, so stream
  collision is not a risk.

## Verification

- Smoke: all five levels, small grid and trajectory count, on one box. Assert
  `p` monotone non-increasing in `tau`, and `tau = 0.0` reproducing across two
  runs bit-for-bit.
- Then Amarel at full width. Amarel does not share the CS filesystem, so the run
  is preceded by a git sync; the checkout there must be at the commit that
  carries this change.
