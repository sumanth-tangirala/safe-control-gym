# Spec: Gymnasium Migration and In-Repo SB3 Training

Date: 2026-07-28
Scope: `safe_control_gym/envs/` (all four envs + `benchmark_env`), the 15 modules
that reference `env.step`, `safe_control_gym/envs/env_wrappers/`,
`safe_control_gym/controllers/pendulum_rl/`, a new
`safe_control_gym/experiments/train_pendulum_sb3.py`, `pyproject.toml`,
`setup.py`, and `tests/`.

## Goal

Make pendulum RL policies reproducible inside this repository.

Concretely: migrate the environments to the Gymnasium step API, add
stable-baselines3 as a dependency, and add a training entry point that produces
a `.pt` the existing `PendulumRL` controller loads unchanged.

## Motivation

No policy in this repository can currently be retrained here.
`safe_control_gym/controllers/pendulum_rl/models/*.pt` were trained in a
different repository under a different interpreter, and
`scripts/extract_pendulum_rl_policies.py` says so explicitly: it must run under
`/common/users/shared/pracsys/st1122/inverted_pendulum/.env/bin/python`
(Python 3.12, stable-baselines3 2.9.0, torch 2.5.1) "because that is the only
place the trained SAC `.zip` models load." The path to a shipped policy is
external `.zip` -> one-time extract -> `.npz` handoff ->
`scripts/convert_pendulum_models_to_pt.py` -> `.pt`.

Reproducing or altering any policy therefore requires another person's
environment and their checkpoints. Every downstream question — retraining under
noise, changing the reward, changing the architecture — is blocked on that.

The blocker is a dependency conflict. `PendulumRL` is a hand-written torch
reproduction of an SB3 SAC actor precisely so that this repository would not
depend on SB3, because SB3 cannot be installed here without moving Gymnasium.

## Prior state (measured)

**stable-baselines3 is absent.** Not in `pyproject.toml` or `setup.py`, not
imported anywhere, not installed in the working environment.

**SB3 forces a Gymnasium major bump.** `pip install --dry-run
stable-baselines3` in the current environment resolves to
`Would install gymnasium-1.3.0 stable_baselines3-2.9.0`. The pin is
`gymnasium = "^0.28"`; the installed version is 0.28.1.

**Every environment speaks the pre-Gymnasium 4-tuple.** `benchmark_env.py:502`,
`inverted_pendulum.py:190`, `cartpole.py:299` and `quadrotor.py:450` all
`return obs, rew, done, info`. Nothing in the package emits
`(obs, reward, terminated, truncated, info)`. SB3 cannot consume these
environments at any Gymnasium version.

**The policy's observation is not the environment's observation.** The
environment emits raw `[theta, theta_dot]`; `PendulumRL` re-encodes to
`[cos theta, sin theta, theta_dot / theta_dot_max]` at `pendulum_rl.py:111`,
and re-queries the policy every `action_repeat` (4) steps.

**`weak` is not a separate policy.** From `models/manifest.json`, `vN_strong` is
`runs2x2/vN_s0/model_best.zip` and `vN_weak` is
`runs2x2/vN_s0/checkpoints/step_{60000,70000,90000}.zip` — an earlier checkpoint
of the same run. Four configurations, one run each, two checkpoints per run.

### Migration spike

Run in a throwaway overlay venv (`--system-site-packages`) so the working
environment was not modified. Same torch (2.8.0), Gymnasium 1.3.0, SB3 2.9.0.

| Suite | gymnasium 0.28.1 | gymnasium 1.3.0 |
| --- | --- | --- |
| `tests/test_inverted_pendulum/` | 74 passed, 1 failed* | 74 passed, 1 failed* |
| `tests/test_examples/` + `test_build.py` | **68 passed**, 2 skipped | **65 failed**, 3 passed, 2 skipped |

\* the same pre-existing failure in both: a subprocess in
`test_pendulum_experiment.py` cannot import `safe_control_gym`. A `PYTHONPATH`
problem, unrelated to this work.

All 65 failures reduce to one cause and three attribute names:

    53x AttributeError: 'RecordDataWrapper' object has no attribute 'GUI'
     6x AttributeError: 'RecordEpisodeStatistics' object has no attribute 'constraints'
     6x AttributeError: 'RecordDataWrapper' object has no attribute 'done_on_out_of_bound'

Gymnasium 1.0 removed `Wrapper.__getattr__` forwarding to the wrapped
environment. Surface: ~11 passthrough call sites, 2 wrapper classes, and
`.unwrapped` appears exactly once in the package today.

Imports and environment construction are otherwise clean: `inverted_pendulum`,
`cartpole` and `quadrotor` all `make()` and `reset()` under 1.3.0, with only
`Box` precision warnings.

**Bound on the spike.** Those 65 tests failed *at* the wrapper attribute access,
so nothing downstream of it executed. The spike measured the first layer only.
It also never exercised the step-API mismatch, because no consumer in this
repository speaks the Gymnasium step protocol — SB3 would be the first.

## Decisions

### Migrate the environments; do not adapt around them

A training-side adapter stack (4-tuple to 5-tuple, obs re-encoding, frame skip)
would let SB3 train without touching the environments, and was considered.

Rejected. It leaves the package speaking an API that Gymnasium removed, and
puts a translation layer between the environments and their only
Gymnasium-native consumer — a layer that can drift from the environments it
wraps without any test noticing. The adapter also hides the defect described
below rather than fixing it.

### `terminated` and `truncated` are already distinguishable

The split is derivable from existing code, not a judgement call:

| Flag | Source | Meaning |
| --- | --- | --- |
| `terminated` | `_get_done()` per env | goal reached, or out-of-bounds when `done_on_out_of_bound` |
| `truncated` | `benchmark_env.py:499`, `ctrl_step_counter >= CTRL_STEPS` | time limit, `EPISODE_LEN_SEC * CTRL_FREQ` |

Per environment, `_get_done()` is goal-reach only for the pendulum
(`inverted_pendulum.py:333`, explicitly no out-of-bounds), and goal-reach or
out-of-bounds for cartpole (`cartpole.py:689`) and quadrotor
(`quadrotor.py:869`, with a per-`QuadType` bounds mask).

`after_step` currently collapses both into one `done`. **This is a live defect,
not only an API mismatch.** Every controller treats a horizon timeout
identically to a goal reach, so value bootstrapping at the end of an episode is
wrong — for exactly the algorithms this work exists to train. Fixing it is the
point of choosing migration over adaptation.

Consequence: RL controllers legitimately produce different numbers after the
migration. That difference is the fix landing, not a regression. LQR is
unaffected and is held bit-exact (see below).

### Safety net, built before anything is touched

The repository already characterises the pendulum to tight tolerances —
`tests/test_inverted_pendulum/fixtures/env_rollouts.json` pins
`(x0, action sequence) -> state sequence` at `atol=1e-9`, `lqr_gain.json` pins
`K` and `(state -> u)` pairs at `1e-6`/`1e-9`, and `rl_golden.json` pins
per-model single-step actions. Cartpole and both quadrotors have no numerical
pins at all; they are covered only by `tests/test_examples/`, which asserts that
examples run, not that they produce the same numbers.

Three oracles, captured under Gymnasium 0.28 and committed **before** any
migration commit:

1. **Golden rollouts extended to all four systems.** Generate `env_rollouts.json`
   for cartpole and quadrotor 2D/3D in the pendulum's existing shape and
   tolerance. Pins physics against a step-API refactor perturbing integration
   order.
2. **A dataset eval slice.** `rollout_seed(base_seed, split_id, index, batch)` is
   a pure function of its coordinates, so a fixed grid reproduces exactly.

   Concretely: `--controller lqr --seed 42`, a coarse grid (`--resolution 0.5`,
   giving 13 x 26 = 338 cells) written to a scratch `--output_dir` outside
   `DATA_ROOT`. Capture per-cell label and terminal state; require an exact
   match after migration.

   `--skip_save` is **not** usable here — it suppresses sequence writing
   (`generate_inverted_pendulum_trajectories.py:252`), so terminal states would
   not be persisted, and terminal states are the quantity the downstream model
   consumes. The slice must be small enough to write in full.

   This is the oracle that protects the datasets, which is what the repository
   exists to produce.
3. **Invariant-set recomputation.** Re-run `compute_invariant_sets.py` for all
   four systems and require `P`, `center` and `c` to match the committed
   `invariant_sets/*.npz`. Cheap, and exercises the closed-loop step map
   directly.

Held bit-exact: golden rollouts, invariant sets, and the **LQR** dataset slice.
Allowed to change: **RL** dataset outputs, per the previous section.

### Flag semantics need their own tests

Golden rollouts cannot catch a `terminated`/`truncated` inversion: dynamics are
identical either way, and only bootstrapping changes. Each environment gets an
explicit test asserting which condition raises which flag, including that both
can be set on the same step (goal reached exactly at the horizon).

### Delivered as one pull request

Steps 1-6 (safety net and migration) would stand alone as a green,
SB3-unused-but-installed state, and splitting there was considered so that the
upstream-touching change could be reviewed on its own.

Rejected in favour of a single pull request. The migration's justification is
the training path — on its own it is a dependency bump with no consumer, which
makes the `terminated`/`truncated` split look like churn rather than the
bootstrapping fix it is. Reviewing them together is what makes the change
legible.

The commit sequence still follows the order of work, so the migration remains
separable by commit if it later needs to be.

### stable-baselines3 is confined to one module

SB3 may be imported only by `safe_control_gym/experiments/train_pendulum_sb3.py`.
`envs/` and `controllers/` stay SB3-free, so `PendulumRL` inference and the whole
collection path continue to work in an environment where SB3 is not installed.

### The observation encoding becomes shared

`[cos theta, sin theta, theta_dot / theta_dot_max]` is currently hardcoded inside
`PendulumRL.select_action`. Training needs the identical transform. It moves to
a single function called by both the training observation wrapper and
`select_action`.

Duplicating it is the rejected alternative, and the reason is that its failure
is silent: a policy trained on a slightly different encoding still trains, still
exports, still loads, and is simply wrong at inference.

### Export contract is unchanged

The trainer writes the existing 8-key checkpoint, so `PendulumRL` and every
generator that names a policy keep working:

| key | value |
| --- | --- |
| `actor_state_dict` | `PendulumActor.state_dict()` |
| `obs_dim` | 3 |
| `act_dim` | 1 |
| `hidden_dims` | `[256, 256]` |
| `activation` | `'relu'` |
| `u_sat` | 0.6371781908344007 |
| `theta_dot_max` | 6.283185307179586 |
| `action_repeat` | 4 |

`models/manifest.json` keeps its shape; provenance fields change from an
external `source_zip` path to git SHA, config, seed and checkpoint step.

The trainer checkpoints periodically as well as on best, because that is what
the `strong`/`weak` axis is (see Prior state) and losing it would make the
existing model set unreproducible even after this work.

## Order of work

1. Capture golden rollouts for cartpole and quadrotor 2D/3D under Gymnasium 0.28. Commit.
2. Capture the dataset eval slice and invariant-set references under 0.28. Commit.
3. Bump Gymnasium to 1.3, add stable-baselines3, in `pyproject.toml` and `setup.py`.
4. Restore wrapper attribute forwarding at the ~11 sites, as explicit properties rather than a blanket `__getattr__`.
5. Split `done` into `terminated`/`truncated` in `benchmark_env` and the three environments; update the 15 modules that reference `env.step`.
6. Add the per-environment flag-semantics tests.
7. Add the shared observation encoding and the training wrappers.
8. Add `train_pendulum_sb3.py` and the exporter.
9. Full verification (below).

Steps 1 and 2 are the ones that are tempting to skip. Skipping them makes every
later step unverifiable.

## Verification

- `tests/test_examples/` + `tests/test_build.py`: back to **68 passed, 2 skipped**.
- `tests/test_inverted_pendulum/`: **74 passed**, with only the known subprocess failure.
- Golden rollouts for all four systems: match at `atol=1e-9`.
- Invariant sets: `P`, `center`, `c` match the committed artifacts.
- LQR dataset slice: labels, terminal states and `p_success` match exactly.
- Flag-semantics tests pass for all four environments.
- Export round-trip: train briefly, export, load through `PendulumRL`, and assert
  its forward matches SB3's deterministic `predict()` on random observations
  within `1e-6`. The existing port achieved `~3e-7`
  (`forward_max_err` in `manifest.json`), so this tolerance is established rather
  than invented.

The pre-existing `test_pendulum_experiment.py` subprocess failure is out of
scope and must remain the only failure; it must not be papered over as part of
this work.

## Out of scope

- **Noise-matched training.** Training policies under a noise preset so the
  closed loop is optimised for the noise it is collected at. This is the
  motivating follow-on and gets its own spec once the training path exists.
- **Reproducing `v1..v4 x strong/weak`.** The original reward, hyperparameters
  and configuration live in the external repository. Matching them may not be
  achievable and would stall this work. The trainer is designed so the attempt
  is possible later.
- **The `test_hpo` suite.** Needs a MySQL-backed Optuna store and does not run
  standalone; unaffected either way.
