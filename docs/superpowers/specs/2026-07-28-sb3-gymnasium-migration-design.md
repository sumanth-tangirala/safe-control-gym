# Spec: Gymnasium Migration and In-Repo SB3 Training

Date: 2026-07-28
Scope: `safe_control_gym/envs/` (all four envs + `benchmark_env`), the 15 modules
that reference `env.step`, `safe_control_gym/envs/env_wrappers/`,
a new `safe_control_gym/experiments/train_sb3.py`, `pyproject.toml`,
`setup.py`, and `tests/`.

## Goal

Make RL controllers trainable inside this repository, for **any** registered
system.

Concretely: migrate the environments to the Gymnasium step API, add
stable-baselines3 as a dependency, and add one training entry point that works
against any registered task — `--task inverted_pendulum`, `cartpole`, or
`quadrotor` — with no per-system glue.

Generality is the reason for migrating rather than adapting. A training-side
adapter would need one implementation per environment; Gymnasium-compliant
environments need none.

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
| `terminated` | `benchmark_env.py:478-482`, `constraints.is_violated(...)` under `DONE_ON_VIOLATION` | constraint violation |
| `truncated` | `benchmark_env.py:497-500`, `ctrl_step_counter >= CTRL_STEPS` | time limit, `EPISODE_LEN_SEC * CTRL_FREQ` |

Note there are **two** termination sources, not one. `_get_done()` is goal-reach
only for the pendulum (`inverted_pendulum.py:333`, explicitly no out-of-bounds),
and goal-reach or out-of-bounds for cartpole (`cartpole.py:689`) and quadrotor
(`quadrotor.py:869`, with a per-`QuadType` bounds mask). Constraint violation is
handled separately, inside `after_step`, and is also termination.

The distinction is **already recorded**, in the pre-Gymnasium convention:

    if self.ctrl_step_counter >= self.CTRL_STEPS:
        info['TimeLimit.truncated'] = not done
        done = True

So the migration promotes an existing `info` key to a first-class return value
rather than recovering information the environments threw away. That lowers the
risk materially: `truncated` can be cross-checked against
`info['TimeLimit.truncated']` during the migration, and the two must agree.

**There is no bootstrapping defect.** An earlier draft of this spec claimed the
collapse into `done` corrupted value bootstrapping. That is wrong, and the code
says so: all six RL controllers already read the info key and compensate.
`sac.py:287-304` is the clearest instance —

    # time truncation is not true termination
    ...
    if 'TimeLimit.truncated' in inff and inff['TimeLimit.truncated']:
        terminal_idx.append(idx)
        terminal_obs.append(inf['terminal_observation'])
    ...
    true_mask[idx] = 1.0

— and `ppo`, `ddpg`, `rarl`, `rap` and `safe_ppo` all reference
`TimeLimit.truncated` the same way.

So the migration's justification is narrower and more honest than "fixing a
bug":

1. It is what lets SB3, and any other Gymnasium-native tool, consume these
   environments at all.
2. It replaces a stringly-typed convention (`info['TimeLimit.truncated']`,
   which every consumer must know to look for and six of them separately
   reimplement) with a first-class return value that cannot be forgotten.

Consequence, and this is a strengthening: because the semantics already hold,
a correct migration must produce **identical numbers everywhere**, RL included.
The earlier draft exempted RL dataset outputs from bit-exactness on the grounds
that the fix would shift them. That exemption is withdrawn — nothing should
shift, and anything that does is a migration bug.

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

Held bit-exact: **everything**. Golden rollouts, invariant sets, and the dataset
slice for both LQR and RL controllers. Nothing is exempt, because the
truncation semantics the migration formalises are already honoured by every
consumer (see the previous section). A number that moves is a migration bug,
not a fix landing.

The dataset slice is therefore captured twice — once with `--controller lqr`
and once with `--controller v3_strong` — so the RL path is covered too.

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

SB3 may be imported only by `safe_control_gym/experiments/train_sb3.py`.
`envs/` and `controllers/` stay SB3-free, so `PendulumRL` inference and the whole
collection path continue to work in an environment where SB3 is not installed.

This is what makes the eventual per-system exporters worth writing: inference
never gains an SB3 dependency, no matter how many systems get trained.

### The trainer is task-agnostic; task-specific shaping is configuration

`train_sb3.py` takes `--task` and `--algo` through the existing `ConfigFactory`,
exactly as every other entry point in `examples/` does. It contains no
per-system branching.

Anything a particular system needs — an observation re-encoding, a frame skip —
is an optional wrapper selected by config, not a hardcoded stage. Concretely,
the pendulum's `[cos theta, sin theta, theta_dot / theta_dot_max]` encoding and
its `action_repeat` of 4 exist only because the shipped policies were trained
that way; cartpole and the quadrotors train on their environment's own
observation with no wrapper at all.

The rejected alternative is baking the pendulum encoding into the trainer. It
would work today and silently mis-train every other system.

### Inference is per-system native export, and is deferred

A single registered `sb3` controller that loads an SB3 model would work for
every system at once. Rejected: it puts an SB3 import on the inference path, and
therefore on the collection path, which contradicts the confinement rule above
and would make every dataset run depend on SB3.

Instead each system gets a native torch controller that a trained SB3 actor is
exported into — the pattern `PendulumRL` already established.

**No exporter is in this spec.** The pendulum exporter, which would write the
existing 8-key `.pt` (`actor_state_dict`, `obs_dim`, `act_dim`, `hidden_dims`,
`activation`, `u_sat`, `theta_dot_max`, `action_repeat`) and keep the
`--controller v3_strong` collection vocabulary working, is the immediate
follow-on.

Consequence, stated plainly: **when this spec lands, a trained policy has no
in-repo consumer.** Training produces SB3-native artifacts and nothing can run
them until the first exporter arrives. Verification therefore rests on API
conformance and training smoke rather than an export round-trip (see
Verification).

The trainer must checkpoint periodically as well as on best, because that is
what the `strong`/`weak` axis is (see Prior state) and losing it would make the
existing model set unreproducible once exporters exist.

## Order of work

1. Capture golden rollouts for cartpole and quadrotor 2D/3D under Gymnasium 0.28. Commit.
2. Capture the dataset eval slice and invariant-set references under 0.28. Commit.
3. Bump Gymnasium to 1.3, add stable-baselines3, in `pyproject.toml` and `setup.py`.
4. Restore wrapper attribute forwarding at the ~11 sites, as explicit properties rather than a blanket `__getattr__`.
5. Split `done` into `terminated`/`truncated` in `benchmark_env` and the three environments; update the 15 modules that reference `env.step`.
6. Add the per-environment flag-semantics tests.
7. Add the optional, config-selected observation and frame-skip wrappers.
8. Add `train_sb3.py`, task-agnostic, driven by `ConfigFactory`.
9. Full verification (below).

Steps 1 and 2 are the ones that are tempting to skip. Skipping them makes every
later step unverifiable.

## Verification

- `tests/test_examples/` + `tests/test_build.py`: back to **68 passed, 2 skipped**.
- `tests/test_inverted_pendulum/`: **74 passed**, with only the known subprocess failure.
- Golden rollouts for all four systems: match at `atol=1e-9`.
- Invariant sets: `P`, `center`, `c` match the committed artifacts.
- Dataset slice, both `lqr` and `v3_strong`: labels and terminal states match exactly.
- The six RL controllers' `TimeLimit.truncated` compensation blocks
  (`ppo`, `ddpg`, `sac`, `rarl`, `rap`, `safe_ppo`) read the new `truncated`
  flag and remain behaviour-identical.
- Flag-semantics tests pass for all four environments, and `truncated` agrees
  with the legacy `info['TimeLimit.truncated']` on every step.
- **`stable_baselines3.common.env_checker.check_env` passes on every registered
  environment.** This is the primary evidence that the migration is correct: it
  validates the Gymnasium API contract directly — tuple arity, `reset` signature,
  space conformance, dtype — rather than inferring correctness from tests that
  happen to pass. It is also generic, so it covers systems that have no golden
  fixtures.
- **Training smoke per task:** a short `train_sb3.py` run on
  `inverted_pendulum`, `cartpole` and `quadrotor` completes and writes a
  loadable SB3 model. Asserts the framework is genuinely task-agnostic rather
  than pendulum-shaped.

There is deliberately no export round-trip check, because no exporter is in
scope. The follow-on that adds the pendulum exporter must assert that its
output, loaded through `PendulumRL`, matches SB3's deterministic `predict()`
within `1e-6`; the existing port achieved `~3e-7` (`forward_max_err` in
`manifest.json`), so that tolerance is established rather than invented.

The pre-existing `test_pendulum_experiment.py` subprocess failure is out of
scope and must remain the only failure; it must not be papered over as part of
this work.

## Out of scope

- **Per-system exporters, including the pendulum's.** The immediate follow-on:
  a native torch controller per system plus the exporter that fills it, starting
  with the pendulum's 8-key `.pt` so `PendulumRL` and the `--controller
  v3_strong` collection vocabulary pick up newly trained policies. Until it
  lands, trained policies cannot be run in-repo.
- **Noise-matched training.** Training policies under a noise preset so the
  closed loop is optimised for the noise it is collected at. This is the
  motivating follow-on and gets its own spec once the training path exists.
- **Reproducing `v1..v4 x strong/weak`.** The original reward, hyperparameters
  and configuration live in the external repository. Matching them may not be
  achievable and would stall this work. The trainer is designed so the attempt
  is possible later.
- **The `test_hpo` suite.** Needs a MySQL-backed Optuna store and does not run
  standalone; unaffected either way.
