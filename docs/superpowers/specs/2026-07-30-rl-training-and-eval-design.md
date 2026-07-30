# Spec: RL Training and Evaluation for Every System

Date: 2026-07-30
Scope: `safe_control_gym/envs/__init__.py`, `safe_control_gym/experiments/`,
`configs/sb3/`, `tests/test_envs/`. No collector changes.

## Goal

Train an RL policy for any system in this repo, and answer "is this policy good
enough to use" with a number rather than a judgement call.

Concretely: composite `(system, task)` environment ids, a run layout that
describes itself, per-system SB3 training configs, and an evaluation CLI that
scores a trained policy against the system's LQR baseline on identical seeded
initial states.

## Motivation

`train_sb3.py` is already task-agnostic — `build_env` is
`make(config.task, **config.task_config)` with nothing pendulum-specific — and
all four system variants pass stable-baselines3's `check_env`
(`tests/test_envs/test_gymnasium_conformance.py:13-21`: `inverted_pendulum`,
`cartpole`, `quadrotor` `quad_type: 2`, `quadrotor` `quad_type: 3`).

But conformance is not trainability. Only the inverted pendulum has ever been
trained through that path. The other three have no training config, no tuned
hyperparameters, and no confirmation that their reward functions yield
stabilization at all.

Evaluation is the larger hole. `BaseExperiment.run_evaluation` and
`compute_metrics` exist and produce episode length, return, RMSE and constraint
violations, but there is no entry point that uses them and no definition of a
passing policy. A policy is currently judged by watching training curves.

That gap blocks the downstream goal directly: a policy is only worth collecting
a dataset with if it succeeds from most of the state space, and today nothing
measures that.

## Prior state (measured)

### Two RL stacks

| stack | algorithms | entry point | proven on |
| --- | --- | --- | --- |
| stable-baselines3 | `sac`, `ppo` | `experiments/train_sb3.py` (92 lines) | inverted pendulum only |
| native | `ppo`, `sac`, `ddpg`, `safe_explorer_ppo`, `rarl`, `rap` | `experiments/train_rl_controller.py` (75 lines) | quadrotor 2D |

`train_sb3.py:45` is `ALGOS = {'sac': SAC, 'ppo': PPO}`, imported from
`stable_baselines3` — vanilla, not Safe Explorer. Safe Explorer PPO is a
separate native controller
(`safe_control_gym/controllers/safe_explorer/safe_ppo.py`), a learned constraint
model that projects actions rather than PPO with a penalty term.

This matters for continuity: the shipped `quadrotor2D_rl` dataset's
`dataset_description.json` records it was "generated using Safe Explorer PPO RL
controller". The one RL dataset in existence came from an algorithm the SB3 path
cannot produce.

### `task` already means two things

`configuration.py:42` defines `--task`, and `configuration.py:67` resolves it to
`config_dict['task_config'] = get_config(args.task)` — so `--task` is the
*registry id* (`cartpole`). Meanwhile the env yamls carry `task: stabilization`,
the `Task` enum at `benchmark_env.py:28` (`STABILIZATION`, `TRAJ_TRACKING`).

The real axes are **system** (cartpole, quadrotor `quad_type: 2`, quadrotor
`quad_type: 3`, inverted_pendulum) x **task** (stabilization, traj_tracking).

### The success signal is NOT uniform — corrected during implementation

The envs all compute `goal_reached` as `||state - X_GOAL|| < tolerance`, but
they do not all *expose* it. `_get_info` in both `cartpole.py` and
`quadrotor.py` gates the key:

```python
if self.TASK == Task.STABILIZATION and self.COST == Cost.QUADRATIC:
    info['goal_reached'] = self.goal_reached
```

RL training uses `cost: rl_reward`, so for cartpole and both quadrotors the key
is **absent**. Only the inverted pendulum emits it unconditionally
(`inverted_pendulum.py:346`). Verified by inspecting a live `step()` info dict on
cartpole: `['constraint_violation', 'current_step', 'mse', 'out_of_bounds']`.

An earlier draft of this spec asserted all four expose it uniformly. That was
wrong, and evaluation built on it would have reported a success rate of exactly
zero for three of four systems while looking healthy.

Evaluation therefore applies the envs' own rule to the state directly rather
than reading the key, which holds whatever the cost is.

This is a goal-*ball* test, and it is **not** the criterion the datasets use —
dataset success is membership of the terminal state in an invariant ellipsoid
(see `.claude/docs/datasets.md`). The two answer different questions and this
spec deliberately uses the goal-ball one; see Decisions.

### Two PyBullet envs in one process corrupted each other

Found while asserting composite/base equivalence. `base_aviary.py`'s
`changeDynamics` call omitted `physicsClientId`, alone among the file's PyBullet
calls, so it targeted client 0. A second env's `reset()` then applied its own
`DRONE_ID`'s damping to the *first* env's client — corrupting that env, and
leaving itself at PyBullet's default damping rather than 0.

Measured: two concurrent quadrotor-2D envs diverged by 0.34 in state within five
steps; sequential envs agreed exactly, because with one env the only client is 0
and the omission is harmless.

This is reachable from ordinary training, not only from tests: SB3's
`EvalCallback` holds an eval env open alongside the training env, and
`DummyVecEnv` holds `n_envs` of them. Fixed, with a falsifiable regression test
(`tests/test_envs/test_concurrent_pybullet_envs.py`).

#### It also means every shipped quadrotor dataset ran at the wrong damping

The quadrotor collectors hold two envs at once — `make('lqr', env_func, ...)`
builds one internally, then `env = env_func()` builds the one actually rolled
out (`generate_quadrotor_2d_trajectories.py:380,386`). The rollout env is the
*second*, so it is the one that never received `linearDamping=0,
angularDamping=0`. It ran at PyBullet's default.

Measured against a single-env reference, five steps, seed 7:

| | rollout env deviation |
| --- | --- |
| with `physicsClientId` | 0.000000 |
| without (as every shipped quadrotor dataset was generated) | 0.069001 |

Confirmed independently by the oracles: after the fix, `test_slice_reproduces`
fails for `quad2d`, `quad2d_rl` and `quad3d` and passes for `cartpole` — exactly
the three collectors that go through `base_aviary.py`. This is the collection
oracles doing the job they were built for.

The fixtures are **not** regenerated. They record what the shipped datasets
contain, and collection work is paused. The three cases are marked
`xfail(strict=True)`, so regenerating them turns the xfail into an XPASS and
fails the suite — the decision has to be taken deliberately, not by re-running a
capture script.

Single-env behaviour is unchanged: golden quadrotor rollouts stay bit-exact, and
the concurrent result now equals the sequential one.

### Noise is not uniform

Two independent mechanisms:

- `safe_control_gym/envs/disturbances.py` — upstream. `DISTURBANCE_TYPES`:
  impulse, step, uniform, white_noise, brownian, periodic, state_dependent.
  Configured by a `disturbances:` block in the env yaml, wired through
  `BenchmarkEnv._setup_disturbances`. Available to cartpole and quadrotor.
- `safe_control_gym/envs/gym_control/pendulum_noise.py` — this fork's.
  `NOISE_PRESETS` (gaussian_act, truncated_gaussian_act, velocity_proportional,
  control_proportional, each low..max), passed as a `noise=` constructor kwarg.
  **Inverted pendulum only.**

`--noise gaussian_act_high` therefore does not exist for cartpole or either
quadrotor.

## Decisions

### Evaluation reports RL-standard episode metrics, not ROA grid success

Mean return, mean episode length, success rate, out-of-bounds rate and
constraint violations over N seeded episodes.

Success is computed by `eval_policy` from the state, not read from
`info['goal_reached']` — that key is cost-gated and absent under `rl_reward` for
three of the four systems (see Prior state). It is evaluated at the **terminal**
state rather than at any step: under `rl_reward` an episode does not stop on
entering the goal ball, so "reached it at some point" would count trajectories
that arrived and then left, and the terminal state is what the downstream model
predicts anyway. `reached_goal_any_step_rate` is reported alongside so the gap
between the two is visible.

The alternative — rolling out a grid and labelling terminal states by
invariant-set membership — is a better predictor of dataset quality, because it
is the dataset's own criterion. It is deliberately deferred: it costs hundreds
to thousands of rollouts per evaluation, and the cheap metric is enough to
answer "did this policy train" and "is it better than LQR", which is what blocks
progress now. `invariant_sets/{cartpole,pendulum,quad2d,quad3d}.npz` all exist,
so adding it later is a metric swap and not a redesign.

Recorded consequence: **the eval metric and the dataset label disagree by
construction.** A policy can pass evaluation and still produce a dataset with a
different success rate. Do not report the two as if they were the same number.

### stable-baselines3 is primary; the native stack survives for what SB3 lacks

All new training goes through `train_sb3.py`. The native path stays reachable
only for `safe_explorer_ppo`, `rarl` and `rap`, which have no SB3 equivalent.

Accepted consequence: a retrained quadrotor-2D policy will be vanilla PPO or
SAC, not Safe Explorer PPO, so it is **not** a reproduction of the controller
that produced the shipped `quadrotor2D_rl` dataset. Any comparison against that
dataset must state this. Porting the Safe Explorer safety layer onto SB3 is a
separate piece of work and is not proposed here.

### All four systems get configs; policies land where they converge

Training configs, registration and evaluation cover all four systems
unconditionally. Actually training each is best-effort: run them, report what
converged and what did not.

Quadrotor-3D is the likely failure — 12 state dimensions, PyBullet dynamics, no
tuned hyperparameters, and no prior evidence that its RL reward stabilizes at
all. A non-converging quad3d is a finding to report, not a reason to hold the
spec.

### Deterministic only

No noise in this spec. The mechanism is pendulum-only (above), so "train under
noise" would mean either restricting to one system or changing the env layer for
cartpole and both quadrotors — a real change to shared code that does not belong
in the same spec as the training and eval framework.

Unifying noise across systems is named as the follow-on, not silently dropped.
Once it lands, noise becomes another config axis and the run layout below already
accommodates it without change.

### Composite registered env ids

One registry id per `(system, task)` pair, each pointing at an existing entry
point with its own yaml.

This works with **zero change** to `registration.py` or `configuration.py`,
because `configuration.py:67` already loads a registered id's yaml into
`task_config` and `train_sb3.py:52` already splats it into `make`. The composite
id needs no new plumbing — only a yaml that pins `quad_type` and `task`.

Chosen over composing run identity from two separate fields because a run
directory should name what it is without a reader opening `config.yml`, and
because an unsupported pair then fails at registry lookup rather than silently
training something unintended. It also matches the RL Baselines3 Zoo layout
already adopted for policy storage.

**`--task` is not renamed.** It is defined once in `configuration.py:42` and
shared by every entry point in the repo; renaming it globally is a wide diff
unrelated to this work. `train_sb3` accepts `--env_id` as an alias. The values
become composite ids either way.

### The acceptance bar is relative to LQR on identical seeded states

Every system has an LQR baseline already — `pendulum_lqr` for the inverted
pendulum, `lqr` for the other three, as used by all five collectors. Evaluation
rolls out the policy and the baseline from the *same* seeded initial states and
reports both.

Chosen over fixed per-system thresholds because those numbers would have to be
invented before anyone knows what is achievable, and a wrong constant either
passes bad policies or blocks good ones. A relative bar calibrates itself.

Known failure mode: **if LQR is itself weak on a system, the bar is vacuous.**
Mitigation is not a cleverer rule but disclosure — `eval.json` and the printed
report always carry absolute numbers for both controllers, so a low baseline is
visible next to the verdict rather than hidden behind it.

## Formats

### Registered ids

| id | entry point | yaml pins |
| --- | --- | --- |
| `inverted_pendulum_stabilization` | `gym_control.inverted_pendulum:InvertedPendulum` | `task: stabilization` |
| `cartpole_stabilization` | `gym_control.cartpole:CartPole` | `task: stabilization` |
| `quadrotor2d_stabilization` | `gym_pybullet_drones.quadrotor:Quadrotor` | `quad_type: 2`, `task: stabilization` |
| `quadrotor3d_stabilization` | `gym_pybullet_drones.quadrotor:Quadrotor` | `quad_type: 3`, `task: stabilization` |

The existing `cartpole`, `quadrotor` and `inverted_pendulum` ids stay registered
and unmodified. Each composite yaml is a copy of its base yaml with the pinned
fields set, so a composite id and its base id with equivalent overrides build the
same environment. `traj_tracking` variants are registered the same way when
wanted; none are added now.

### Run layout

```
logs/<algo>/<env_id>_<run>/
    best_model.zip        # EvalCallback
    checkpoints/          # CheckpointCallback, periodic
    config.yml            # fully merged config, as trained
    args.yml              # parsed CLI arguments
    command.txt           # verbatim argv
    eval.json             # written by the eval CLI
```

`<run>` auto-increments so a re-run never clobbers a previous one. `config.yml`
is what makes the run rebuildable: the eval CLI reconstructs the environment and
its shaping wrappers from it rather than re-deriving them from flags.

Periodic checkpoints are kept alongside `best_model.zip`, not replaced by it:
the shipped strong/weak pendulum model pairs are best-vs-intermediate
checkpoints of one run, so discarding intermediates would make that pairing
unreproducible.

### Training configs

`configs/sb3/<env_id>_<algo>.yaml`, passed through the existing `--overrides`
mechanism. Holds `algo`, `seed`, and the `sb3_config` block
(`total_timesteps`, `net_arch`, `save_freq`, `eval_freq`).

### `eval.json`

| key | type | notes |
| --- | --- | --- |
| `env_id` | str | composite id the run was trained on |
| `algo` | str | `sac` or `ppo` |
| `n_episodes` | int | episodes per controller |
| `seed` | int | seeds the shared initial states |
| `policy` | object | metrics block, below |
| `baseline` | object | same block, for `lqr` / `pendulum_lqr` |
| `baseline_id` | str | which LQR was used |
| `margin` | float | allowed shortfall against the baseline |
| `verdict` | str | `PASS` or `FAIL` |

Metrics block, per controller: `success_rate`, `mean_return`,
`mean_episode_length`, `out_of_bounds_rate`, `constraint_violation_rate`,
`terminated_frac`, `truncated_frac`.

`verdict` is `PASS` when `policy.success_rate >= baseline.success_rate - margin`.

## Interface

```
python -m safe_control_gym.experiments.train_sb3 \
    --env_id cartpole_stabilization --algo sac \
    --overrides configs/sb3/cartpole_stabilization_sac.yaml \
    --output_dir logs --use_gpu

python -m safe_control_gym.experiments.eval_policy \
    --run logs/sac/cartpole_stabilization_1 --n_episodes 100 --seed 0
```

Pass `--use_gpu` when one is free. Measured on an idle ilab2 (64 cores, RTX
A4500), SAC on the pendulum with `net_arch: [256, 256]`: 65.6 steps/s on CPU
against 111.0 steps/s on CUDA, 1.69x. The environment is not the bottleneck
either way — the pendulum steps at ~12,500/s.

## Order of work

1. Register the four composite ids and their yamls; extend the conformance
   parametrisation to cover them. Nothing else depends on a working trainer, so
   this lands first and independently.
2. Add the run layout and `EvalCallback` to `train_sb3.py`; add the `--env_id`
   alias.
3. Write `eval_policy.py` against the pendulum, whose trained policy already
   exists — so the CLI is proven before any new training runs.
4. Write the four training configs.
5. Train the four systems. Independent of each other; run in parallel.
6. Evaluate each, record `eval.json`, report what converged.

Steps 1-3 are the framework and are the part that must be correct. Steps 5-6 are
best-effort by the scope decision above.

## Verification

- All four composite ids pass SB3's `check_env`.
- A composite id and its base id plus equivalent overrides produce environments
  whose `observation_space`, `action_space` and `state_space` compare equal, and
  which return identical observations for a fixed seed. This is what makes the
  registration a rename rather than a behaviour change.
- `eval_policy` is deterministic: the same `--seed` and `--n_episodes` produce
  byte-identical `eval.json` metrics across runs.
- The policy and the baseline are evaluated from *identical* initial states —
  asserted directly, not assumed from a shared seed.
- A training run writes all of `config.yml`, `args.yml`, `command.txt` and
  `best_model.zip`.
- `tests/test_envs/` and `tests/test_inverted_pendulum/` stay green.
- No file under `/common/users/shared/pracsys/genMoPlan/data_trajectories` is
  written, and `invariant_sets/*.npz` stay byte-identical.

## Out of scope

- **Any collector change.** Paused pending a conversation with the dataset's
  other authors. Nothing here writes `roa_labels.txt`, `eval_states.txt` or a
  trajectory.
- **Noise.** Deterministic only, by the decision above. Unifying
  `NOISE_PRESETS` and `disturbances.py` into one mechanism across all four
  systems is the named follow-on.
- **ROA grid evaluation.** Deferred with the metric decision; the invariant sets
  it needs already exist.
- **Porting Safe Explorer PPO to SB3.** The native controller stays where it is.
- **Trajectory-tracking tasks.** Registrable by the same pattern; none added.
- **Hyperparameter search.** Configs are hand-written starting points, not tuned
  results.

## Risks

**Quadrotor-3D may not converge.** 12 dimensions, PyBullet, untuned, and no
evidence its RL reward stabilizes. Handled by the scope decision: report it.

**The LQR bar may be vacuous where LQR is weak.** Handled by disclosure, not by
a cleverer rule — both controllers' absolute numbers are always reported.

**The eval metric is not the dataset metric.** A passing policy is not a
guarantee of a good dataset. Stated in the decision, and worth restating
wherever a `verdict` is quoted.

## Related

- `.claude/docs/datasets.md` — invariant-set success labelling, the criterion
  this spec deliberately does not use.
- `docs/superpowers/specs/2026-07-28-sb3-gymnasium-migration-design.md` — the
  migration that made `train_sb3.py` possible.
- `docs/superpowers/specs/2026-07-29-collection-oracles-design.md` — the
  collector oracles this spec is explicitly not touching.
