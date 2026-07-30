# Architecture

Load when touching `safe_control_gym/`, or adding an environment, controller, or
safety filter.

## Two layers, deliberately separate

- **`safe_control_gym/`** — the library. Upstream `safe-control-gym` (UofT DSL)
  plus this fork's inverted pendulum. Envs, controllers and safety filters are
  import-only; the two exceptions are `experiments/train_rl_controller.py` and
  `experiments/train_sb3.py`, both runnable training scripts.
- **Repo root `*.py`** — this fork's scripts. Dataset generators, invariant-set
  computation, calibration, visualisation. They import the library, never the
  reverse.

## The Gymnasium step API

`gymnasium = "^1.3"` (`pyproject.toml`, `setup.py`). All four registered
environments (`benchmark_env` and the three concrete envs) return the
Gymnasium 5-tuple `(obs, rew, terminated, truncated, info)`, not the
pre-Gymnasium `(obs, rew, done, info)`. `reset(self, seed=None, options=None)`
on all three; `options` is accepted and ignored.

`terminated` has **two sources**, not one:

| Flag | Source | Meaning |
| --- | --- | --- |
| `terminated` | `_get_done()`, per env | goal reached, or out-of-bounds when `done_on_out_of_bound` |
| `terminated` | `benchmark_env.py`'s `after_step`, `constraints.is_violated(...)` under `DONE_ON_VIOLATION` | constraint violation |
| `truncated` | `after_step`, `ctrl_step_counter >= CTRL_STEPS` | time limit (`EPISODE_LEN_SEC * CTRL_FREQ`) |

`info['TimeLimit.truncated']` is deliberately still set alongside `truncated`
— six controllers (`ppo`, `ddpg`, `sac`, `rarl`, `rap`, `safe_ppo`) already read
that key to compensate a bootstrap target for time truncation, and the flag
promotes what was already a working convention rather than replacing it.
Both flags can be true on the same step (goal reached exactly at the horizon);
neither implementation may mask the other.

Both termination sources and the truncation path are exercised by
`tests/test_envs/test_truncation_semantics.py`; see
`.claude/docs/workflows.md` for the full oracle set.

## Two RL stacks coexist

The repo's own native controllers — `ppo`, `sac`, `ddpg`, `rarl`, `rap`,
`safe_explorer_ppo`, `pendulum_rl` — are unchanged by the Gymnasium migration
and are what every example under `examples/` still uses.

`stable-baselines3 = "^2.9"` is a second, training-only stack, added so that
*some* system can be trained in-repo without a per-environment adapter (see the
Motivation in
`docs/superpowers/specs/2026-07-28-sb3-gymnasium-migration-design.md`). It is
confined by rule to one module: **`envs/` and `controllers/` must stay
SB3-free**, so `PendulumRL` inference and the whole dataset-collection path
keep working in an environment where SB3 is not installed. A test
(`tests/test_envs/test_train_sb3.py::test_sb3_not_imported_by_library`) asserts
this by blanking `sys.modules['stable_baselines3']` and importing both
packages.

The seam back is a **per-system exporter**. `scripts/export_sb3_pendulum.py`
reads the SB3 `.zip` and writes the 8-key `.pt` that `pendulum_rl` loads, so a
pendulum policy trained here can be run and collected with via
`--controller <name>`. Export scripts may import SB3 — they are scripts, never
imported by the library — but `envs/` and `controllers/` may not.

**Only the pendulum has this.** Cartpole and the quadrotors can be trained by
`train_sb3.py`, but have no native actor controller to export *into*, so their
policies are inspectable and retrainable yet cannot be run in-repo. Closing that
for a second system means writing its actor controller first, and for cartpole
also teaching its generator the `disturbances` noise mechanism — the pendulum's
`pendulum_noise.py` is pendulum-only.

Keeping a generator's logic in the root script is the existing convention. Do
not push collection policy (grids, splits, stopping rules, success labels) down
into `safe_control_gym/` — the library models systems and controllers, the
scripts decide what to collect.

## The registry

`safe_control_gym/utils/registration.py` provides `register(...)` and
`make(idx, ...)`. Everything is looked up by string id.

- Envs registered in `safe_control_gym/envs/__init__.py`. Three name a
  *system*: `cartpole`, `quadrotor`, `inverted_pendulum`. Four name a
  `(system, task)` pair: `cartpole_stabilization`,
  `inverted_pendulum_stabilization`, `quadrotor2d_stabilization`,
  `quadrotor3d_stabilization`.

  The composite ids exist because `--task` carried two axes at once — the
  registry id and the `task:` field inside its yaml (the `Task` enum,
  `stabilization` or `traj_tracking`) — so a run directory named `quadrotor_3`
  said neither which `quad_type` nor which task it was. A composite id is its
  base `entry_point` plus a yaml with both axes pinned; it needs no new plumbing
  because `configuration.py` already resolves `--task <id>` to that id's yaml.
  The base ids are unchanged, so every collector and example still works.

  Each composite yaml must stay a faithful copy of its base — building either id
  with the same config must give the same environment, which
  `tests/test_envs/test_composite_env_ids.py` asserts observation-by-observation.
  Training values go in `configs/sb3/<env_id>_<algo>.yaml`, never in the env yaml.
- Controllers registered in `safe_control_gym/controllers/__init__.py`:
  `lqr`, `ilqr`, `mpc`, `linear_mpc`, `gp_mpc`, `mpc_acados`, `pid`, `ppo`,
  `sac`, `ddpg`, `safe_explorer_ppo`, `rarl`, `rap`, plus this fork's
  `pendulum_lqr` and `pendulum_rl`.
- Safety filters in `safe_control_gym/safety_filters/`: `mpsc` variants, `cbf`,
  `cbf_nn`.

Each registration carries a `config_entry_point`, either
`module:file.yaml` or `module:dict_name`. That yaml is the *defaults* for the
component; nothing else supplies them.

## Config resolution

`safe_control_gym/utils/configuration.py::ConfigFactory.merge()` builds the
config every example uses:

1. Base bookkeeping dict (`tag`, `seed`, `use_gpu`, `output_dir='results'`, `restore`).
2. CLI args: `--algo`, `--task`, `--safety_filter`.
3. Registered defaults for whichever of those were named.
4. `--overrides a.yaml b.yaml …`, applied in order.
5. `--kv_overrides 'algo_config.training=False'` — dotted-path scalar overrides,
   applied last.

Result is a `munch` object, so `config.task_config.noise` and
`config.get('n_episodes', 5)` both work. `examples/inverted_pendulum/pendulum_experiment.py`
is the shortest complete illustration of the whole flow.

## Package map

| Path | Role |
| --- | --- |
| `envs/benchmark_env.py` | Base env: constraints, disturbances, symbolic model plumbing. |
| `envs/gym_control/` | Non-PyBullet systems: `cartpole.py`, `inverted_pendulum.py`, and `pendulum_noise.py`. |
| `envs/gym_pybullet_drones/` | `quadrotor.py` (2D and 3D) over `base_aviary.py`. |
| `envs/constraints.py`, `envs/disturbances.py` | Constraint and disturbance objects referenced from task yaml. |
| `envs/env_wrappers/` | Episode-statistics recording, the vectorised env family, `forwarding.py` (`AttributeForwardingMixin`), `shaping.py` (optional SB3 observation/cadence wrappers). |
| `controllers/base_controller.py` | Interface every controller implements. |
| `controllers/pendulum_lqr/`, `controllers/pendulum_rl/` | This fork's pendulum controllers. RL policies ship as native torch `.pt` state-dicts. |
| `safety_filters/` | MPSC and CBF filters over a base controller. |
| `experiments/base_experiment.py` | `BaseExperiment.run_evaluation(...)` — the rollout + metrics loop. |
| `experiments/train_rl_controller.py` | Native-stack RL training entry point (`ppo`, `sac`, `ddpg`, `rarl`, `rap`, `safe_explorer_ppo`). |
| `experiments/train_sb3.py` | Task-agnostic stable-baselines3 training entry point. The only module permitted to import SB3. |
| `math_and_models/` | Symbolic models, transformations, metrics. `transformations.py` is exempted from most lint rules. |
| `utils/` | `registration.py`, `configuration.py`, `logging.py`, `plotting.py`, `utils.py`. |
| `hyperparameters/` | Optuna-based HPO. Has its own tests under `tests/test_hpo/`. |

## Adding a component

Adding an environment or controller means three edits, not one:

1. The implementation module.
2. A sibling `<name>.yaml` holding its defaults.
3. A `register(...)` call in the package's `__init__.py`.

Then a test under `tests/`. `tests/test_inverted_pendulum/test_registration.py`
is the pattern for asserting a component is reachable through `make`.

## Pendulum noise

`envs/gym_control/pendulum_noise.py` ports the external pendulum repo's noise
families and names them the same way (`truncated_gaussian_act_med`,
`control_proportional_high`, …). `NOISE_PRESETS` maps preset name to
`(NoiseModel subclass, params)`; `build_noise_model` also accepts a
`{'type': ..., **params}` dict, a `NoiseModel` instance, or `None`.

Levels, weakest to strongest: `low`, `med`, `high`, `xhigh`, `xxhigh`, `ultra`,
`max`. Select noise the safe-control-gym way — an `--overrides` file under
`examples/inverted_pendulum/config_overrides/noise/` — not by constructing a
model inline.

---

Related: [datasets.md](datasets.md) for what the root scripts actually collect, [conventions.md](conventions.md) for the scope rule this page states, [glossary.md](glossary.md) for the terms.
