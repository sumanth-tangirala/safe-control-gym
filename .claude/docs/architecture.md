# Architecture

Load when touching `safe_control_gym/`, or adding an environment, controller, or
safety filter.

## Two layers, deliberately separate

- **`safe_control_gym/`** — the library. Upstream `safe-control-gym` (UofT DSL)
  plus this fork's inverted pendulum. Import-only; no CLI entry points.
- **Repo root `*.py`** — this fork's scripts. Dataset generators, invariant-set
  computation, calibration, visualisation. They import the library, never the
  reverse.

Keeping a generator's logic in the root script is the existing convention. Do
not push collection policy (grids, splits, stopping rules, success labels) down
into `safe_control_gym/` — the library models systems and controllers, the
scripts decide what to collect.

## The registry

`safe_control_gym/utils/registration.py` provides `register(...)` and
`make(idx, ...)`. Everything is looked up by string id.

- Envs registered in `safe_control_gym/envs/__init__.py`:
  `cartpole`, `quadrotor`, `inverted_pendulum`.
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
| `envs/env_wrappers/` | Episode-statistics recording and the vectorised env family. |
| `controllers/base_controller.py` | Interface every controller implements. |
| `controllers/pendulum_lqr/`, `controllers/pendulum_rl/` | This fork's pendulum controllers. RL policies ship as native torch `.pt` state-dicts. |
| `safety_filters/` | MPSC and CBF filters over a base controller. |
| `experiments/base_experiment.py` | `BaseExperiment.run_evaluation(...)` — the rollout + metrics loop. |
| `experiments/train_rl_controller.py` | RL training entry point. |
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
