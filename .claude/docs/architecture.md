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

## Noise: two unrelated mechanisms

### Pendulum noise (`pendulum_noise.py`)

`envs/gym_control/pendulum_noise.py` ports the external pendulum repo's noise
families and names them the same way (`truncated_gaussian_act_med`,
`control_proportional_high`, …). `NOISE_PRESETS` maps preset name to
`(NoiseModel subclass, params)`; `build_noise_model` also accepts a
`{'type': ..., **params}` dict, a `NoiseModel` instance, or `None`.

Levels run `low`, `med`, `high`, `xhigh`, `xxhigh`, `ultra`, `max` — but the
ordering is WRONG in two families: `gaussian_act_ultra` is var 3.0 against
`_max`'s 2.0, and `truncated_gaussian_act_ultra` is 2.0 against `_max`'s 1.0.
`max` is the *weaker* of the pair in both. Since the collector derives the output
directory from the suffix (`noise_level()`), a shipped dataset labelled `max` is
mis-ordered relative to `ultra`. Select noise the safe-control-gym way — an
`--overrides` file under `examples/inverted_pendulum/config_overrides/noise/` —
not by constructing a model inline.

**Its dynamics families measure the noise, not the controller.** They add `eps`
to the state *after* integration (`inverted_pendulum.py:173-177`), same sigma on
theta (rad) and thetadot (rad/s), independent draws, redrawn every substep. A
force cannot do this: a generalised force enters the acceleration row only, so a
physical disturbance moves theta at second order in `dt` while moving thetadot at
first order. Measured on a 161x161 grid over the whole state space, LQR, matched
sigma:

    deterministic                       mean p 0.386
    velocity_proportional_high          mean p 0.431   <- success RISES
    gaussian_act_high (torque)          mean p 0.256   <- success falls
    uniform torque, same sigma          mean p 0.247

Success *rises* under the state-additive families because one draw moves theta by
~0.083 against a goal radius of 0.075 — the noise can place the state in the goal
set rather than driving it there. Per-step theta movement, measured from recorded
trajectories: controller alone 0.041, torque noise 0.041 (unchanged), state
noise 0.559.

This is the MECHANISM, not the success rule. Re-running with a per-channel box
and a 10-step dwell requirement still gives 0.426 against 0.386, because
`velocity_proportional`'s sigma scales with `|thetadot|` and collapses to 0.008
at the goal — the noise carries the state there, then switches itself off. So the
shipped noisy pendulum datasets CANNOT be repaired by relabelling stored
trajectories; the inflation is in the trajectories.

Fixes, cheapest first: apply the noise on `disturbances['action']` (a torque);
or keep the state write but only on `thetadot`, which is exactly an impulsive
torque and is what `legged_gym` does ("emulates an impulse by setting a
randomized base velocity"); or derive the full covariance from a torque spectral
density, which gives `sigma_theta/sigma_thetadot = dt/sqrt(3)` and correlation
0.87 rather than equal independent draws (Van Loan discretisation; Sarkka &
Solin, *Applied SDEs*, Ex. 6.3).

### The `disturbances` mechanism (upstream, every env)

`envs/disturbances.py` is the other one, and the only one the cartpole and
quadrotors have. `DISTURBANCE_TYPES` is `impulse`, `step`, `uniform`,
`white_noise`, `periodic`, `signal_dependent` — `brownian` and `state_dependent`
are stubs, absent from the registry. **`periodic` does not do what its name says**: it redraws the
phase on every `apply()` call (`disturbances.py`), so the `t` dependence is
swamped and it yields i.i.d. arcsine-distributed noise in `[-scale, scale]` with
`frequency` having no effect. A genuine periodic disturbance would draw the phase
once in `reset()`. Consequence: the library currently offers NO temporally
correlated disturbance — `periodic` would have been it. Each env declares `DISTURBANCE_MODES`; cartpole's is
`{'observation': dim 4, 'action': dim 1, 'dynamics': dim 2}`.

The three modes are different physics, not three intensities. The split matters
because a physical disturbance on a mechanical system is a generalised force, and
generalised forces enter the acceleration rows only — the kinematic row `qdot = v`
is a definition, not a law to perturb. Noise on a *position* is sensor noise, and
belongs on `observation`, which perturbs the measurement and leaves the true state
alone. No major simulator exposes a per-step state-write channel: MuJoCo documents
`ctrl`, `qfrc_applied` and `xfrc_applied`; Isaac Lab randomises observations,
actions, sim params and actor params; Gymnasium touches the RNG only in `reset()`.

- **`action`** — added to the physical command before the `physical_action_bounds`
  clip (`cartpole.py:543`). Units are Newtons, since it runs after
  `denormalize_action`. **Matched**: lies in `range(B)`.
- **`dynamics`** — `p.applyExternalForce` on the *pole* link at its COM, world
  frame, `[Fx, 0, Fz]` (`cartpole.py:592-610`). Drawn once per control step and
  held across substeps. Uncapped — bypasses the action clip. **Unmatched**.
- **`observation`** — post-hoc on the returned obs only. Yields a POMDP; the
  dynamics stay deterministic.

`adversary_disturbance` reuses the `dynamics`/`action` channels for a learned
RARL adversary; it needs `env.set_adversary_control` each step and is not a
passive noise source.

Nothing under `configs/` enables any of this — every env is deterministic today.

**Cartpole's chosen axis** is `dynamics` masked to `Fx` with `uniform`: unmatched,
so the ROA estimate is not optimistic, and bounded, so the ROA can be set-valued.
See `docs/superpowers/specs/2026-07-31-cartpole-stochastic-dynamics-design.md`
for the derivation and the four rejected alternatives. Two traps recorded there:
`Disturbance.seed` binds `env.np_random` itself rather than a child stream, which
breaks the pure-function-of-`rollout_seed` invariant; and one Newton at the pole
COM is `M/m = 10` times the angular effect of one Newton at the cart, with
opposite sign, so magnitudes do not transfer between modes or between systems.

---

Related: [datasets.md](datasets.md) for what the root scripts actually collect, [conventions.md](conventions.md) for the scope rule this page states, [glossary.md](glossary.md) for the terms.

## The `dynamics` disturbance mode, as actually used

Documented previously as one of cartpole's three modes but unused anywhere. The
2026-08-14/15 quadrotor collections are the first use, and two properties matter.

`DISTURBANCE_MODES['dynamics']['dim']` is `int(QuadType)` — 2 for quad2d, 3 for
quad3d — and the force enters `_advance_simulation` as `applyExternalForce` at
the COM link in `WORLD_FRAME`, **inside** the substep loop. Re-applying every
substep is required, not incidental: PyBullet clears external forces after each
`stepSimulation`, so applying once per control step would act for 1 substep of 50
and silently under-apply the disturbance 50x.

Applied at the COM, the force produces no moment, so the disturbance input matrix
has nonzero entries only in the linear-acceleration rows. That is what makes it
unmatched: `range(B_d)` is not contained in `range(G(x))`, since the vehicle can
only produce lateral force by tilting first. Contrast the `action` mode, which
perturbs rotor thrusts and therefore lands in the same channel the controller
commands.

`BrownianNoise` and `StateDependentDisturbance` exist as classes but are **not**
registered in `DISTURBANCE_TYPES`, so the six reachable types are `impulse`,
`step`, `uniform`, `white_noise`, `periodic`, `signal_dependent`. A
velocity-dependent (drag-like) disturbance is therefore not expressible through
this path without registering one.

`SignalDependentNoise` is the fork's addition (2026-08-15): Gaussian noise whose
scale is a function of the signal it perturbs,

```
w ~ Normal(0, alpha + beta * |target|)
```

`WhiteNoise` cannot express this — its `std` is fixed at construction, so it
necessarily has the same sigma at the goal as far from it. The two constants are
separate mechanisms rather than one magnitude: `alpha` is the floor surviving as
the command goes to zero, `beta` the effort-proportional term that bites only
during the transient. The sum is a **standard deviation, not a variance**; the
two readings differ by ~5x at the values in use and put the family on opposite
sides of the `tau` sweep, so the class docstring, the JSON field `scale_is` and
`pend_sig_validate.py sigma` all assert it. `|target|`, not `target`: a signed
command sends the scale negative at `u = -0.637`, and the constructor rejects a
negative `alpha` or `beta` for the same reason.

### Where an action disturbance sits relative to saturation

The pendulum takes `external_action_disturbance` (default `False`), which
selects between two placements in `_preprocess_control`:

| | applied | physical claim |
| --- | --- | --- |
| `False` | `sat(u + w)` | noise **inside** the actuator — command, current, quantisation. The motor cannot be driven past `u_sat` by it. |
| `True` | `sat(u) + w` | an external torque on the **shaft** — wind, contact, friction. Still matched, still through `B`, but the actuator's limit does not apply to something the actuator is not producing. |

This is a different physical claim, not a magnitude setting, and it decides
whether noise can help at all — see `datasets.md`. Under `True` the disturbance
is deliberately **not** re-clipped, and any signal-dependent scale reads the
*saturated* command, which is the torque the actuator is really producing.

`u_sat` is untouched by the switch and must stay that way: it is a property of
the plant, and at an authority ratio of 0.866 raising it by 15% makes the
pendulum fully actuated and collapses the whole region-of-attraction structure.
The question is never how big the actuator is, only where `w` lives.
