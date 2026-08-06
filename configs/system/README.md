# configs/system/ — one plant per system

A system file holds what is true of the dynamical system regardless of regime:
the plant constructor kwargs (`task_config_overrides`), the start region, the
observation contract, and the curriculum. Regime files under
`configs/collection/` and `configs/physical/` pull it in with an `extends:` key
(resolved by `load_collection_bounds` in `train_sb3.py`, child keys win) and
add only their termination box and any dataset reference numbers.

## Migration status

| System | Status |
| --- | --- |
| cartpole | Migrated 2026-07-31. Plant verified against the shipped dataset: 596/596 label reproduction. |
| inverted_pendulum | Migrated 2026-07-31. Plant verified against the shipped dataset: 556/556 label reproduction (300 random + 256 balanced). The mismatch it caught was `pyb_freq` — the dataset integrates three explicit-Euler substeps per control step, the env default is one. Neither regime carries an `env_attributes` block, because this system has no termination box. |
| quadrotor2d | Migrated 2026-07-31. Plant verified against the shipped dataset: 555/556 label reproduction (300 random + a balanced 128/128, which is 256/256 on its own); the single miss is a state whose shipped terminal sits 0.00078 inside the 0.2 goal ball. Three mismatches caught: `ctrl_freq`/`pyb_freq` (env default 60/240, dataset 100/5000), `episode_len_sec` (default 5 s = 500 steps against a dataset horizon of 1200, and shipped trajectories run to 708 steps), and the goal tolerance (`task_info.stabilization_goal_tolerance` default 0.05 against the dataset's 0.2 — under `terminate_on_goal` the tolerance decides the terminal state, hence the label). The kill box lives in `state_space_bounds`, not `env_attributes`: the quadrotors terminate on `state_space`. **One plant fact is not expressible in config** — see below. |
| quadrotor3d | Migrated 2026-07-31. Plant verified against the shipped dataset: **999,954/1,000,000** label reproduction over the FULL eval set (balanced 1024 is 1024/1024). All 46 residuals are ties at the 0.05 tolerance boundary — each within 4.2e-7 of exactly 0.05, against a terminal-state reproduction error of 1.08e-6 median — not config error; perturbing states by ±5e-7 flipped 0/1024. Mismatches caught: `ctrl_freq`/`pyb_freq` (env default 60/240 against the dataset's 100/5000) and the damping below. Two things the description gets wrong: gravity (it says 9.81, `base_aviary.py:77` sets 9.8) and its quaternion `goal_state` (the test runs on the 12-D Euler state). `episode_len_sec` is deliberately 10 s rather than the dataset's 1000 s — never binding either way, longest shipped trajectory 636 states, asserted in the tests. **`eval_states.txt` mixes two angular-velocity frames** (93.69% body, 6.23% world); a replay must handle both. |

All four systems are migrated. The hazard this closed was the one cartpole had
before 2026-07-31: trained at 50 Hz / 10 N against a 100 Hz / 2000 N reference.
To migrate a new system: read its `dataset_description.json`, write
`configs/system/<name>.yaml`, slim both regime files to `extends:` + kills, then
verify by replaying labelled states — see `.claude/log.md` 2026-07-31 for the
cartpole procedure. Verify against the *trajectories*, not only the description:
on three of four systems the decisive parameter was absent from it or wrong.

## The decisive parameter differed on every system

| System | What no default would have given you |
| --- | --- |
| cartpole | `control_bound` 2000 N against the env's `action_scale` default of 10 N — a 200x authority error |
| inverted_pendulum | three explicit-Euler substeps per control step, not one |
| quadrotor2d | `ctrl_freq`/`pyb_freq`, a 1200-step horizon, a 0.2 m goal tolerance, and damping |
| quadrotor3d | `ctrl_freq`/`pyb_freq` and damping |

In each case the signature was the same: a **one-directional** label miss
(shipped-1 / replay-0, with ~none the other way). That asymmetry means a starved
or mis-parameterised plant, not a mislabelled dataset — and it is the fastest
way to tell a wrong plant from a wrong horizon.

## The quadrotors cannot be put back into their datasets' plant by config

Every shipped quadrotor dataset ran at PyBullet's **default** damping
(`linearDamping = angularDamping = 0.04`), because `base_aviary`'s
`changeDynamics` call omitted `physicsClientId` and so targeted client 0 while
the rollout env lived on client 1. The library is fixed, so an env built from
`configs/system/quadrotor2d.yaml` today runs at zero damping — a *different
plant* from the one that produced `reference_success`.

Damping is a post-construction call that `_housekeeping` re-applies on every
reset, so no constructor kwarg can express it. A replay harness has to re-impose
0.04 after each reset deliberately. Measured 2026-07-31, one open-loop step from
each recorded state of three long shipped trajectories: 1.1e-6 deviation at
damping 0.04 (the 6-decimal text-format floor), 1.2e-2 at damping 0, and 3.1e-4
at either 0.039 or 0.041 — a sharp minimum at PyBullet's default.

`dataset_description.json` is not sufficient on its own. The pendulum's records
only `dt: 0.01`, which is the control hold, not the integrator step; the
substep count had to be recovered by replaying shipped trajectory files step by
step and sweeping it. Verify against the trajectories, not just the description.

## Threshold kwargs are silently ignored (2026-08-06)

`x_threshold`, `x_dot_threshold`, `theta_threshold_radians` and
`theta_dot_threshold` passed to `make('cartpole', ...)` are swallowed by
`**kwargs` and never applied. The env keeps its defaults:

| passed | actually used |
| --- | --- |
| `x_threshold=6.0` | **2.4** |
| `x_dot_threshold=5.0` | **20** |
| `theta_threshold_radians=inf` | **pi/2** |
| `theta_dot_threshold=5.0` | **20** |

Nothing warns. Everything else in the same call takes effect
(`action_scale`, `ctrl_freq`, `pyb_freq`, `obs_wrap_angle`,
`x_dot_limit`, `theta_dot_limit`), which is what makes it convincing.

The consequence is severe and one-directional: with `x_threshold` at 2.4, any
start beyond `|x| > 2.4` is out of bounds *before the first step* and dies
immediately. A probe built this way reproduced only 66.5% of the shipped
cartpole labels, missing 130 of 400 in one direction (shipped success, probe
failure) and 4 in the other. Setting the same four values as ATTRIBUTES after
construction gives **400/400**.

This is why `configs/collection/cartpole.yaml` carries them under
`env_attributes:` rather than in `task_config_overrides:`. That file already
knew; the trap is reaching for the constructor instead.

Same lesson shape as the `control_bound: 2000.0` miss recorded above: the
information needed was in `dataset_description.json` all along, and the failure
mode was a one-directional label miss.
