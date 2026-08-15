# dm_control cartpole-swingup vs this repo's cartpole RL

Load when comparing this repo's cartpole training to dm_control's
cartpole-swingup, or when tuning `Cost.SHAPED`, whose structure is copied from
it.

Verified against upstream source
(`dm_control/suite/cartpole.py` and `cartpole.xml`, google-deepmind/dm_control
main, fetched 2026-07-31). dm_control is not installed in the project env.

## The dm_control system

- MuJoCo, `timestep="0.01"`, `integrator="RK4"`, `_DEFAULT_TIME_LIMIT = 10`
  — 1000 steps per episode at 100 Hz.
- Pole: 1 m, mass 0.1. Slider joint range ±1.8 m. Motor `gear="10"`,
  `ctrlrange="-1 1"` — ±10 N of force authority.
- Observation: `bounded_position()` = (cart x, cos θ, sin θ) plus velocities —
  5 channels.
- Swingup initial state: always hanging — `qpos['hinge_1'] = π + .01·randn()`,
  cart at `.01·randn()`, velocities `.01·randn()`. One start state with tiny
  noise, not a region.
- **No early termination of any kind.** Episodes always run the full 1000
  steps; failure scores low rather than ending the episode.

Dense swingup reward — a product of four factors, each with a floor:

```
upright        = (cos θ + 1) / 2
centered       = (1 + tolerance(x, margin=2)) / 2            # floor 0.5
small_control  = (4 + tolerance(u, margin=1, quadratic)) / 5 # floor 0.8
small_velocity = (1 + tolerance(θ̇, margin=5)) / 2           # floor 0.5
reward = upright * centered * small_control * small_velocity
```

Sparse variant (`Balance` with `sparse=True`): 1 per step while cart is in
±0.25 **and** cos θ in (0.995, 1], else 0. Still no termination — it pays
*holding* the region, per step.

## Correspondences (convergent design)

| axis | dm_control swingup | this repo (physical regime) |
| --- | --- | --- |
| force authority | gear 10 × ctrl ±1 = ±10 N | `action_scale` 10 N, action ±1 |
| pole mass | 0.1 | 0.1 (URDF) |
| angle encoding | cos/sin native | `AngleObservation` wrapper |
| obs channels | 5 | 5 |
| hold-the-ball reward | sparse variant, +1/step in region | `Cost.SPARSE` + `terminate_on_goal: False`: `goal_reached` pays `sparse_goal_reward` every step held |
| velocity handling | shaped term, margin 5, never terminal | `theta_dot`/`x_dot` termination dropped; shaped `settled` term |

The last row was measured independently here: terminating on velocity moved
61% of cartpole crashes to a bound the URDF does not declare, and dropping it
merely relocated failures (7% → 8% LQR success, x_dot crashes became x
crashes). dm_control never terminated on it in the first place.

## Structural differences and why

- **Termination.** dm_control has none, so the dense-reward/termination
  inversion this repo measured (+4.46 return for tighter terminal error vs
  −181.2 for the termination it triggers; SAC hovered outside the ball at
  0.000 success) cannot arise there. This repo keeps termination because the
  downstream flow-matching model consumes *terminal states* — the episode has
  to end somewhere meaningful. That single constraint forces the sparse/shaped
  reward design, the `|oob| > |step|·H` invariant, and γ = 0.995 sized to the
  400-step horizon.
- **Start distribution.** Swingup always starts hanging; this repo samples the
  full box (x ±6, ẋ ±5, θ ±π, θ̇ ±5) because the product is an ROA-labelled
  dataset over a state grid, not a single-task policy. The initial-state
  curriculum (start 0.005 of range, step 0.15, threshold 0.5) exists because
  sparse reward over the full box produced zero goal entries in 300 episodes.
- **Success metric.** dm_control has no success label — return over 1000 steps
  is the benchmark number. This repo scores terminal-state goal-ball
  membership in `eval_policy` and invariant-ellipsoid membership for datasets,
  which disagree by construction.
- **Reward margins.** dm_control's tolerance margins are in raw units per
  channel (2 m, 5 rad/s) and its factors have floors (0.5, 0.8) so the product
  never collapses. `Cost.SHAPED` instead normalises the state error by the
  state-space span and uses single margins — raw-unit margins failed here
  (median shaped reward 0.0000 over 16k random steps before normalising) — has
  no floor on the proximity term, and adds `sparse_goal_reward` /
  `sparse_oob_reward` on top because termination still exists.
- **Rail.** ±1.8 m there, ±6 m workspace here (URDF rail is ±15); with equal
  force authority the cart has far more room, and recovering from x ≈ 6
  is a longer task than anything swingup poses.
- **Sim.** MuJoCo RK4 at 100 Hz control vs PyBullet semi-implicit Euler at
  50 Hz control (`pyb_freq` = `ctrl_freq` = 50). Horizon 1000 steps vs 400
  (8 s sized from measured worst-case LQR steps-to-goal, 249).
- **Task factoring.** dm_control splits balance / swingup / swingup_sparse
  into separate tasks; this repo's single stabilization task from θ ±π with
  θ bounds at ∞ subsumes swing-up and balance in one.

## Status of the shaped cost

As of 2026-07-31, `Cost.SHAPED` is implemented in
`safe_control_gym/envs/benchmark_env.py` (`_tolerance`, `_shaped_reward`) but
no `configs/sb3/*.yaml` trains with it yet — `cartpole_stabilization_sac.yaml`
still sets `cost: sparse`. The comment trail in `configs/physical/cartpole.yaml`
records the dm_control-derived reasoning for dropping velocity termination.

Related: `.claude/docs/architecture.md` (env step API, RL stacks),
`.claude/docs/datasets.md` (invariant-set success criterion),
`.claude/docs/glossary.md` (terminated/truncated, U_SAT).
