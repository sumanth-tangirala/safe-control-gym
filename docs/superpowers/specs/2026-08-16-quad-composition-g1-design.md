# Spec: Quadrotor Controller Composition — Is a Practical `G1` Naturally Non-Subsumed?

Status: design, awaiting review.
Systems: quadrotor-2D (controller 2 = `safe_explorer_ppo`), quadrotor-3D (controller 2 = LQR).

## Goal

Show that for quadrotors, a handoff region `G1` chosen on **purely practical
grounds** — what a recovery controller can actually guarantee — naturally
satisfies

```
G1 ∩ RoA2 ≠ ∅        and        G1 ⊄ RoA2
```

where `RoA2` is the region of attraction of the controller that already collected
the system's dataset. The non-subsumption fraction `1 − P(controller 2 succeeds | handoff ∈ G1)`
is the **primary measured result**. Composed ROA enlargement is a secondary result.

## Motivation

The existing `pendulum/rl_to_lqr` composition dataset establishes the two-stage
structure, but its `G1` is contrived: the V4 region is a union of two ellipses
deliberately centred **off** the upright equilibrium, on the arms of the LQR ROA,
sized so that 15.5% of it falls outside that ROA. The interesting property —
that a handoff can land outside `RoA2` and fail — was *designed in*, not
discovered. That weakens any downstream argument that sequential composition
requires reasoning about per-stage ROAs.

Quadrotors can make the property emerge instead of being assumed, because of an
asymmetry the pendulum does not have:

- A recovery ("flip") controller has authority over **attitude**. Its goal region
  is therefore naturally attitude-defined: upright, rotating slowly.
- `RoA2` additionally depends on **position and translational velocity**.
- Arresting translational velocity requires tilting, which directly opposes the
  attitude objective. So the flip controller cannot fix the variables that decide
  `RoA2` membership.

Two sets defined over different variables have no reason for one to contain the
other. That is the mechanism to demonstrate.

## Prior state

Existing on the shared drive under `data_trajectories/deterministic/`:

| dataset | controller | states | success |
|---|---|---|---|
| `quadrotor2D_rl` | `safe_explorer_ppo` | 489,789 (grid) | 8.02% |
| `quadrotor3D_lqr` | LQR | 1,000,000 (eval set) | 21.97% |
| `pendulum/{lqr,rl,rl-weak,rl_to_lqr,rl-weak_to_lqr}` | — | 49,770 | — |

Pretrained checkpoints for both systems exist under `examples/rl/models/{ppo,sac,safe_explorer_ppo}/`.
`quadrotor2D_rl` was generated with `safe_explorer_ppo_model_quadrotor_2D_stab.pt`.

## Evidence gathered during design

All numbers below were measured from the existing `eval_states.txt` files and the
`cf2x.urdf` parameters. They motivate the design and are the baseline the work
will be compared against.

### Failure structure

Both controllers fail predominantly on **velocity** termination bounds, not on the
position box, and both fail overwhelmingly at high tilt:

```
quad3D (LQR):  67.3% velocity-only, 30.0% position-only
               success vs tilt: 0.572 (<15°) → 0.064 (>120°)
               39.5% of failures start above 90° tilt
quad2D (RL):   81.4% velocity-only, 16.0% position-only
               success vs |θ|: 0.475 (9.1°) → 0.000 (≥90°)
               ZERO successes above 90°, out of ~245,000 states
```

### Actuator authority — the two baselines do NOT share it

The physical Crazyflie 2.x gives total thrust 0.11265 – 0.59337 N against a
0.26487 N weight (TWR 0.425 – 2.240) and, on a 0.02807 m lever, α_max = 482 rad/s².
But the two existing datasets were generated with **different** action-space
configurations:

| dataset | config | TWR max | α_max |
|---|---|---|---|
| `quadrotor2D_rl` | `normalized_rl_action_space=True`, `norm_act_scale=0.1` | **1.100** | **53.1 rad/s²** |
| `quadrotor3D_lqr` | `normalized_rl_action_space=False` (physical) | 2.240 | 482 rad/s² |

`norm_act_scale=0.1` is the `Quadrotor.__init__` default and is never overridden
in `generate_quadrotor_2d_trajectories_rl.py:588`. The denormalisation is
`thrust = (1 + norm_act_scale·a)·hover_thrust`, so controller 2 commands only
±10% around hover.

**This explains the 2D tilt cliff.** Controller 2's zero successes above 90° are
not a learned limitation: at TWR 1.10 it can barely hold altitude, and at
α = 53 rad/s² a 180° rotation needs ≥0.54 s. It physically cannot right itself.
It also partly explains 21.97% vs 8.02% between the two baselines.

### Flip feasibility budget

With thrust scheduled optimally (max while `cos θ > 0`, min while `cos θ < 0`,
since thrust points downward when inverted), vertical acceleration is negative
throughout `|θ| ∈ (63.5°, 180°)` regardless of command. Traversing that arc at the
maximum permitted rate gives the least possible loss of vertical velocity:

```
Δż_min(θ₀) = (1/ω_max) ∫ from θ_target to θ₀ of  z̈*(θ) dθ
z̈*(θ) = (T_max/m)·cos θ − g   for θ < 90°
         (T_min/m)·cos θ − g   for θ ≥ 90°
```

Evaluated with `θ_target = 10°` (the tightest attitude-only `G1` the 2D grid can
express) for 2D, and `30°` for 3D:

| | actuator | ω_max | ż band | max recoverable tilt (ż = +bound / 0 / −½bound) | ceiling |
|---|---|---|---|---|---|
| **2D as specified (D6)** | restricted, TWR 1.10 | 8 rad/s | 2.0 m/s | **138° / 107° / 85°** | 8.02% → **54.0%** |
| 2D if given physical | TWR 2.24 | 8 rad/s | 2.0 m/s | 180° / 150° / 133° | 8.02% → 76.6% |
| 3D | physical, TWR 2.24 | 24 rad/s | 6.0 m/s | 180° / 180° / 180° | 21.97% → 91.2% |

Under D6, 52.4% of failing states above 10° are budget-feasible. 2D cannot
recover from full inversion at either authority level without starting with
upward velocity; 3D can from anywhere. These ceilings ignore the horizontal
velocity bound, the altitude box, and the rate required at handoff, so they are
strict upper bounds.

### Non-subsumption, attitude-only `G1`

`G1 = {tilt < tilt_c, |ω| < w_c}`. Entry is `frac outside = 1 − P(controller 2 succeeds | G1)`:

| quad2D | w_c=1 | 2 | 3 | 5 | | quad3D | w_c=2 | 4 | 6 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| tilt<10° | 0.417 | 0.420 | 0.440 | 0.490 | | tilt<10° | 0.013 | 0.142 | 0.256 | 0.293 |
| tilt<23° | 0.475 | 0.474 | 0.499 | 0.547 | | tilt<20° | 0.021 | 0.176 | 0.265 | 0.303 |
| tilt<41° | 0.641 | 0.643 | 0.661 | 0.692 | | tilt<30° | 0.049 | 0.244 | 0.310 | 0.350 |

The separating variables inside `G1` are exactly the ones the flip controller does
not govern:

```
quad2D, inside {tilt<10°, |ω|<1}:  succ 0.815 → 0.203 across |v| quintiles
                                   succ 0.672 → 0.459 across |x| quintiles
quad3D, inside {tilt<30°, |ω|<10}: succ 0.873 → 0.406 across |v| quintiles
                                   succ 0.821 → 0.375 across r_xy quintiles
```

**2D is the strong case:** no attitude-only `G1` expressible on the grid is
subsumed; the floor is 41.7%. **3D is the tunable case:** at `|ω|<2` it is 1.3%
outside (effectively subsumed), at `|ω|<10` it is 35%. The 3D argument therefore
depends on showing that tight `w_c` is *not achievable* by the flip maneuver, not
merely undesirable.

### Sampling-grid limitation (2D)

`quadrotor2D_rl` eval states are a full Cartesian grid, not random samples:

```
x: 7 (Δ0.28)   z: 7 (Δ0.23)   θ: 12 (Δ0.55 rad = 31.5°)
ẋ: 7 (Δ0.28)   ż: 7 (Δ0.28)   θ̇: 17 (Δ0.9)         →  7·7·12·7·7·17 ≈ 489.6k
```

`|θ|` takes only 12 values, the smallest being 9.08°. Consequences: the tilt axis
of every table above has one sample per bin; `G1` cannot be resolved below 9.08°;
and any `G1` calibration must come from a dedicated rollout experiment rather than
from this grid.

## Decisions

### D1 — `G1`'s form is fixed a priori; its parameters come from controller capability

This is the single most important decision in this spec, and it must not become
circular. Two separate things:

- **Form** — chosen a priori and never revisited: `G1 = {|tilt| < tilt_c, |ω| < w_c}`,
  attitude only. This is the non-contrived choice because attitude is the only
  thing a recovery controller has authority over. No position or velocity term
  ever enters `G1`.
- **Parameters** `(tilt_c, w_c)` — set by what controller 1 can *reliably deliver*,
  measured in a calibration pass, with **no reference to `RoA2`** at any point.

Procedure, in this order:

1. Train controller 1 against a nominal attitude target `G_nom` (D2, D6).
2. Calibration pass: run it from held-out initial states and measure the
   distribution of attitude states at which it stabilises. Set `(tilt_c, w_c)` to a
   high quantile of that distribution — the tightest region it hits reliably.
3. Freeze `G1`. Generate the `*_flip/` and `*_flip_to_*/` datasets with handoff
   latching on first entry.
4. Only then measure `P(ctrl2_success | flip_success)`.

Step 2 never looks at controller 2, so the relationship between `G1` and `RoA2` in
step 4 is a discovery. Recording `(tilt_c, w_c)` and the calibration distribution in
`dataset_description.json` is what makes that auditable.

### D2 — Controller 1 reward is attitude-only

Controller 1 is trained to reach `{|θ| small, |θ̇| small}` and nothing else. It
gets **no** position or translational-velocity term. Adding one would pull `G1`
back toward `RoA2` and reintroduce exactly the contrivance this work exists to
avoid. The controller optimises attitude because attitude is what it has
authority over.

### D3 — Composition is sequential with a latch on first `G1` entry

Controller 1 runs from the initial state until the first state that enters `G1`;
control then latches to controller 2 permanently. Labels are per-initial-state, so
a latched (non-memoryless) law is well-posed. Matches `pendulum/rl_to_lqr`.

### D4 — System and limits are unchanged

No change to state bounds, termination thresholds, success radius, or control
frequency on either system. Everything that makes the composed result comparable
to the existing baseline depends on the generating process differing only in the
control law.

```
quad2D:  x ±1.0, z ∈ [0.1,1.5], |v| ≤ 1.0, |θ̇| ≤ 8,  goal radius 0.2, 100 Hz
         normalized_rl_action_space=True, norm_act_scale=0.1  (TWR 1.10, α 53.1)
quad3D:  x,y ±1.8, z ∈ [0.1,3.0], |v| ≤ 3.0, |ω| ≤ 24, goal radius 0.05, 100 Hz
         normalized_rl_action_space=False                     (TWR 2.24, α 482)
```

Actuator authority is part of "the limits" and is per-system, inherited from
whichever configuration generated that system's baseline.

### D5 — Controller 2 is the existing checkpoint

quad2D: `safe_explorer_ppo_model_quadrotor_2D_stab.pt`, the exact model that
generated `quadrotor2D_rl`. quad3D: LQR as configured in
`generate_quadrotor_3d_trajectories.py`. No retraining, so `RoA2` is the already
characterised set.

### D6 — Controller 1 is a learned policy on controller 2's action space

- **`normalized_rl_action_space=True`, `norm_act_scale=0.1` — identical to
  controller 2.** Controller 1 gets TWR 1.10 and α_max 53.1 rad/s², exactly what
  generated `quadrotor2D_rl`. Giving it the physical range would nearly double the
  composed ROA (ceiling 76.6% vs 54.0%) but would make the gain attributable to
  actuation rather than control, which is not a claim this work wants to defend.
- Note the tradeoff direction: a weaker controller 1 delivers *looser* attitude at
  handoff, hence a looser `G1`, hence a **higher** non-subsumption fraction. The
  restriction strengthens the primary result while shrinking the secondary one.
- Any future variant that changes actuator authority must regenerate controller 2's
  baseline under the same authority, or the comparison is confounded.
- Observation replaces the stored angle representation, whose discontinuity sits
  exactly on the region of interest: 2D feeds `(cos θ, sin θ)` (7-dim); 3D feeds
  the rotation matrix (18-dim) rather than the `qw ≥ 0` canonicalised quaternion.
- Reward: potential-based shaping on attitude distance to the **nominal** target
  `G_nom` (D1 step 1), terminal bonus on entry, terminal penalty on out-of-bounds,
  no control cost. `G_nom` is a training-time target only; the `G1` that triggers
  handoff is fixed later by the calibration pass and may be looser or tighter.
- Initial states uniform over the closed state space; training-only timeout ~2 s.
- SAC first; `safe_explorer_ppo` is the fallback, being the proven path in this repo.

### D7 — Dataset layout

Mirrors `pendulum/`, per system:

```
quadrotor2D_rl/            existing, unchanged — controller 2 alone = RoA2 baseline
quadrotor2D_flip/          NEW — controller 1 alone, truncated at first G1 entry
quadrotor2D_flip_to_rl/    NEW — full composite, EVALUATION ONLY
```

All three share the same initial states (the existing ~489.6k grid) so comparisons
are paired. 3D follows with `quadrotor3D_flip/` and `quadrotor3D_flip_to_lqr/`.

### D8 — Formats

- `eval_states.txt`: `init(n), final(n), flip_success, ctrl2_success`
- `roa_labels.txt`: one line per trajectory — `init(n), flip_success, ctrl2_success`
- `handoff_states.txt`: `init(n), handoff_state(n)` or a `-1` sentinel row when no
  handoff fired. This is the raw material for fitting `G1` and for measuring
  non-subsumption on real handoffs rather than on grid states.
- `dataset_description.json` carrying the `G1` definition, both label semantics,
  and the note that `(flip_success=0, ctrl2_success=1)` cannot occur.

## Metrics

Primary:

```
non_subsumption = 1 − P(ctrl2_success = 1 | flip_success = 1)
```

measured on real handoff states, with a bootstrap interval. Reported per system.
The claim is that it is bounded away from both 0 and 1.

Secondary:

- Composed success rate vs. baseline on shared initial states (paired).
- `|Flip⁻¹(G1) \ RoA2|` — states the composition wins that controller 2 alone
  cannot reach.
- Fraction of flip attempts terminating on each bound, to show whether the limits
  or the controller cap the result.

## Risks

1. **3D `G1` may be subsumable.** At `|ω|<2`, only 1.3% of `G1` lies outside
   `RoA2`. The 3D argument holds only if the flip maneuver cannot deliver rates
   that tight. If it can, 3D contributes an enlarged composed ROA but not the
   non-subsumption result, and the claim rests on 2D.
2. **2D floor unconfirmed below 9.08° tilt.** The strong 2D claim needs the
   dedicated rollout experiment at finer attitude resolution.
3. **Handoff states are not uniform in `G1`.** All non-subsumption figures above
   come from grid initial states. Real handoffs cluster, so the true fraction may
   differ in either direction; only the composition dataset settles it.
4. **Controller 1 may not train.** Under D6's restricted actuator, recovery is
   budget-infeasible above ~107° at `ż = 0` and above ~85° at `ż = −0.5`, so
   roughly half the failing state space is genuinely unrecoverable and the reward
   is sparse there. Shaping must not be mistaken for progress in that region — the
   validation below checks measured `Δż` against the analytic budget precisely so
   an unreachable region is not misread as a training failure.

## Validation

- Flip controller reproduces the analytic budget: measured `Δż` over the negative
  acceleration arc within ~15% of `∫ z̈* dθ / ω_max`.
- Re-running controller 2 alone on the shared grid reproduces `quadrotor2D_rl`'s
  8.02% to within sampling noise — confirms the harness matches the original.
- `(flip_success=0, ctrl2_success=1)` count is exactly zero.
- Trajectory in `*_flip/` is a prefix of the corresponding `*_flip_to_*/`
  trajectory up to and including the handoff index.
- `(tilt_c, w_c)` were fixed before any composition rollout ran — verifiable from
  the calibration record in `dataset_description.json`.

## Out of scope

- Regenerating `quadrotor3D_lqr` or `quadrotor2D_rl` with the composed policy.
  The composition datasets are evaluation-only, per the pendulum precedent.
- Retraining controller 2.
- Any change to state bounds or success criteria.
- Downstream ROA/final-state prediction models consuming these datasets.
