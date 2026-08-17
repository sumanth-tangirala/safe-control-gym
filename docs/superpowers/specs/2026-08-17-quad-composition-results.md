# Quadrotor Controller Composition — Overnight Results

Branch: `quad-composition-g1`. Written 2026-08-17 ~02:00, while training continues.
Spec: `docs/superpowers/specs/2026-08-16-quad-composition-g1-design.md`.
Plan: `docs/superpowers/plans/2026-08-16-quad2d-composition.md`.

---

## The headline

**The project's central claim holds, measured in 3D.**

```
non_subsumption = 1 - P(ctrl2_success | flip_success) = 0.195
95% Wilson CI [0.169, 0.224]      n = 800 handoffs / 2000 initial states
```

A handoff region `G1` chosen on purely practical grounds — attitude only, because that is
all a recovery controller has authority over — is **neither subsumed by nor disjoint from**
the second controller's region of attraction. Unlike the pendulum's `G1`, this was not
constructed to have that property.

Roughly uniform across initial tilt, so it is not an artifact of extreme attitudes:

| initial tilt | 0–30° | 30–60° | 60–90° | 90–120° | 120–150° | 150–180° |
|---|---|---|---|---|---|---|
| non-subsumption | 0.173 | 0.262 | 0.187 | 0.161 | 0.180 | 0.139 |
| n handoffs | 75 | 195 | 209 | 174 | 111 | 36 |

All intervals overlap heavily.

### Why it happens — the mechanism, not just the number

Success rate across quintiles of each variable **at the handoff instant**:

| variable | spread | quintiles (low → high) | |
|---|---|---|---|
| tilt | 0.156 | 0.719 → 0.831 → 0.812 → 0.875 → 0.787 | **constrained by G1**, non-monotone |
| \|ω\| | 0.188 | 0.812 → 0.850 → 0.775 → 0.887 → 0.700 | **constrained by G1**, non-monotone |
| translational speed | **0.306** | 0.944 → 0.875 → 0.838 → 0.731 → 0.637 | free, monotone |
| horizontal distance | **0.294** | 0.887 → 0.894 → 0.844 → 0.800 → 0.600 | free, monotone |
| altitude error | 0.200 | 0.831 → 0.875 → 0.825 → 0.819 → 0.675 | free, monotone |

The variables `G1` does **not** constrain cleanly predict handoff failure. The ones it
**does** constrain do not, because `G1` has already removed their variation. That is the
predicted mechanism, confirmed: an attitude-only region cannot be contained in a region of
attraction that also depends on position and translational velocity.

`n_handoffs_at_index_zero = 0` — every handoff was real. The trivial-handoff concern the 2D
design flagged and never resolved does not arise, because `G_NOM_3D` is tight and uniform
SO(3) sampling essentially never starts inside it.

Reproduce: `analyze_quad3d_composition.py`; result in `results/quad3d_composition.json`.

---

## A methodological finding worth carrying into the paper

**Ranking controllers on flip rate selects the worse composition.** Observed in three
independent paired comparisons:

| controller | S1 (reached G1) | **S1→S2** | S1→F2 |
|---|---|---|---|
| SAC (selected) | 42.1% | **33.7%** | 8.4% |
| hand-coded geometric | **50.9%** | 18.9% | 32.0% |
| SAC `s0_tfull` @150k | **46.3%** | 28.0% | 18.3% |

The geometric controller flips *more often* and completes *far less often*: 63% of its
handoffs fail, against SAC's 20%. It reaches `G1` at speed and off-centre, which satisfies an
attitude-only region while leaving LQR nothing to work with.

This matters practically because `ep_return` — the number anyone watching a training curve
would use — tracks `G1` entry, i.e. S1. It is the misleading metric. Select on S1→S2.

McNemar, SAC vs geometric on S1→S2: 166 SAC-only vs 33 geometric-only, χ² = 87.6.

---

## The 2D result: an impossibility, not a failure

**quadrotor-2D cannot be flipped from inversion by any controller, as configured.** Three
independent lines agree:

1. **Analytic budget.** Rotating 180° takes ≥ π/8 = 0.39 s at `|θ̇| ≤ 8`. Through a half-turn
   `⟨cos θ⟩ = 0`, so the drone free-falls ~3.8 m/s against a band only 2.0 m/s wide.
2. **The RL baseline.** `safe_explorer_ppo` scores **zero** successes above 90° tilt — 0 of
   ~245,000 states.
3. **A hand-coded bang-bang flip**, which is not a learning problem: 0.593 / 0.194 / 0.061 at
   0–30 / 30–60 / 60–90°, and **0.000** above 90°.

Ablating one limit at a time (bang-bang success from \|θ\| ≥ 90°):

```
BASELINE (quadrotor2D_rl exactly)          0.000
rate cap 8 -> 16 / 24 / 40 rad/s           0.000 / 0.000 / 0.000
|v| 1.0 -> 3.0 m/s                         0.013
actuator TWR 1.10 -> 2.00                  0.000
altitude box [0.1,1.5] -> [0.1,3.0]        0.000
rate 24 AND |v| 3.0 (both 3D values)       0.150   <-- only the conjunction works
```

The limits are **mutually contradictory** for a flip: the rate cap forces the maneuver to be
slow, the velocity band requires it to be fast. Relaxing either alone leaves the other
binding. 3D flips because it was calibrated with both.

Relaxed 2D (rate 24, `|v|` 3.0, everything else unchanged) was tried and reaches only ~0.144.
A larger position box does not help (0.156); rate 40 or `|v|` 5.0 make it *worse*.

---

## What was built

| artifact | what it is |
|---|---|
| `quad_composition/{g1,rollout2d,flip_env2d,budget}.py` | 2D core: handoff region, rollout core, attitude-only training env, analytic flip budget |
| `quad_composition/{rollout3d,flip_env3d,geometric_flip3d}.py` | 3D core plus a hand-coded geometric flip controller |
| `train_quadrotor_{2d,3d}_flip.py` | SAC training |
| `calibrate_quad{2d,3d}_g1.py` | `G1` calibration from measured exits |
| `generate_quadrotor_2d_composition.py` | flip / composite / baseline datasets |
| `analyze_quad{2d,3d}_composition.py` | non-subsumption + paired composed gain |
| `visualize_quad{2d,3d}_composition.py` | the four-category videos |
| `compare_ctrl1_3d.py`, `verify_quad3d_composition_videos.py` | paired controller comparison; frame-decoding video verifier |

Selected controller: `models/quad3d_ctrl1_selected.pt` (`quad3d_s5_tfull` @ 75k steps).
Videos: `rollout_visualizations/quad3d_composition/` — 12 clips, 4 categories × 3, all
genuine inversions (initial tilt 138–176°), the two sets disjoint by construction.
Test suite: 232 passing.

---

## Bugs found that would have silently corrupted the result

1. **Gimbal-folded pitch (Critical).** `p.getEulerFromQuaternion` folds pitch to ±π/2, so a
   drone at true pitch 3.0 rad read back as 0.14 rad — *upright*. Every attitude decision was
   computed on the wrong quantity, and the reward would have trained the policy to **stay
   inverted**. Fixed by deriving attitude from the rotation matrix. Verified: this survived
   eight tasks and eight clean reviews because every attitude test used synthetic state vectors.
2. **Unlearnable observation.** Controller 1 could not distinguish upright from inverted —
   the same folding, in its observation. Spec D6 mandated `(cos θ, sin θ)`; the plan had lost
   it. 3D now uses the rotation matrix (18-dim).
3. **Out-of-bounds reward leak.** `info['out_of_bounds']` depends on position and velocity,
   so penalising it made the reward depend on exactly the variables `G1` must not — biasing
   `G1` toward `RoA2`, the contrivance the project exists to avoid.
4. **Open state space.** Relaxing termination bounds left *sampling* bounds pinned, so
   training drew from a smaller box than it could reach. Bounds are now derived from
   `env.state_space`.
5. **No intermediate checkpoints.** `sac.yaml` sets `log_interval`, `save_interval` and
   `eval_interval` all to 0, so a run killed before `max_env_steps` left nothing at all.

---

## Open items

- **Training still running** — 16 seeds to 1M steps, ~24% at time of writing, ETA ~08:15.
  One re-selection pass already found no significant improvement (χ² = 1.36, p = 0.24).
- **Calibrated `G1` is worse than the nominal one.** The spec's D1 procedure (fit `G1` to
  measured exits) produced `tilt_c = 120.9°`, which fires the handoff while still badly tilted
  and drops S1→S2 from 18.5% to 8.0%. This is a real finding against the spec's approach.
- **3D datasets in progress** (spec D7): flip / composite / regenerated-baseline at 40k states.
- **Environment:** torch here is built against NumPy 1.x but runs under 2.x. Patched for SAC
  following the existing `safe_ppo_utils` precedent, but `ddpg_utils.py:174` and likely
  PPO/RARL remain broken. The honest fix is aligning the conda env.
- **2D composition datasets** were never generated; 2D is an impossibility result now.
