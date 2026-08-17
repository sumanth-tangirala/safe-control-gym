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

### Caveat: the magnitude depends on `G1`, so never quote it alone

Non-subsumption is substantially a measure of **how loose `G1` is**. Sweeping `G1` and scoring
each on what actually matters (n=400 paired, selected controller):

| `G1` | S1 | **S1→S2** | S1→F2 | non-subsumption |
|---|---|---|---|---|
| `G_NOM_3D` (10°, 4) | 40.8% | 33.2% | 7.5% | **0.184** |
| exit q0.50 (34.7°, 7.6) | 49.2% | 28.5% | 20.8% | 0.421 |
| exit q0.75 (93°, 19) | 68.0% | 13.8% | 54.2% | 0.798 |
| exit q0.90 — **spec D1** (123°, 28) | 86.0% | 9.0% | 77.0% | **0.895** |
| tightened 20°/6 | 45.5% | 33.0% | 12.5% | 0.275 |
| tightened 15°/5 | 44.2% | **34.8%** | 9.5% | 0.215 |

So 0.195 is a property of *(system, `G1`)*, not of the system alone. The qualitative claim
survives — every sensible attitude-only region lands in 0.18–0.28, and `G_NOM_3D` was fixed a
priori rather than tuned — but **always report the (`G1`, non-subsumption, S1→S2) triple
together.** A loose `G1` yields a more dramatic 0.895 while the composition collapses to 9%
end-to-end; that is a weaker result presented as a stronger one.

### Spec D1's calibration procedure should be withdrawn

D1 sets `G1` from a high quantile of controller 1's exit attitudes. Tested fairly — calibrated
from the *selected* controller, not the weak geometric one — it yields `tilt_c = 123.45°`,
`w_c = 28.06` rad/s (which exceeds the ±24 rate bound, admitting any rate).

The exit quantiles show why: p50 34.7°, p75 93.2°, p90 123.4°. The policy genuinely improves
attitude (median best-attitude 34.7° vs ~90° uncontrolled), but a **high quantile of a
distribution that includes the ~57% of rollouts that fail is dominated by failures, not by
deliveries.** The procedure is monotonically harmful in this system: no quantile beats a plain
a-priori region. If retained at all, it must fit on successful rollouts only.

### Replicated on an independent seed

The headline initially rested on one seed. Re-drawn:

| seed | non-subsumption | 95% CI | n |
|---|---|---|---|
| 7 | 0.1950 | [0.1690, 0.2239] | 800 |
| 20260817 | 0.1873 | [0.1618, 0.2158] | 801 |

Each sits inside the other's interval. The **mechanism** replicates too — on both seeds
independently, every G1-free variable has a larger spread than *both* G1-constrained ones
(seed 7: min free 0.200 > max constrained 0.188; seed 2: 0.232 > 0.107). The explanation is
not a seed artifact.

---

## Secondary result: the composition more than doubles the ROA

Paired over 40,000 identical initial states (the shipped `quadrotor3D_lqr` eval states):

```
baseline (LQR alone)   21.20%
composed               46.36%       +25.16 points,  2.19x
won 11116 | lost 1051 | both 7429 | neither 20404        (10.6 : 1)
```

Non-subsumption implied here is `1 - 18545/22432 = 0.173`, consistent with the two
uniform-SO(3) measurements on a different state distribution.

Datasets: `quadrotor3D_lqr_regenerated`, `quadrotor3D_flip`, `quadrotor3D_flip_to_lqr` under
`data_trajectories/deterministic/`, 40k states each. Full 1M extrapolates to ~10 h.

---

## A guard does not help — and the reason is structural

The composition always runs controller 1 first, so 1051 of the above are states LQR alone
would have solved. The obvious fix is the supervisory guard the design called for: run LQR
where it works, flip elsewhere. It was built, fitted and measured. It makes things **worse**.

```
baseline                 8.00%      (uniform SO(3) sample, n=8000)
unguarded composition   31.46%
guarded composition     31.20%      <- worse
oracle guard            33.45%      <- upper bound for ANY guard
```

The guard is a competent classifier — held-out accuracy 0.839 against a 0.785 majority floor,
precision 0.721, recall 0.412 — and still recovers only 2 of 159 losses while causing 23 new
ones, capturing **−13.2%** of the achievable gain. A threshold sweep across two model families
never turned positive. Cross-checked on in-distribution held-out rows (27/225 recovered,
167/2198 wrongly regressed, net −140), so this is not distribution shift.

**Why:** the population a guard must protect (2.6% recoverable) is outnumbered **~10.6:1** by
the population it can damage (27.8% genuinely needing the flip). At that ratio a guard needs
near-perfect precision *and* recall to break even, and no scalar attitude/velocity feature
separates the two populations that sharply.

Note the trap this closes: a guard with imperfect *recall* declines to intervene on states
that needed the flip, and each is a lost success. It is not a free safety net. The `guard=`
parameter remains as opt-in infrastructure (default `None`, zero behaviour change) for a
better-featured attempt — e.g. one using a learned ROA estimate rather than hand-picked
scalars — but the 1051 losses are a genuine cost of composition, not a bug to engineer away.

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
