# quad2d stochastic collection under an altitude-gated one-sided sidewind

Date: 2026-08-17
System: quadrotor 2D, `safe_explorer_ppo`, stabilization
Status: draw law: per-rollout sinusoid; ladder chosen 2026-08-18 (0.05/0.08/0.13/0.30); collection pending

## What this is

A disturbance **corridor**: a band of disturbed air at a fixed altitude that a
trajectory has to cross to reach the goal. The force is one-sided, always +x,
and its magnitude is gated by altitude so that the band is localised and the
goal sits in calm air above it.

Two things here are new to this repo. It is the first **biased** disturbance --
every existing family is zero-mean and symmetric, `uniform(-f, +f)` or
`Normal(0, sigma)`. And it is the first whose scale is a function of **env
state** rather than of the commanded signal; `SignalDependentNoise` keys off
`|target|`, which is a different mechanism (see Rejected alternatives).

The object being measured is unchanged: a stochastic ROA, published as
`p_success` over K trials at fixed eval start states, plus a train split of
labelled trajectories.

## The noise model

```
sigma(z) = F_max * exp(-0.5 * ((z - 0.55) / 0.12)**2)     # newtons, the draw bound
F_x      ~ U(0, sigma(z_t))                                # fresh draw, every control step
F_z      = 0
```

`sigma(z)` is the **upper bound of the draw**, not a scale on a fixed
distribution -- the support itself moves with altitude. Mean load is
`sigma(z)/2`, standard deviation `sigma(z)/(2*sqrt(3))`. The two are locked at
a ratio of `1/sqrt(3)`, so a single-parameter `U(0, sigma)` cannot set the push
and the variability independently. Accepted: the corridor's job is to be a
hazard of a given strength, not to disentangle bias from spread.

### The envelope is a strategy

`sigma(z)` is not hard-coded. The disturbance class resolves it from a named
profile registry, `ALTITUDE_PROFILES`, with the Gaussian bump above registered
as `'gaussian'` -- the only entry for now. A future profile (a different shape,
a different placement) is a new registry entry plus its parameters in the
config; nothing else changes. Likewise the draw law is isolated in a single
`_draw(bound)` method so an alternative law (see the sine entry under Rejected
alternatives) is a subclass override, not a rewrite. Every
`dataset_description.json` records the profile name beside its parameters,
because two datasets with the same `f_max` and different profiles are not
comparable.

Injected through the existing `dynamics` channel, so: a world-frame force at
the COM, producing **no torque**, re-applied on every one of the 50 PyBullet
substeps because PyBullet clears external forces per `stepSimulation`
(`base_aviary.py:280-288`). Unmatched -- `range(B_d)` is not contained in
`range(G(x))`, and the vehicle can only oppose a lateral force by tilting first.

For `TWO_D` the disturbance vector is 2-D and maps to `[Fx, 0, Fz]`
(`quadrotor.py:465`), so a `[1, 0]` mask restricts it to +x.

### Band geometry

| quantity | value |
| --- | --- |
| centre | z = 0.55 |
| width parameter | 0.12 m |
| core, >= 60% of peak | z in [0.43, 0.67], 0.24 m |
| extent, > 1% of peak | z in [0.186, 0.914], 0.728 m |
| sigma at the goal (z = 1.0) | 0.09% of peak -- effectively zero |
| goal ball altitude span | z in [0.8, 1.2] |

"Narrow" is relative. At the 1% threshold the band spans more than half the
usable altitude range; it is a broad gradient with a concentrated core.

## Why the goal must be calm

This is the constraint that drove the whole design, and it is not obvious.

`goal_reached` is `||state - X_GOAL|| < 0.2` over the **full 6-D state**
(`quadrotor.py:920`), and for `TWO_D` stabilization `X_GOAL = [0, 0, 1, 0, 0, 0]`
(`quadrotor.py:294-297`). **Theta = 0 is inside the success test.** The
out-of-bounds mask exempts theta as periodic; the goal test does not.

Holding station against a mean lateral load requires a standing tilt
`theta_ss = atan((sigma/2) / mg)`, with `mg = 0.027 * 9.8 = 0.2646 N`. That tilt
is charged against the same 0.2 ball that defines success:

| mean load | theta_ss | share of the 0.2 ball |
| --- | --- | --- |
| 0.005 N | 0.019 rad | 9% |
| 0.010 N | 0.038 rad | 19% |
| 0.025 N | 0.094 rad | 47% |

So a profile with appreciable `sigma` at the goal cannot be satisfied: the tilt
that cancels the force is scored against the vehicle. Placing the corridor below
the goal removes the conflict -- tilt during transit is free, because only the
state at the moment of the goal test is measured.

## Rejected alternatives

**Profile A, a bump spanning the full altitude range** (zeros at z = 0.1 and
1.5, peak at 0.8). Rejected on measurement: the goal ball spans z in [0.8, 1.2]
and A peaks at 0.8, so the peak sits *inside* the success region -- 81% of peak
force at z = 1.0. Keeping the standing tilt under a quarter of the ball caps
`F_max` at 0.033 N, an order of magnitude below the 0.070-0.200 N levels the
existing zero-mean quad2d family already uses. Most of any sweep would read
`p_success ~ 0` for reasons that have nothing to do with the ROA.

**Profiles B and C** (peak below the goal; twin bands either side). Both satisfy
the tilt constraint. Rejected in favour of D because neither is localised --
they span the whole approach, so there is no "outside the corridor" and no
control group of trajectories that never meet the disturbance.

**A profile carrying a small non-zero sigma at the goal.** Proposed as a fifth
candidate to preserve the terminal asymmetry a biased force produces. Not
adopted: it optimises for asymmetry in the terminal-state distribution, which is
a different experiment from a crossing hazard. Recorded because the trade is
real -- a corridor deliberately gives the controller calm air in which to undo
the asymmetry imprinted during the crossing.

**Deterministic field** (`F_x = sigma(z)`, no draw). Rejected: given a start
state the outcome is fixed, `p_success` collapses to {0, 1}, and K trials buy
nothing.

**Frozen random field indexed by z**, drawn once per rollout. Attractive because
it makes the disturbance a property of the airspace rather than of the clock,
and because it defeats the averaging described below. Rejected for cost: it
needs a correlation length as a second parameter and a field-generation path
with its own reproducibility story, and the existing per-step mechanism is
already demonstrated to produce usable spread in this exact setup
(`fraction_interior` 0.122 at `f = 0.150` in the shipped quad2d family).

**Per-rollout scalar scale** (`F_x = sigma(z) * u`, `u ~ U(0,1)` at reset).
Rejected as the coarsest of the three; it makes every crossing a scaled copy of
the same gust.

**Per-rollout sinusoid** (`F_x = sigma(z) * [0.5 + 0.5 * A * sin(omega*t + phi)]`,
`A ~ U(0,1)` and `phi ~ U(-pi, pi)` drawn at reset, `omega` fixed). The measured
fallback if the per-step draw's transition shell proves too thin. On the
standard crossing (z_dot = 0.5, 1.46 s in band) the delivered-impulse spread is
53.5% with phase alone and 30.9% with phase+amplitude, against 6.3% for per-step
i.i.d. -- the coherence defeats the averaging, and the amplitude decouples mean
(set by the 0.5 bias term) from spread (set by A). Rejected as the default for
five reasons, in order: omega is a timescale whose effect depends on crossing
time and is therefore entangled with the start-state distribution; the
randomness collapses to two per-rollout scalars, making p_success a 2-D integral
over a fixed gust family rather than robustness to noise; a single tone
concentrates all disturbance energy at one frequency, so any closed-loop
resonance there becomes a dataset-wide artifact; it changes two things at once
relative to the flat `f_*` family (gating and temporal structure), breaking
attribution; and its mean is sigma/2 only per full period, so effective
difficulty varies with entry timing. Revive it -- as a `_draw` override, see
"The envelope is a strategy" -- only if the sweep's fraction_interior stop
condition triggers.

## The sweep falsified the per-step draw

Ran 2026-08-18: 400 random grid states, 30 paired trials, fourteen levels
0.002-0.80 N. Retention spans 1.00 down to 0.06, but `fraction_interior`
peaks at **0.025** (10/400 states, at 0.13 N) and never approaches the 0.122
of the shipped zero-mean family. Raising the force does not help: at 0.50 N
retention hits 0.09 -- the structural floor set by the 9.25% of successes that
never enter the band -- with 283 horizon-hits (the quad3d deadline artifact),
and at 0.80 N interior is exactly 0. The averaging mechanism below is real and
no F_max escapes it. Decision [user, 2026-08-18]: switch the draw law to the
banked per-rollout sinusoid; the ladder is re-swept under it.

The switch is `AltitudeGatedSineNoise(AltitudeGatedNoise)`, registered as
`'altitude_gated_sine'`: `reset()` draws `phi ~ U(-pi, pi)` and `A ~ U(0, 1)`
once per episode from the seeded stream; `_draw` returns
`bound * (0.5 + 0.5 * A * sin(2*pi/PERIOD * t + phi))` with `t` the control
step time and `PERIOD = 2.0 s` (about 1.4x the median band-crossing time --
inside the control bandwidth, long enough not to average out, short enough
that a crossing sees a partial cycle rather than a frozen offset). Mean force
is unchanged (`E[0.5 + 0.5*A*sin] = 0.5`), so the tilt-budget analysis
carries over. The five objections recorded against the sinusoid below were
weighed against the measured degeneracy and accepted as the lesser cost.

### The collected family: two sine+ambient configs

CORRECTION 2026-08-19: the sine_gauss law described below was PAUSED
mid-implementation by the user and never landed; the paragraph is kept as a
considered-variant record, not as the adopted law. What was actually adopted,
swept and collected is the ADDITIVE two-term model:

```
F_x = sigma(z) * (0.5 + 0.5 * A * sin(2*pi/2.0 * t + phi))  +  N(0, alpha)
      [corridor: per-rollout coherent gust]                    [ambient wobble,
                                                                every step, everywhere]
```

registered as NOISE_MODELS['sine+ambient']. A 16-cell (F_max x alpha) grid
plus per-state rescued/broken analysis and dense (x, z) heatmap slices chose
TWO configs, both collected [user, 2026-08-19]:

| level name | F_max | alpha | retention | fuzzy | character |
| --- | --- | --- | --- | --- | --- |
| sharp | 0.08 | 0.06 | 0.77 | 7.01% | corridor-dominant: hard failure lobes, thin fuzzy rim |
| smooth | 0.05 | 0.09 | 0.76 | 8.96% | wobble-dominant: wide 0-to-1 probability field |

MEASURED 2026-08-19, on the full collected set after the shortcut repair. The
row above previously read 0.74/8.5% and 0.76/10.0%, taken from the 400-state
sweep and from eval shards carrying the falsified reachability shortcut. Those
are superseded.

The numbers come from `q2_corridor_reduce.py` over all 489,789 eval states per
config, each at exactly 20 trials, `trials_shortcut` 0 for both. Fuzzy is
`fraction_interior` directly. Retention is the config's `mean_p_success`
divided by the baseline's, so 0.0602/0.0778 and 0.0593/0.0778.

The shortcut inflated fuzziness by about 1.2 points in each config and left
retention almost untouched. That asymmetry is expected: a rescued start moves
p up from 0 and a broken one moves p down from 1, so the two roughly cancel in
the mean while both add to the interior count. Measured over the repaired
shards, `sharp` has 8,786 rescued against 20,531 broken and `smooth` 4,478
against 5,486.

The pair still brackets what the design wanted, near-equal retention with
different texture, and `smooth` remains the fuzzier of the two.

plus a same-code baseline (F=0, alpha=0). Eval runs K=20 with the trial-window
mechanism (--trial_lo): a later top-up to K=50 draws bit-identical trials
(window-sum == single-run, verified). Publication target
[user, 2026-08-19]:

```
DATA_ROOT/stochastic/quadrotor2D/corridor_sine_ambient/rl/{baseline,sharp,smooth}/
```

with st1122 granted read access at placement, and the announcement DM sent
only after access is verified.

### The sine-enveloped Gaussian (considered, paused, not collected)

Final revision [user, 2026-08-18]: the campaign law is `sine_gauss`,

```
F(z, t) = sigma(z) * wave(t) * |N(0, 1)| * 0.6267
wave(t) = 0.5 + 0.5 * A * sin(2*pi/2.0 * t + phi),  A ~ U(0,1), phi ~ U(-pi,pi) per rollout
```

`0.6267 = 0.5 / sqrt(2/pi)` keeps `E[F] = sigma(z) * wave / 2`, so the mean
force -- and with it the tilt budget and the ladder -- matches both earlier
laws. The trial-to-trial spread still comes from (A, phi); the per-step
Gaussian adds a broadband floor to the single tone (measured contribution to
impulse spread: none -- 29.9% with dither vs 30.9% without -- it is texture,
not fuzziness). Two properties change against the pure sine: the force
history is no longer deterministic within a rollout, and `sigma(z)` becomes a
SCALE rather than a hard bound -- `|N|*0.6267` exceeds 1 with probability
~0.11, so instantaneous force can exceed sigma(z). Both recorded in every
description JSON.

## Known cost of the per-step draw

Per-step i.i.d. draws average along a crossing. Measured analytically for D at
`F_max = 0.05 N`:

| z_dot | steps in band | mean delta-v_x | std of delta-v_x | spread |
| --- | --- | --- | --- | --- |
| 0.25 | 291 | 1.114 m/s | 0.049 m/s | 4.4% |
| 0.50 | 146 | 0.557 m/s | 0.035 m/s | 6.3% |
| 1.00 | 73 | 0.279 m/s | 0.025 m/s | 8.9% |

The consequence is a **thin transition shell** rather than a degenerate
`p_success`: states far from the success boundary read 0 or 1, and only a narrow
band of marginal states reads intermediate. This is accepted, but the sweep must
measure the realised spread rather than trust this estimate.

Note also that the mean impulse is large. At `F_max = 0.05 N` a crossing at
`z_dot = 0.5` delivers 0.557 m/s of lateral velocity against an `x_dot` bound of
+/-1.0, which is why the level bracket below sits well under the existing
family's.

## Corridor coverage

Over all 489,789 shipped eval start states (`roa_labels.txt`, column 1):

| region | share | count |
| --- | --- | --- |
| below the band (z < 0.186) | 14.29% | 69,968 |
| inside the band | 42.86% | 209,912 |
| above the band (z > 0.914) | 42.86% | 209,909 |

The near-exact fractions come from the stratified grid the shipped set was
sampled on.

Measured on the undisturbed (`f_max = 0`) trajectory from 20,000 states sampled
uniformly at random from the shipped set (`q2_corridor_entry.py`):

| region | n | entered |
| --- | --- | --- |
| below the band | 2,791 | 5.63% |
| inside the band | 8,616 | 100.00% |
| above the band | 8,593 | 6.85% |
| overall | 20,000 | 46.81% |

Starts outside the band almost never enter it: below the band the trajectory
either terminates near the floor or never climbs into the corridor; above the
band it flies directly to the goal, which sits at z=1.0 above the band's top
edge (0.914), without dropping through. So the disturbance can only reach
about 47% of eval states, not the 100% a start-state count would suggest.
Under the corridor framing the states that do cross are not wasted -- they are
the control group, and the contrast between "had to cross" and "did not" is
the signal. But a state whose trajectory never enters the band gives the same
answer on all K trials, so skipping it is worth roughly half the eval compute.

**Mitigation to implement:** roll each eval state once deterministically, and
spend the K trials only on states whose trajectory enters the band, with a
margin in z to cover trajectories that noise could push in. States outside the
margin get `p_success` from the single roll.

**FALSIFIED 2026-08-19, for any model carrying an ambient term.** The claim
above that "a state whose trajectory never enters the band gives the same
answer on all K trials" holds for the corridor gust alone, which `sigma(z)`
gates on altitude. It is false for the adopted `sine+ambient` family. `build()`
adds the ambient term as `white_noise` on the same `[1, 0]` mask with no
altitude condition, so it acts on a start at any height, at every one of up to
1200 control steps.

Measured, not argued. Ten skipped starts from the `sharp` run (f_max 0.08,
ambient 0.06) were re-flown 20 times each. Seven varied, having been recorded
as certain successes: they came back 13, 15, 17, 17, 17, 17 and 18 of 20. The
error runs both ways. In shard 447, 7 of 25 starts recorded as certain failures
were also fuzzy, so 15 of its 34 skipped starts carried a wrong label. That 44%
is a ceiling rather than a typical rate, because shard 447's few skipped starts
hug the margin boundary where flipping is likeliest.

No wider MARGIN repairs this. There is no altitude at which the ambient term
switches off, so the premise never becomes true again. The shortcut now applies
only when `ambient` is 0, which covers the `sine` and `uniform` models and the
zero-disturbance baseline. Population effect on `fraction_interior`, measured
over the repaired shards, is +0.34 points for `sharp` and +0.53 for `smooth`.

Cost of dropping it is 1.63x the flights, 5,735,693 to 9,375,960 for one eval
pass. Wall-clock barely moves: the finish time is set by the fully-reachable
shards, which the shortcut never touched, and the freed cores were idle anyway.
The already-collected runs were repaired rather than discarded, via
`q2_corridor_topup.py` writing the missing flights as a second k-window that
the reducer sums. That is sound because `rollout_seed` is a pure function of
(base, split, state index, trial), so trial k draws the same noise whichever
process rolls it. Verified against real collector output at a nonzero shard
offset: five fuzzy starts at global indices 390,980 to 391,214 reproduced
exactly, at 16, 4, 17, 4 and 9 of 20.

## Implementation

One new `Disturbance` subclass. `SignalDependentNoise` cannot serve: its scale
is `alpha + beta * |target|`, and on the `dynamics` channel `target` is the
zero vector `disturb_force` (`quadrotor.py:450`), so it would collapse to a
constant `alpha`. The new class reads altitude off the env instead --
`apply(target, env)` already receives `env`, and the TWO_D state order is
`[x, x_dot, z, z_dot, theta, theta_dot]`, so altitude is `env.state[2]`.

Registered in `DISTURBANCE_TYPES` alongside the existing six, and configured the
same way as every other family:

```python
disturbances = {'dynamics': [{'disturbance_func': 'altitude_gated',
                              'f_max': F_MAX, 'centre': 0.55, 'width': 0.12,
                              'mask': [1, 0]}]}
```

Everything else is transcribed from `q2_common.py` unchanged: horizon 1200,
tolerance 0.2 entry-cut, `cost='quadratic'`, bounds x +/-1.0, z [0.1, 1.5],
velocities +/-1.0, theta_dot +/-8.0, `safe_explorer_ppo` with the shipped model
and a frozen `obs_normalizer`.

Collection follows the existing q2 pipeline shape: a collector, a level-0 gate
that must reproduce the deterministic labels before any noisy level runs, a
sweep that chooses the levels, and a reducer that writes the dataset layout.

## To be measured before collection

1. **Level ladder -- measured.** The impulse-derived 0.002-0.020 N bracket
   (`q2_corridor_sweep.py`, `q2_sweep.py`'s pattern) measured too weak and was
   extended, under the same per-step uniform draw, up to 0.80 N. That sweep is
   labeled **falsified** below -- see "The sweep falsified the per-step draw".
   The per-rollout sinusoid (`altitude_gated_sine`, commit `c10cf210`) was
   **adopted** in its place and re-swept on the same 400 states and paired
   seeds.

   **Uniform (falsified), baseline `p_success` = 0.0825 (`f_max = 0`):**

   | F_max (N) | p_success | retention | fraction_interior | hit_horizon |
   | --- | --- | --- | --- | --- |
   | 0.002 | 0.0825 | 1.00 | 0.0000 | 0 |
   | 0.004 | 0.0825 | 1.00 | 0.0000 | 0 |
   | 0.006 | 0.0822 | 1.00 | 0.0025 | 0 |
   | 0.009 | 0.0820 | 0.99 | 0.0025 | 0 |
   | 0.012 | 0.0807 | 0.98 | 0.0025 | 0 |
   | 0.016 | 0.0800 | 0.97 | 0.0000 | 0 |
   | 0.020 | 0.0800 | 0.97 | 0.0000 | 0 |
   | 0.030 | 0.0793 | 0.96 | 0.0050 | 0 |
   | 0.050 | 0.0769 | 0.93 | 0.0075 | 0 |
   | 0.080 | 0.0664 | 0.80 | 0.0225 | 0 |
   | 0.130 | 0.0520 | 0.63 | 0.0250 | 5 |
   | 0.200 | 0.0409 | 0.50 | 0.0125 | 0 |
   | 0.300 | 0.0326 | 0.40 | 0.0175 | 1 |
   | 0.500 | 0.0077 | 0.09 | 0.0100 | 283 |
   | 0.800 | 0.0050 | 0.06 | 0.0000 | 84 |

   `fraction_interior` peaks at 0.025 (0.13 N) and never approaches the shipped
   zero-mean family's 0.122; retention saturates near the 0.09 structural floor
   set by the 9.25% of successes that never enter the band, so raising `F_max`
   further does not help.

   **Sine (adopted), same baseline and paired seeds:**

   | F_max (N) | p_success | retention | fraction_interior | hit_horizon |
   | --- | --- | --- | --- | --- |
   | 0.050 | 0.0752 | 0.91 | 0.0250 | 0 |
   | 0.080 | 0.0668 | 0.81 | 0.0450 | 0 |
   | 0.130 | 0.0513 | 0.62 | 0.0475 | 0 |
   | 0.200 | 0.0422 | 0.51 | 0.0375 | 2 |
   | 0.300 | 0.0346 | 0.42 | 0.0375 | 20 |

   **Chosen ladder:** draw law `sine`, four noisy levels `0.05 / 0.08 / 0.13 /
   0.30` N (retention `0.91 / 0.81 / 0.62 / 0.42`), plus the `f_max = 0`
   baseline.

   **Dilution point.** Only 46.8% of states ever enter the band
   (`q2_corridor_entry.py`); the rest are the corridor's designed control
   group and read `p` in `{0, 1}` on every trial regardless of level, diluting
   the aggregate `fraction_interior`. The sine family's aggregate 0.0475 (at
   0.13 N) is roughly 10% of the *touchable* 46.8% of states -- on par with the
   shipped zero-mean family's 0.122, which has no untouchable states to dilute
   against.
2. **Realised trial-to-trial spread**, and the `fraction_interior` it produces.
   If the transition shell is too thin to be useful, the frozen-field
   alternative above is the fallback and this spec should be revised rather than
   the levels pushed higher.
3. **Entry rate.** What fraction of trajectories from each start region actually
   enter the band, which sets both the mitigation margin and the interpretation
   of any level-averaged `p_success`.

## Level naming

`f_*` in the existing families means peak newtons of a flat uniform bound. Here
`F_max` is the peak of an altitude-varying bound, reached only at z = 0.55.
A single scalar therefore does not describe the disturbance. Levels are named by
`F_max`, and each `dataset_description.json` must additionally record the centre,
the width, `sigma` at the goal, and the band extent -- otherwise the level names
are not comparable with any other family, or with each other under a later change
of geometry.
