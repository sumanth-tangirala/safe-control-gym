# Glossary

Load when a term in a spec or plan is unfamiliar. These appear throughout the
sources without definition.

**ROA — region of attraction.** The set of initial states from which the closed
loop reaches the goal. Collecting a labelled grid over the state space is
approximating it. `roa_labels.txt` in the older datasets is `(init_state, label)`
per grid cell.

**Invariant terminal set.** An ellipsoid `{(s - s0)' P (s - s0) <= c}` around the
closed-loop attractor `s0` that the closed loop cannot leave once entered.
Membership of *any* state — including one a model predicted — therefore soundly
implies task success. Obtained by finite-differencing the true closed-loop step
map at `s0` and solving `A_d' P A_d - P = -Q`, then raising `c` to the largest
level that survives boundary sampling. Stored in `invariant_sets/<system>.npz`
as `P`, `center`, `c`.

**Non-normal closed loop.** `A A' != A' A`. Consequence here: a state entering a
radius-R Euclidean ball transiently excurses to `gain x R` before converging
(measured: pendulum 2.6, quad2D 3.2, quad3D 4.9, cartpole 5.4, roughly constant
in R). This is why no Euclidean ball is invariant and why the ellipsoid scheme
exists.

**Entry-cut.** The noisy-collection success rule, now used by every stochastic
family: a rollout succeeds if it *ever* entered the goal set, and the stored
trajectory is truncated at (and includes) that entry state. Used because under
noise a rollout can enter and drift back out, which would otherwise break the
"label is a function of the terminal state" property. The goal set differs per
system — pendulum a per-channel box at 0.05, cartpole an L2 ball at 0.05, quad2d
an L2 ball at 0.2, quad3d an L2 ball at 0.05 — so a success rate in one says
nothing about a success rate in another. The signature of an entry-cut set in
shipped data is that successes end pressed against the threshold from below
(cartpole 0.0497-0.0500, quad2d 0.1972-0.1998, quad3d 0.0494-0.0498); a dwell
requirement would leave them well inside instead.

**Labels cannot validate a success rule.** Converging and diverging trajectories
are separated by a wide gap — measured on quad3d, the converging set tops out at
3.14e-4 and the non-converging set bottoms out at 1.537 — so many different rules
partition the same states identically. A cartpole gate scored 300/300 against a
rule that was wrong in two ways (per-channel tolerances instead of an L2 ball, and
a 10-step dwell that was never implemented). Only the stored **final states**
discriminate, because they reveal *where* the rollout was cut. Compare final
states, not labels, when validating a reproduction.

**Bounded-time reach probability.** What `p_success` actually measures: reached
the goal set *within the horizon*, not reached it at all. Load-bearing because
under persistent noise the controllers largely still converge, just later — given
quad3d's own 100,000-step allowance, success at `f = 0.072` is ~0.24 against ~0.25
at `f = 0`, while at H=1000 it reads 0.058. Roughly 15% of those rollouts would
succeed with unlimited time. Not comparable to an asymptotic reach probability,
and not transferable across horizons.

**Interior fraction.** The share of eval states with `0 < p_success < 1` — the
ones carrying information the deterministic labels do not already contain.
Depends on K as well as on the state: with small K, a state whose true `p` is 0.97
reads as a flat 1.0. Measured at f=0.032 on quad3d, K=100 resolves 18.3% as
interior where K=10 predicted 6.0%; at f=0.048, where probabilities sit further
from the boundaries, the two agree (25.1% vs 24.5%). So extra trials buy
resolution specifically at low noise levels.

**Noise floor.** The stationary distance from upright that a noisy closed loop
settles into. Under the state-additive presets it exceeds the 0.075 goal radius
at `high`/`xhigh` (p50 distance 0.086 / 0.139), which is why the invariant-set
scheme does not apply to *those* datasets.

That is a property of the mechanism, not of noise as such. Under **torque**
noise the closed loop stays confined: 24 runs x 5000 settled steps per level,
zero escapes at every level, with a settled region measured at
`|theta| <= 0.0385*tau`, `|theta_dot| <= 0.4031*tau` — linear in `tau` to within
1.3%, and elongated roughly **10:1 along theta_dot** because a torque perturbs
the acceleration row and reaches the angle only by integration. So an invariant
success set does exist there; it is simply not a ball or a square box. The
worst-case robust invariant ellipsoid is far looser than the observed region
(`|theta_dot| <= 3.11` vs 0.20 at `tau = 0.5`, ~15x), which is the usual gap
between an adversarial bound and unbiased noise.

**`alpha` / `beta` (signal-dependent noise).** `sigma = alpha + beta*|u|`, a
standard deviation. They control different things and only `alpha` acts *at the
goal*: `beta` scales with the commanded torque, so it goes quiet exactly where a
stabilising run is finishing, while `alpha` is the floor surviving as `u -> 0`.
A `beta`-only sweep therefore cannot say anything about whether the settled
region fits inside the success box — that is an `alpha` question.

The landmark is `|theta_dot| <~ 0.70*alpha` for the settled spread, so
`alpha ~ 0.07` is where the floor reaches the 0.05 box. Measured (full grid,
K = 20, three betas), p does **not** turn over there or anywhere in `0 <= alpha
<= 0.8` — it rises monotonically throughout. The reason is the success rule, not
the physics: entry-cut scores *entry* with no dwell, so a floor too large for the
pendulum to sit inside the box still helps it stumble in, and stumbling in is all
that counts. A dwell requirement would restore the turn. This is the clearest
case on record of a label choice determining a result that looks physical.

**Scale mixture.** What signal-dependent noise is: every draw has its own sigma,
`w_t ~ Normal(0, alpha + beta*|u_t|)`. Its delivered standard deviation is
therefore `sqrt(E[sigma^2])`, **not** `E[sigma]`, and with a heavy-tailed command
the two diverge badly — measured on cartpole at `alpha = 0, beta = 2`, 10.76
against 2.52, a factor of four. Quoting the mean sigma understates a level.

**Matched variance vs matched difficulty.** Two ways to align a new noise family
with an existing one, and they are not the same. Matching *variance* fixes
`sqrt(E[sigma^2])`; matching *difficulty* fixes `p_success`. On cartpole,
gaussian levels matched in variance to the uniform ones came out 24-77% easier
(p 0.4567 vs 0.3692 at the `low` pairing, widening with strength), because what
kills a run is noise at the goal and the signal-dependent family goes quiet
there. A published level set must say which it matched.

**Level naming.** Published levels are `low`/`med`/`high` [user, 2026-08-16] with
the constants in a `README.md` beside them and in each description's `level_name`
and `noise_model.parameters`. Two earlier conventions failed: `beta_<b>` with
`alpha` implicit, which silently misleads once `alpha` varies, and
`a<alpha>_b<beta>`, which is explicit but unreadable as a ladder. The rule that
survives is that a name is either fully explicit or carries no parameters at all
— never partially explicit.

**Noise preset.** A named entry in `NOISE_PRESETS`
(`safe_control_gym/envs/gym_control/pendulum_noise.py`), mirroring the source
repo's Hydra config names: `<family>_<level>`, e.g. `truncated_gaussian_act_med`,
`control_proportional_high`. Levels are *intended* to run `low`, `med`, `high`,
`xhigh`, `xxhigh`, `ultra`, `max` — but in two families `max` is WEAKER than
`ultra` (`gaussian_act`: 2.0 vs 3.0; `truncated_gaussian_act`: 1.0 vs 2.0). The
collector names its output directory from the suffix, so those two datasets are
mis-ordered on disk. Pendulum-only; the other systems use `disturbances`.

**State-additive vs force noise.** Two mechanisms that are NOT interchangeable.
Force noise (`disturbances`, any mode) perturbs a generalised force, so it enters
the acceleration rows and position follows by integration. State-additive noise
(`pendulum_noise.py`'s dynamics families) writes the state directly, including
position, which no physical disturbance can do. Measured on the pendulum at
matched sigma, the two move the region of attraction in OPPOSITE directions —
0.386 -> 0.256 for torque, 0.386 -> 0.431 for state-additive, because the latter
can place the state in the goal set. See `architecture.md`.

**Reach probability vs region of attraction.** Under constant-magnitude additive
noise the origin is not an equilibrium of the closed loop, so the asymptotic ROA
is empty for every threshold and `p(success)` must be a *finite-horizon* quantity.
What the noisy datasets measure is a reach(-avoid) probability over a fixed
horizon, not a region of attraction. The distinction is load-bearing whenever the
number is reported as a control result.

**Reach / reach-avoid / reach-avoid-stay.** Three named points in the
Manna-Pnueli temporal hierarchy, which is the actual classification:
safety (`[]p`), **guarantee** (`<>p`), obligation (boolean combinations of the
two), recurrence (`[]<>p`), **persistence** (`<>[]p`), reactivity. Reach =
guarantee; reach-avoid = obligation; reach-avoid-stay = persistence plus a
safety conjunct. The pendulum has no unsafe set — theta wraps, theta_dot clips —
so its datasets are pure **guarantee**. The cartpole has real kill thresholds
(`x_threshold`, `x_dot_threshold`), so its datasets are **obligation**. The two
are therefore not the same kind of measurement, and their success rates are not
interchangeable. See `queue.md` for the papers.

**Recurrence vs invariance.** `[]<>S` (visits S infinitely often) is strictly
weaker than `<>[]S` (enters S and stays). Under persistent noise a closed loop
can be recurrent in a set without being invariant in it — measured on the
pendulum at `tau = 0.5`: one trajectory entered the 0.05 box 217 separate times
for 365 of its 1000 steps and never held it for 10 consecutive steps. Scoring
that as failure measures the box, not the controller. This is why the torque
datasets score **first entry with no dwell**.

**Matched / unmatched uncertainty.** A perturbation is *matched* if it lies in
`range(B)` — the same direction the controller commands — and unmatched
otherwise. Cartpole's `action` mode is matched; its `dynamics` mode is not.

Matchedness alone does NOT imply the ROA is biased toward the nominal one — an
earlier version of this page said it did, and the external-torque pendulum family
refutes it. That family is matched, enters through the same `B`, and *gains* up
to 30,561 cells. What biases an ROA toward the nominal is **saturation
placement**, not matchedness. See below.

**Saturation placement.** Where an action disturbance sits relative to the
actuator clip, and the single largest determinant of whether noise can help:

```
sat(u + w)   noise inside the actuator  -- command/current noise. The motor
             cannot be driven past u_sat by it.
sat(u) + w   noise outside -- an external shaft torque. Still matched, but the
             actuator's limit does not bound something the actuator is not
             producing.
```

Under `sat(u + w)`, whenever the command is saturated the clip discards every
positive draw and passes every negative one, so the noise can only *subtract*
control authority. Measured on the pendulum LQR, the command is saturated on
70–98% of steps depending on level, and the realised noise std collapses to
28–71% of what was drawn. A start state that fails for want of authority
therefore cannot be rescued by a disturbance that can only remove authority —
which is the mechanism behind the zero-gain result, and it is a property of the
clip rather than of noise, of matchedness, or of pendulums.

Measured at `alpha = 0.008, beta = 1.6` on 200 deterministically-failing cells ×
K = 10, the *same* `w` rescues **0 of 2000** rollouts inside the clip and **956 of
2000** outside it.

**Zero gain (pre-saturation families).** Across every pre-saturation pendulum
family — the seven published `tau` levels, the `beta` sweep, and the `beta = 1.6`
full grid — **24,643,200 rollouts started from deterministically-failing cells and
none reached the goal**; 95% upper bound on the mean gain probability 1.2e-7. Two
independent reasons, both measured: the clip asymmetry above, and the fact that
failing states park at the hanging equilibrium (72.8% of steps) with a settled
energy of 14–21% of the 1.4715 J barrier — and *more* noise gives *less* energy,
because more of it is clipped away. Not a horizon artifact: still zero at 8,000
steps, ten times the dataset horizon.

**Internal vs external uncertainty.** Internal: the plant is not what was
modelled (parametric mismatch, unmodelled structure such as friction) —
`randomized_inertial_prop`. External: the plant is right and something pushes it
— the `dynamics` disturbance mode. "Unmodelled dynamics" in the robust-control
sense is internal; strictly it means omitted *structure*, so randomising three
inertial parameters is the weaker cousin, parametric uncertainty.

**Stochastic ROA.** Under a stochastic plant the ROA stops being a set and
becomes a field `p(success | x0, H)`, thresholded at some `alpha`. Both `alpha`
and `H` are choices, not properties of the system, and must be reported. A
set-valued answer requires *bounded* noise: with unbounded support no bounded set
is invariant with probability 1.

**Split.** `train` (random starts, full trajectories stored) or `eval` (fixed
grid, only a per-cell success probability stored). Independent processes.
`TRAIN_SPLIT_ID = 0`, `EVAL_SPLIT_ID = 1`, used as a coordinate in
`rollout_seed`.

**Batch (eval).** One rollout from every grid cell — 49,770 rollouts for the
pendulum. The unit of publication: only whole batches are written, so every cell
always has the same `trials`.

**mean-SE stopping rule.** `mean_i sqrt(p_i (1 - p_i) / B)` over all cells,
where `p_i = successes_i / B`. Measures how much the estimate could still move.
Near-monotone in `B`, so unlike a drift statistic it cannot trip early by
chance. Eval stops when it drops below `--se_tol` (default 0.01).

**Drift.** `mean_i |p_i(B) - p_i(B - check_every)|`. Logged as a diagnostic,
deliberately *not* used to gate stopping.

**Half-open grid.** `lo + resolution * arange(ceil((hi - lo) / resolution))`.
For theta this is correct rather than merely convenient: theta is periodic, so
`-pi` and `+pi` are the same physical state and including both would duplicate a
column. The earlier `arange(lo, hi + resolution, resolution)` overshot the
domain and silently duplicated cells.

**Horizon.** Rollout length in steps at 100 Hz (`dt = 0.01`). It differs per
family and is not a shared constant: **800** for the published pendulum tree
(the superseded `lqr_legacy_20260806/` used 1000), **1000** for cartpole and
quad3d, **1200** for quad2d, inherited from its deterministic set rather than
chosen. `DEFAULT_HORIZON = {'lqr': 600,
'rl': 1100}` for the invariant scheme, set to the old maximum success length plus
a settle buffer. The horizon is a load-bearing parameter of the label, not a
safety margin — see bounded-time reach probability — and levels calibrated at one
horizon do not transfer to another.

**`U_SAT`.** Pendulum control saturation, `0.6371781908344007`. Not a round
number because it is inherited from the source repo's model.

**Terminal-state model.** The downstream flow-matching consumer of these
datasets. It predicts where a trajectory ends up, which is why every labelling
decision here is judged by whether the label remains a function of the terminal
state.

**`terminated` / `truncated`.** The Gymnasium 5-tuple's two done-flags,
replacing the pre-migration single `done`. `terminated` means the episode
ended for a reason internal to the task — goal reached, out-of-bounds when
`done_on_out_of_bound`, or a constraint violation under `DONE_ON_VIOLATION`.
`truncated` means only the time limit was hit (`ctrl_step_counter >=
CTRL_STEPS`). Both can be true on the same step (goal reached exactly at the
horizon). `info['TimeLimit.truncated']` is the pre-migration convention the
split was promoted from and is kept for the six controllers that already read
it. See `.claude/docs/architecture.md` for the two-source table.

**`check_env`.** `stable_baselines3.common.env_checker.check_env` — validates
that an environment satisfies the Gymnasium API contract (tuple arity, `reset`
signature, space/dtype conformance) directly, rather than inferring
correctness from tests that happen to pass. Run in
`tests/test_envs/test_gymnasium_conformance.py` against all four registered
environments; it is the primary evidence the Gymnasium migration is correct,
because it also covers systems (cartpole, both quadrotors) that have no golden
rollout fixtures of their own.

---

Related: [datasets.md](datasets.md) where most of these terms are load-bearing, [architecture.md](architecture.md) for the code they name.
