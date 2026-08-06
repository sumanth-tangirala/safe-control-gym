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

**Entry-cut.** The noisy-collection success rule: a rollout succeeds if it *ever*
entered the 0.075 L2 goal ball, and the stored trajectory is truncated at (and
includes) that entry state. Used because under noise a rollout can enter and
drift back out, which would otherwise break the "label is a function of the
terminal state" property.

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
otherwise. Load-bearing for ROA estimation: a controller partially rejects
matched noise through its own input channel, so an ROA measured under it is
biased toward the nominal, noise-free ROA. Cartpole's `action` mode is matched;
its `dynamics` mode is not.

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

**Horizon.** Rollout length in steps at 100 Hz (`dt = 0.01`). 1000 for the noisy
scheme; `DEFAULT_HORIZON = {'lqr': 600, 'rl': 1100}` for the invariant scheme,
set to the old maximum success length plus a settle buffer.

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
