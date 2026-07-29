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
settles into. At `high`/`xhigh` it exceeds the 0.075 goal radius (p50 distance
0.086 / 0.139), which is why the invariant-set scheme does not apply to noisy
datasets — no invariant success set exists there.

**Noise preset.** A named entry in `NOISE_PRESETS`
(`safe_control_gym/envs/gym_control/pendulum_noise.py`), mirroring the source
repo's Hydra config names: `<family>_<level>`, e.g. `truncated_gaussian_act_med`,
`control_proportional_high`. Levels weakest to strongest: `low`, `med`, `high`,
`xhigh`, `xxhigh`, `ultra`, `max`.

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
