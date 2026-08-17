# Log

Append-only, newest first. One entry per ingest, per filed query answer, per
lint pass. Keep the `## [2026-08-17] ingest | gaussian_signal becomes the standard family; cartpole port

`gaussian_signal` is now the canonical stochastic family for both pendulum and
cartpole [user, 2026-08-17]. `noisy_torque` and the state-additive presets are
historical. Pages updated to present it as the default rather than as the newest
of several alternatives.

**Published naming changed twice and has settled.** Both trees are now
`low`/`med`/`high` [user, 2026-08-16], with constants in a `README.md` beside the
levels and in each description's `level_name`. Pendulum: alpha 0.05/0.10/0.20,
beta 0.16/0.64/1.00, mean p 0.3869/0.4067/0.5457. Cartpole: sigma 8/11/18.
Non-published levels moved to `archive_alpha_0.008/` and `archive/`. The rule
that survived two failed conventions: a level name is either fully explicit or
carries no parameters at all, never partially explicit.

**Cartpole port.** `cp_collect.build()` takes alpha/beta; `cp_gauss_sweep.py` and
`sbatch_cartpole_gauss_sweep.sh` added; design and Amarel runbook in
`docs/superpowers/specs/2026-08-17-cartpole-gaussian-signal-collection.md`.
Three findings worth keeping:

- **Placement is inert on cartpole.** Its LQR demands a median 0.27 N against a
  2000 N `action_scale` and never saturates in 16,494 measured steps, so
  sat(u + w) and sat(u) + w are the same function. That absence explains why the
  published uniform cartpole family gains 743-911 cells under pre-saturation
  noise where the pendulum's gains nothing.
- **The action is on the cart and reaches the pole through cos(theta)**, measured
  1.46 rad/s^2 per N upright falling to 0.02 at 89 degrees. Constant force noise
  is therefore already state-dependent on the pole.
- **Matched variance is not matched difficulty.** Gaussian levels matched in
  delivered std to the uniform ones came out 24/47/77% easier, widening with
  strength, because what kills a run is noise at the goal and this family goes
  quiet there. A level set has to declare which it matched.

**Corrections to my own reasoning, both measured rather than argued.** I claimed
beta was inert on cartpole; it is inert only at pendulum-scale values, because
|u| there is skewed (median 0.27 N, p99 28.9) rather than uniformly small. And I
blamed cartpole's slowness on the per-reset URDF rewrite; the profile refuted it
-- reset is 2% of a rollout. The real cost is pyb_freq 5000 giving 50 simulator
substeps per control step, ~15x the pendulum per rollout. The URDF caching fix
landed anyway (bit-exact, 540/540 hashes) but is honestly recorded as buying no
speedup outside NFS.

New glossary entries: scale mixture and why delivered std is sqrt(E[sigma^2])
rather than E[sigma] (they differ 4x at alpha = 0); matched variance vs matched
difficulty; level naming.

Pages touched: `datasets.md`, `glossary.md`, `architecture.md`, `workflows.md`,
`compute.md`, `INDEX.md`.

Not yet done: the cartpole family is specced and smoke-tested but not collected
-- it is intended to run on a collaborator's own Amarel allocation. The level
criterion (matched variance or matched difficulty) is still open.

## [2026-08-15] ingest | signal-dependent and external-torque pendulum families; the placement result

Two new pendulum families collected and one library capability added, plus a
result that corrects a claim this wiki has been carrying.

`SignalDependentNoise` registered in `DISTURBANCE_TYPES` (commit 734394d4):
`w ~ Normal(0, alpha + beta*|u|)`, a standard deviation not a variance. Gated by
reproducing the shipped `tau_0.00` labels on all 49,770 eval cells with
`alpha = beta = 0`, and by an empirical sigma check within 0.2%.

`external_action_disturbance` added to the pendulum env (7ea66acf), selecting
`sat(u + w)` from `sat(u) + w`. Default `False`, gated at 300/300 against the
already-collected `beta = 1.6` labels so no existing dataset moves.

Collected, all at horizon 800 / `ctrl_freq` 100 / `pyb_freq` 300 / seed 42 / K=100
so every pendulum family stays comparable cell-for-cell:

- `signal_dependent/pendulum/lqr/beta_{0,0.2,0.4,0.8,1.6,3.2}` at `alpha = 0.008`
- `external_torque/pendulum/lqr/a{0.050_b0.160, 0.008_b0.640, 0.100_b0.640}`,
  plus four earlier `beta_*` levels at `alpha = 0.008` under the superseded
  naming

**What this overturns.** `glossary.md` said a controller partially rejects
matched noise through its own input channel, so an ROA measured under it is
biased toward the nominal. That is wrong. The external-torque family is matched —
same `B` — and gains up to 30,561 cells. The cause is the saturation clip: under
`sat(u + w)` a saturated command discards every positive draw and passes every
negative one, so noise can only subtract authority. The same `w` at
`alpha = 0.008, beta = 1.6` rescues 0 of 2000 rollouts inside the clip and 956 of
2000 outside it. `datasets.md` stated the same result as a property of "the
physically admissible channel"; corrected to a property of placement.

The zero-gain result itself is now quantified rather than asserted: 24,643,200
rollouts from deterministically-failing cells across every pre-saturation family,
none successful, 95% bound 1.2e-7. Not a horizon artifact — still zero at 8,000
steps.

Second correction, this one to a prediction rather than a page: the alpha sweep
(3 betas x 8 alphas, full grid, K=20) was expected to show p turning over near
`alpha ~ 0.07` where the settled spread reaches the success box. It does not turn
anywhere in `0 <= alpha <= 0.8`. Entry-cut scores entry with no dwell, so a floor
too large to sit inside the box still helps a trajectory stumble in. Filed in
`glossary.md` as the clearest case here of a label choice producing a result that
looks physical.

`compute.md` gains a fourth way a scheduler produces nothing useful: a job
running stale code because `git pull | tail -1` hid an aborted merge, the error
being on stderr and the reassuring `Updating A..B` on stdout. Cost one sweep that
silently reproduced the previous sweep's numbers.

Pages touched: `architecture.md`, `datasets.md`, `glossary.md`, `workflows.md`,
`compute.md`, `INDEX.md`.

Not yet done: none of these datasets is published to `DATA_ROOT` — all six
signal-dependent levels and all seven external levels are still on cluster
scratch. The 2026-08-15 spec still says an alpha sweep is out of scope, which is
now the opposite of what happened.

## [2026-08-15] ingest | unmatched-force quadrotor datasets; cartpole re-collection

Three stochastic families collected 2026-08-14/15 and placed at
`DATA_ROOT/stochastic/{quadrotor3D/noisy_dynamics/lqr, quadrotor2D/noisy_dynamics/rl,
cartpole/noisy_torque/lqr}`. Five levels each including a same-code baseline.
First use anywhere in this repo of the `dynamics` disturbance mode — an unmatched
world-frame force at the COM, as against the matched `action`/torque channel the
pendulum and cartpole families use.

Level-0 reproduction of the corresponding deterministic sets: cartpole **1.0000**
over all 116,242 eval states, quad2d 0.9949, quad3d 0.9702. quad3d's gap is chaos
amplification over ~500-step tumbling trajectories and cannot be closed here — the
collector that produced the shipped set is not in this repo in runnable form.

The cartpole re-collection corrects six defects in
`stochastic/cartpole/noisy_action/lqr/`, which is left in place but superseded.
The consequential one is a control bound of 100 N where the deterministic set uses
2000 N — 20x too little authority, which also breaks the noise scale.

Changed in the wiki:

- `datasets.md` — new section for the three families, the two mechanisms and
  `B_d`; the cartpole defect table; that the deterministic cartpole description
  states its own success rule wrongly and that labels cannot detect it; that noise
  levels are coupled to the success rule and horizon; the two rate-injection
  directions for quad3d.
- `glossary.md` — **entry-cut** generalised from the pendulum's 0.075 ball to the
  four goal sets now in use, with the shipped-data signature that identifies one;
  **horizon** corrected, quad2d is 1200 not 1000; new terms *labels cannot
  validate a success rule*, *bounded-time reach probability*, *interior fraction*.
- `compute.md` — three ways a scheduler reports success and produces nothing, and
  the expected-vs-actual shard check that is the only thing catching them.
- `architecture.md` — the `dynamics` mode as actually used: per-substep
  re-application, no torque at the COM, and the two unregistered disturbance
  classes.
- `INDEX.md` — four page summaries extended.

Measurements worth carrying: retention at quad3d `f = 0.14` is 0.618 under a 0.1
per-channel box and 0.015 under the 0.05 L2 ball, which is why levels do not
transfer across rules. Every shipped cartpole success ends with `||state||` in
[0.0497, 0.0500] and none satisfies `|x| < 0.01`, which is how the description's
claimed rule was falsified. A cartpole gate scored 300/300 against that wrong
rule; the final-state comparison under the real rule matches at median 4.97e-07.

## [2026-08-06] ingest | torque-noise pendulum datasets; specification classes

Five pendulum datasets under a NEW noise mechanism, placed at
`DATA_ROOT/stochastic/pendulum/noisy_torque/lqr/tau_{0.00,0.10,0.15,0.30,0.50}`.
Uniform noise on the commanded torque, pre-saturation — the physically admissible
channel, replacing `pendulum_noise.py`'s state-additive presets which write into
`(theta, theta_dot)` directly and make success RISE with noise.

Measured: p falls monotonically 0.3878 -> 0.3529 -> 0.3372 -> 0.2997 -> 0.2593,
and across all 49,770 cells at all four noisy levels the largest gain over the
noiseless field is +0.000 (3,047,000 rollouts from failing cells, zero successes).
Spec: `docs/superpowers/specs/2026-08-06-pendulum-torque-noise-datasets-design.md`.

Changed in the wiki:

- `datasets.md` — new section for the family; corrected the claim that a noisy
  closed loop has no invariant success set (true of the state-additive presets,
  false under torque noise).
- `glossary.md` — same correction on `noise floor`, with the measured settled
  region `|theta| <= 0.0385*tau`, `|theta_dot| <= 0.4031*tau` (10:1 elongated,
  zero escapes in 120,000 settled steps per level); two new entries for the
  Manna-Pnueli classes and for recurrence vs invariance.

Decisions worth carrying:

- **No dwell.** The 10-step hold was carried over from the cartpole and broke the
  terminal-state invariant: 9,863 of 100,000 trajectories at tau=0.5 ended inside
  the box with label 0, because at 79% of saturation the plant is recurrent in
  the box but not invariant in it (one trajectory: 217 visits, 365 steps inside,
  longest run 7). Scoring first entry restores the invariant by construction.
- **The pendulum is a `guarantee` (reach) problem, the cartpole an `obligation`
  (reach-avoid) one** — the cartpole has real kill thresholds, the pendulum none.
  Their success rates are not the same kind of number.
- **The noise stream depends on `disturbances.py`.** These datasets predate
  `b2705cb1` (spawned child streams) and replay only against `2e0b9ddc`; the
  producing commit is recorded in each `dataset_description.json`.

Open, not done: the cartpole calibration probe still does not reproduce the
shipped 0.1797 (66.5% label agreement, 130/400 one-directional misses) — two
bugs found and fixed so far (env goal-termination left on; `env.state` assignment
silently discarded because the cartpole's state lives in PyBullet), at least one
remaining.

## [YYYY-MM-DD] <op> | <subject>` prefix exactly — it is
what makes the log greppable:

```bash
grep '^## \[' .claude/log.md | head -5      # what happened recently
grep '^## \[' .claude/log.md | grep ingest  # what has been read into the wiki
```

`<op>` is one of `ingest`, `query`, `lint`.

---

## [2026-08-03] query | LQR authority sweep: 50 N is free

Calibration for the low-authority arm: same LQR gains (they do not depend on
the clip), same 400 random eval states, action_scale in {50, 100, 200, 500,
2000}, reach < 0.1, both kill boxes. Coverage is IDENTICAL at every authority
-- collection 0.1725, physical 0.2525 (one single-state flip at 100 N) -- so
2000 N buys the reference controller nothing and the 10 N force starvation
(6.6% of successes, measured 2026-07-31) is fully healed by 50 N. LQR's
closed loop lives under ~50 N; the cap only removes unused headroom, so
reference_success is effectively authority-invariant down to 50 N (confirm at
scale before leaning hard on it). For RL the cap tames even divergence: peak
|x_dot| across failed physical rollouts 33 m/s at 50 N vs 100 at 2000, and
reaching any speed now costs sustained effort rather than one control step.
Next arm per the accumulated lessons: action_scale 50, dmc reward + sparse
goal bonus at the 0.05 ball, physical kills, RoA-selected checkpoints.

## [2026-07-31] query | redo arms, final table: velocity pricing verdicts

All three dmc-recipe arms trained to 1M and evaluated on the same seed-3
29,060-state sample, reach < 0.1, both kill boxes. Final table
(collection / physical): LQR 0.1804 / 0.2688; control @120k 0.0172 / 0.9771;
control @1M 0.0166 / 0.9180; lqrvel(scratch, additive -0.02*(x_dot^2 +
theta_dot^2)) @1M 0.0000 / 0.0000; lqrvel-ft (same reward, warm-started from
control@540k) @1M 0.0003 / 0.0008. Three lessons, all measured:

1. OVERTRAINING: the control lost 6 points of physical coverage (9 on
   LQR-solvable states) between 120k and 1M at flat eval reward ~990 --
   the training metric cannot see coverage degradation. Checkpoint selection
   must use the RoA metric; SB3's reward-based best_model selected a 60k
   checkpoint. Best known coverage policy: cartpole_stabilization_6
   checkpoint step_120000 (0.9771).
2. ADDITIVE VELOCITY PRICING KILLS REACH: lqrvel learned calm balance
   (speeds 6.5/6.8 vs the control's 18/27, reward ~+120-140) but parks
   WHEREVER it catches the pole and never travels to the origin -- 0.0000
   from all 29,060 states. dm_control's centered factor pays ~0.002/step to
   go home; the toll costs ~0.5/step in transit. Velocity pricing without
   real position income un-teaches destinations. Next reward needs a goal
   bonus at the ball (sparse hold) or kills instead of tolls.
3. WARM START INTO A CHANGED REWARD FAILS: lqrvel-ft crashed to -3502 at its
   first tick (donor critic valued slam behaviour at +900; new reward values
   it deeply negative; actor chased the re-fit through garbage) and plateaued
   at -80..-99 -- worse than scratch -- despite learning_starts 25000 and lr
   5e-5. The I5 lesson survives its own mitigations when the REWARD changes,
   not just the task. A real lambda schedule needs fresh buffers per stage.

Artifact (RoA slice maps, LQR vs RL, both kill boxes):
https://claude.ai/code/artifact/fcf9589d-1a23-42f4-9044-dace54e32039

## [2026-07-31] query | the 2x2: controller x kill box, exact, one sample

LQR under physical kills completes the table -- same harness, same seed-3
random 25% of eval_states.txt (n=29,060), reach < 0.1, 1000 steps. Rows are
controllers, columns kill boxes: LQR 0.1804 / 0.2688; dmc-recipe RL @120k
0.0172 / 0.9771. LQR/physical keeps all 5,243 collection successes (exact
monotonicity at scale) and flips 10.79% of the 23,817 failures -- the n=300
morning estimate (~26.3%) was within half a point. The asymmetry, quantified:
relaxing the velocity kills buys LQR +8.8 points (its failures are competence
-- one linear law cannot swing up) and buys RL +96 points (its failures are
style -- slam trajectories the collection box kills). Four quadrants: LQR =
style-compatible, competence-limited; this RL arm = competence-rich,
style-incompatible; the lqrvel arms are training toward the empty quadrant.

## [2026-07-31] query | dmc-recipe snapshot over 29,060 random eval states

Full-population follow-up to the RoA-sample probe (same harness, same 120k
checkpoint, reach < 0.1, 1000-step horizon, random 25% of eval_states.txt,
n=29,060). Model under physical kills: 0.9771 overall -- and UNIFORM across
the label boundary (0.9805 on LQR successes, 0.9764 on LQR failures): it
solves 23,254 states LQR cannot, missing 102 LQR gets. Under collection kills:
0.0172. LQR anchor: 0.1804 vs reference 0.1797, and per-label PERFECT --
1.0000 on label-1, 0.0000 on label-0 -- so first-entry-at-0.1 reproduces the
shipped label column exactly on all 29,060 states, a 50x-scale extension of
the 556-state verification and evidence the label boundary is robust to
doubling the ball. Consequence: the dmc-recipe policy's physical-kills ROA is
~5.4x LQR's over the same region; the entire gap to its collection-kills
number is movement style (velocity kills), which the lqrvel arm
(shaped_dmc_velocity_weight 0.02, run cartpole-dmc-lqrvel) is training to
close. Also observed on the control run: training success flipped 1 -> 0
between 120k and 240k at flat reward ~989 -- the dense product visits our
goal set incidentally, it does not anchor it; checkpoint selection for OUR
metric cannot ride SB3's reward-based best_model.

## [2026-07-31] query | dmc-recipe snapshot over the LQR RoA; encoding bug in build_env

Probed the dmc-recipe arm's 120k-step checkpoint (run cartpole_stabilization_6,
trained ONLY from hanging starts, dense shaped_dmc, physical kills, 2000 N
plant) over 5221 states -- a 25% sample of the shipped LQR RoA (label-1 rows of
eval_states.txt) -- success = first entry of the wrapped L2 goal error under
0.1, 1000-step horizon. Results: physical kills 5116/5221 = 0.980; collection
kills 509/5221 = 0.098; LQR anchor on the same states under collection kills
5221/5221 (by construction -- its own RoA, looser ball; validates the
harness). The 10x gap is movement style, not competence: the policy's first
action from rest is a ~1850 N slam that puts x_dot at ~18 m/s in one control
step, legal under physical kills, dead at step one under the collection
box. dm_control's reward cannot price this: small_velocity watches theta_dot
only (x_dot appears nowhere), has a 0.5 floor, and multiplies a near-zero
upright term during the swing -- upstream's real velocity regulariser was the
10 N actuator, which our plant deliberately does not have. Also: training
success flipped 0 -> 1 between the 60k and 120k evals at flat reward ~988.6,
so the policy parks inside the 0.05 ball from hanging starts despite the flat
centered gradient.

Found and fixed en route: build_env(config, regime=...) applied the
evaluation regime's kill box to state_space BEFORE NormalizeObservation
captured its scales, so cross-regime evaluation re-scaled the velocity
channels 20 -> 5 -- the policy saw x_dot 4x too large (measured: state
[3,2,0,2] read 0.4 for the trained-on 0.1) and scored 0 from its own training
distribution. Fix: encoding bounds applied first, always; the regime applied
after the wrappers, moving thresholds and state_space but not the captured
scales. Regression test: tests/test_envs/test_build_env_encoding.py (same
state, both stacks, identical observation; kill box must still move).

## [2026-07-31] ingest | curriculum arm ledger, distilled before deleting configs/sb3/archive

The archived RL configs (all sized for the wrong 50 Hz plant, most never
committed) are deleted for the fresh redo. Their comments were the only record
of the curriculum experiment tree; the measured outcomes, distilled, since the
lessons are plant-independent:

- Arm A (physical, sparse reach, init-only additive curriculum): mastered the
  near-goal stage, then FORGOT it after one widening -- the additive schedule's
  first step is 0.005 -> 0.155, 31x per axis, ~1e6x in 4-dim start volume.
- Arm B (plain sparse, collection regime): advance gate stalled at stage 0,
  training confined to near-goal starts.
- Arm B' (B + tolerance curriculum, ball 0.5 -> 0.05 at x0.85 per cleared
  tick): six consecutive 100% ticks under the collection kills, no forgetting
  -- the tolerance-first lever validated. Later hit a wall at ball 0.16, where
  one x0.85 tightening spans the force-moderation transition; B'f probed it
  with x0.92 steps.
- Arm E (full box always, tolerance-only from 0.5): a 0.5 ball from full-box
  starts demands swing-up-plus-catch before any reward pays -- dm_control's
  classic sparse hard-exploration case. E' (ball starts 2.0, reachable by
  undirected exploration) sat ~12 ticks at 2.0 scoring 0.40-0.45, permanently
  one hit under the 0.5 advance gate; E'' probed gate 0.4.
- Arm G (geometric start widening, x2 per stage = constant 16x volume ratio):
  the fix for the additive cliff both A and B' died on.
- Arm F (tolerance-first product policy): reached the 0.3 ball from 71% of
  full-box starts; ceiling at ball 0.2 measured 45%.
- Arms I/I4/I5/I6 (warm-start fine-tuning of F, ratchet from 0.3): I4 held 9
  ticks at ball 0.2167 in the 0.1-0.5 band; I5 (gate 0.4) drifted 0.80 -> 0.40
  -> 0.00 as the ball tightened -- fine-tuning at the from-scratch 3e-4 rate
  walks the actor off its donor; I6 dropped to 5e-5. learning_starts 25000
  protects restored weights from empty-buffer updates (arm J: 0.8 first tick,
  0.00 two ticks later without it).
- Dense-reward exploit, measured (kept in git history in the tracked configs):
  under rl_reward, reaching the goal ends the episode and forfeits remaining
  steps -- tightening terminal error was worth +4.46 while the termination it
  triggers cost -181.2, so SAC hovered just outside the ball at 3.8x LQR
  return and 0.000 success. Sparse makes the goal the only positive term.

Sizing rules (also in commit b018429d): horizon from measured worst-case LQR
steps-to-goal with margin; sparse_step_reward = -0.5/H; gamma sized so the
terminal reward is visible at episode start. All step-denominated numbers must
be resized for the 100 Hz plant (800-step episodes).

## [2026-07-31] query | reach vs hold labels coincide for the reference LQR

Same replay harness, same 556 sampled eval_states.txt rows, relabelled with
terminate_on_goal true (success = first entry into the 0.05 L2 goal ball):
556/556 agreement with the shipped labels, identical to the hold-rule replay.
For the 2000 N LQR, ball entry and the 10-step box hold are the same label --
post-entry non-normal excursions (up to ~5.4x the radius) stay 20x inside the
velocity kills and always re-settle. The equivalence is a property of the
controller, not the task; eval_policy's task-mismatch withholding stays.
Script bug worth remembering: `info['goal_reached']` is only populated under
cost='quadratic' (cartpole.py:806); under cost='rl_reward' read
`env.goal_reached` directly or every reach label silently reads 0.

## [2026-07-31] query | cartpole plant sync, regime inheritance, ROA relaxation measured

Uncommitted work, logged so the measurements survive until ingest. (1) Cartpole
regime files were instantiating a different plant than the reference dataset:
env defaults ctrl_freq 50 / pyb_freq 50 (one 20 ms PyBullet step per control
step) vs the dataset's 100 / 5000 (0.2 ms integrator, 10 ms hold), and the
physical regime additionally ran 10 N vs the dataset's 2000 N. Both now carry
the full plant via `task_config_overrides`. (2) The shared plant moved to a new
`configs/system/cartpole.yaml`; regime files inherit it through an `extends:`
key resolved recursively in `load_collection_bounds` (child wins). Merged dicts
verified bit-identical to the pre-refactor files. (3) Replayed eval_states.txt
samples with LQR under the refactored collection config: 596/596 label
agreement (340 random + 256 balanced). Under the physical config on the same
samples: zero successes lost and 25/300 random flipped to success — dropping
the invented velocity kills grows LQR's ROA ~18% -> ~26%. Also: every
`configs/sb3/` training config archived to `configs/sb3/archive/` (sized for
the old 50 Hz plant; two collection-regime copies had silently dropped
action_scale and trained at 10 N); RL restarts on the synced regimes. Replay
trap for next time: `terminate_on_goal` defaults True and ends episodes at
first goal entry, before the dataset's 10-step hold — reference_run needs it
false.

---

## [2026-07-31] ingest | cartpole stochastic dynamics spec

Read `docs/superpowers/specs/2026-07-31-cartpole-stochastic-dynamics-design.md`,
written this session from a design discussion.

Taken from it: the cartpole's stochastic axis is the `dynamics` disturbance mode
masked to `Fx`, with `uniform` rather than `white_noise`. Chosen on
matchedness — `action` noise lies in `range(B)`, so an ROA measured under it is
optimistic — and on boundedness, since unbounded support admits no invariant set.
Domain randomisation plus friction is the physically realistic account of
unmodelled dynamics and was rejected as *primary* only: no target hardware, a
different dataset axis, and a mixture-of-deterministic-plants sampling structure
rather than a smooth field.

Changed: `architecture.md`'s "Pendulum noise" section became "Noise: two
unrelated mechanisms" — the page previously documented only `pendulum_noise.py`
and never explained the `disturbances` mechanism that every other env has, which
was a gap given `architecture.md:70` already referred to it. `glossary.md` gained
matched/unmatched uncertainty, internal vs external uncertainty, and stochastic
ROA. `INDEX.md` updated for both.

Not yet true of the code: nothing under `configs/` enables any disturbance, and
the spec records two blockers — `Disturbance.seed` binds `env.np_random` itself
(`disturbances.py:33-35`), violating resident invariant 3, and the magnitude is
uncalibrated. Every rejection in the spec is analytic; no rollouts were run.

## [2026-07-31] query | cartpole RL vs dm_control cartpole-swingup

Compared this repo's cartpole + SB3 SAC training (physical regime, sparse
cost, curriculum) against dm_control's cartpole-swingup, verifying upstream
constants from google-deepmind/dm_control main (`suite/cartpole.py`,
`cartpole.xml`). Same force authority (±10 N), pole mass and cos/sin
observation; structurally different on termination (they have none),
start distribution (single hanging state vs full box + curriculum) and
success metric (return vs terminal goal-ball / invariant ellipsoid). Filed as
`.claude/docs/dm-control-cartpole.md`. Noted `Cost.SHAPED` copies their
tolerance-product structure but no config trains with it yet.

## [2026-07-30] ingest | composite env ids, eval CLI, and two bugs they surfaced

Source: `docs/superpowers/specs/2026-07-30-rl-training-and-eval-design.md` and
its implementation.

Taken from it. Four composite `(system, task)` env ids
(`{cartpole,inverted_pendulum,quadrotor2d,quadrotor3d}_stabilization`), because
`--task` was carrying both the registry id and the `Task` enum at once and a run
directory named `quadrotor_3` identified neither. They need no plumbing:
`configuration.py:67` already resolves an id to its yaml. An RL-Zoo run layout,
`<output_dir>/<algo>/<env_id>_<run>/` with `config.yml`/`args.yml`/`command.txt`.
A new `eval_policy.py` scoring a policy against its system's LQR from identical
seeded initial states. Per-system training configs under `configs/sb3/`.

Two bugs the work surfaced, both recorded with their measurement:

- `base_aviary.py`'s `changeDynamics` omitted `physicsClientId`, so with two
  envs alive the second wrote damping to the first's client. The quadrotor
  collectors hold exactly two, and the one they roll out is the second — so
  every shipped quadrotor dataset ran at PyBullet's default damping instead of
  zero. Rollout-env deviation from a single-env reference: `0.069001` without
  the fix, `0.000000` with it. Fixed, because `EvalCallback` holds a second env
  and would otherwise train against corrupted dynamics. Fixtures deliberately
  not regenerated; the three quadrotor slices are `xfail(strict=True)` so
  regenerating them fails the suite.
- `info['goal_reached']` is gated on `COST == Cost.QUADRATIC`, so it is absent
  under `rl_reward` for cartpole and both quadrotors. An earlier draft of the
  spec claimed all four expose it uniformly; evaluation built on that would have
  reported zero success for three of four systems while looking healthy.
  `eval_policy` computes success from the state instead, at the terminal step.

Changed: `architecture.md` (registry section — composite ids and the
faithful-copy invariant), `datasets.md` (no shared collector output contract,
the out-of-repo backfill scripts, `cal_set`/`test_set` having no producer, and
the damping finding), `workflows.md` (four new test files, the run layout, the
eval CLI), `INDEX.md` (three summaries).

## [2026-07-29] ingest | SB3-to-native pendulum exporter + wiki staleness check

Filed `scripts/export_sb3_pendulum.py` into `workflows.md` and the closed
train->export->run->collect loop into `architecture.md`. Added a staleness
advisory to `wiki_lint.py`: it reports source commits landed since the wiki was
last touched, and `session_start.sh` surfaces it every session. Advisory, not a
failure -- whether a source change needs a wiki edit is a judgement call, and
failing the lint on every commit would train people to ignore it.

## [2026-07-29] ingest | gymnasium/SB3 migration lands in architecture, workflows, glossary, compute

The prior entry below described the migration's code but only actually filed
the quad2d ellipsoid finding into `docs/datasets.md`; `architecture.md`,
`workflows.md` and `glossary.md` still described the pre-migration 4-tuple
world with zero mentions of gymnasium, SB3 or `terminated`/`truncated` (the
plan's own "Follow-on work" section says as much). This entry does that
ingest, from `docs/superpowers/specs/2026-07-28-sb3-gymnasium-migration-design.md`,
`plans/sb3-gymnasium-migration.md`, `git log --oneline main..HEAD` (31 commits),
and the landed code.

`architecture.md`: added the Gymnasium 5-tuple / `terminated`+`truncated`
section (two termination sources, one truncation source, `info['TimeLimit.truncated']`
kept for six controllers), a "Two RL stacks coexist" section (native
controllers unchanged; SB3 confined to `train_sb3.py`; no exporter yet), and
package-map rows for `train_sb3.py`, `forwarding.py`, `shaping.py`. Also fixed
a claim that predates this migration and that the new addition would have
made worse to leave: "import-only; no CLI entry points" was already false —
`train_rl_controller.py` has had a `__main__` block all along.

`workflows.md`: corrected the pendulum test bar to **75 passed, 0 known
failures** (verified directly) — the old "74 + 1 known failure" was a broken
editable install (`safe_control_gym.pth` pointed at a deleted sibling clone),
fixed by `pip install -e .`, not a real failure. Documented the new
`tests/test_envs/` oracle directory (golden rollouts, dataset-slice at
`atol=1e-12`, truncation semantics, wrapper forwarding, `check_env`
conformance, episode-flag initialisation, SB3 training smoke) and why
`test_invariant_sets.py` is deliberately absent from it. Added the
`train_sb3.py` command with the corrected GPU measurement.

`glossary.md`: added `terminated`/`truncated` and `check_env`.

`compute.md`: replaced the unmeasured "RL training wants a GPU" framing with
the measured number — SB3 SAC on the pendulum, `net_arch [256, 256]`, idle
ilab2, threads pinned: **cpu 65.6 steps/s vs cuda 111.0 steps/s, GPU 1.69x
faster** — and the correction of the earlier wrong claim (asserted from
general principle, first re-check ran on a loaded host and measured
contention, not devices).

`datasets.md` was checked against the sources and found already consistent
(the quad2d finding and the read-only `invariant_sets/*.npz` note match the
current code and the absence of `tests/test_envs/test_invariant_sets.py`); no
edit needed there this time.

Found in passing, not part of this migration: an untracked
`scripts/export_sb3_pendulum.py` and `tests/test_envs/test_export_sb3_pendulum.py`
exist in the working tree but are not committed to any branch. The spec and
plan are explicit that no exporter is in scope and a trained SB3 policy has no
in-repo consumer yet; those uncommitted files look like separate, unlanded
work on that follow-on and were not treated as source for this ingest.

`INDEX.md` summaries updated for architecture.md, workflows.md, compute.md and
glossary.md to match.

## [2026-07-29] ingest | quad2d ellipsoid finding

Migrated all four envs to the Gymnasium 5-tuple and added task-agnostic SB3
training (`plans/sb3-gymnasium-migration.md`). Filed the quad2d invariant-set
finding into `docs/datasets.md`: its ellipsoid is not reproducible because a
ReLU policy puts a crease inside the finite-difference stencil, and that is
expected rather than a defect, because `validate()` is what establishes
invariance. Human ruled no investigation needed; recorded so it is not
re-derived.

## [2026-07-28] lint | split machine-specific facts out of the wiki

Moved the conda env path and the cluster/machine guidance to CLAUDE.local.md
(gitignored) ahead of opening a PR against the shared repo. Added two checks to
wiki_lint.py: committed pages may name no absolute path except DATA_ROOT, and
every page must carry a title, a "Load when" line and a Related line -- the
second added after a bad test edit truncated a page and every existing check
still passed. Fixed conventions.md and glossary.md, which had never matched the
"Load when" convention.

## [2026-07-28] lint | bootstrap pass

First run of `.claude/wiki_lint.py`. 6 pages, 8 constants verified against
`generate_inverted_pendulum_trajectories.py`. Clean.

## [2026-07-28] ingest | karpathy/llm-wiki gist

Read `gist.github.com/karpathy/442a6bf555914893e9891c11519de94f`. Restructured
the wiki to the pattern it describes: `CLAUDE.md` became the schema (three
layers, three operations), `INDEX.md` became a content catalog rather than a
static repo map, added this log and `wiki_lint.py`.

Deviations recorded in `CLAUDE.md` under "Deviations from the pattern": raw
sources here are the repo itself rather than a curated document collection, and
the index is hand-ordered by task rather than by entity.

## [2026-07-28] ingest | plans/invariant-terminal-sets-recollection.md

Non-normality excursion gains (pendulum 2.6, quad2D 3.2, quad3D 4.9, cartpole
5.4), the discrete-Lyapunov ellipsoid construction, and the fixed-horizon
rationale filed into `docs/datasets.md` and `docs/glossary.md`.

## [2026-07-28] ingest | docs/superpowers/specs/2026-07-25-noisy-pendulum-collection-design.md

Train/eval split semantics, the mean-SE stopping rule, the entry-cut success
rule, the float32-vs-int16 measurement, the half-open grid fix, and the
atomic-publication scheme filed into `docs/datasets.md`. Terms filed into
`docs/glossary.md`.

## [2026-07-28] ingest | repository bootstrap

Initial pass over `safe_control_gym/`, the root generators,
`.pre-commit-config.yaml`, `tests/`, and the git log. Produced
`docs/architecture.md`, `docs/workflows.md`, `docs/conventions.md`,
`docs/compute.md`.
