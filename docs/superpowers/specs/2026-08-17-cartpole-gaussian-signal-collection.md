# Cartpole gaussian_signal: design and Amarel runbook

**Date:** 2026-08-17
**Status:** SUPERSEDED below — the levels in "Collect these three" are wrong and
must not be collected. See "2026-08-17 revision" for what replaces them.
**Scope:** `cp_collect.py`, `cp_gauss_sweep.py`, `cp_interior_sweep.py`,
`scripts/sbatch_cartpole_gauss_sweep.sh`,
`scripts/sbatch_cartpole_interior_sweep.sh`,
`safe_control_gym/envs/gym_control/cartpole.py`

A cartpole stochastic family whose noise scale grows with the commanded force,
the counterpart to the pendulum's `gaussian_signal`. Intended to be run on Amarel
by someone with the full 6720-core allocation.

---

## 2026-08-17 revision: two findings that invalidate the sections below

**1. The env could not step at all.** `cartpole.py::_get_reward` branched on
`Cost.SHAPED_DMC` and `Cost.SHAPED`, and `inverted_pendulum.py` on
`Cost.SHAPED`, as the *first* test in the function. Neither member existed in
the enum, so the attribute lookup raised `AttributeError: SHAPED_DMC` on every
step of every rollout, in every cost mode including the `quadratic` this
collection uses. `_shaped_reward()` is defined on no branch in this
repository's history.

Fixed: `SHAPED_DMC` added to the enum (`_shaped_dmc_reward()` is fully
implemented, so the dispatch was the only missing piece); the `SHAPED` branches
removed from both envs. The gate then reproduces the deterministic labels 58/58.

Note the implication for "What the smoke test already showed": that table cannot
have been produced by this commit. Its numbers are from a checkout that predates
the breakage, which is the stale-checkout hazard `.claude/docs/compute.md`
documents.

**2. The chosen levels do not deliver the variance they claim.** The calibration
rests on `E|u| = 1.581`, `E[u^2] = 26.436`. Measured over 25,845 control steps
of the noiseless LQR on 320 uniformly-drawn cells of the actual eval grid:

| | spec | measured | ratio |
| --- | --- | --- | --- |
| `E|u|` | 1.581 | 6.234 | 3.94x |
| `E[u^2]` | 26.436 | 552.975 | 20.92x |
| p50 | 0.27 | 0.526 | |
| p99 | 28.9 | 137.6 | |

Since delivered std is `sqrt(alpha^2 + 2*alpha*beta*E|u| + beta^2*E[u^2])`, the
three levels deliver **3.41x** their target — 15.7 / 21.6 / 35.4 N against the
intended 4.62 / 6.35 / 10.39 N. The ratio is identical across levels because
both constants scale with `k`, which is a self-consistency check on the
measurement. Confirmed end to end: instrumenting the disturbance during real
rollouts gives 13.6 / 17.4 / 24.3 N on a smaller sample, while the uniform
family measures 4.64 / 6.39 / 10.50 N on the same cells — exactly as claimed,
because its noise is state-independent.

### What replaces it: match the pendulum's uncertainty band

The requirement is no longer "same variance as the uniform cartpole levels" but
"same *noisiness* as the pendulum `gaussian_signal` set" — the separatrix blur
that makes those `p_success` fields look the way they do. The measurable form of
that is the **interior fraction**: cells with `0 < p < 1` at K = 100.

From `stochastic/pendulum/gaussian_signal/lqr/README.md`, reproduced exactly
from each level's `eval_success_prob.npz`:

| level | alpha | beta | interior | mean p |
| --- | --- | --- | --- | --- |
| low | 0.05 | 0.16 | 11.2% | 0.3869 |
| med | 0.10 | 0.64 | 64.6% | 0.4067 |
| high | 0.20 | 1.00 | 82.4% | 0.5457 |

Those three numbers are the targets. `cp_interior_sweep.py` searches for the
cartpole `(alpha, beta)` that hit them, and
`scripts/sbatch_cartpole_interior_sweep.sh` runs it.

Two things it does differently from `cp_gauss_sweep.py`:

- **Uniform sampling, not stratified.** Interior is a whole-grid rate; the 50/50
  stratification by deterministic label over-represents the boundary, which is
  where interior cells are, so its interior number is not comparable to the
  pendulum's.
- **The ratio `alpha/beta` is swept, not assumed.** It sets the character:
  `alpha` is all that survives at the goal. Variance-matching forces
  `alpha = 3.80*beta`, which makes `alpha` dominate everywhere but the tail —
  i.e. nearly constant noise, discarding the one property that distinguishes
  this family. At the median command (0.53 N) a pendulum-like balance wants
  `alpha ~ 0.27*beta`. The sweep covers 0.27, 1.00 and 3.80 so the choice is
  read off measurements rather than assumed.

Levels are therefore **not yet chosen**, and the sweep is a prerequisite again,
not optional confirmation.

### Also corrected in the runbook

- `--cpus-per-task=64` is wrong for `main-redhat`, which is heterogeneous:
  excluding `halk*` it holds 44 nodes of 32 cores, 50 of 40, 48 of 52 and 198 of
  64. Pinning 64 makes an array schedulable on 58% of the partition. Use
  `--exclusive` and size the pool from `nproc`; the collectors already read
  `sched_getaffinity`.
- The sbatch scripts hardcoded `/home/dm1487` for the repo, interpreter, log
  directory and scratch root, so every array task failed immediately for any
  other account. Now `CP_REPO`/`PYTHON` with per-user defaults.
- `CP_SIGMA0` is required by `sbatch_cartpole_gauss_collect.sh` but never read
  by `cp_collect.py`; only `cp_gauss_sweep.py` uses it. Copying `sigma_0.npz` is
  needed for the old sweep, not for collection.

---

## Collect these three

```
level   alpha    beta    delivered std
low     2.413    0.635      4.62 N
med     3.317    0.873      6.35 N
high    5.428    1.429     10.39 N
```

Matched in **delivered standard deviation** to the published uniform cartpole
levels (sigma 8 / 11 / 18). Derived from the measured command distribution, not
guessed, and all three are smoke-tested — see "What the smoke test already
showed". The sweep in the runbook is **optional confirmation**, not a
prerequisite: go straight to collection unless you want the matched-*difficulty*
numbers instead, which are a different set (`k ~ 0.9 / 1.1 / 1.8`, estimated
rather than measured).

Why matched variance rather than matched difficulty: it makes the comparison
against the uniform family answer a clean question — does *when* the noise
arrives matter, holding how much constant? The smoke test says yes, emphatically
(24-77% more success at the same variance). Matched difficulty is the better
choice only if these are meant as drop-in replacements for a downstream model,
in which case run the sweep and read the levels off it.

Command form:

```
--alpha 2.413 --beta 0.635      # low
--alpha 3.317 --beta 0.873      # med
--alpha 5.428 --beta 1.429      # high
```

## The noise

```
xdot = f(x, sat(u + w)),   w ~ Normal(0, alpha + beta*|u|)
```

`alpha + beta*|u|` is a **standard deviation**, not a variance. `alpha` is the
floor surviving as the command goes to zero; `beta` is effort-proportional.

Delivered noise is a **scale mixture** — every draw has its own sigma — so its
standard deviation is `sqrt(E[sigma^2])`, NOT `E[sigma]`. At `alpha = 0` those
differ by 4x (measured 10.76 against 2.52). Quoting `E[sigma]` badly understates
a level.

## Why placement does not appear here

The pendulum family's central distinction — noise inside vs outside the actuator
saturation — is **absent on cartpole**, and the collector does not offer it.

Measured over 16,494 control steps: the LQR's demand is a median of **0.27 N**
against an `action_scale` of **2000 N**, p99 28.9, max 57.8. It never saturates.
So `sat(u + w)` and `sat(u) + w` are the same function, and the clip is inert.

That is also why this system's noise can *rescue* failing states under
pre-saturation noise, which the pendulum's cannot: the pendulum is saturated
70-98% of steps, and a saturated clip discards every positive draw. The published
uniform cartpole family already gains 743-911 cells for exactly this reason.

`action_scale = 2000` is inherited from the deterministic cartpole set and is not
physically motivated — a 1 kg cart does not have a 2000 N actuator. The clip
would only start binding near 10 N (3% of steps) or 1 N (22%).

## How the action reaches the pole

The action is a force on the **cart**. It reaches the **pole** through a coupling
proportional to `cos(theta)`, finite-differenced through the simulator:

| theta (deg) | d(theta_ddot)/dF | relative | cos(theta) |
| --- | --- | --- | --- |
| 0 | -1.463 | 1.000 | 1.000 |
| 30 | -1.244 | 0.850 | 0.866 |
| 60 | -0.693 | 0.474 | 0.500 |
| 85 | -0.119 | 0.081 | 0.087 |
| 89 | -0.023 | 0.016 | 0.017 |

With the pole horizontal a cart force does nothing to it. So **constant force
noise is already a state-dependent disturbance on the pole**, before any
signal-dependence is added. Sign: positive force gives negative `theta_ddot` —
the pole is driven by the cart accelerating out from under it.

## Levels: one knob

Two free constants make a poor ladder. Fixing the share of noise **variance**
coming from the signal term at 50% fixes the ratio, against the measured command
distribution (`E|u| = 1.581`, `E[u^2] = 26.436` under the noiseless LQR):

```
beta = k,   alpha = 3.80 * k
```

Delivered std is then linear in `k`:

| k | alpha | beta | delivered std | sigma at median \|u\| | sigma at p99 \|u\| |
| --- | --- | --- | --- | --- | --- |
| 0.318 | 1.208 | 0.318 | 2.31 N | 1.29 | 10.4 |
| 0.635 | 2.413 | 0.635 | 4.62 N | 2.58 | 20.8 |
| 0.873 | 3.317 | 0.873 | 6.35 N | 3.55 | 28.6 |
| 1.429 | 5.428 | 1.429 | 10.39 N | 5.81 | 46.7 |

The three middle rows are **matched in delivered standard deviation** to the
published uniform levels `low`/`med`/`high` (sigma 8/11/18 = 4.62/6.35/10.39 N).
"Matched" means the second moment only — uniform is bounded and flat in time,
this is unbounded and concentrated in the transient.

## What the smoke test already showed

240 stratified eval cells, K = 10. Gate `alpha = beta = 0` reproduced the
`sigma_0` labels 240/240.

| k | gaussian p | uniform counterpart | uniform p | gaussian broke | uniform broke |
| --- | --- | --- | --- | --- | --- |
| 0.635 | 0.4567 | low (sigma 8) | 0.3692 | 63 | 109 |
| 0.873 | 0.3804 | med (sigma 11) | 0.2592 | 110 | 120 |
| 1.429 | 0.1825 | high (sigma 18) | 0.1029 | 120 | 120 |

**At matched variance the gaussian family is substantially gentler** — 24%, 47%
and 77% more success — and the gap widens with strength. Timing matters, not just
variance: what kills a run is noise *at* the goal preventing entry into the 0.05
ball, and this family goes quiet exactly there.

Consequence: **matching variance does not match difficulty.** To place the
gaussian levels at the same `p_success` as the uniform ones, `k` is roughly
`{0.9, 1.1, 1.8}` — read off the sweep, not from this table.

Caveats on those numbers: the 240 cells are stratified 50/50 by deterministic
label, so `p` is not comparable to the published full-grid values, only within
the table. K = 10 is too small to measure `gained` at all.

## Runbook (Amarel)

**Use the whole allocation.** The QOS allows 6720 CPUs and 500 submitted jobs;
the sizings below deliberately spend most of that, and there is no reason to be
modest. Every stage here is embarrassingly parallel — shards are independent,
`rollout_seed` is a pure function of its coordinates, so the same rollouts happen
whichever node runs them and a shard can be re-run without changing a result.

Practically that means: prefer more array tasks over more cores per task
(`--nodes=1 --exclusive --cpus-per-task=64` per task, then as many tasks as the
work divides into), raise `NSHARDS` rather than waiting, and resubmit stragglers
instead of letting one slow shard hold the campaign. 80 tasks is 5,120 cores and
still leaves headroom; a full collection can go to 105 tasks (6,720 cores) before
hitting the cap. If a stage looks like it will take an hour, shard it further —
the only real cost of another shard is one more file to merge.

Two caveats on going wide. The 500-job submit cap is per user and counts array
elements, so a single array of 105 is fine but several large arrays at once will
be rejected — submit them in sequence or use a deferred launcher. And exclude the
`halk*` nodes (below); adding nodes that produce nothing makes a campaign slower,
not faster, because the work silently never happens.

### 0. Inputs the cluster cannot see

Amarel has no access to the iLab filesystem. Two files must be copied across,
and both are read through environment variables.

Run this **from an iLab machine** (arrakis, ilab*, rlab*) — Amarel cannot see
`/common/users/shared`, which is the whole reason the copy is needed.

```bash
D=/common/users/shared/pracsys/genMoPlan/data_trajectories
ssh amarel mkdir -p '~/cpdet'
scp $D/deterministic/cartpole_pybullet/eval_states.txt \
    amarel:cpdet/eval_states.txt
scp $D/stochastic/cartpole/noisy_torque/archive/sigma_0/eval_success_prob.npz \
    amarel:cpdet/sigma_0.npz
```

`eval_states.txt` is 8.7 MB (116,242 rows), `sigma_0.npz` 5.4 MB. Verify with
`ssh amarel 'ls -l ~/cpdet'` before submitting — a missing file fails every
array task identically and looks like a cluster problem.

Then on Amarel, for every job:

```bash
export CP_DET_DIR=$HOME/cpdet          # supplies eval_states.txt
export CP_SIGMA0=$HOME/cpdet/sigma_0.npz   # deterministic labels
```

Without these the job fails immediately with `eval_states.txt not found` —
which is the correct behaviour, but check for it before assuming a node problem.

### 1. Sweep to choose levels

```bash
cd ~/scg-repo
sbatch --exclude="$(sinfo -p main-redhat -h -o '%n' | grep '^halk' | sort -u | paste -sd, -)" \
       --export=ALL,MODE=collect,CP_GAUSS_ROOT=/scratch/$USER/cp_gauss_sweep,\
CP_DET_DIR=$HOME/cpdet,CP_SIGMA0=$HOME/cpdet/sigma_0.npz \
       scripts/sbatch_cartpole_gauss_sweep.sh
```

80 tasks (10 configs x 8 shards), 64 cores each = **5,120 cores**, roughly
**10-15 minutes**. Then:

```bash
MODE=merge CP_GAUSS_ROOT=/scratch/$USER/cp_gauss_sweep \
    bash scripts/sbatch_cartpole_gauss_sweep.sh
```

Read `low`/`med`/`high` off that table by whichever criterion is wanted —
matched variance or matched difficulty. **Levels are never chosen a priori
here**: they are coupled to the success rule and the horizon and transfer across
neither.

### 2. Collect

One array, all three levels, the whole allocation:

```bash
sbatch --exclude="$(sinfo -p main-redhat -h -o '%n' | grep '^halk' | sort -u | paste -sd, -)" \
       --export=ALL,CP_DET_DIR=$HOME/cpdet,CP_SIGMA0=$HOME/cpdet/sigma_0.npz,\
CP_GAUSS_OUT=/scratch/$USER/cartpole_gaussian_signal \
       scripts/sbatch_cartpole_gauss_collect.sh
```

105 tasks x 64 cores = **6,720 cores**, the full QOS. Per level 5 train shards
and 30 eval shards; eval is sharded far harder because it is 99% of the work.
The alpha/beta values are baked into the script — edit `ALPHAS`/`BETAS` there if
the sweep is run and different levels are wanted.

Shards are idempotent, so a partial run resubmits verbatim.

The underlying calls, if a shard needs running by hand:

```bash
python cp_collect.py --split train --alpha A --beta B \
    --shard S --nshards N --out <dir>/train_sS.npz
python cp_collect.py --split eval  --alpha A --beta B --trials 100 \
    --shard S --nshards N --out <dir>/eval_sS.npz
```

### 3. Cost

A cartpole rollout under noise costs about **1.07 core-seconds** — 1000 control
steps at 1.067 ms each, every one stepping PyBullet 50 times at `pyb_freq 5000`.
That is ~15x the pendulum's per-rollout cost.

| stage | rollouts | core-hours | wall clock at 6720 cores |
| --- | --- | --- | --- |
| sweep (10 configs, 10k cells, K=20) | 2.0M | ~600 | ~10 min |
| collect 3 levels, eval K=100 | 34.9M | ~10,400 | ~1.6 h |
| collect 3 levels, train | 0.35M | ~100 | minutes |

The collection figure is an **upper bound**: it assumes every rollout runs the
full horizon, whereas noiseless ones average 94 steps and heavily-noised ones
often die early on the out-of-bounds thresholds. The expensive case is the
middle.

Those wall-clock figures assume the allocation is actually used. At 64 cores on
one node the same collection is ~160 hours, so the difference between spreading
across the cluster and not is the difference between an afternoon and a week.
Shard the eval split by at least 20 ways per level, and run levels concurrently
rather than in sequence.

### 4. Operational notes

- **Exclude `halk*` nodes.** They run and write nothing; this cost several
  hours across two campaigns. The `--exclude` above is not optional.
- **Never pipe `git pull` through `tail`.** An aborted merge prints its error on
  stderr and `Updating A..B` on stdout, so `tail -1` reads as success. Check
  `git rev-parse HEAD` against `origin/<branch>` after pulling.
- **Shards are idempotent** — an existing `--out` is skipped, so a partial run
  can be resubmitted as-is.
- `output_dir` must be node-local (`$SLURM_TMPDIR` or `/tmp`); `cp_collect`
  already does this. Cartpole reloads its URDF on reset and on NFS that is a
  15-45x penalty.

## Rejected

**Keying the scale on `|u|` without rescaling.** The pendulum uses `beta ~ 1`
because `|u|` there sits pinned at `u_sat = 0.637`. On cartpole `|u|` has a
median of 0.27 N and a p99 of 28.9, so the same `beta` contributes ~0.02 N and
does nothing. The measured distribution is what sets the ratio here, not the
pendulum's value.

**Reducing `action_scale` to make the clip bind.** It would recreate the
pendulum's saturation structure, and it is a different plant — every published
cartpole dataset uses 2000 N. Out of scope; noted because it is the only way to
make placement meaningful on this system.

**Matching the two families by difficulty rather than variance.** Defensible if
the datasets are meant to be drop-in substitutes for a downstream model. Not
chosen because matched variance makes the comparison answer a clean question:
does *when* the noise arrives matter, holding how much constant? The answer,
from the smoke test, is yes.
