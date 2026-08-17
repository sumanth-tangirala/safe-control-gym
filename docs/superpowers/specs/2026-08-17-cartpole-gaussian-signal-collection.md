# Cartpole gaussian_signal: design and Amarel runbook

**Date:** 2026-08-17
**Status:** ready to run
**Scope:** `cp_collect.py`, `cp_gauss_sweep.py`, `scripts/sbatch_cartpole_gauss_sweep.sh`,
`safe_control_gym/envs/gym_control/cartpole.py`

A cartpole stochastic family whose noise scale grows with the commanded force,
the counterpart to the pendulum's `gaussian_signal`. Intended to be run on Amarel
by someone with the full 6720-core allocation.

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

### 0. Inputs the cluster cannot see

Amarel has no access to the iLab filesystem. Two files must be copied across,
and both are read through environment variables:

```bash
# on a machine that can see /common/users/shared:
D=/common/users/shared/pracsys/genMoPlan/data_trajectories
scp $D/deterministic/cartpole_pybullet/eval_states.txt \
    amarel:~/cpdet/eval_states.txt
scp $D/stochastic/cartpole/noisy_torque/archive/sigma_0/eval_success_prob.npz \
    amarel:~/cpdet/sigma_0.npz
```

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

`cp_collect.py` takes `--alpha/--beta` in place of `--level`. Per level:

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
