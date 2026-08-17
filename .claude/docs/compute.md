# Compute

Load when deciding *where* a job runs.

This page says what this repo's jobs **need**. It deliberately names no
machines — that is site-specific and belongs in `CLAUDE.local.md`, which is not
committed. Read that first if it exists on this checkout.

## What this repo's jobs actually need

**Collection is CPU-bound and embarrassingly parallel.** The generators fan out
with `multiprocessing.Pool` over `--num_workers`; `get_available_cpus()` respects
process affinity, so a `taskset`- or SLURM-restricted allocation is honoured
automatically. Core count is the thing to optimise for, not GPU memory.

- Pendulum LQR rollouts: pure numpy + the env step. No GPU.
- Pendulum RL: small native-torch MLP policies (`.pt` state-dicts) evaluated
  per step. CPU is fine and usually faster than paying transfer overhead.
- Quadrotor collection: PyBullet stepping, CPU.
- `compute_invariant_sets.py`: finite differencing plus a discrete Lyapunov
  solve and boundary sampling. Minutes, single machine, no GPU.

The one thing that genuinely wants a GPU is RL *training* — either
`safe_control_gym/experiments/train_rl_controller.py` (the native stack) or
`safe_control_gym/experiments/train_sb3.py` (stable-baselines3) — not
collection.

For SB3 specifically, "wants" is now measured rather than assumed: SAC on the
pendulum, `net_arch: [256, 256]`, measured back to back on an idle ilab2 with
threads pinned, is **1.69x faster on GPU** (cpu 65.6 steps/s, cuda 111.0
steps/s). An earlier draft of `train_sb3.py`'s docstring asserted the opposite
from the general principle that small MLP policies favour CPU — that was
wrong, and the first attempt to check it ran on a loaded host and measured
contention rather than devices. `--use_gpu` still defaults off; pass it when a
GPU is actually free.

## Storage

Everything written goes to `DATA_ROOT`
(`/common/users/shared/pracsys/genMoPlan/data_trajectories`, set in the
generators), which must be visible from wherever the job runs. A train split is
roughly 1.6 GB at
300k trajectories and the observed mean of ~671 states each; an eval split is
about 1.6 MB. Check free space before launching a fleet of train splits.

## Job shape

Train and eval splits are independent processes and are meant to run
concurrently — two jobs, not one job with two phases.

Eval is safe to preempt: the published dataset is the checkpoint and
`dataset_description.json` records `converged: false`. That makes it a good fit
for a preemptible or short-partition allocation. Train is not checkpointed the
same way; size its wall-clock request accordingly.

---

Related: [workflows.md](workflows.md) for the commands, [architecture.md](architecture.md) for the two RL stacks this measures, [datasets.md](datasets.md) for the sizes those commands write, `CLAUDE.local.md` for which machine (uncommitted, may not exist here).

## The cartpole writes a URDF on every reset

`cartpole.py::reset` writes a per-process URDF into `output_dir` each time it is
called, so a collection performs one small file write per rollout. On shared NFS
that dominated everything else — measured 15-45x slower than the same job with
`output_dir` on node-local disk (60-180 min per batch against 4.5 min). Set it to
`$SLURM_TMPDIR` or `/tmp`, and note that `$TMPDIR` on some hosts points back at
NFS, so it is not a safe default. The directory must exist: `reset` raises
`FileNotFoundError` from `ElementTree.write` rather than creating it.

Applies to cartpole only; the quadrotors load a static URDF from the package.

## Per-rollout cost differs 15x between systems

Sizing a campaign from the pendulum's numbers underestimates cartpole badly.

| | ctrl_freq | pyb_freq | substeps/step | horizon | cost per rollout |
| --- | --- | --- | --- | --- | --- |
| pendulum | 100 | 300 | 3 | 800 | ~0.07 core-s |
| cartpole | 100 | 5000 | 50 | 1000 | ~1.07 core-s |

Measured: a cartpole control step is 1.067 ms, `reset()` only 0.71 ms (2% of a
rollout — it is not the URDF reload, which is the intuitive but wrong suspect).
There is a second multiplier: a *noiseless* cartpole rollout averages 94 steps
because it reaches the goal fast, while a noisy one runs far closer to the cap,
so adding noise makes each rollout roughly 10x dearer on top of the 50 substeps.

Consequence: three cartpole levels at K = 100 over its 116,242-cell grid is
~10,400 core-hours, against ~700 for the equivalent pendulum campaign. Quote it
as an upper bound — heavily-noised rollouts often die early on the out-of-bounds
thresholds rather than timing out.

## Inputs a cluster cannot see

Amarel has its own filesystem and cannot read `/common/users/shared`, where the
deterministic reference sets live. Any collector that compares against them must
take a path override, and the files must be copied across first — quad3d needed
this, and cartpole needs `CP_DET_DIR` plus `CP_SIGMA0`. The failure is loud
(`eval_states.txt not found`) but identical on every array task, so it reads like
a cluster fault rather than a missing input.

## A job that runs the wrong code

Distinct from the failure modes below: the job runs, exits clean, and produces
plausible output — from a stale checkout. Seen 2026-08-15, where a sweep
"reproduced" the previous sweep's numbers to four decimals because the cluster's
copy of the repo was three commits behind.

The cause was a `git pull` that had aborted:

```
error: Your local changes to the following files would be overwritten by merge:
        safe_control_gym/envs/gym_control/inverted_pendulum.py
Aborting
Updating 2e3dda0e..1ef51d2c
```

The error goes to **stderr** and the reassuring `Updating A..B` to **stdout**, so
`git pull ... | tail -1` shows only the latter and reads as success. Two rules
follow: never pipe a pull through `tail`, and before submitting, assert the code
is actually there — check `git rev-parse HEAD` against `origin/<branch>` and
import the symbol the job depends on. A remote checkout accumulates local edits
from previous debugging and will block a fast-forward eventually.

## Three ways a scheduler reports success and produces nothing

Measured during the 2026-08-14/15 collections. All three look identical to job
state: `COMPLETED`, exit code 0, or a state that dependencies treat as fine. None
is detectable from `sacct` alone.

**Nodes that cannot write.** A subset of the CPU partition ran tasks to
completion and wrote **no output file and no data shard** — 14 of 200 in one
array, with perfect correlation to node name prefix (163/163 shards from one
prefix, 0/14 from the other). `/home` and the compute-node mount were the same
filesystem with identical contents and there were no partial `.tmp` files, so it
was not a path or partial-write problem. Consequences: `--requeue` never fires,
because nothing failed; and `afterok` dependencies are *satisfied* by the empty
tasks, so the next stage launches on incomplete input. Mitigation is
`--exclude=<those nodes>`, applied to pending arrays with
`scontrol update jobid=<id> ExcNodeList=...` before they start.

**Preemption without requeue.** Arrays submitted without `--requeue` lose
preempted tasks silently — the shard is simply absent and the state is
`PREEMPTED`, which is easy to omit from a failure filter. Six tasks were lost this
way, two of them at 4h17m of a ~6h run. Always set `--requeue`, and include
`PREEMPT` and `CANCELLED` in any watchdog pattern.

**Stdout that never lands.** A 57-minute job exited `COMPLETED` having produced
no output file at all, losing results that existed only as printed text. Anything
that is a deliverable must be written by the job itself — npz, not stdout.

**The check that catches all three** is comparing *expected shards against shards
on disk*, not job states. The operational signal is a queue count that falls while
the shard count does not. Every collector here writes one file per shard and skips
a shard whose file already exists, which makes recovery a matter of resubmitting
the same array — finished work is never redone.

Two scheduler limits worth knowing before sizing an array: a per-user cap on
*submitted* jobs (500 here) that rejects the submission outright rather than
queueing it, and a CPU cap that is the real concurrency limit. Sizing a wave to
land exactly on the CPU cap is worthwhile; a deferred submitter job is the way to
chain past the submit cap when one collection must follow another.
