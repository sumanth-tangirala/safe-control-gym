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

The one thing that genuinely wants a GPU is RL *training*
(`safe_control_gym/experiments/train_rl_controller.py`), not collection.

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

Related: [workflows.md](workflows.md) for the commands, [datasets.md](datasets.md) for the sizes those commands write, `CLAUDE.local.md` for which machine (uncommitted, may not exist here).
