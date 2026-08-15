#!/bin/bash
#SBATCH --job-name=pend_sig_sweep
#SBATCH --partition=main-redhat
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/dm1487/scg-repo/logs/pend_sig_sweep_%A_%a.out

# Beta sweep for the signal-dependent pendulum family, one beta per array task.
#
# One task per level rather than one process looping over levels: the levels are
# independent, and a stalled or preempted level then costs only itself. Each
# task writes its own npz, so a partial sweep is still a readable curve.

set -euo pipefail

cd /home/dm1487/scg-repo

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY=${PYTHON:-/home/dm1487/envs/scg/bin/python}
ROOT=${SIG_SWEEP_ROOT:-/scratch/dm1487/pendulum_signal_sweep_20260815}
N_CELLS=${N_CELLS:-2000}
K=${K:-20}
BETAS=(0.04 0.1 0.2 0.4 0.8 1.6 3.2 6.4)

mkdir -p "$ROOT"
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}
NCPU=$($PY -c 'import os; print(len(os.sched_getaffinity(0)))')

echo "task=$SLURM_ARRAY_TASK_ID beta=$BETA cells=$N_CELLS K=$K cores=$NCPU host=$(hostname) start=$(date -Is)"

SIG_SWEEP_OUT="$ROOT/beta_${BETA}.npz" \
    $PY pend_sig_sweep.py "$N_CELLS" "$K" "$BETA"

echo "task=$SLURM_ARRAY_TASK_ID beta=$BETA done=$(date -Is)"
