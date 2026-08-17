#!/bin/bash
#SBATCH --job-name=cp_iconf
#SBATCH --partition=main-redhat
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/cp_iconf_%A_%a.out

# Find the cartpole gaussian_signal levels whose uncertainty band matches the
# pendulum's: interior (0 < p < 1 at K=100) of 11.2% / 64.6% / 82.4%.
#
# Two knobs. The RATIO alpha/beta sets the character -- how much noise survives
# at the goal, where |u| -> 0 -- and the SCALE sets the strength. This sweeps a
# log ladder in beta at three ratios, because the right ratio is exactly what is
# not known a priori:
#
#   0.27  beta dominates at the median command (0.53 N), so the noise goes
#         quiet at the goal. This is the pendulum-like character.
#   1.00  the two terms are comparable there.
#   3.80  what matching delivered variance produces -- alpha dominates
#         everywhere except the tail, so the noise is nearly constant. Included
#         to show what that choice costs, not because it is expected to win.
#
# 11 configs x 8 shards = 88 tasks. Each config is 2,000 uniformly-sampled
# cells at K=100 = ~59 core-hours, so the sweep is ~650 core-hours total; a few
# minutes of wall clock across the partition.
#
# NOT --cpus-per-task=64: main-redhat is heterogeneous (44 nodes x 32 cores,
# 50 x 40, 48 x 52, 198 x 64 excluding halk). Pinning 64 makes the array
# schedulable on 58% of the partition and leaves the rest idle. --exclusive
# takes whole nodes of any size and NPROC below adapts to what arrived.
#
# BEFORE SUBMITTING: CP_DET_DIR must point at a directory holding
# eval_states.txt -- Amarel cannot see the shared dataset root. Without it every
# task fails identically and it reads like a cluster fault.
#
#   sbatch --exclude="$(sinfo -p main-redhat -h -o '%n' | grep '^halk' \
#            | sort -u | paste -sd, -)" \
#          --export=ALL,CP_DET_DIR=$HOME/cpdet \
#          scripts/sbatch_cartpole_interior_sweep.sh
#
# Shards are idempotent -- an existing --out is skipped.

set -euo pipefail

CP_REPO=${CP_REPO:-$HOME/Projects/safe-control-gym}
cd "$CP_REPO"
mkdir -p logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NPROC=${NPROC:-$(nproc)}

PY=${PYTHON:-$HOME/miniforge3/envs/scg/bin/python}
ROOT=${CP_SWEEP_OUT:-/scratch/$USER/cp_interior_confirm}
: "${CP_DET_DIR:?set CP_DET_DIR to the directory holding eval_states.txt}"
mkdir -p "$ROOT"

# alpha beta, one config per line, index = config number.
CONFIGS=(
  "0.340   1.26"     # candidate low: interpolated to interior 0.112, confirm it
  "0.300   1.10"     # one step either side, in case the interpolation is off
)
NSHARDS=8

I=$SLURM_ARRAY_TASK_ID
C=$((I / NSHARDS))
S=$((I % NSHARDS))
read -r A B <<< "${CONFIGS[$C]}"

echo "task=$I config=$C alpha=$A beta=$B shard=$S host=$(hostname) cores=$NPROC start=$(date -Is)"

$PY cp_interior_sweep.py --alpha "$A" --beta "$B" \
    --shard "$S" --nshards "$NSHARDS" \
    --out "$ROOT/a${A}_b${B}_s$(printf '%02d' "$S").npz"

echo "task=$I done=$(date -Is)"
