#!/bin/bash
#SBATCH --job-name=cp_gcol
#SBATCH --partition=main-redhat
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --array=0-104
#SBATCH --output=logs/cp_gcol_%A_%a.out

# Full collection of the cartpole gaussian_signal family at the three chosen
# levels. sigma = alpha + beta*|u| on the commanded cart force.
#
#   low    alpha 0.405  beta 1.500    mid 0.0380   interior 0.1375
#   med    alpha 0.945  beta 3.500    mid 0.1270   interior 0.2165
#   high   alpha 1.080  beta 4.000    mid 0.1520   interior 0.2120
#
# Matched on MID -- the fraction of eval cells with p in [0.2, 0.8] at K=100 --
# against the pendulum gaussian_signal set's 0.0376 / 0.1334 / 0.6342. low is a
# direct match and med is within 5%; high is not reachable and 0.1520 is this
# ratio's ceiling.
#
# Not matched on `interior` (0 < p < 1), which was tried and abandoned: it
# counts uncertain cells without regard to how uncertain they are. A level with
# interior 0.1115, matching the pendulum low exactly, turned out to have 84.9%
# of those cells below 0.1 or above 0.9 with a median of 0.98 -- a hard edge
# with one frayed pixel, not a gradient.
#
# Ratio alpha = 0.27*beta throughout. A large alpha is a constant floor that
# converts p=1 cells into p=0.98, which inflates `interior` cheaply but adds no
# depth; beta acts in the transient and genuinely randomises the outcome. The
# ratio is also what keeps the ladder monotone in delivered noise: at matched
# mid, ratio 3.80 reaches the same haze at a third of the noise, so mixing the
# two would put med above high in strength.
#
# Every value above is measured at 2,000 uniformly-sampled cells, K=100, by
# cp_interior_sweep.py + cp_sweep_mid.py. See
# docs/superpowers/specs/2026-08-17-cartpole-gaussian-signal-collection.md.
#
# 3 levels x (5 train shards + 30 eval shards) = 105 tasks x 64 cores = 6,720
# cores -- the whole allocation, which is the point. Eval is sharded 30 ways
# because it is 99% of the work: 116,242 cells x K=100 x 3 levels = 34.9M
# rollouts at ~1.07 core-seconds each.
#
# BEFORE SUBMITTING: this cluster cannot see the shared dataset root, so
# CP_DET_DIR (holding eval_states.txt) must be set and the file copied across.
# Without it every task fails identically with 'eval_states.txt not found',
# which reads like a node fault. CP_SIGMA0 is NOT needed here -- cp_collect.py
# never reads it; only cp_gauss_sweep.py does.
#
#   sbatch --exclude="$(sinfo -p main-redhat -h -o '%n' | grep '^halk' \
#            | sort -u | paste -sd, -)" \
#          --export=ALL,CP_DET_DIR=$HOME/cpdet \
#          scripts/sbatch_cartpole_gauss_collect.sh
#
# Shards are idempotent -- an existing --out is skipped -- so a partial run can
# be resubmitted verbatim.

set -euo pipefail

# Repo and interpreter are per-user; override with CP_REPO / PYTHON rather than
# editing this file. The defaults are the layout compute.md documents for this
# account (dedicated clone under ~/Projects, miniforge in ~).
CP_REPO=${CP_REPO:-$HOME/Projects/safe-control-gym}
cd "$CP_REPO"
mkdir -p logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# NOT $(nproc): it honours OMP_NUM_THREADS, which is pinned to 1 four lines
# above, so it returns 1 and the whole node runs a single worker.
export NPROC=${NPROC:-${SLURM_CPUS_ON_NODE:-$(nproc --all)}}

PY=${PYTHON:-$HOME/miniforge3/envs/scg/bin/python}
ROOT=${CP_GAUSS_OUT:-/scratch/$USER/cartpole_gaussian_signal}
: "${CP_DET_DIR:?set CP_DET_DIR to the directory holding eval_states.txt}"

NAMES=(low med high)
ALPHAS=(0.405 0.945 1.080)
BETAS=(1.500 3.500 4.000)
TRAIN_SHARDS=5
EVAL_SHARDS=30
PER_LEVEL=$((TRAIN_SHARDS + EVAL_SHARDS))

I=$SLURM_ARRAY_TASK_ID
L=$((I / PER_LEVEL))
J=$((I % PER_LEVEL))
NAME=${NAMES[$L]}
A=${ALPHAS[$L]}
B=${BETAS[$L]}
OUT=$ROOT/$NAME
mkdir -p "$OUT"

echo "task=$I level=$NAME alpha=$A beta=$B host=$(hostname) start=$(date -Is)"

if [ "$J" -lt "$TRAIN_SHARDS" ]; then
    $PY cp_collect.py --split train --alpha "$A" --beta "$B" \
        --shard "$J" --nshards "$TRAIN_SHARDS" \
        --out "$OUT/train_s$(printf '%02d' "$J").npz"
else
    S=$((J - TRAIN_SHARDS))
    $PY cp_collect.py --split eval --alpha "$A" --beta "$B" --trials 100 \
        --shard "$S" --nshards "$EVAL_SHARDS" \
        --out "$OUT/eval_s$(printf '%02d' "$S").npz"
fi

echo "task=$I done=$(date -Is)"
