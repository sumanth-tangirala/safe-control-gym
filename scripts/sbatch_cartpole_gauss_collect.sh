#!/bin/bash
#SBATCH --job-name=cp_gcol
#SBATCH --partition=main-redhat
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --array=0-104
#SBATCH --output=/home/dm1487/scg-repo/logs/cp_gcol_%A_%a.out

# Full collection of the cartpole gaussian_signal family at the three chosen
# levels. sigma = alpha + beta*|u| on the commanded cart force.
#
#   low    alpha 2.413  beta 0.635    delivered std  4.62 N
#   med    alpha 3.317  beta 0.873                   6.35 N
#   high   alpha 5.428  beta 1.429                  10.39 N
#
# Matched in delivered standard deviation to the published uniform levels
# (sigma 8 / 11 / 18). See
# docs/superpowers/specs/2026-08-17-cartpole-gaussian-signal-collection.md.
#
# 3 levels x (5 train shards + 30 eval shards) = 105 tasks x 64 cores = 6,720
# cores -- the whole allocation, which is the point. Eval is sharded 30 ways
# because it is 99% of the work: 116,242 cells x K=100 x 3 levels = 34.9M
# rollouts at ~1.07 core-seconds each.
#
# BEFORE SUBMITTING: this cluster cannot see the shared dataset root, so
# CP_DET_DIR (holding eval_states.txt) and CP_SIGMA0 (the deterministic labels)
# must both be set and the files copied across. Without them every task fails
# identically with 'eval_states.txt not found', which reads like a node fault.
#
#   sbatch --exclude="$(sinfo -p main-redhat -h -o '%n' | grep '^halk' \
#            | sort -u | paste -sd, -)" \
#          --export=ALL,CP_DET_DIR=$HOME/cpdet,CP_SIGMA0=$HOME/cpdet/sigma_0.npz \
#          scripts/sbatch_cartpole_gauss_collect.sh
#
# Shards are idempotent -- an existing --out is skipped -- so a partial run can
# be resubmitted verbatim.

set -euo pipefail

cd /home/dm1487/scg-repo

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY=${PYTHON:-/home/dm1487/envs/scg/bin/python}
ROOT=${CP_GAUSS_OUT:-/scratch/$USER/cartpole_gaussian_signal}
: "${CP_DET_DIR:?set CP_DET_DIR to the directory holding eval_states.txt}"
: "${CP_SIGMA0:?set CP_SIGMA0 to the sigma_0 eval_success_prob.npz}"

NAMES=(low med high)
ALPHAS=(2.413 3.317 5.428)
BETAS=(0.635 0.873 1.429)
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
