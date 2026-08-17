#!/bin/bash
#SBATCH --job-name=cp_gauss
#SBATCH --partition=main-redhat
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --array=0-79
#SBATCH --output=logs/cp_gauss_%A_%a.out

# Cartpole gaussian_signal level sweep: sigma = alpha + beta*|u| on the cart
# force, against the published uniform family on the same cells.
#
# 10 configs x 8 shards = 80 tasks, 64 cores each = 5,120 cores.
#
# Sized from measurement, not guesswork: a cartpole rollout under noise runs to
# the horizon far more often than a noiseless one, so it costs ~1.07 s of a core
# (1000 control steps x 1.07 ms, each stepping PyBullet 50 times at
# pyb_freq 5000). The full 116,242-cell grid at K = 20 would be ~2,450
# core-hours PER CONFIG; a 10,000-cell stratified subsample is ~210, which
# across 8 shards is around 7-15 minutes a shard.
#
# The subsample is stratified by the deterministic label because the cartpole
# grid is 82% deterministic failures -- a uniform draw would spend most of the
# budget re-confirming that the far exterior stays dead.

set -euo pipefail

CP_REPO=${CP_REPO:-$HOME/Projects/safe-control-gym}
cd "$CP_REPO"
mkdir -p logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY=${PYTHON:-$HOME/miniforge3/envs/scg/bin/python}
MODE=${MODE:-collect}
ROOT=${CP_GAUSS_ROOT:-/scratch/$USER/cartpole_gauss_sweep}
NSHARDS=8

mkdir -p "$ROOT"

if [ "$MODE" = "merge" ]; then
    $PY cp_gauss_sweep.py --merge --out-dir "$ROOT"
    exit 0
fi

I=$SLURM_ARRAY_TASK_ID
CFG=$((I / NSHARDS))
SHARD=$((I % NSHARDS))
LABEL=$($PY -c "import cp_gauss_sweep as m; print(m.CONFIGS[$CFG][0])")
OUT=$ROOT/${LABEL}_s$(printf '%02d' "$SHARD").npz

echo "task=$I config=$CFG ($LABEL) shard=$SHARD host=$(hostname) start=$(date -Is)"
$PY cp_gauss_sweep.py --config "$CFG" --shard "$SHARD" --nshards "$NSHARDS" --out "$OUT"
echo "task=$I done=$(date -Is)"
