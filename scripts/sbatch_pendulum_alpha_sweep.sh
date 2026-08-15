#!/bin/bash
#SBATCH --job-name=pend_alpha
#SBATCH --partition=main-redhat
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --array=0-95
#SBATCH --output=/home/dm1487/scg-repo/logs/pend_alpha_%A_%a.out

# Alpha sweep for the external-torque family, at each of the three betas that
# will be collected. Full 49,770-cell grid, K = 20, so the alpha axis is measured
# on the same footing as the beta axis already was.
#
# alpha and beta do different things and only alpha acts AT THE GOAL. beta scales
# with the commanded torque, so it goes quiet exactly where a stabilising
# controller is finishing; alpha is the floor that survives as u -> 0. Whether
# the settled region fits inside the 0.05 success box is therefore an alpha
# question and not a beta question at all.
#
# The range brackets the point where the floor reaches the box. Under torque
# noise the settled spread is |theta_dot| <~ 0.70*alpha, so alpha ~ 0.07 is where
# it touches the 0.05 edge. But these datasets score ENTRY with no dwell, so a
# floor may well help a trajectory stumble into the box before it starts
# preventing it from staying -- which is a measurement, not a prediction.
#
# 3 betas x 8 alphas x 4 shards x 5 batches = 96 tasks.

set -euo pipefail

cd /home/dm1487/scg-repo

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY=${PYTHON:-/home/dm1487/envs/scg/bin/python}
MODE=${MODE:-collect}
ROOT=${PEND_ALPHA_ROOT:-/scratch/dm1487/pendulum_alpha_sweep_20260815}
NCPU=$($PY -c 'import os; print(len(os.sched_getaffinity(0)))')

BETAS=(0.16 0.64 1.00)
BNAMES=(0.160 0.640 1.000)
ALPHAS=(0.000 0.008 0.020 0.050 0.100 0.200 0.400 0.800)
NALPHA=${#ALPHAS[@]}
SHARDS=4
BATCHES_PER_SHARD=5

echo "mode=$MODE task=${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) cores=$NCPU start=$(date -Is)"

case "$MODE" in
    collect)
        I=$SLURM_ARRAY_TASK_ID
        S=$((I % SHARDS))
        C=$((I / SHARDS))              # combo index
        B=$((C / NALPHA))
        A=$((C % NALPHA))
        OUT=$ROOT/b${BNAMES[$B]}_a${ALPHAS[$A]}
        mkdir -p "$OUT"
        echo "beta=${BETAS[$B]} alpha=${ALPHAS[$A]} shard=$S"
        $PY generate_inverted_pendulum_trajectories.py \
            --split eval --controller lqr \
            --noise_alpha "${ALPHAS[$A]}" --noise_beta "${BETAS[$B]}" --external_noise \
            --output_dir "$OUT" --seed 42 --horizon 800 \
            --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
            --parallel --num_workers "$NCPU" \
            --batch_offset "$((S * BATCHES_PER_SHARD))" --batch_count "$BATCHES_PER_SHARD"
        ;;
    finalize)
        for B in $(seq 0 $((${#BETAS[@]} - 1))); do
            for A in $(seq 0 $((NALPHA - 1))); do
                OUT=$ROOT/b${BNAMES[$B]}_a${ALPHAS[$A]}
                [ -d "$OUT" ] || continue
                $PY generate_inverted_pendulum_trajectories.py \
                    --split eval --controller lqr \
                    --noise_alpha "${ALPHAS[$A]}" --noise_beta "${BETAS[$B]}" \
                    --external_noise \
                    --output_dir "$OUT" --seed 42 --horizon 800 \
                    --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
                    --se_tol 0.01 --min_batches 5 --max_batches 20 --check_every 5 \
                    --merge_eval_shards >/dev/null
                echo "merged $OUT"
            done
        done
        ;;
    *)
        echo "unknown MODE=$MODE (expected collect or finalize)" >&2
        exit 2
        ;;
esac

echo "mode=$MODE task=${SLURM_ARRAY_TASK_ID:-none} done=$(date -Is)"
