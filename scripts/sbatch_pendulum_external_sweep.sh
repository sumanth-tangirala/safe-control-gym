#!/bin/bash
#SBATCH --job-name=pend_ext
#SBATCH --partition=main-redhat
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --array=0-31
#SBATCH --output=/home/dm1487/scg-repo/logs/pend_ext_%A_%a.out

# Level sweep for the EXTERNAL-torque pendulum family: sat(u) + w, where w is a
# torque on the shaft rather than part of the actuator, so u_sat does not bound
# it. See docs/superpowers/specs for the design.
#
# This sweeps the FULL 49,770-cell eval grid rather than a subsample. A 2,000-cell
# estimate costs ~1/25th as much, which mattered when a sweep was the expensive
# step; against 6720 cores the whole grid is a few minutes, and it gives exact
# per-cell gains instead of an estimate of them. Gains are the reason this family
# exists, and they are concentrated in whatever region of the grid the
# deterministic controller fails in -- exactly the structure a uniform subsample
# smears out.
#
# K = 20 rather than 100: enough to resolve gains and rank levels, a fifth of the
# cost. The chosen levels get K = 100 at collection time.
#
# 8 levels x 4 shards x 5 batches = 32 tasks.

set -euo pipefail

cd /home/dm1487/scg-repo

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY=${PYTHON:-/home/dm1487/envs/scg/bin/python}
MODE=${MODE:-collect}
ROOT=${PEND_EXT_ROOT:-/scratch/dm1487/pendulum_external_sweep_20260815}
ALPHA=${ALPHA:-0.008}
NCPU=$($PY -c 'import os; print(len(os.sched_getaffinity(0)))')

# An order of magnitude below the internal family's levels: the same w is far
# more potent outside the clip, because none of it is discarded. beta ~ sigma at
# saturation as a fraction of u_sat, so 0.16 is a disturbance worth ~16% of the
# motor's authority and 1.6 is one that outguns it.
BETAS=(0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.60)
NAMES=(0.010 0.020 0.040 0.080 0.160 0.320 0.640 1.600)
NLEVEL=${#BETAS[@]}
SHARDS=4
BATCHES_PER_SHARD=5

eval_shard() {
    local beta=$1 out=$2 offset=$3 count=$4
    $PY generate_inverted_pendulum_trajectories.py \
        --split eval --controller lqr \
        --noise_alpha "$ALPHA" --noise_beta "$beta" --external_noise \
        --output_dir "$out" --seed 42 --horizon 800 \
        --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
        --parallel --num_workers "$NCPU" \
        --batch_offset "$offset" --batch_count "$count"
}

echo "mode=$MODE task=${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) cores=$NCPU start=$(date -Is)"

case "$MODE" in
    collect)
        I=$SLURM_ARRAY_TASK_ID
        T=$((I / SHARDS))
        S=$((I % SHARDS))
        OUT=$ROOT/beta_${NAMES[$T]}
        mkdir -p "$OUT"
        echo "beta=${BETAS[$T]} shard=$S batches=[$((S * BATCHES_PER_SHARD)),$(((S + 1) * BATCHES_PER_SHARD)))"
        eval_shard "${BETAS[$T]}" "$OUT" "$((S * BATCHES_PER_SHARD))" "$BATCHES_PER_SHARD"
        ;;
    finalize)
        for T in $(seq 0 $((NLEVEL - 1))); do
            OUT=$ROOT/beta_${NAMES[$T]}
            $PY generate_inverted_pendulum_trajectories.py \
                --split eval --controller lqr \
                --noise_alpha "$ALPHA" --noise_beta "${BETAS[$T]}" --external_noise \
                --output_dir "$OUT" --seed 42 --horizon 800 \
                --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
                --se_tol 0.01 --min_batches 5 --max_batches 20 --check_every 5 \
                --merge_eval_shards
        done
        $PY - "$ROOT" <<'PY'
import json
import os
import sys

import numpy as np

root = sys.argv[1]
DET = ('/common/users/shared/pracsys/genMoPlan/data_trajectories/stochastic/pendulum/'
       'noisy_torque/lqr/tau_0.00/eval_success_prob.npz')
det = np.load(DET)['p_success'] > 0
print(f'{"beta":>7} {"sigma_sat":>10} {"p":>8} {"interior":>9} {"gain cells":>11} '
      f'{"gain rate":>10} {"mean p|fail":>12}')
for name in sorted(os.listdir(root)):
    d = os.path.join(root, name)
    z = np.load(os.path.join(d, 'eval_success_prob.npz'))
    ed = json.load(open(os.path.join(d, 'eval_description.json')))
    assert ed['signal_noise']['placement'] == 'sat(u) + w', ed['signal_noise']['placement']
    p = z['p_success']
    gain = (~det) & (p > 0)
    print(f'{ed["signal_noise"]["beta"]:7.3f} {ed["signal_noise"]["sigma_at_u_sat"]:10.4f} '
          f'{p.mean():8.4f} {((p > 0) & (p < 1)).mean():9.4f} {int(gain.sum()):11d} '
          f'{gain.mean():10.4%} {p[~det].mean():12.5f}')
PY
        ;;
    *)
        echo "unknown MODE=$MODE (expected collect or finalize)" >&2
        exit 2
        ;;
esac

echo "mode=$MODE task=${SLURM_ARRAY_TASK_ID:-none} done=$(date -Is)"
