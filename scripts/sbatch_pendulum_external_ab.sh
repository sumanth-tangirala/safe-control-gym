#!/bin/bash
#SBATCH --job-name=pend_ab
#SBATCH --partition=main-redhat
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --array=0-62
#SBATCH --output=/home/dm1487/scg-repo/logs/pend_ab_%A_%a.out

# Full collection of the external-torque family at three chosen (alpha, beta)
# pairs [user, 2026-08-15]:
#
#   alpha 0.050  beta 0.16     floor 0.035 vs a 0.05 box -- a real floor
#   alpha 0.008  beta 0.64     floor 0.006 -- effectively none, the control
#   alpha 0.100  beta 0.64     floor 0.070 -- floor at the box edge
#
# alpha and beta are varied together rather than as a grid, so the LEVEL
# DIRECTORY NAMES CARRY BOTH. The earlier external tree named levels beta_<b>
# with alpha implicit, which was true when alpha was fixed at 0.008 and becomes
# a trap the moment it is not.
#
# Measured on the full-grid alpha sweep (job 60605673), K = 20, against a
# deterministic p of 0.3860:
#
#   a0.050_b0.160   p .3869   interior  8.3%   rescued  2,200
#   a0.008_b0.640   p .3962   interior 26.4%   rescued  8,596
#   a0.100_b0.640   p .4066   interior 39.7%   rescued 14,430
#
# The middle pair is already collected at K = 100 under the old naming as
# beta_0.640. It is recollected here rather than copied: same seed, same
# parameters, so the result is bit-identical, and one clean provenance beats a
# tree that needs a footnote.
#
# Everything else matches the other pendulum families so they stay comparable
# cell-for-cell: horizon 800, ctrl_freq 100, pyb_freq 300, seed 42, resolution
# 0.04, 100k train trajectories, K = 100 eval trials.
#
# 3 pairs x (1 train + 20 eval shards) = 63 tasks.

set -euo pipefail

cd /home/dm1487/scg-repo

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY=${PYTHON:-/home/dm1487/envs/scg/bin/python}
MODE=${MODE:-collect}
ROOT=${PEND_AB_ROOT:-/scratch/dm1487/pendulum_external_ab_20260815/lqr}
NCPU=$($PY -c 'import os; print(len(os.sched_getaffinity(0)))')

ALPHAS=(0.050 0.008 0.100)
BETAS=(0.16 0.64 0.64)
NAMES=(a0.050_b0.160 a0.008_b0.640 a0.100_b0.640)
NPAIR=${#ALPHAS[@]}
SHARDS=20
BATCHES_PER_SHARD=5

collect_train() {
    local alpha=$1 beta=$2 out=$3 n_traj=$4
    $PY generate_inverted_pendulum_trajectories.py \
        --split train --controller lqr \
        --noise_alpha "$alpha" --noise_beta "$beta" --external_noise \
        --output_dir "$out" --seed 42 --horizon 800 \
        --ctrl_freq 100 --pyb_freq 300 --num_trajs "$n_traj" \
        --parallel --num_workers "$NCPU"
}

collect_eval_shard() {
    local alpha=$1 beta=$2 out=$3 offset=$4 count=$5
    $PY generate_inverted_pendulum_trajectories.py \
        --split eval --controller lqr \
        --noise_alpha "$alpha" --noise_beta "$beta" --external_noise \
        --output_dir "$out" --seed 42 --horizon 800 \
        --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
        --parallel --num_workers "$NCPU" \
        --batch_offset "$offset" --batch_count "$count"
}

merge_eval() {
    local alpha=$1 beta=$2 out=$3
    $PY generate_inverted_pendulum_trajectories.py \
        --split eval --controller lqr \
        --noise_alpha "$alpha" --noise_beta "$beta" --external_noise \
        --output_dir "$out" --seed 42 --horizon 800 \
        --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
        --se_tol 0.01 --min_batches 10 --max_batches 100 --check_every 10 \
        --merge_eval_shards
}

echo "mode=$MODE task=${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) cores=$NCPU start=$(date -Is) root=$ROOT"

case "$MODE" in
    collect)
        I=$SLURM_ARRAY_TASK_ID
        if [ "$I" -lt "$NPAIR" ]; then
            OUT=$ROOT/${NAMES[$I]}
            mkdir -p "$OUT"
            echo "train alpha=${ALPHAS[$I]} beta=${BETAS[$I]}"
            collect_train "${ALPHAS[$I]}" "${BETAS[$I]}" "$OUT" 100000
        else
            E=$((I - NPAIR))
            T=$((E / SHARDS))
            S=$((E % SHARDS))
            OUT=$ROOT/${NAMES[$T]}
            mkdir -p "$OUT"
            echo "eval alpha=${ALPHAS[$T]} beta=${BETAS[$T]} shard=$S"
            collect_eval_shard "${ALPHAS[$T]}" "${BETAS[$T]}" "$OUT" \
                "$((S * BATCHES_PER_SHARD))" "$BATCHES_PER_SHARD"
        fi
        ;;
    finalize)
        for T in $(seq 0 $((NPAIR - 1))); do
            merge_eval "${ALPHAS[$T]}" "${BETAS[$T]}" "$ROOT/${NAMES[$T]}"
        done
        $PY prepare_stochastic_layout.py --root "$ROOT" --n_cal 1000
        $PY - "$ROOT" <<'PY'
import json
import os
import sys

import numpy as np

root = sys.argv[1]
expected = {'a0.050_b0.160': (0.05, 0.16),
            'a0.008_b0.640': (0.008, 0.64),
            'a0.100_b0.640': (0.10, 0.64)}
assert set(os.listdir(root)) == set(expected), sorted(os.listdir(root))
for name, (alpha, beta) in sorted(expected.items()):
    out = os.path.join(root, name)
    train = np.load(os.path.join(out, 'train.npz'))
    ev = np.load(os.path.join(out, 'eval_success_prob.npz'))
    td = json.load(open(os.path.join(out, 'train_description.json')))
    ed = json.load(open(os.path.join(out, 'eval_description.json')))
    assert len(train['labels']) == 100_000, len(train['labels'])
    assert len(ev['p_success']) == 49_770, len(ev['p_success'])
    assert np.all(ev['trials'] == 100), np.unique(ev['trials'])
    for d in (td, ed):
        sn = d['signal_noise']
        # The directory name is the only place both constants appear together,
        # so check the name against the payload rather than trusting either.
        assert abs(sn['alpha'] - alpha) < 1e-12, (name, sn['alpha'])
        assert abs(sn['beta'] - beta) < 1e-12, (name, sn['beta'])
        assert sn['placement'] == 'sat(u) + w', sn['placement']
        assert d['ctrl_freq'] == 100 and d['pyb_freq'] == 300
        assert d['horizon_steps'] == 800
    assert os.path.exists(os.path.join(out, 'dataset_description.json'))
    assert os.path.exists(os.path.join(out, 'train_test_splits', 'shuffled_indices_0.txt'))
    p = ev['p_success']
    print(f'{name} verified alpha={alpha} beta={beta} '
          f'train_success={float(train["labels"].mean()):.4f} '
          f'eval_p={float(p.mean()):.4f} interior={float(((p > 0) & (p < 1)).mean()):.4f}')
PY
        ;;
    *)
        echo "unknown MODE=$MODE (expected collect or finalize)" >&2
        exit 2
        ;;
esac

echo "mode=$MODE task=${SLURM_ARRAY_TASK_ID:-none} done=$(date -Is)"
