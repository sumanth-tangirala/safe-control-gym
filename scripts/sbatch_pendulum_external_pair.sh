#!/bin/bash
#SBATCH --job-name=pend_pair
#SBATCH --partition=main-redhat
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --array=0-20
#SBATCH --output=/home/dm1487/scg-repo/logs/pend_pair_%A_%a.out

# One (alpha, beta) pair of the external-torque pendulum family, full collection.
#
# Generalised from sbatch_pendulum_external_ab.sh, which hard-codes the three
# published pairs. That script stays as the record of what those were; this one
# takes ALPHA and BETA from the environment so a single extra level can be added
# without editing a committed level list.
#
#   ALPHA=0.2 BETA=1.0 sbatch scripts/sbatch_pendulum_external_pair.sh
#   MODE=finalize ALPHA=0.2 BETA=1.0 bash scripts/sbatch_pendulum_external_pair.sh
#
# Everything matches the other pendulum families so they stay comparable
# cell-for-cell: horizon 800, ctrl_freq 100, pyb_freq 300, seed 42, resolution
# 0.04, 100k train trajectories, K = 100 eval trials, entry-cut box rule.
#
# 1 train + 20 eval shards x 5 batches = 21 tasks.

set -euo pipefail

cd /home/dm1487/scg-repo

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY=${PYTHON:-/home/dm1487/envs/scg/bin/python}
MODE=${MODE:-collect}
ALPHA=${ALPHA:?set ALPHA}
BETA=${BETA:?set BETA}
ROOT=${PEND_PAIR_ROOT:-/scratch/dm1487/pendulum_external_pair_20260816/lqr}
NCPU=$($PY -c 'import os; print(len(os.sched_getaffinity(0)))')

# Name carries BOTH constants -- the earlier tree named levels beta_<b> with
# alpha implicit, which silently misleads the moment alpha varies.
# printf, not python: an f-string with nested single quotes is a syntax error
# before python 3.12, and this ran on 3.10.
NAME=$(printf 'a%.3f_b%.3f' "$ALPHA" "$BETA")
OUT=$ROOT/$NAME
SHARDS=20
BATCHES_PER_SHARD=5

echo "mode=$MODE task=${SLURM_ARRAY_TASK_ID:-none} alpha=$ALPHA beta=$BETA name=$NAME host=$(hostname) cores=$NCPU start=$(date -Is)"

mkdir -p "$OUT"

case "$MODE" in
    collect)
        I=$SLURM_ARRAY_TASK_ID
        if [ "$I" -eq 0 ]; then
            $PY generate_inverted_pendulum_trajectories.py \
                --split train --controller lqr \
                --noise_alpha "$ALPHA" --noise_beta "$BETA" --external_noise \
                --output_dir "$OUT" --seed 42 --horizon 800 \
                --ctrl_freq 100 --pyb_freq 300 --num_trajs 100000 \
                --parallel --num_workers "$NCPU"
        else
            S=$((I - 1))
            $PY generate_inverted_pendulum_trajectories.py \
                --split eval --controller lqr \
                --noise_alpha "$ALPHA" --noise_beta "$BETA" --external_noise \
                --output_dir "$OUT" --seed 42 --horizon 800 \
                --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
                --parallel --num_workers "$NCPU" \
                --batch_offset "$((S * BATCHES_PER_SHARD))" \
                --batch_count "$BATCHES_PER_SHARD"
        fi
        ;;
    finalize)
        $PY generate_inverted_pendulum_trajectories.py \
            --split eval --controller lqr \
            --noise_alpha "$ALPHA" --noise_beta "$BETA" --external_noise \
            --output_dir "$OUT" --seed 42 --horizon 800 \
            --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
            --se_tol 0.01 --min_batches 10 --max_batches 100 --check_every 10 \
            --merge_eval_shards
        $PY prepare_stochastic_layout.py --root "$ROOT" --n_cal 1000
        $PY - "$OUT" "$ALPHA" "$BETA" <<'PY'
import json
import os
import sys

import numpy as np

out, alpha, beta = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
train = np.load(os.path.join(out, 'train.npz'))
ev = np.load(os.path.join(out, 'eval_success_prob.npz'))
td = json.load(open(os.path.join(out, 'train_description.json')))
ed = json.load(open(os.path.join(out, 'eval_description.json')))
assert len(train['labels']) == 100_000, len(train['labels'])
assert len(ev['p_success']) == 49_770, len(ev['p_success'])
assert np.all(ev['trials'] == 100), np.unique(ev['trials'])
for d in (td, ed):
    sn = d['signal_noise']
    assert abs(sn['alpha'] - alpha) < 1e-12, sn['alpha']
    assert abs(sn['beta'] - beta) < 1e-12, sn['beta']
    assert sn['placement'] == 'sat(u) + w', sn['placement']
    assert d['ctrl_freq'] == 100 and d['pyb_freq'] == 300
    assert d['horizon_steps'] == 800
assert os.path.exists(os.path.join(out, 'dataset_description.json'))
assert os.path.exists(os.path.join(out, 'train_test_splits', 'shuffled_indices_0.txt'))
p = ev['p_success']
print(f'{os.path.basename(out)} verified alpha={alpha} beta={beta} '
      f'train_success={float(train["labels"].mean()):.4f} '
      f'eval_p={float(p.mean()):.4f} '
      f'interior={float(((p > 0) & (p < 1)).mean()):.4f}')
PY
        ;;
    *)
        echo "unknown MODE=$MODE (expected collect or finalize)" >&2
        exit 2
        ;;
esac

echo "mode=$MODE task=${SLURM_ARRAY_TASK_ID:-none} done=$(date -Is)"
