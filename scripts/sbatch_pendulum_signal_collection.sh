#!/bin/bash
#SBATCH --job-name=pend_sig
#SBATCH --partition=main-redhat
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --array=0-125
#SBATCH --output=/home/dm1487/scg-repo/logs/pend_sig_%A_%a.out

# Collect the signal-dependent pendulum family: alpha fixed at 0.008, beta swept.
#
# Levels come from pend_sig_sweep.py (job 60602794), not from a guess -- they are
# coupled to the success rule and the horizon and do not transfer across either.
# Measured p on a 2000-cell subsample against a deterministic 0.3860:
#
#   beta   0.04    0.1     0.2     0.4     0.8     1.6     3.2     6.4
#   p      0.3722  0.3491  0.3129  0.2739  0.1958  0.1033  0.0370  0.0139
#   int.   0.014   0.024   0.033   0.048   0.090   0.128   0.102   0.067
#
# 0.04 and 0.1 are dropped as too close to deterministic and 6.4 as p = 0.014
# with a collapsing interior fraction. beta = 0 is kept at alpha = 0.008: it is
# the control this sweep needs, a constant sigma floor with no signal
# dependence, which separates what beta adds from what the floor does. The
# deterministic reference is the published tau_0.00 and is not recollected.
#
# Everything else matches noisy_torque/: horizon 800, ctrl_freq 100, pyb_freq
# 300, seed 42, resolution 0.04, 100k train trajectories, K = 100 eval trials.

set -euo pipefail

cd /home/dm1487/scg-repo

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY=${PYTHON:-/home/dm1487/envs/scg/bin/python}
MODE=${MODE:-collect}
ROOT=${PEND_SIG_ROOT:-/scratch/dm1487/pendulum_signal_dependent_20260815/lqr}
ALPHA=${ALPHA:-0.008}
NCPU=$($PY -c 'import os; print(len(os.sched_getaffinity(0)))')

BETAS=(0 0.2 0.4 0.8 1.6 3.2)
NAMES=(0.000 0.200 0.400 0.800 1.600 3.200)
NLEVEL=${#BETAS[@]}
SHARDS=20
BATCHES_PER_SHARD=5

collect_train() {
    local beta=$1 out=$2 n_traj=$3
    $PY generate_inverted_pendulum_trajectories.py \
        --split train --controller lqr \
        --noise_alpha "$ALPHA" --noise_beta "$beta" \
        --output_dir "$out" --seed 42 --horizon 800 \
        --ctrl_freq 100 --pyb_freq 300 --num_trajs "$n_traj" \
        --parallel --num_workers "$NCPU"
}

collect_eval_shard() {
    local beta=$1 out=$2 batch_offset=$3 batch_count=$4
    $PY generate_inverted_pendulum_trajectories.py \
        --split eval --controller lqr \
        --noise_alpha "$ALPHA" --noise_beta "$beta" \
        --output_dir "$out" --seed 42 --horizon 800 \
        --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
        --parallel --num_workers "$NCPU" \
        --batch_offset "$batch_offset" --batch_count "$batch_count"
}

merge_eval() {
    local beta=$1 out=$2
    $PY generate_inverted_pendulum_trajectories.py \
        --split eval --controller lqr \
        --noise_alpha "$ALPHA" --noise_beta "$beta" \
        --output_dir "$out" --seed 42 --horizon 800 \
        --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
        --se_tol 0.01 --min_batches 10 --max_batches 100 --check_every 10 \
        --merge_eval_shards
}

echo "mode=$MODE task=${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) cores=$NCPU start=$(date -Is) root=$ROOT"

case "$MODE" in
    collect)
        I=$SLURM_ARRAY_TASK_ID
        if [ "$I" -lt "$NLEVEL" ]; then
            OUT=$ROOT/beta_${NAMES[$I]}
            mkdir -p "$OUT"
            collect_train "${BETAS[$I]}" "$OUT" 100000
        else
            E=$((I - NLEVEL))
            T=$((E / SHARDS))
            S=$((E % SHARDS))
            OUT=$ROOT/beta_${NAMES[$T]}
            mkdir -p "$OUT"
            collect_eval_shard "${BETAS[$T]}" "$OUT" "$((S * BATCHES_PER_SHARD))" \
                "$BATCHES_PER_SHARD"
        fi
        ;;
    finalize)
        for T in $(seq 0 $((NLEVEL - 1))); do
            merge_eval "${BETAS[$T]}" "$ROOT/beta_${NAMES[$T]}"
        done
        $PY prepare_stochastic_layout.py --root "$ROOT" --n_cal 1000
        $PY - "$ROOT" <<'PY'
import json
import os
import sys

import numpy as np

root = sys.argv[1]
expected = {'beta_0.000': 0.0, 'beta_0.200': 0.2, 'beta_0.400': 0.4,
            'beta_0.800': 0.8, 'beta_1.600': 1.6, 'beta_3.200': 3.2}
assert set(os.listdir(root)) == set(expected), sorted(os.listdir(root))
for name, beta in sorted(expected.items()):
    out = os.path.join(root, name)
    train = np.load(os.path.join(out, 'train.npz'))
    ev = np.load(os.path.join(out, 'eval_success_prob.npz'))
    td = json.load(open(os.path.join(out, 'train_description.json')))
    ed = json.load(open(os.path.join(out, 'eval_description.json')))
    assert len(train['labels']) == 100_000, len(train['labels'])
    assert len(ev['p_success']) == 49_770, len(ev['p_success'])
    assert np.all(ev['trials'] == 100), np.unique(ev['trials'])
    for d in (td, ed):
        assert d['signal_noise']['alpha'] == 0.008
        assert d['signal_noise']['beta'] == beta
        assert d['ctrl_freq'] == 100 and d['pyb_freq'] == 300
        assert d['horizon_steps'] == 800
    assert os.path.exists(os.path.join(out, 'dataset_description.json'))
    assert os.path.exists(os.path.join(out, 'train_test_splits', 'shuffled_indices_0.txt'))
    p = ev['p_success']
    print(f'{name} verified train_success={float(train["labels"].mean()):.4f} '
          f'eval_p={float(p.mean()):.4f} interior={float(((p > 0) & (p < 1)).mean()):.4f}')
PY
        ;;
    *)
        echo "unknown MODE=$MODE (expected collect or finalize)" >&2
        exit 2
        ;;
esac

echo "mode=$MODE task=${SLURM_ARRAY_TASK_ID:-none} done=$(date -Is)"
