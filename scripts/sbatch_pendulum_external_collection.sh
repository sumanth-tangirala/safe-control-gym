#!/bin/bash
#SBATCH --job-name=pend_extc
#SBATCH --partition=main-redhat
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --array=0-104
#SBATCH --output=/home/dm1487/scg-repo/logs/pend_extc_%A_%a.out

# Collect the EXTERNAL-torque pendulum family: sat(u) + w, alpha = 0.008, beta
# swept. w is a torque on the shaft rather than part of the actuator, so u_sat
# does not bound it and the applied torque can exceed the motor's limit.
#
# Levels come from the full-grid sweep (job 60604165), K = 20 over all 49,770
# cells, against a deterministic p of 0.3860:
#
#   beta        0.010  0.020  0.040  0.080  0.160  0.320  0.640  1.600
#   sigma/u_sat  2.3%   3.3%   5.3%   9.3%  17.3%  33.3%  65.3% 161.3%
#   mean p      .3860  .3860  .3860  .3861  .3864  .3878  .3962  .6502
#   gain rate    0.42%  0.57%  0.92%  1.66%  3.08%  6.44% 17.27% 61.40%
#
# Unlike every family collected before it, this one GAINS cells: start states the
# deterministic controller fails from that noise rescues. Gains and losses nearly
# cancel at the low end -- 824 gained against 772 lost at beta = 0.08 -- so mean
# p is preserved while thousands of individual cells change status. The noisy
# region of attraction is a RESHAPING of the deterministic one rather than an
# erosion of it, which is the property the internal family cannot produce.
#
# beta = 1.6 is deliberately past the point where the disturbance outguns the
# motor (161% of u_sat, p = 0.65 against a deterministic 0.3860). It is included
# as the noise-dominated extreme and its description says so; it is not a
# credible external-disturbance magnitude.
#
# Everything else matches the other pendulum families so they stay comparable
# cell-for-cell: horizon 800, ctrl_freq 100, pyb_freq 300, seed 42, resolution
# 0.04, 100k train trajectories, K = 100 eval trials.
#
# 5 levels x (1 train + 20 eval shards) = 105 tasks.

set -euo pipefail

cd /home/dm1487/scg-repo

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY=${PYTHON:-/home/dm1487/envs/scg/bin/python}
MODE=${MODE:-collect}
ROOT=${PEND_EXT_ROOT:-/scratch/dm1487/pendulum_external_20260815/lqr}
ALPHA=${ALPHA:-0.008}
NCPU=$($PY -c 'import os; print(len(os.sched_getaffinity(0)))')

BETAS=(0.08 0.16 0.32 0.64 1.60)
NAMES=(0.080 0.160 0.320 0.640 1.600)
NLEVEL=${#BETAS[@]}
SHARDS=20
BATCHES_PER_SHARD=5

collect_train() {
    local beta=$1 out=$2 n_traj=$3
    $PY generate_inverted_pendulum_trajectories.py \
        --split train --controller lqr \
        --noise_alpha "$ALPHA" --noise_beta "$beta" --external_noise \
        --output_dir "$out" --seed 42 --horizon 800 \
        --ctrl_freq 100 --pyb_freq 300 --num_trajs "$n_traj" \
        --parallel --num_workers "$NCPU"
}

collect_eval_shard() {
    local beta=$1 out=$2 offset=$3 count=$4
    $PY generate_inverted_pendulum_trajectories.py \
        --split eval --controller lqr \
        --noise_alpha "$ALPHA" --noise_beta "$beta" --external_noise \
        --output_dir "$out" --seed 42 --horizon 800 \
        --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
        --parallel --num_workers "$NCPU" \
        --batch_offset "$offset" --batch_count "$count"
}

merge_eval() {
    local beta=$1 out=$2
    $PY generate_inverted_pendulum_trajectories.py \
        --split eval --controller lqr \
        --noise_alpha "$ALPHA" --noise_beta "$beta" --external_noise \
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
expected = {'beta_0.080': 0.08, 'beta_0.160': 0.16, 'beta_0.320': 0.32,
            'beta_0.640': 0.64, 'beta_1.600': 1.6}
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
        assert abs(d['signal_noise']['beta'] - beta) < 1e-12
        # The whole point of the family -- assert it rather than trust the flag.
        assert d['signal_noise']['placement'] == 'sat(u) + w', d['signal_noise']['placement']
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
