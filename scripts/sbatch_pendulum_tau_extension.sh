#!/bin/bash
#SBATCH --job-name=pend_tau_ext
#SBATCH --partition=main-redhat
#SBATCH --account=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --array=0-62
#SBATCH --output=/home/dm1487/scg-repo/logs/pend_tau_ext_%A_%a.out

# Extend the active stochastic pendulum sweep with tau={1,2,5} without ever
# writing to the published dataset tree.  Collection and verification happen
# under Amarel scratch; publication to iLab is a separate guarded rsync.

set -euo pipefail

cd /home/dm1487/scg-repo

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY=${PYTHON:-/home/dm1487/envs/scg/bin/python}
MODE=${MODE:-collect}
ROOT=${PEND_TAU_ROOT:-/scratch/dm1487/pendulum_tau_extension_20260813/lqr}
NCPU=$($PY -c 'import os; print(len(os.sched_getaffinity(0)))')

TAUS=(1.0 2.0 5.0)
NAMES=(1.00 2.00 5.00)

collect_train() {
    local tau=$1
    local out=$2
    local n_traj=$3
    $PY generate_inverted_pendulum_trajectories.py \
        --split train --controller lqr --torque_noise "$tau" \
        --output_dir "$out" --seed 42 --horizon 800 \
        --ctrl_freq 100 --pyb_freq 300 --num_trajs "$n_traj" \
        --parallel --num_workers "$NCPU"
}

collect_eval_shard() {
    local tau=$1
    local out=$2
    local batch_offset=$3
    local batch_count=$4
    $PY generate_inverted_pendulum_trajectories.py \
        --split eval --controller lqr --torque_noise "$tau" \
        --output_dir "$out" --seed 42 --horizon 800 \
        --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
        --parallel --num_workers "$NCPU" \
        --batch_offset "$batch_offset" --batch_count "$batch_count"
}

merge_eval() {
    local tau=$1
    local out=$2
    local min_batches=$3
    local max_batches=$4
    $PY generate_inverted_pendulum_trajectories.py \
        --split eval --controller lqr --torque_noise "$tau" \
        --output_dir "$out" --seed 42 --horizon 800 \
        --ctrl_freq 100 --pyb_freq 300 --resolution 0.04 \
        --se_tol 0.01 --min_batches "$min_batches" \
        --max_batches "$max_batches" --check_every 10 \
        --merge_eval_shards
}

echo "mode=$MODE task=$SLURM_ARRAY_TASK_ID host=$(hostname) cores=$NCPU start=$(date -Is) root=$ROOT"

case "$MODE" in
    smoke)
        if [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
            echo 'smoke uses array task 0 only' >&2
            exit 2
        fi
        SMOKE_ROOT=${PEND_TAU_ROOT:-/scratch/dm1487/pendulum_tau_extension_smoke_20260813/lqr}
        OUT=$SMOKE_ROOT/tau_5.00
        mkdir -p "$OUT"
        START=$SECONDS
        collect_train 5.0 "$OUT" 256
        TRAIN_SECONDS=$((SECONDS - START))
        START=$SECONDS
        collect_eval_shard 5.0 "$OUT" 0 1
        EVAL_SECONDS=$((SECONDS - START))
        merge_eval 5.0 "$OUT" 1 1
        $PY prepare_stochastic_layout.py --root "$SMOKE_ROOT" --n_cal 1000
        $PY - "$OUT" <<'PY'
import json
import os
import sys
import numpy as np

out = sys.argv[1]
train = np.load(os.path.join(out, 'train.npz'))
ev = np.load(os.path.join(out, 'eval_success_prob.npz'))
td = json.load(open(os.path.join(out, 'train_description.json')))
ed = json.load(open(os.path.join(out, 'eval_description.json')))
assert len(train['labels']) == 256
assert len(ev['p_success']) == 49_770
assert np.all(ev['trials'] == 1)
for d in (td, ed):
    assert d['torque_noise'] == 5.0
    assert d['ctrl_freq'] == 100 and d['pyb_freq'] == 300
    assert d['horizon_steps'] == 800
print('smoke verification passed')
PY
        echo "smoke train_seconds=$TRAIN_SECONDS eval_batch_seconds=$EVAL_SECONDS"
        ;;
    collect)
        I=$SLURM_ARRAY_TASK_ID
        if [ "$I" -lt 3 ]; then
            T=$I
            OUT=$ROOT/tau_${NAMES[$T]}
            mkdir -p "$OUT"
            collect_train "${TAUS[$T]}" "$OUT" 100000
        else
            E=$((I - 3))
            T=$((E / 20))
            S=$((E % 20))
            OUT=$ROOT/tau_${NAMES[$T]}
            mkdir -p "$OUT"
            collect_eval_shard "${TAUS[$T]}" "$OUT" "$((S * 5))" 5
        fi
        ;;
    finalize)
        if [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
            echo 'finalize uses array task 0 only' >&2
            exit 2
        fi
        for T in 0 1 2; do
            merge_eval "${TAUS[$T]}" "$ROOT/tau_${NAMES[$T]}" 10 100
        done
        $PY prepare_stochastic_layout.py --root "$ROOT" --n_cal 1000
        $PY - "$ROOT" <<'PY'
import json
import os
import sys
import numpy as np

root = sys.argv[1]
expected = {'tau_1.00': 1.0, 'tau_2.00': 2.0, 'tau_5.00': 5.0}
assert set(os.listdir(root)) == set(expected)
for name, tau in expected.items():
    out = os.path.join(root, name)
    train = np.load(os.path.join(out, 'train.npz'))
    ev = np.load(os.path.join(out, 'eval_success_prob.npz'))
    td = json.load(open(os.path.join(out, 'train_description.json')))
    ed = json.load(open(os.path.join(out, 'eval_description.json')))
    assert len(train['labels']) == 100_000
    assert len(ev['p_success']) == 49_770
    assert np.all(ev['trials'] == 100)
    assert td['torque_noise'] == ed['torque_noise'] == tau
    for d in (td, ed):
        assert d['ctrl_freq'] == 100 and d['pyb_freq'] == 300
        assert d['horizon_steps'] == 800
    assert os.path.exists(os.path.join(out, 'dataset_description.json'))
    assert os.path.exists(os.path.join(out, 'train_test_splits', 'shuffled_indices_0.txt'))
    print(name, 'verified', 'train_success=', float(train['labels'].mean()),
          'eval_success=', float(ev['p_success'].mean()))
PY
        ;;
    *)
        echo "unknown MODE=$MODE (expected smoke, collect, or finalize)" >&2
        exit 2
        ;;
esac

echo "mode=$MODE task=$SLURM_ARRAY_TASK_ID done=$(date -Is)"
