#!/usr/bin/env python3
'''Lay out the torque-noise datasets the way the adaptive-FM consumer reads them.

The consumer (olympics-classifier, `system=pendulum_stoch`, `pool_format: npz`)
has TWO data paths, and the stochastic one wants almost nothing from us:

  deterministic (text)          stochastic (npz)          <- ours
  ------------------------      ---------------------------------
  trajectories/sequence_i.txt   one flat train.npz
  shuffled_indices: FILENAMES   shuffled_indices: INTEGER ROW IDS
  eval_states: start,end,label  eval_states: theta,theta_dot,p_success

So no trajectory directory is unpacked -- `train.npz` is already the delivered
artifact, keys and all (`states`, `offsets`, `starts`, `labels`, `seeds`). This
script only adds the small index and probability files beside it:

    train_test_splits/shuffled_indices_0.txt   row ids into train.npz, shuffled
    train_test_splits/shuffled_labels_0.txt    aligned 0/1
    eval_states.txt                            theta, theta_dot, p_success
    cal_set.txt / test_set.txt                 a partition of eval_states

Formats copied from the consumer's own prep script
(olympics-classifier/scripts/prepare_stochastic_pendulum.py): "%.6f,%.6f,%.4f"
for the 3-column files, "%d" for the index files, shuffle seed 0.

ONE STRUCTURAL DIFFERENCE from that script, and it is deliberate. It splits a
single 49,770-cell grid into train starts and eval starts, with 10 rollouts per
start. Ours collects the two splits SEPARATELY: train is 100k uniformly random
starts rolled ONCE each, eval is the full grid rolled 100 times per cell. So

  * every eval cell is available -- none are spent on training, and
  * train and eval start states are disjoint by construction, so there is no
    start-state leakage between them.

That means `shuffled_indices_0.txt` is a plain permutation of rollout ids with
no block structure, which is what the v2 datasets already look like to the
consumer ("plain rollout-level shuffle, no block structure").
'''
import argparse
import datetime
import json
import os

import numpy as np

SHUFFLE_SEED = 0
N_CAL = 10_000            # matches the consumer's N_CAL_STARTS
FMT_3COL = '%.6f,%.6f,%.4f'


def prepare(level_dir, n_cal=N_CAL, seed=SHUFFLE_SEED):
    train = np.load(os.path.join(level_dir, 'train.npz'))
    ev = np.load(os.path.join(level_dir, 'eval_success_prob.npz'))
    splits = os.path.join(level_dir, 'train_test_splits')
    os.makedirs(splits, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- training pool: a permutation of rollout row ids into train.npz -------
    labels = train['labels'].astype(np.int64)
    n_roll = len(labels)
    idx = rng.permutation(n_roll)
    np.savetxt(os.path.join(splits, 'shuffled_indices_0.txt'), idx, fmt='%d')
    np.savetxt(os.path.join(splits, 'shuffled_labels_0.txt'), labels[idx], fmt='%d')

    # --- eval grid: theta, theta_dot, p_success, in shuffle order -------------
    # Shuffled so cal/test are a random partition rather than a slice of the
    # grid -- a contiguous slice would be a band of the state space, not a
    # sample of it.
    rows = np.column_stack([ev['starts'], ev['p_success']])
    order = rng.permutation(len(rows))
    rows = rows[order]
    if n_cal >= len(rows):
        raise ValueError(f'n_cal {n_cal} >= {len(rows)} eval cells')
    np.savetxt(os.path.join(level_dir, 'eval_states.txt'), rows, fmt=FMT_3COL)
    np.savetxt(os.path.join(level_dir, 'cal_set.txt'), rows[:n_cal], fmt=FMT_3COL)
    np.savetxt(os.path.join(level_dir, 'test_set.txt'), rows[n_cal:], fmt=FMT_3COL)

    # --- dataset_description.json: REQUIRED by the consumer -------------------
    # PendulumSystem.__init__ reads achieved_bounds from a file with exactly this
    # name and raises FileNotFoundError otherwise -- it normalises the state by
    # these limits, so they must be the bounds the DATA actually reaches, not the
    # sampling box. Measured over every stored state, not the start states.
    st = train['states']
    achieved = {
        'theta': {'min': float(st[:, 0].min()), 'max': float(st[:, 0].max()), 'unit': 'rad'},
        'theta_dot': {'min': float(st[:, 1].min()), 'max': float(st[:, 1].max()),
                      'unit': 'rad/s'},
    }
    train_desc_path = os.path.join(level_dir, 'train_description.json')
    train_desc = json.load(open(train_desc_path)) if os.path.exists(train_desc_path) else {}
    dataset_desc = {
        'dataset_name': train_desc.get('dataset_name', os.path.basename(level_dir)),
        'achieved_bounds': achieved,
        'state_space': {'state_order': ['theta', 'theta_dot'],
                        'total_dimensions': 2},
        'manifold_structure': {'type': 'Product', 'notation': 'S^1 x R'},
        'collection': train_desc,
        'note': ('achieved_bounds are the min/max over every stored state in '
                 'train.npz, which is what the consumer normalises by. theta is '
                 'wrapped to [-pi, pi] by the env, so its range is the wrap range '
                 'rather than an achieved extreme.'),
    }
    json.dump(dataset_desc, open(os.path.join(level_dir, 'dataset_description.json'), 'w'),
              indent=2, default=str)

    # --- record what was done, beside the collection's own description --------
    desc_path = os.path.join(level_dir, 'eval_description.json')
    desc = json.load(open(desc_path)) if os.path.exists(desc_path) else {}
    desc['prep'] = {
        'date': datetime.date.today().isoformat(),
        'script': 'prepare_stochastic_layout.py',
        'shuffle_seed': seed,
        'n_train_rollouts': int(n_roll),
        'n_eval_cells': int(len(rows)),
        'n_cal': int(n_cal),
        'n_test': int(len(rows) - n_cal),
        'note': ('train and eval start states are disjoint by construction '
                 '(uniform-random starts vs a grid), so no start-state leakage; '
                 'shuffled_indices_0 is a plain rollout permutation with no block '
                 'structure.'),
        'achieved_bounds': achieved,
        'files': {
            'train.npz': 'flat pool: states, offsets, starts, labels, seeds',
            'dataset_description.json': 'achieved_bounds -- REQUIRED by PendulumSystem',
            'train_test_splits/shuffled_indices_0.txt': 'row ids into train.npz',
            'train_test_splits/shuffled_labels_0.txt': 'aligned 0/1 labels',
            'eval_states.txt': 'theta, theta_dot, p_success per grid cell, shuffle order',
            'cal_set.txt': f'first {n_cal} rows of eval_states.txt',
            'test_set.txt': 'the remainder',
        },
    }
    json.dump(desc, open(desc_path, 'w'), indent=2, default=str)
    return n_roll, len(rows)


def verify(level_dir, n_cal=N_CAL):
    '''Re-read what was written and check it against the source arrays.'''
    train = np.load(os.path.join(level_dir, 'train.npz'))
    ev = np.load(os.path.join(level_dir, 'eval_success_prob.npz'))
    splits = os.path.join(level_dir, 'train_test_splits')

    idx = np.loadtxt(os.path.join(splits, 'shuffled_indices_0.txt'), dtype=np.int64)
    lab = np.loadtxt(os.path.join(splits, 'shuffled_labels_0.txt'), dtype=np.int64)
    n_roll = len(train['labels'])
    assert len(idx) == n_roll, 'indices do not cover the pool'
    assert np.array_equal(np.sort(idx), np.arange(n_roll)), 'indices are not a permutation'
    assert np.array_equal(lab, train['labels'].astype(np.int64)[idx]), 'labels misaligned'
    assert idx.max() < n_roll and idx.min() >= 0, 'row id out of range'

    dd_path = os.path.join(level_dir, 'dataset_description.json')
    assert os.path.exists(dd_path), 'dataset_description.json missing (consumer requires it)'
    dd = json.load(open(dd_path))
    b = dd['achieved_bounds']
    st = train['states']
    assert abs(b['theta_dot']['max'] - float(st[:, 1].max())) < 1e-6, 'bounds stale'
    assert abs(b['theta']['min'] - float(st[:, 0].min())) < 1e-6, 'bounds stale'

    es = np.loadtxt(os.path.join(level_dir, 'eval_states.txt'), delimiter=',')
    cal = np.loadtxt(os.path.join(level_dir, 'cal_set.txt'), delimiter=',')
    tst = np.loadtxt(os.path.join(level_dir, 'test_set.txt'), delimiter=',')
    assert es.shape[1] == 3, 'eval_states must be 3-column for the stochastic reader'
    assert len(es) == len(ev['p_success']), 'eval_states lost cells'
    assert np.array_equal(np.vstack([cal, tst]), es), 'cal + test != eval_states'
    assert len(cal) == n_cal
    # p_success survived the round trip through 4 decimals
    assert abs(es[:, 2].mean() - float(ev['p_success'].mean())) < 5e-5, 'p_success drifted'
    return float(es[:, 2].mean()), len(idx), len(es)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, help='directory holding the tau_* level dirs')
    ap.add_argument('--n_cal', type=int, default=N_CAL)
    args = ap.parse_args()

    levels = sorted(d for d in os.listdir(args.root)
                    if os.path.isdir(os.path.join(args.root, d)))
    for lv in levels:
        d = os.path.join(args.root, lv)
        if not os.path.exists(os.path.join(d, 'train.npz')):
            print(f'{lv}: no train.npz, skipped')
            continue
        n_roll, n_cells = prepare(d, args.n_cal)
        mean_p, n_idx, n_es = verify(d, args.n_cal)
        print(f'{lv}: {n_roll} train rollouts, {n_cells} eval cells, '
              f'mean p_success {mean_p:.4f}  [verified]')
