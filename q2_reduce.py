'''Aggregate the quad2d shards into one dataset directory per noise level.

Mirrors q3_reduce.py, with the quad2d schema:

  <root>/f_<level>/
      train.npz               states(float32,6) offsets starts labels seeds
      roa_labels.txt          6-D state + p_success  (7 cols, 6dp / 4dp)
      eval_states.txt         same content, family-conventional name
      cal_set.txt             10,000 rows   (matches the deterministic split)
      test_set.txt            479,789 rows
      eval_success_prob.npz   starts successes trials p_success det_labels
      train_test_splits/      shuffled_indices_0.txt shuffled_labels_0.txt
      {dataset,train,eval}_description.json

The deterministic quad2d roa_labels.txt is already
`x,z,theta,x_dot,z_dot,theta_dot,label` -- 7 columns -- so our output is that
schema with the label column turned float.
'''
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from q2_common import HORIZON, STATE_BOUNDS, TOL  # noqa: E402

ROOT = os.environ.get('Q2_OUT', os.path.expanduser('~/scg-repo/q2out'))
DEST = os.environ.get('Q2_DATASET', os.path.expanduser('~/scg-repo/q2dataset'))
CAL_N = 10_000
SPLIT_SEED = 20260813
WEIGHT_N = 0.027 * 9.81


def reduce_train(level, out_dir):
    files = sorted(glob.glob(os.path.join(ROOT, 'train', f'L{level}_s*.npz')))
    if not files:
        return None
    states, offsets, starts, labels, seeds, det = [], [0], [], [], [], []
    for f in files:
        d = np.load(f)
        states.append(d['states'])
        offsets.extend((d['offsets'][1:] + offsets[-1]).tolist())
        starts.append(d['starts'])
        labels.append(d['labels'])
        seeds.append(d['seeds'])
        det.append(d['det_labels'])
    states = np.concatenate(states)
    labels = np.concatenate(labels)
    det = np.concatenate(det)
    offsets = np.asarray(offsets, dtype=np.int64)
    assert offsets[-1] == len(states), 'offsets do not span the state array'
    assert len(offsets) - 1 == len(labels), 'offset/label count mismatch'
    np.savez(os.path.join(out_dir, 'train.npz'),
             states=states, offsets=offsets, starts=np.concatenate(starts),
             labels=labels, seeds=np.concatenate(seeds), det_labels=det)
    os.makedirs(os.path.join(out_dir, 'train_test_splits'), exist_ok=True)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(labels))
    with open(os.path.join(out_dir, 'train_test_splits', 'shuffled_indices_0.txt'), 'w') as fh:
        fh.write('\n'.join(f'sequence_{i}.txt' for i in order) + '\n')
    with open(os.path.join(out_dir, 'train_test_splits', 'shuffled_labels_0.txt'), 'w') as fh:
        fh.write('\n'.join(str(int(labels[i])) for i in order) + '\n')
    lengths = np.diff(offsets)
    return dict(num_trajectories=int(len(labels)),
                success_count=int(labels.sum()),
                success_rate=float(labels.mean()),
                agreement_with_deterministic=float((labels == det).mean()),
                mean_length=float(lengths.mean()), max_length=int(lengths.max()),
                hit_horizon=int((lengths - 1 >= HORIZON).sum()),
                total_states=int(len(states)), shards=len(files))


def reduce_eval(level, out_dir):
    files = sorted(glob.glob(os.path.join(ROOT, 'eval', f'L{level}_s*.npz')))
    if not files:
        return None
    starts, hits, det, trials = [], [], [], None
    for f in files:
        d = np.load(f)
        starts.append(d['starts'])
        hits.append(d['hits'])
        det.append(d['det_labels'])
        trials = int(d['trials'])
    starts = np.concatenate(starts)
    hits = np.concatenate(hits)
    det = np.concatenate(det)
    p = hits / trials

    body = '\n'.join(','.join(f'{v:.6f}' for v in r) + f',{q:.4f}'
                     for r, q in zip(starts, p))
    for name in ('roa_labels.txt', 'eval_states.txt'):
        with open(os.path.join(out_dir, name), 'w') as fh:
            fh.write(body + '\n')
    lines = body.split('\n')
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(lines))
    cal_n = min(CAL_N, len(lines) // 10)
    for name, idx in (('cal_set.txt', perm[:cal_n]), ('test_set.txt', perm[cal_n:])):
        with open(os.path.join(out_dir, name), 'w') as fh:
            fh.write('\n'.join(lines[i] for i in idx) + '\n')

    np.savez(os.path.join(out_dir, 'eval_success_prob.npz'),
             starts=starts, successes=hits,
             trials=np.full(len(hits), trials, dtype=np.int32),
             p_success=p, det_labels=det)
    return dict(num_states=int(len(p)), trials=trials,
                mean_p_success=float(p.mean()),
                fraction_interior=float(((p > 0) & (p < 1)).mean()),
                agreement_with_deterministic=float(((hits > 0).astype(int) == det).mean()),
                deterministic_rate=float((det == 1).mean()), shards=len(files))


def describe(level, tr, ev):
    return {
        'dataset_name': f'2D Quadrotor RL (safe_explorer_ppo) under planar force noise, f={level}',
        'mechanism': {
            'kind': 'dynamics', 'dim': 2, 'frame': 'world',
            'applied_as': '[Fx, 0, Fz] at the COM link -- planar, and NO torque',
            'distribution': 'uniform', 'low': -float(level), 'high': float(level),
            'hold': 'zero-order, redrawn each control step (100 Hz)',
            'matched': False,
            'reference_scale': {'body_weight_N': WEIGHT_N,
                                'level_as_fraction_of_weight': float(level) / WEIGHT_N},
            'note': ('Unmatched external forcing, not a wind model: independent '
                     'of drone velocity, no moment, white in time.'),
        },
        'controller': {'type': 'safe_explorer_ppo',
                       'model': 'safe_explorer_ppo_model_quadrotor_2D_stab.pt',
                       'obs_normalizer': 'frozen (set_read_only) and applied every step',
                       'note': "info['constraint_values'] is seeded before the first step; "
                               'the policy reads it internally'},
        'success_criteria': {
            'type': 'radius', 'threshold': TOL,
            'goal_state': [0, 1, 0, 0, 0, 0],
            'entry_cut': True,
            'note': ('Stops at first entry, so the label is a function of the '
                     'terminal state. Verified on the shipped set: successful '
                     'trajectories end at 0.1972-0.1998 against the 0.2 threshold.'),
        },
        'horizon': {
            'steps': HORIZON, 'seconds': HORIZON / 100.0,
            'inherited': True,
            'note': ('INHERITED from the deterministic set, which already used '
                     '1200 as a real limit (longest trajectory 709, timeouts 0). '
                     'LOAD-BEARING under noise: rollouts start hitting the cap '
                     'from f=0.020 upward, so p_success is a BOUNDED-TIME reach '
                     'probability and part of the decline is the deadline rather '
                     'than failure.'),
        },
        'termination_thresholds': {
            'x': 1.0, 'z_min': 0.1, 'z_max': 1.5, 'theta': 'inf',
            'x_dot': 1.0, 'z_dot': 1.0, 'theta_dot': 8.0,
            'source': ('deterministic dataset_description.json; the quad2d RL '
                       'generator sets state_space explicitly, so unlike quad3d '
                       'these are unambiguous'),
            'env_state_indices': {str(k): list(v) for k, v in STATE_BOUNDS.items()},
        },
        'plant': {'quad_type': 2, 'ctrl_freq': 100, 'pyb_freq': 5000,
                  'cost': 'quadratic', 'randomized_init': False,
                  'normalized_rl_action_space': True,
                  'constraints': 'SAFE_EXPLORER_CONSTRAINTS, done_on_violation=False'},
        'data_format': {
            'state_order': ['x', 'z', 'theta', 'x_dot', 'z_dot', 'theta_dot'],
            'angular_velocity_frame': ('world -- for TWO_D the env stores ang_v[1] '
                                       'directly with no body conversion, so there is '
                                       'none of the quad3d sampler-vs-stored asymmetry'),
            'theta': 'wrapped to [-pi, pi] when stored',
            'train': 'train.npz -- states(float32,6) offsets starts labels seeds',
            'eval': 'roa_labels.txt / eval_states.txt -- 6 state cols + p_success',
            'precision': {'state': 6, 'p_success': 4},
        },
        'sampling': {'type': 'stratified_grid',
                     'note': ('The same 489,789 grid states serve BOTH splits: train '
                              'takes one rollout each and keeps the trajectory, eval '
                              'takes K and keeps the success count. The shipped set is '
                              'likewise 1:1 between trajectories and roa_labels rows.')},
        'reproducibility': {
            'seed_fn': 'rollout_seed(base, index, trial), base 20260813',
            'note': ('The level is excluded from the seed, so levels are paired '
                     'under common random numbers.'),
        },
        'known_discrepancy': {
            'level_0_vs_shipped': ('At f=0 this reproduces the shipped labels to '
                                   '~99% (measured 99.2% on a balanced sample, '
                                   '98.0-99.0% on random samples). Use THIS f=0 '
                                   'level as the baseline for the noisy levels, '
                                   'since it is the only one produced by the same '
                                   'code path.'),
            'theta_pi_singularity': ('Inherited from the deterministic set: states '
                                     'with theta within ~0.01 rad of +/-pi get spurious '
                                     'successes from a PyBullet quaternion-Euler '
                                     'roundtrip resolving gimbal lock across roll and '
                                     'yaw. Affects a thin strip only; documented in the '
                                     'deterministic description under known_artifacts.'),
        },
        'train_statistics': tr,
        'eval_statistics': ev,
    }


def main():
    levels = sys.argv[1:] or ['0', '0.070', '0.100', '0.150', '0.200']
    for lv in levels:
        fl = float(lv)
        out_dir = os.path.join(DEST, f'f_{fl:.3f}')
        os.makedirs(out_dir, exist_ok=True)
        tr = reduce_train(lv, out_dir)
        ev = reduce_eval(lv, out_dir)
        desc = describe(fl, tr, ev)
        for name, payload in (('dataset_description.json', desc),
                              ('train_description.json', {**desc, 'split': 'train'}),
                              ('eval_description.json', {**desc, 'split': 'eval'})):
            with open(os.path.join(out_dir, name), 'w') as fh:
                json.dump(payload, fh, indent=2)
        print(f'f_{fl:.3f}: train={tr} eval={ev}', flush=True)


if __name__ == '__main__':
    main()
