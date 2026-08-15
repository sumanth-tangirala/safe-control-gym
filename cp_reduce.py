'''Aggregate the cartpole shards into one dataset directory per noise level.

Mirrors q3_reduce.py, with the cartpole schema:

  <root>/f_<level>/
      train.npz               states(float32,6) offsets starts labels seeds
      roa_labels.txt          6-D state + p_success  (7 cols, 6dp / 4dp)
      eval_states.txt         same content, family-conventional name
      cal_set.txt             10,000 rows   (matches the deterministic split)
      test_set.txt            479,789 rows
      eval_success_prob.npz   starts successes trials p_success det_labels
      train_test_splits/      shuffled_indices_0.txt shuffled_labels_0.txt
      {dataset,train,eval}_description.json

The deterministic cartpole roa_labels.txt is already
`x,z,theta,x_dot,z_dot,theta_dot,label` -- 7 columns -- so our output is that
schema with the label column turned float.
'''
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from cp_collect import BOUNDS, CUT, FORCE, HORIZON, N_EVAL, N_TRAIN, TOL  # noqa: E402

ROOT = os.environ.get('CP_OUT', os.path.expanduser('~/scg-repo/cpout'))
DEST = os.environ.get('CP_DATASET', os.path.expanduser('~/scg-repo/cpdataset'))
CAL_N = 10_000   # matches the deterministic cartpole cal/test split
SPLIT_SEED = 20260813
WEIGHT_N = 0.027 * 9.81


def reduce_train(level, out_dir):
    files = sorted(glob.glob(os.path.join(ROOT, 'train', f'S{level}_s*.npz')))
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
    files = sorted(glob.glob(os.path.join(ROOT, 'eval', f'S{level}_s*.npz')))
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
        'dataset_name': f'CartPole LQR under action noise (sigma={level} N)',
        'mechanism': {
            'kind': 'action', 'dim': 1,
            'applied_as': 'uniform on the commanded cart force, PRE-saturation',
            'distribution': 'uniform', 'low': -float(level), 'high': float(level),
            'hold': 'zero-order, redrawn each control step (100 Hz)',
            'matched': True,
            'note': ('Matched uncertainty: the disturbance enters through the same '
                     'channel the controller commands, so the LQR partially rejects '
                     'it and the measured region is biased toward the noise-free one. '
                     'Same mechanism class as the pendulum torque noise; the '
                     'quadrotor datasets instead use an UNMATCHED external force.'),
            'control_bound_N': FORCE,
            'level_as_fraction_of_bound': float(level) / FORCE,
            'saturation_caveat': ('Noise is added before the clip to +/-2000 N, so near '
                                  'saturation the nominally zero-mean disturbance becomes '
                                  'control-dependent. In practice the LQR commands small '
                                  'forces near the goal and half-retention occurs at '
                                  '0.55% of the bound, so this is far from binding.'),
        },
        'controller': {'type': 'LQR', 'q_lqr': [1, 1, 1, 1], 'r_lqr': [0.1],
                       'discrete_dynamics': True, 'goal_state': [0, 0, 0, 0]},
        'success_criteria': {
            'type': 'radius', 'threshold': TOL, 'entry_cut': True,
            'computed_over': 'the full 4-D state, env-native goal_reached',
            'IMPORTANT': (
                'The deterministic cartpole_pybullet description claims per-channel '
                'tolerances (x < 0.01, others < 0.05) held for 10 consecutive steps. '
                'That was NEVER IMPLEMENTED. Every shipped success ends with '
                '||state|| in [0.0497, 0.0500] and not one satisfies |x| < 0.01 -- '
                'the signature of first entry into an L2 ball of radius 0.05 with no '
                'dwell. This dataset uses the real rule. Note that LABELS CANNOT '
                'DISTINGUISH THE TWO: a gate scored 300/300 against the wrong rule. '
                'Only the stored final states discriminate.'),
            'verification': ('Reproduces the deterministic set exactly: labels 300/300 '
                             'on a balanced sample, final states median 4.97e-07 (the '
                             '6-decimal storage floor), and at full scale sigma=0 gives '
                             'agreement 1.0000 over all 116,242 eval states.'),
        },
        'horizon': {'steps': HORIZON, 'seconds': HORIZON / 100.0,
                    'note': ('Matches the deterministic max_steps of 1000. Load-bearing '
                             'under noise: rollouts reach the cap from sigma=4 upward, '
                             'against a deterministic max of 600, so p_success is a '
                             'BOUNDED-TIME reach probability.')},
        'termination_thresholds': dict(CUT, theta='inf (periodic)',
                                       source='deterministic dataset_description.json',
                                       note=('The previously shipped stochastic set '
                                             'relaxed x_dot/theta_dot to 20.0; this '
                                             'restores the deterministic 5.0.')),
        'plant': {'gravity': 9.8, 'cart_mass': 1.0, 'pole_mass': 0.1,
                  'pole_length': 0.5, 'ctrl_freq': 100, 'pyb_freq': 5000,
                  'cost': 'quadratic',
                  'cost_note': ("cost='quadratic' is REQUIRED: without it goal_reached "
                                'never reaches info, so the env terminates in the right '
                                'place while every label reads 0.'),
                  'damping': 0, 'joint_motor': 'disabled'},
        'sampling': {
            'train': {'kind': 'random', 'n': N_TRAIN, 'bounds': BOUNDS,
                      'rejection': 'states at or beyond a termination threshold dropped',
                      'note': 'the same filter that reduced the raw grid to 116,242'},
            'eval': {'kind': 'the exact deterministic eval states', 'n': N_EVAL,
                     'source': 'deterministic/cartpole_pybullet/eval_states.txt cols 0:4',
                     'note': ('row i indexes the same physical state in both datasets. '
                              'The previously shipped stochastic set used 131,859 '
                              'states, aligned with nothing.')},
        },
        'data_format': {
            'state_order': ['x', 'theta', 'x_dot', 'theta_dot'],
            'env_state_order': ['x', 'x_dot', 'theta', 'theta_dot'],
            'permutation': [0, 2, 1, 3],
            'note': ('File order is written directly by the collector. The previously '
                     'shipped set stored env order and was reordered post-hoc via a '
                     'conversion_manifest.'),
            'train': 'train.npz -- states(float32,4) offsets starts labels seeds',
            'eval': 'roa_labels.txt / eval_states.txt -- 4 state cols + p_success',
            'precision': {'state': 6, 'p_success': 4},
        },
        'reproducibility': {
            'seed_fn': 'rollout_seed(split_id, index, trial), base 20260815',
            'note': 'level excluded from the seed, so levels are paired under CRN',
        },
        'corrections_vs_previously_shipped': {
            'control_bound': '100 N -> 2000 N (the deterministic value; 20x error)',
            'success_rule': 'uniform 0.1 per-channel box -> L2 ball radius 0.05',
            'termination': 'x_dot/theta_dot 20.0 -> 5.0',
            'eval_states': '131,859 unaligned -> the exact 116,242 deterministic states',
            'baseline': 'no sigma=0 level -> sigma=0 included',
            'state_order': 'post-hoc canonicalization -> written in file order directly',
        },
        'collection_note': {
            'halk_nodes': ('Amarel main-redhat halk* nodes exit COMPLETED with exit code '
                           '0 while writing no output and no data shard. SLURM reports '
                           'success, so --requeue never fires and afterok dependencies '
                           'are satisfied by empty tasks. 14 train shards were lost this '
                           'way and recovered by a shard-presence check; those nodes are '
                           'excluded in the sbatch files.'),
        },
        'train_statistics': tr,
        'eval_statistics': ev,
    }


def main():
    levels = sys.argv[1:] or ['0', '6', '8', '11', '18']
    for lv in levels:
        fl = float(lv)
        out_dir = os.path.join(DEST, f's_{fl:g}')
        os.makedirs(out_dir, exist_ok=True)
        tr = reduce_train(lv, out_dir)
        ev = reduce_eval(lv, out_dir)
        desc = describe(fl, tr, ev)
        for name, payload in (('dataset_description.json', desc),
                              ('train_description.json', {**desc, 'split': 'train'}),
                              ('eval_description.json', {**desc, 'split': 'eval'})):
            with open(os.path.join(out_dir, name), 'w') as fh:
                json.dump(payload, fh, indent=2)
        print(f's_{fl:g}: train={tr} eval={ev}', flush=True)


if __name__ == '__main__':
    main()
