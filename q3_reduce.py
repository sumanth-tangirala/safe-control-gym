'''Aggregate the sharded collection into one dataset directory per noise level.

Layout follows the stochastic family (pendulum, cartpole) for the train split
and the deterministic quadrotor3D_lqr set for the eval split:

  <root>/f_<level>/
      train.npz               states offsets starts labels seeds
      roa_labels.txt          13-D state + p_success   (14 cols, 6dp / 4dp)
      eval_states.txt         same content, family-conventional name
      cal_set.txt             10,000 rows, random subset
      test_set.txt            990,000 rows, the remainder
      eval_success_prob.npz   starts successes trials p_success det_labels
      train_test_splits/      shuffled_indices_0.txt shuffled_labels_0.txt
      train_description.json eval_description.json dataset_description.json

roa_labels.txt is the deterministic set's own schema with the label column
turned float: verified there that roa_labels[:, :13] == eval_states[:, :13] and
roa_labels[:, 13] == eval_states[:, 26], bit-identical over 200k rows.
'''
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from generate_quadrotor_3d_noisy import BOUNDS, HORIZON, STATE_BOUNDS, TASK_INFO  # noqa: E402

ROOT = os.environ.get('Q3_OUT', os.path.expanduser('~/scg-repo/q3out'))
DEST = os.environ.get('Q3_DATASET', os.path.expanduser('~/scg-repo/q3dataset'))
CAL_N = 10_000
SPLIT_SEED = 20260813


def level_tag(level):
    return f'f_{level:.3f}'


def reduce_train(level, out_dir):
    files = sorted(glob.glob(os.path.join(ROOT, 'train', f'L{level}_s*.npz')))
    if not files:
        return None
    states, offsets, starts, labels, seeds = [], [0], [], [], []
    for f in files:
        d = np.load(f)
        states.append(d['states'])
        # Each shard's offsets are shard-local; rebase onto the running total.
        offsets.extend((d['offsets'][1:] + offsets[-1]).tolist())
        starts.append(d['starts'])
        labels.append(d['labels'])
        seeds.append(d['seeds'])
    states = np.concatenate(states)
    labels = np.concatenate(labels)
    offsets = np.asarray(offsets, dtype=np.int64)
    assert offsets[-1] == len(states), 'offsets do not span the state array'
    assert len(offsets) - 1 == len(labels), 'offset/label count mismatch'
    np.savez(os.path.join(out_dir, 'train.npz'),
             states=states, offsets=offsets,
             starts=np.concatenate(starts), labels=labels,
             seeds=np.concatenate(seeds))
    os.makedirs(os.path.join(out_dir, 'train_test_splits'), exist_ok=True)
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(labels))
    with open(os.path.join(out_dir, 'train_test_splits',
                           'shuffled_indices_0.txt'), 'w') as fh:
        fh.write('\n'.join(f'sequence_{i}.txt' for i in order) + '\n')
    with open(os.path.join(out_dir, 'train_test_splits',
                           'shuffled_labels_0.txt'), 'w') as fh:
        fh.write('\n'.join(str(int(labels[i])) for i in order) + '\n')
    lengths = np.diff(offsets)
    return dict(num_trajectories=int(len(labels)),
                success_count=int(labels.sum()),
                success_rate=float(labels.mean()),
                mean_length=float(lengths.mean()),
                max_length=int(lengths.max()),
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

    rows = np.hstack([starts, p[:, None]])
    body = '\n'.join(
        ','.join(f'{v:.6f}' for v in r[:13]) + f',{r[13]:.4f}' for r in rows)
    for name in ('roa_labels.txt', 'eval_states.txt'):
        with open(os.path.join(out_dir, name), 'w') as fh:
            fh.write(body + '\n')

    # cal/test is a random partition, matching the deterministic set (its
    # cal_set indices are scattered from 1,734 to 997,396, not a prefix).
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(rows))
    lines = body.split('\n')
    # Guard the degenerate case: on a partial reduce there may be fewer rows
    # than CAL_N, which would put everything in cal and leave test empty.
    cal_n = min(CAL_N, len(rows) // 10)
    for name, idx in (('cal_set.txt', perm[:cal_n]), ('test_set.txt', perm[cal_n:])):
        with open(os.path.join(out_dir, name), 'w') as fh:
            fh.write('\n'.join(lines[i] for i in idx) + '\n')

    np.savez(os.path.join(out_dir, 'eval_success_prob.npz'),
             starts=starts, successes=hits,
             trials=np.full(len(hits), trials, dtype=np.int32),
             p_success=p, det_labels=det)
    interior = float(((p > 0) & (p < 1)).mean())
    return dict(num_states=int(len(p)), trials=trials,
                mean_p_success=float(p.mean()),
                fraction_interior=interior,
                agreement_with_deterministic=float(((hits > 0).astype(int) == det).mean()),
                deterministic_rate=float(det.mean()), shards=len(files))


def describe(level, train_stats, eval_stats):
    return {
        'dataset_name': f'3D Quadrotor LQR under external force noise, f={level}',
        'mechanism': {
            'kind': 'dynamics', 'dim': 3, 'frame': 'world',
            'distribution': 'uniform', 'low': -float(level), 'high': float(level),
            'hold': 'zero-order, redrawn each control step (100 Hz)',
            'applied': 'applyExternalForce at the COM link, so it exerts NO torque',
            'matched': False,
            'note': ('Unmatched external forcing. This is NOT a wind model: the '
                     'force is independent of the drone velocity, produces no '
                     'moment, and is white in time.'),
            'reference_scale': {'body_weight_N': 0.2646,
                                'level_as_fraction_of_weight': float(level) / 0.2646},
        },
        'success_criteria': {
            'type': 'radius', 'threshold': 0.05,
            'computed_over': ('the env 12-D internal state with EULER angles '
                              '[x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r], '
                              'NOT the 13-D quaternion state that the files store'),
            'entry_cut': True,
            'note': ('Trajectory stops at first entry, so the label is a function '
                     'of the terminal state. The norm sums metres, m/s, radians '
                     'and rad/s; that is dimensionally incoherent but is what '
                     'generated the deterministic dataset, so it is inherited.'),
        },
        'horizon': {
            'steps': HORIZON, 'seconds': HORIZON / 100.0,
            'note': ('LOAD-BEARING. Labels are BOUNDED-TIME reach probabilities. '
                     'The deterministic run allowed 100,000 steps and never '
                     'needed more than 636. Under this disturbance the '
                     'controller mostly still reaches the goal but takes far '
                     'longer: given unlimited time, success at f=0.072 is ~0.24 '
                     'against f=0 at ~0.25, while at H=1000 it reads 0.058. '
                     'About 15% of f=0.072 rollouts would succeed with unlimited '
                     'time. These numbers are NOT asymptotic reach probabilities.'),
        },
        'termination_thresholds': {'x': 1.8, 'y': 1.8, 'z_min': 0.1, 'z_max': 3.0,
                                   'x_dot': 3.0, 'y_dot': 3.0, 'z_dot': 3.0,
                                   'p': 24.0, 'q': 24.0, 'r': 24.0,
                                   'angles': 'infinite (periodic, no termination)',
                                   'source': 'deterministic dataset_description.json',
                                   'caveat': ('The dataset-era collector never set '
                                              'state_space at all and its achieved_bounds '
                                              'show p reaching 39.1 against a stated 24, so '
                                              'rates likely did not terminate in the original '
                                              'run. Configurations differ by only 2 labels in '
                                              '400; the documented thresholds were chosen.')},
        'plant': {'ctrl_freq': 100, 'pyb_freq': 5000, 'cost': 'quadratic',
                  'randomized_init': False, 'quad_type': 3,
                  'gravity': 9.8, 'mass': 0.027,
                  'note': 'gravity is the env GRAVITY_ACC 9.8; the deterministic '
                          'description says 9.81 but outcomes are identical at both'},
        'controller': {'type': 'LQR', 'q_lqr': [1] * 12, 'r_lqr': [0.1] * 4,
                       'discrete_dynamics': True,
                       'goal_state_xyz': TASK_INFO['stabilization_goal']},
        'initial_state_bounds': {k: float(v) for k, v in BOUNDS.items()},
        'state_bounds_indices': {str(k): list(v) for k, v in STATE_BOUNDS.items()},
        'data_format': {
            'trajectory_state_order': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz',
                                       'x_dot', 'y_dot', 'z_dot', 'p', 'q', 'r'],
            'quaternion': 'scalar-first, canonicalised qw >= 0',
            'angular_velocity_frame': 'body',
            'train': 'train.npz -- states(float32,13) offsets starts labels seeds',
            'eval': 'roa_labels.txt / eval_states.txt -- 13 state cols + p_success',
            'precision': {'state': 6, 'p_success': 4},
        },
        'reproducibility': {
            'seed_fn': 'rollout_seed(base, split_id, index, trial), base 20260813',
            'note': ('The level is deliberately excluded from the seed, so every '
                     'level sees the same noise stream per (start, trial) and '
                     'levels are paired under common random numbers.'),
            'injection': ('TWO PATHS. Train starts come from the sampler and are '
                          'passed to resetBaseVelocity as world rates, repeating '
                          'the original collector conflation (converting instead '
                          'drops agreement 393/400 -> 367/400). Eval starts are '
                          'read from the shipped eval_states.txt, which the env '
                          'wrote as body rates, so they need world = R @ body '
                          '(skipping it drops 148/150 -> 138/150).'),
            'sampler_order': ('generate_random_initial_states returns GROUPED '
                              'order [x, y, z, phi, theta, psi, x_dot, y_dot, '
                              'z_dot, p, q, r], not the env interleaved order.'),
        },
        'known_discrepancy': {
            'level_0_vs_shipped': ('At f=0 this reproduces the shipped deterministic '
                                   'labels to 98-99%, not 100%. The code that '
                                   'generated the shipped data is not in the repo in '
                                   'runnable form -- its env_func omits task_info, '
                                   'which the 3D branch indexes, so it raises '
                                   'IndexError on construction. The residual is '
                                   'chaos amplification over ~500-step trajectories '
                                   'plus boundary ties. Use THIS f=0 level as the '
                                   'baseline for these noisy levels, since it is the '
                                   'only one produced by the same code path.'),
        },
        'train_statistics': train_stats,
        'eval_statistics': eval_stats,
    }


def main():
    levels = sys.argv[1:] or ['0', '0.032', '0.048', '0.060', '0.072']
    for lv in levels:
        fl = float(lv)
        out_dir = os.path.join(DEST, level_tag(fl))
        os.makedirs(out_dir, exist_ok=True)
        tr = reduce_train(lv, out_dir)
        ev = reduce_eval(lv, out_dir)
        desc = describe(fl, tr, ev)
        for name, payload in (('dataset_description.json', desc),
                              ('train_description.json', {**desc, 'split': 'train'}),
                              ('eval_description.json', {**desc, 'split': 'eval'})):
            with open(os.path.join(out_dir, name), 'w') as fh:
                json.dump(payload, fh, indent=2)
        print(f'{level_tag(fl)}: train={tr} eval={ev}', flush=True)


if __name__ == '__main__':
    main()
