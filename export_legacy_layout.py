#!/usr/bin/env python3
'''Export a `--split train`/`--split eval` dataset into the shipped DATA_ROOT layout.

The split collectors write two npz files. Every dataset already under DATA_ROOT is
instead a directory of text files, and nothing downstream reads npz. This converts
one to the other so a new dataset drops in beside the old ones.

Formats are taken from the shipped `deterministic/pendulum/lqr`, byte for byte:

  trajectories/sequence_<i>.txt   `%.6g,%.6g`  -- six SIGNIFICANT digits, so small
                                  values come out as `7.04173e-05`. Note this is
                                  NOT the `%.6f` that generate_*.py's own
                                  save_trajectory uses; the shipped sets were
                                  written by the source repo, and matching them
                                  matters more here than matching ourselves.
  trajectory_labels.txt           `sequence_<i>.txt,<0|1>` -- the direct label map
                                  dataset_split_randomizer prefers. Its fallback,
                                  roa_labels.txt, matches initial states by KD-tree;
                                  we know the labels exactly, so we skip it.
  eval_states.txt                 `%.6f,%.6f,%.6f,%.6f,%d`  -- label as an int

Two files in the shipped layout are NOT produced, here or anywhere: `cal_set.txt`
and `test_set.txt` have no producer in this repo or in the scripts under
DATA_ROOT's parent. They are not omissions in this exporter.

`train_test_splits/` is produced by DATA_ROOT's own dataset_split_randomizer.py
(seed 42) rather than reimplemented, so the split matches every other dataset.
Run it afterwards; --print_split_command prints the invocation.

The eval split's probability field has no counterpart in the shipped layout --
every shipped label is a binary 0/1. `eval_success_prob.npz` and
`success_probabilities.txt` are copied across as additions, not conversions.
'''
import argparse
import json
import os
import shutil

import numpy as np

RANDOMIZER = '/common/users/shared/pracsys/genMoPlan/dataset_split_randomizer.py'


def fmt_traj(row):
    '''Six significant digits, matching the shipped trajectory files.'''
    return f'{row[0]:.6g},{row[1]:.6g}'


def export_trajectories(train, out_dir, limit=None):
    '''Write trajectories/ plus the three per-trajectory index files.'''
    traj_dir = os.path.join(out_dir, 'trajectories')
    os.makedirs(traj_dir, exist_ok=True)
    states, offsets, labels = train['states'], train['offsets'], train['labels']
    starts = train['starts']
    n = len(labels) if limit is None else min(limit, len(labels))

    roa, evs, succ = [], [], []
    for i in range(n):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        seq = states[lo:hi]
        with open(os.path.join(traj_dir, f'sequence_{i}.txt'), 'w') as f:
            f.write('\n'.join(fmt_traj(r) for r in seq) + '\n')
        s, t, lab = starts[i], seq[-1], int(labels[i])
        roa.append(f'{s[0]:.6f},{s[1]:.6f},{float(lab):.6f}')
        evs.append(f'{s[0]:.6f},{s[1]:.6f},{t[0]:.6f},{t[1]:.6f},{lab}')
        succ.append(f'sequence_{i}.txt,{lab}')

    for name, rows in [('trajectory_labels.txt', succ), ('eval_states.txt', evs)]:
        with open(os.path.join(out_dir, name), 'w') as f:
            f.write('\n'.join(rows) + '\n')
    return n


def build_description(train_desc, eval_desc, n_traj, out_dir):
    '''The shipped description schema, filled from our own run metadata.'''
    src = train_desc or {}
    tau = src.get('torque_noise')
    lengths = None
    return {
        'dataset_name': f'Pendulum LQR Trajectories (torque tau={tau})',
        'description': (
            'Inverted pendulum trajectories under uniform noise on the commanded '
            'torque, U(-tau, tau) per control step applied before the u_sat clip. '
            'Start states are sampled uniformly over the full state space.'),
        'generation_parameters': {
            'initial_state_bounds': {
                'theta': {'min': -np.pi, 'max': np.pi, 'unit': 'rad',
                          'description': 'Pendulum angle (full rotation range, 0 = upright)'},
                'theta_dot': {'min': -2 * np.pi, 'max': 2 * np.pi, 'unit': 'rad/s',
                              'description': 'Angular velocity (range: -2pi to 2pi)'},
            },
            'sampling_strategy': {
                'method': 'random',
                'description': 'Initial states sampled uniformly at random over the state space'},
            'noise': {
                'mechanism': 'uniform on commanded torque, pre-saturation',
                'tau': tau,
                'u_sat': src.get('u_sat'),
                'fraction_of_saturation': (None if tau is None or not src.get('u_sat')
                                           else tau / src['u_sat']),
                'description': (
                    'Enters through the same column as the control (matched), so it '
                    'perturbs the acceleration row only. Distinct from the '
                    'state-additive pendulum_noise.py presets, which write into '
                    '(theta, theta_dot) directly and are not comparable.')},
            'termination_conditions': {
                'success': src.get('success_rule'),
                'failure': {'name': None,
                            'condition': 'No failure condition - trajectories that never '
                                         'reach the goal run to max_steps'},
                'timeout': {'name': 'Maximum Episode Length',
                            'max_steps': src.get('horizon_steps')},
            },
            'simulation_parameters': {
                'dt': src.get('dt'), 'ctrl_freq': src.get('ctrl_freq'),
                'pyb_freq': src.get('pyb_freq'), 'max_steps': src.get('horizon_steps')},
            'controller': {'type': 'LQR', 'task': 'stabilization', 'goal_state': [0, 0]},
            'seed': src.get('seed'),
        },
        'dataset_statistics': {'total_trajectories': n_traj},
        'state_space': {
            'total_dimensions': 2,
            'components': {'theta': {'index': 0, 'unit': 'rad'},
                           'theta_dot': {'index': 1, 'unit': 'rad/s'}},
            'state_order': ['theta', 'theta_dot']},
        'manifold_structure': {'type': 'Product', 'notation': 'S^1 x R'},
        'data_format': {'file_format': 'text (.txt)',
                        'file_naming': 'sequence_{i}.txt where i is the trajectory index',
                        'line_format': 'theta,theta_dot',
                        'precision': '6 significant digits (%.6g)',
                        'state_order': ['theta', 'theta_dot']},
        'additional_files': {
            'trajectory_labels.txt': {'format': 'sequence_<i>.txt,label',
                                      'label_meaning': '1 = success, 0 = failure',
                                      'total_entries': n_traj},
            'eval_states.txt': {'format': 'theta,theta_dot,final_theta,final_theta_dot,label'},
            'success_probabilities.txt': (
                {'format': 'theta,theta_dot,p_success',
                 'description': 'Per-cell success PROBABILITY over the eval grid, from '
                                'repeated rollouts under independent noise draws. Has no '
                                'counterpart in the older datasets, whose labels are all '
                                'binary.',
                 'grid_cells': eval_desc.get('num_cells'),
                 'rollouts_per_cell': eval_desc.get('n_batches'),
                 'success_rule': eval_desc.get('success_rule')} if eval_desc else None),
        },
        'not_produced': {
            'cal_set.txt': 'no producer exists in this repo or under DATA_ROOT',
            'test_set.txt': 'no producer exists in this repo or under DATA_ROOT'},
        'lengths': lengths,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='directory holding train.npz / eval_success_prob.npz')
    ap.add_argument('--dst', required=True, help='directory to write the shipped layout into')
    ap.add_argument('--limit', type=int, default=None,
                    help='export only the first N trajectories (the shipped pendulum set has '
                         '49,770; 100k means 100k files)')
    ap.add_argument('--print_split_command', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    train_path = os.path.join(args.src, 'train.npz')
    if not os.path.exists(train_path):
        raise SystemExit(f'no train.npz in {args.src}')
    train = np.load(train_path)

    def load_json(name):
        p = os.path.join(args.src, name)
        return json.load(open(p)) if os.path.exists(p) else None

    n = export_trajectories(train, args.dst, args.limit)

    # The probability field is carried across unchanged -- it is an addition to the
    # layout, not a conversion of anything in it.
    for name in ['eval_success_prob.npz', 'success_probabilities.txt']:
        src_p = os.path.join(args.src, name)
        if os.path.exists(src_p):
            shutil.copy2(src_p, os.path.join(args.dst, name))

    desc = build_description(load_json('train_description.json'),
                             load_json('eval_description.json'), n, args.dst)
    with open(os.path.join(args.dst, 'dataset_description.json'), 'w') as f:
        json.dump(desc, f, indent=2, default=str)

    print(f'exported {n} trajectories to {args.dst}')
    print('NOT produced (no producer anywhere): cal_set.txt, test_set.txt')
    if args.print_split_command:
        print(f'\nnext, for train_test_splits/:\n  python {RANDOMIZER} --dataset {args.dst}')


if __name__ == '__main__':
    main()
