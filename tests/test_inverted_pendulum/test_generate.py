'''Smoke tests for generate_inverted_pendulum_trajectories.py: it must produce a
valid dataset (sequence files, roa labels, description) for both the LQR and an
RL controller.'''

import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _check_dataset(out_dir, controller, n):
    seqs = sorted(glob.glob(os.path.join(out_dir, 'trajectories', 'sequence_*.txt')))
    assert len(seqs) == n, f'expected {n} sequence files, got {len(seqs)}'
    # Every state line is 'theta,theta_dot' (2 comma-separated floats).
    with open(seqs[0]) as f:
        lines = [ln for ln in f.read().strip().split('\n') if ln]
    assert len(lines) >= 1
    first = lines[0].split(',')
    assert len(first) == 2
    theta = float(first[0])
    assert -np.pi - 1e-6 <= theta <= np.pi + 1e-6, 'theta must be wrapped to [-pi, pi]'
    # roa labels: one per trajectory, 'theta,theta_dot,label'.
    with open(os.path.join(out_dir, 'roa_labels.txt')) as f:
        labels = [ln for ln in f.read().strip().split('\n') if ln]
    assert len(labels) == n
    assert len(labels[0].split(',')) == 3
    assert labels[0].split(',')[-1] in ('0', '1')
    # description metadata.
    desc = json.load(open(os.path.join(out_dir, 'dataset_description.json')))
    assert desc['controller'] == controller
    assert desc['num_trajectories'] == n
    assert desc['state_order'] == ['theta', 'theta_dot']


def test_generate_lqr_dataset(tmp_path):
    from generate_inverted_pendulum_trajectories import generate
    out = str(tmp_path / 'lqr')
    stats = generate('lqr', out, num_trajs=3, random_init=True, seed=0,
                     horizon=200, parallel=False)
    assert stats['total_count'] == 3
    _check_dataset(out, 'lqr', 3)


def test_generate_rl_dataset(tmp_path):
    from generate_inverted_pendulum_trajectories import generate
    out = str(tmp_path / 'v1_strong')
    stats = generate('v1_strong', out, num_trajs=2, random_init=True, seed=1,
                     horizon=200, parallel=False)
    assert stats['total_count'] == 2
    _check_dataset(out, 'v1_strong', 2)


def test_generate_is_resumable(tmp_path):
    from generate_inverted_pendulum_trajectories import generate
    out = str(tmp_path / 'resume')
    generate('lqr', out, num_trajs=2, random_init=True, seed=2, horizon=200, parallel=False)
    seq1 = os.path.join(out, 'trajectories', 'sequence_1.txt')
    mtime_before = os.path.getmtime(seq1)
    # Re-run: existing sequence files should not be regenerated.
    generate('lqr', out, num_trajs=2, random_init=True, seed=2, horizon=200, parallel=False)
    assert os.path.getmtime(seq1) == mtime_before, 'existing trajectory should not be rewritten'


def test_generate_with_noise_records_it(tmp_path):
    from generate_inverted_pendulum_trajectories import generate
    out = str(tmp_path / 'noisy')
    stats = generate('lqr', out, num_trajs=3, random_init=True, seed=0,
                     horizon=200, noise='gaussian_act_high')
    _check_dataset(out, 'lqr', 3)
    desc = json.load(open(os.path.join(out, 'dataset_description.json')))
    assert desc['noise'] == 'gaussian_act_high'
