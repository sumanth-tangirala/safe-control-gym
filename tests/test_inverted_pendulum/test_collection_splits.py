'''Tests for the split train/eval collection scheme.

Spec: docs/superpowers/specs/2026-07-25-noisy-pendulum-collection-design.md
'''

import glob
import math
import os
import sys

import numpy as np
import pytest

SHIPPED_LQR_LABELS = ('/common/users/shared/pracsys/genMoPlan/data_trajectories/'
                      'deterministic/pendulum/lqr/roa_labels.txt')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# --- grid -------------------------------------------------------------------

def test_grid_is_half_open_and_stays_within_the_domain():
    '''theta is periodic, so [-pi, pi) must not include both endpoints.'''
    from generate_inverted_pendulum_trajectories import sample_initial_states
    grid = sample_initial_states(0, False, 0, 2 * math.pi, 0.04)
    theta = np.unique(grid[:, 0])
    theta_dot = np.unique(grid[:, 1])
    assert theta.max() < math.pi, 'theta must not reach or exceed +pi'
    assert theta_dot.max() < 2 * math.pi, 'theta_dot must not reach or exceed +2pi'
    assert len(theta) == 158
    assert len(theta_dot) == 315
    assert len(grid) == 49770


def test_grid_reproduces_the_shipped_deterministic_dataset():
    '''The 49,770 grid points must be the ones the shipped datasets were built on.

    Compared as sorted point sets: roa_labels.txt rows are in trajectory-index
    order, not grid order. The residual is ~3e-6 because the external repo
    started from -3.14159 (pi truncated to 5 dp) rather than -pi.
    '''
    if not os.path.exists(SHIPPED_LQR_LABELS):
        pytest.skip('shipped dataset not mounted')
    from generate_inverted_pendulum_trajectories import sample_initial_states
    reference = np.loadtxt(SHIPPED_LQR_LABELS, delimiter=',')[:, :2]
    grid = sample_initial_states(0, False, 0, 2 * math.pi, 0.04)
    assert grid.shape == reference.shape
    order = np.lexsort  # sort both by (theta_dot, theta) so rows correspond
    grid = grid[order((grid[:, 1], grid[:, 0]))]
    reference = reference[order((reference[:, 1], reference[:, 0]))]
    assert np.abs(grid - reference).max() < 1e-5


# --- seeded rollouts --------------------------------------------------------

def _noisy_env_and_ctrl(noise='control_proportional_med'):
    from generate_inverted_pendulum_trajectories import make_controller, make_env_func
    env_func = make_env_func({'ctrl_freq': 100, 'pyb_freq': 100, 'episode_len_sec': 11,
                              'noise': noise, 'invariant': False})
    return env_func(), make_controller('lqr', env_func)


def test_rollout_is_reproducible_from_its_seed():
    '''Same seed must replay the same noise; a different seed must not.'''
    from generate_inverted_pendulum_trajectories import run_trajectory
    env, ctrl = _noisy_env_and_ctrl()
    init = np.array([2.0, 0.5])
    same_a, _, _ = run_trajectory(env, ctrl, init, 50, seed=1234)
    same_b, _, _ = run_trajectory(env, ctrl, init, 50, seed=1234)
    other, _, _ = run_trajectory(env, ctrl, init, 50, seed=99)
    assert np.array_equal(np.array(same_a), np.array(same_b)), 'same seed must replay exactly'
    assert not np.array_equal(np.array(same_a), np.array(other)), 'different seed must diverge'


def test_rollout_seeds_are_deterministic_and_independent():
    '''Seeds are a pure function of (base, split, index, batch), all distinct.

    Purity is what lets a resumed eval run draw exactly the noise an
    uninterrupted run would have drawn.
    '''
    from generate_inverted_pendulum_trajectories import rollout_seed
    assert rollout_seed(42, 0, 7, 0) == rollout_seed(42, 0, 7, 0)
    assert rollout_seed(42, 0, 7, 0) != rollout_seed(43, 0, 7, 0), 'base seed must matter'
    combos = [(s, i, b) for s in (0, 1) for i in range(40) for b in range(6)]
    seeds = {rollout_seed(42, s, i, b) for s, i, b in combos}
    assert len(seeds) == len(combos), 'seeds must be distinct across split/index/batch'


# --- train split ------------------------------------------------------------

def test_train_split_writes_a_flat_float32_dataset(tmp_path):
    '''train.npz keeps the shipped key layout, with states in float32.'''
    from generate_inverted_pendulum_trajectories import collect_train
    out = str(tmp_path / 'train')
    collect_train('lqr', out, num_trajs=5, seed=0, horizon=40,
                  noise='control_proportional_med', parallel=False)
    data = np.load(os.path.join(out, 'train.npz'))
    assert data['states'].dtype == np.float32
    assert data['starts'].shape == (5, 2)
    assert data['labels'].shape == (5,)
    assert data['seeds'].shape == (5,)
    assert data['offsets'].shape == (6,)
    assert data['offsets'][0] == 0
    assert data['offsets'][-1] == len(data['states'])
    for i in range(5):
        traj = data['states'][data['offsets'][i]:data['offsets'][i + 1]]
        assert len(traj) >= 1
        assert np.allclose(traj[0], data['starts'][i], atol=1e-5), 'traj must open on its start'


def test_train_labels_follow_the_first_hit_rule(tmp_path):
    '''Label 1 iff the trajectory was cut on entering the 0.075 goal ball.'''
    from generate_inverted_pendulum_trajectories import collect_train
    out = str(tmp_path / 'train')
    horizon = 400
    collect_train('lqr', out, num_trajs=12, seed=3, horizon=horizon,
                  noise='control_proportional_med', parallel=False)
    data = np.load(os.path.join(out, 'train.npz'))
    assert data['labels'].max() == 1, 'expected at least one success in 12 rollouts'
    for i, label in enumerate(data['labels']):
        traj = data['states'][data['offsets'][i]:data['offsets'][i + 1]]
        terminal_dist = float(np.linalg.norm(traj[-1]))
        if label == 1:
            assert terminal_dist < 0.075, 'a success must end inside the goal ball'
        else:
            assert len(traj) == horizon + 1, 'a failure must run the full horizon'


def test_train_resumes_from_completed_shards_without_recomputing(tmp_path):
    '''A re-run must reuse finished shards and reproduce the same dataset.'''
    from generate_inverted_pendulum_trajectories import collect_train
    out = str(tmp_path / 'train')
    kwargs = dict(num_trajs=6, seed=5, horizon=40,
                  noise='control_proportional_med', parallel=False, batch_size=3)
    collect_train('lqr', out, **kwargs)
    first = dict(np.load(os.path.join(out, 'train.npz')))

    shards = sorted(glob.glob(os.path.join(out, '_shards', '*.npz')))
    assert len(shards) == 2, 'expected one shard per batch'
    mtimes = {p: os.path.getmtime(p) for p in shards}

    os.remove(os.path.join(out, 'train.npz'))
    collect_train('lqr', out, **kwargs)

    assert {p: os.path.getmtime(p) for p in shards} == mtimes, 'shards must not be recomputed'
    second = dict(np.load(os.path.join(out, 'train.npz')))
    for key in first:
        assert np.array_equal(first[key], second[key]), f'{key} must be reproduced exactly'
