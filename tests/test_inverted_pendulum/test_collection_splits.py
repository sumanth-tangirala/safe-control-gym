'''Tests for the split train/eval collection scheme.

Spec: docs/superpowers/specs/2026-07-25-noisy-pendulum-collection-design.md
'''

import glob
import json
import math
import os
import subprocess
import sys
import time

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


# --- atomic writes ----------------------------------------------------------

def test_atomic_savez_keeps_the_previous_file_when_a_write_fails(tmp_path):
    '''A failed write must not damage what is already published.'''
    from generate_inverted_pendulum_trajectories import atomic_savez
    path = str(tmp_path / 'published.npz')
    atomic_savez(path, values=np.arange(4))
    with pytest.raises(Exception):
        atomic_savez(path, values=[[1, 2], [3]])  # ragged: fails while writing
    assert np.array_equal(np.load(path)['values'], np.arange(4)), 'published file was damaged'


# --- eval split -------------------------------------------------------------

# A coarse grid keeps these tests quick: ceil(2pi/1.0) x ceil(4pi/1.0) = 7 x 13.
COARSE_RESOLUTION = 1.0
COARSE_CELLS = 7 * 13


def test_eval_split_stores_probabilities_and_no_trajectories(tmp_path):
    '''The whole point: per-cell probabilities, not rollouts.'''
    from generate_inverted_pendulum_trajectories import collect_eval
    out = str(tmp_path / 'eval')
    collect_eval('lqr', out, seed=0, horizon=30, noise='control_proportional_med',
                 resolution=COARSE_RESOLUTION, min_batches=2, max_batches=2,
                 check_every=1, parallel=False)
    data = np.load(os.path.join(out, 'eval_success_prob.npz'))
    assert 'states' not in data.files, 'eval must not store trajectories'
    assert data['starts'].shape == (COARSE_CELLS, 2)
    assert data['successes'].shape == (COARSE_CELLS,)
    assert data['trials'].shape == (COARSE_CELLS,)
    assert int(data['n_batches']) == 2
    assert np.all(data['trials'] == 2), 'only whole batches are published'
    assert np.all(data['successes'] <= data['trials'])
    assert np.allclose(data['p_success'], data['successes'] / data['trials'])
    with open(os.path.join(out, 'success_probabilities.txt')) as f:
        lines = [ln for ln in f.read().strip().split('\n') if ln]
    assert len(lines) == COARSE_CELLS
    assert len(lines[0].split(',')) == 3


def test_eval_stops_once_the_estimate_has_settled(tmp_path):
    '''Convergence must happen on its own, well before the batch cap.

    Deterministic dynamics put every cell at s = 0 or s = B, so the Jeffreys SD
    is sqrt(p(1-p)/(B+2)) with p = 0.5/(B+1) -- a pure function of B. It crosses
    0.05 between B=12 (0.0514) and B=14 (0.0449), and checks run every 2.
    '''
    from generate_inverted_pendulum_trajectories import collect_eval
    out = str(tmp_path / 'eval')
    stats = collect_eval('lqr', out, seed=0, horizon=30, noise=None,
                         resolution=COARSE_RESOLUTION, se_tol=0.05, min_batches=2,
                         max_batches=60, check_every=2, parallel=False)
    assert stats['converged'] is True
    assert stats['n_batches'] == 14, 'must stop when the estimate settles, not at the cap'
    description = json.load(open(os.path.join(out, 'eval_description.json')))
    assert description['converged'] is True
    assert description['n_batches'] == 14
    assert description['stopping_rule']['se_tol'] == 0.05


def test_eval_labels_an_unconverged_run_honestly(tmp_path):
    '''Hitting the batch cap must not look like convergence.'''
    from generate_inverted_pendulum_trajectories import collect_eval
    out = str(tmp_path / 'eval')
    stats = collect_eval('lqr', out, seed=0, horizon=30,
                         noise='control_proportional_xhigh', se_tol=1e-9,
                         resolution=COARSE_RESOLUTION, min_batches=1, max_batches=2,
                         check_every=1, parallel=False)
    assert stats['converged'] is False
    description = json.load(open(os.path.join(out, 'eval_description.json')))
    assert description['converged'] is False
    assert description['n_batches'] == 2


def test_eval_resume_matches_an_uninterrupted_run(tmp_path):
    '''Two runs of 2 batches must equal one run of 4, cell for cell.

    This is what purity of rollout_seed buys: a resumed run draws exactly the
    noise the uninterrupted run would have drawn.
    '''
    from generate_inverted_pendulum_trajectories import collect_eval
    shared = dict(seed=0, horizon=30, noise='control_proportional_med',
                  resolution=COARSE_RESOLUTION, se_tol=1e-9, min_batches=1,
                  check_every=1, parallel=False)
    straight, resumed = str(tmp_path / 'straight'), str(tmp_path / 'resumed')
    collect_eval('lqr', straight, max_batches=4, **shared)
    collect_eval('lqr', resumed, max_batches=2, **shared)
    collect_eval('lqr', resumed, max_batches=4, **shared)

    a = np.load(os.path.join(straight, 'eval_success_prob.npz'))
    b = np.load(os.path.join(resumed, 'eval_success_prob.npz'))
    assert int(a['n_batches']) == 4 and int(b['n_batches']) == 4
    assert np.array_equal(a['trials'], b['trials'])
    assert np.array_equal(a['successes'], b['successes']), 'resume must reproduce exactly'


def test_eval_dataset_stays_functional_when_the_run_is_killed(tmp_path):
    '''SIGKILL at an arbitrary moment must leave a complete, loadable dataset.'''
    from generate_inverted_pendulum_trajectories import collect_eval  # noqa: F401  (import check)
    for attempt, extra_delay in enumerate((0.0, 0.4, 0.9)):
        out = str(tmp_path / f'killed_{attempt}')
        npz_path = os.path.join(out, 'eval_success_prob.npz')
        code = (f'import sys; sys.path.insert(0, {REPO_ROOT!r})\n'
                'from generate_inverted_pendulum_trajectories import collect_eval\n'
                f'collect_eval("lqr", {out!r}, seed=0, horizon=30,\n'
                '             noise="control_proportional_med",\n'
                f'             resolution={COARSE_RESOLUTION}, se_tol=1e-9, min_batches=1,\n'
                '             max_batches=100000, check_every=1, parallel=False)\n')
        proc = subprocess.Popen([sys.executable, '-c', code])
        try:
            # Wait for the first publication rather than guessing a delay: the
            # child spends ~1.6 s on casadi warmup, and on a loaded machine that
            # stretches unpredictably. Then let a little more run, so the kill
            # lands at a different point each attempt.
            deadline = time.time() + 120
            while not os.path.exists(npz_path) and time.time() < deadline:
                assert proc.poll() is None, 'collector exited before publishing'
                time.sleep(0.02)
            assert os.path.exists(npz_path), 'no batch published within 120 s'
            time.sleep(extra_delay)
        finally:
            proc.kill()
            proc.wait()
        data = np.load(npz_path)  # must not raise: no torn files
        n_batches = int(data['n_batches'])
        assert n_batches >= 1
        assert np.all(data['trials'] == n_batches), 'only whole batches are published'
        assert np.all(data['successes'] <= data['trials'])
        assert np.allclose(data['p_success'], data['successes'] / data['trials'])
        assert len(data['starts']) == COARSE_CELLS
        # A kill during staging can leave a *.tmp behind. That is debris, not
        # corruption: temps have fixed names, so they cannot accumulate, and
        # nothing ever reads them. What matters is that no published file is a
        # partial one, which the checks above and below cover.

        # The npz is committed last, so its presence must imply the mirrors are
        # already in place -- a kill can leave them at most one batch ahead.
        txt_path = os.path.join(out, 'success_probabilities.txt')
        description_path = os.path.join(out, 'eval_description.json')
        assert os.path.exists(txt_path), 'npz published without its txt mirror'
        assert os.path.exists(description_path), 'npz published without its description'
        with open(txt_path) as f:
            assert len([ln for ln in f.read().strip().split('\n') if ln]) == COARSE_CELLS
        description = json.load(open(description_path))
        assert description['n_batches'] - n_batches in (0, 1)


# --- descriptions, layout and CLI -------------------------------------------

def test_train_split_writes_a_description(tmp_path):
    from generate_inverted_pendulum_trajectories import collect_train
    out = str(tmp_path / 'train')
    collect_train('lqr', out, num_trajs=4, seed=0, horizon=30,
                  noise='control_proportional_med', parallel=False)
    description = json.load(open(os.path.join(out, 'train_description.json')))
    assert description['split'] == 'train'
    assert description['controller'] == 'lqr'
    assert description['noise'] == 'control_proportional_med'
    assert description['num_trajectories'] == 4
    assert description['data_format']['states_dtype'] == 'float32'


def test_default_output_dir_follows_the_on_disk_layout():
    from generate_inverted_pendulum_trajectories import default_output_dir
    assert default_output_dir('lqr', 'control_proportional_med').endswith(
        'noisy/pendulum/lqr/med')
    assert default_output_dir('v3_strong', 'control_proportional_xhigh').endswith(
        'noisy/pendulum/v3_strong/xhigh')
    assert default_output_dir('lqr', None).endswith('deterministic/pendulum/lqr')


def _run_cli(*args):
    return subprocess.run([sys.executable,
                           os.path.join(REPO_ROOT, 'generate_inverted_pendulum_trajectories.py'),
                           *args], cwd=REPO_ROOT, capture_output=True, text=True)


def test_cli_collects_the_train_split(tmp_path):
    out = str(tmp_path / 'train')
    done = _run_cli('--split', 'train', '--controller', 'lqr', '--num_trajs', '4',
                    '--horizon', '30', '--noise', 'control_proportional_med',
                    '--output_dir', out)
    assert done.returncode == 0, done.stderr
    assert len(np.load(os.path.join(out, 'train.npz'))['labels']) == 4


def test_cli_collects_the_eval_split(tmp_path):
    out = str(tmp_path / 'eval')
    done = _run_cli('--split', 'eval', '--controller', 'lqr', '--horizon', '30',
                    '--noise', 'control_proportional_med', '--resolution', '1.0',
                    '--min_batches', '1', '--max_batches', '2', '--check_every', '1',
                    '--output_dir', out)
    assert done.returncode == 0, done.stderr
    data = np.load(os.path.join(out, 'eval_success_prob.npz'))
    assert int(data['n_batches']) == 2
    assert len(data['p_success']) == COARSE_CELLS


def test_cli_without_split_keeps_the_legacy_sequence_output(tmp_path):
    '''Omitting --split must still produce the old txt dataset.'''
    out = str(tmp_path / 'legacy')
    done = _run_cli('--controller', 'lqr', '--num_trajs', '3', '--random_init',
                    '--horizon', '30', '--output_dir', out)
    assert done.returncode == 0, done.stderr
    assert len(glob.glob(os.path.join(out, 'trajectories', 'sequence_*.txt'))) == 3
