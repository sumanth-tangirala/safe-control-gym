#!/usr/bin/env python3
'''Generate inverted-pendulum trajectory datasets.

Rolls out one of the ported controllers -- the bounds-normalized LQR
(``--controller lqr``) or a standalone trained SAC swing-up policy
(``--controller v1_strong ... v4_weak``) -- from sampled initial states, and
writes trajectories in the same text format as the cartpole/quadrotor
generators.

State is ``[theta, theta_dot]`` with ``theta`` wrapped to ``[-pi, pi]``. Because
the pendulum clips ``theta_dot`` and wraps ``theta`` (no out-of-bounds failure),
a trajectory ends only on reaching the upright goal (label 1) or on timeout
(label 0): trajectories are cut at (and include) the first state within the
goal threshold (0.075).

With ``--invariant_terminal_sets`` (off by default), goal termination is
disabled instead: every trajectory runs for exactly the fixed horizon T and
label 1 iff the final state lies in the strictly invariant Lyapunov ellipsoid
``(s - s0)' P (s - s0) <= c`` (artifact ``invariant_sets/pendulum.npz``; see
plans/invariant-terminal-sets-recollection.md). By invariance this is
equivalent to "ever entered the ellipsoid", and successful terminal states are
settled at the upright equilibrium, deep inside the classification region.

``--split`` selects a purpose-built collection mode instead (see
docs/superpowers/specs/2026-07-25-noisy-pendulum-collection-design.md):

  * ``--split train`` -- ``--num_trajs`` rollouts from *random* start states,
    written to ``train.npz`` as a flat float32 state array plus offsets,
    starts, labels and seeds. Sharded, so an interrupted run recomputes
    nothing.
  * ``--split eval`` -- batches over the *grid*, where one batch is one rollout
    from every cell, keeping only the per-cell probability of success. It runs
    until that estimate settles, and republishes the complete dataset
    atomically after every batch, so the run can be killed at any moment and
    still leave a functional dataset.

Examples:
    python generate_inverted_pendulum_trajectories.py --controller lqr \
        --random_init --num_trajs 100000 --parallel --seed 42
    python generate_inverted_pendulum_trajectories.py --controller v3_strong \
        --random_init --num_trajs 50000 --parallel --seed 42

    # train and eval are independent processes, meant to run concurrently
    python generate_inverted_pendulum_trajectories.py --split train \
        --controller lqr --noise control_proportional_med --parallel --num_workers 24 &
    python generate_inverted_pendulum_trajectories.py --split eval \
        --controller lqr --noise control_proportional_med --parallel --num_workers 48 &
'''

import argparse
import json
import math
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from safe_control_gym.envs.gym_control.pendulum_noise import NOISE_PRESETS
from safe_control_gym.utils.registration import make

THETA_DOT_MAX = 2 * math.pi
U_SAT = 0.6371781908344007
VARIANTS = ['v1', 'v2', 'v3', 'v4']
STRENGTHS = ['strong', 'weak']
VALID_CONTROLLERS = ['lqr'] + [f'{v}_{s}' for v in VARIANTS for s in STRENGTHS]
CACHE_NAME = '_results_cache.json'
TRAIN_SPLIT_ID, EVAL_SPLIT_ID = 0, 1
GRID_RESOLUTION = 0.04  # 158 x 315 = 49,770 states, matching the shipped datasets
DEFAULT_NUM_TRAJS = 300000
DATA_ROOT = '/common/users/shared/pracsys/genMoPlan/data_trajectories'
NOISE_LEVELS = ('low', 'med', 'high', 'xhigh', 'xxhigh', 'ultra', 'max')
INVARIANT_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'invariant_sets', 'pendulum.npz')
# Fixed horizons (steps): old max success length + settle buffer
# (plans/invariant-terminal-sets-recollection.md).
DEFAULT_HORIZON = {'lqr': 600, 'rl': 1100}


def load_invariant_set(path=INVARIANT_SET_PATH):
    '''Load the success ellipsoid (P, center, c) from its artifact.'''
    data = np.load(path)
    return data['P'], data['center'], float(data['c'])


def in_invariant_set(state, P, center, c):
    dev = np.asarray(state, dtype=np.float64) - center
    return bool(dev @ P @ dev <= c)


def noise_level(noise):
    '''Short level name for a preset (``control_proportional_med`` -> ``med``).'''
    if noise is None:
        return None
    tail = noise.rsplit('_', 1)[-1]
    return tail if tail in NOISE_LEVELS else noise


def default_output_dir(controller, noise):
    '''Dataset location, following the ``<family>/pendulum/<controller>/`` layout.'''
    if noise is None:
        return os.path.join(DATA_ROOT, 'deterministic', 'pendulum', controller)
    return os.path.join(DATA_ROOT, 'noisy', 'pendulum', controller, noise_level(noise))


def rollout_seed(base_seed, split_id, index, batch=0):
    '''Per-rollout seed, a pure function of its coordinates.

    ``split_id`` is ``TRAIN_SPLIT_ID``/``EVAL_SPLIT_ID``, ``index`` the
    trajectory (train) or grid-cell (eval) index, ``batch`` the eval batch
    number. Purity is what lets a resumed run draw exactly the noise an
    uninterrupted run would have drawn.
    '''
    seq = np.random.SeedSequence([int(base_seed), int(split_id), int(index), int(batch)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def normalize_angle(angle):
    '''Normalize an angle to [-pi, pi].'''
    return math.atan2(math.sin(angle), math.cos(angle))


def get_available_cpus():
    '''Number of CPUs available to this process (respects affinity/taskset).'''
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return cpu_count()


def grid_axis(lo, hi, resolution):
    '''Half-open ``[lo, hi)`` axis at ``resolution``.

    The count is computed up front rather than left to ``np.arange``'s float
    endpoint handling, so the axis length is exact.
    '''
    return lo + resolution * np.arange(int(math.ceil((hi - lo) / resolution)))


def sample_initial_states(num_trajs, random_init, seed, theta_dot_max, resolution):
    '''Sample initial ``[theta, theta_dot]`` states.

    ``random_init``: ``num_trajs`` uniform samples over the full state space.
    Otherwise: a discretized grid at ``resolution`` (``num_trajs`` ignored).

    The grid is **half-open** in both coordinates. For ``theta`` that is
    required, not merely tidy: it is periodic, so ``-pi`` and ``+pi`` are the
    same physical state and including both would duplicate a column. At
    ``resolution=0.04`` this yields the 158 x 315 = 49,770 state grid the
    shipped pendulum datasets use.
    '''
    if random_init:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=num_trajs)
        thetadot = rng.uniform(-theta_dot_max, theta_dot_max, size=num_trajs)
        return np.stack([theta, thetadot], axis=1)
    thetas = grid_axis(-math.pi, math.pi, resolution)
    thetadots = grid_axis(-theta_dot_max, theta_dot_max, resolution)
    return np.array([[t, d] for t in thetas for d in thetadots], dtype=np.float64)


def make_env_func(env_config):
    # In invariant mode goal_threshold=0 disables goal termination (episodes
    # run the full horizon); otherwise the env default (0.075) terminates at
    # first goal entry.
    kwargs = {}
    if env_config.get('invariant'):
        kwargs['goal_threshold'] = 0.0
    return partial(make, 'inverted_pendulum',
                   ctrl_freq=env_config['ctrl_freq'],
                   pyb_freq=env_config['pyb_freq'],
                   episode_len_sec=env_config['episode_len_sec'],
                   cost='quadratic',
                   gui=False,
                   randomized_init=False,
                   noise=env_config.get('noise'),
                   **kwargs)


def make_controller(controller, env_func):
    '''Instantiate the LQR or the requested standalone RL policy.'''
    if controller == 'lqr':
        return make('pendulum_lqr', env_func, q_lqr=[1, 1], r_lqr=[1])
    ctrl = make('pendulum_rl', env_func, model_path=controller)
    ctrl.obs_normalizer.set_read_only()
    return ctrl


def run_trajectory(env, ctrl, init_state, max_steps, invariant=False, seed=None):
    '''Roll out one trajectory from ``init_state``.

    Default: terminate at (and include) the first state within the goal
    threshold; returns ``(trajectory, success, timeout)``.

    ``invariant=True``: no early termination; roll exactly ``max_steps`` steps
    and return ``(trajectory, None, False)`` -- the success label is decided
    afterwards from the terminal state.

    ``seed`` reseeds the env's RNG, which is what the noise model draws from,
    making a noisy rollout exactly reproducible. ``None`` leaves the RNG alone
    (the stream continues from wherever the previous rollout left it).
    '''
    env.reset(seed=seed)
    env.state = np.array(init_state, dtype=np.float64)
    obs = env._get_observation()
    info = None
    if hasattr(ctrl, 'reset'):
        ctrl.reset()

    trajectory = [[normalize_angle(env.state[0]), float(env.state[1])]]
    success = None if invariant else False
    timeout = False
    for _ in range(max_steps):
        obs_in = ctrl.obs_normalizer(obs) if hasattr(ctrl, 'obs_normalizer') else obs
        action = ctrl.select_action(obs_in, info)
        obs, _, done, info = env.step(action)
        trajectory.append([normalize_angle(env.state[0]), float(env.state[1])])
        if not invariant and done:
            success = bool(info.get('goal_reached', False))
            break
    else:
        timeout = not invariant
    return trajectory, success, timeout


def save_trajectory(trajectory, filepath):
    '''Save a trajectory (one comma-separated state per line, 6 decimals).'''
    with open(filepath, 'w') as f:
        for state in trajectory:
            f.write(','.join(f'{v:.6f}' for v in state) + '\n')


def _process_batch(args_tuple):
    '''Worker: roll out a batch of trajectories with a shared env + controller.'''
    batch, controller, env_config, trajectories_dir, skip_save = args_tuple
    invariant = bool(env_config.get('invariant'))
    env_func = make_env_func(env_config)
    ctrl = make_controller(controller, env_func)
    env = env_func()
    if invariant:
        P, center, c = load_invariant_set()
    records = []
    for idx, init_state in batch:
        trajectory, success, timeout = run_trajectory(
            env, ctrl, init_state, env_config['max_steps'], invariant=invariant)
        terminal_v_over_c = None
        if invariant:
            dev = np.array(trajectory[-1]) - center
            terminal_v_over_c = float(dev @ P @ dev) / c
            success = terminal_v_over_c <= 1.0
        if not skip_save:
            save_trajectory(trajectory, os.path.join(trajectories_dir, f'sequence_{idx}.txt'))
        records.append((idx, {
            'init_state': [normalize_angle(init_state[0]), float(init_state[1])],
            'label': 1 if success else 0,
            'success': bool(success),
            'timeout': bool(timeout),
            'terminal_v_over_c': terminal_v_over_c,
            'length': len(trajectory),
        }))
    env.close()
    ctrl.close()
    return records


# --- atomic writers ---------------------------------------------------------

def stage_npz(path, **arrays):
    '''Write an npz to a temp file; returns the ``(tmp, final)`` pair to commit.'''
    tmp = path + '.tmp.npz'
    np.savez(tmp, **arrays)
    return (tmp, path)


def stage_text(path, text):
    '''Write a text file to a temp file; returns the ``(tmp, final)`` pair.'''
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    return (tmp, path)


def commit_staged(staged):
    '''Move staged files into place, in order.

    ``os.replace`` is atomic within a filesystem, so no reader (and no kill)
    ever sees a half-written file. Committing a *set* of files is not atomic,
    so order matters: put the authoritative file last, and its presence then
    implies the others are already in place.
    '''
    for tmp, path in staged:
        os.replace(tmp, path)


def atomic_savez(path, **arrays):
    '''Write a single npz atomically.'''
    commit_staged([stage_npz(path, **arrays)])


def atomic_write_text(path, text):
    '''Write a single text file atomically.'''
    commit_staged([stage_text(path, text)])


# --- train split ------------------------------------------------------------

def _shard_is_current(path, fingerprint):
    '''True if ``path`` is a shard already written for this exact config.'''
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as shard:
            return str(shard['fingerprint']) == fingerprint
    except (OSError, ValueError, KeyError):
        return False  # truncated or foreign file: recompute


def _train_worker(args_tuple):
    '''Worker: roll out one batch of training trajectories into its shard.

    Shards are the resume unit -- a shard already written for this config is
    left alone, so an interrupted run recomputes nothing.
    '''
    batch, controller, env_config, base_seed, shard_path, fingerprint = args_tuple
    if _shard_is_current(shard_path, fingerprint):
        return shard_path
    env_func = make_env_func(env_config)
    ctrl = make_controller(controller, env_func)
    env = env_func()
    indices, states, lengths, labels, seeds = [], [], [], [], []
    for idx, init_state in batch:
        seed = rollout_seed(base_seed, TRAIN_SPLIT_ID, idx)
        trajectory, success, _ = run_trajectory(
            env, ctrl, init_state, env_config['max_steps'], seed=seed)
        states.append(np.asarray(trajectory, dtype=np.float32))
        indices.append(idx)
        lengths.append(len(trajectory))
        labels.append(int(bool(success)))
        seeds.append(seed)
    env.close()
    ctrl.close()
    atomic_savez(shard_path,
                 fingerprint=np.array(fingerprint),
                 indices=np.array(indices, dtype=np.int64),
                 states=np.concatenate(states, axis=0),
                 lengths=np.array(lengths, dtype=np.int64),
                 labels=np.array(labels, dtype=np.uint8),
                 seeds=np.array(seeds, dtype=np.int64))
    return shard_path


def collect_train(controller, output_dir, num_trajs=DEFAULT_NUM_TRAJS, seed=42,
                  horizon=1000, noise=None, parallel=False, num_workers=None,
                  batch_size=256, verbose=False):
    '''Collect the training split: ``num_trajs`` rollouts from random starts.

    Each trajectory is cut at (and includes) the first state inside the goal
    ball (label 1) or runs the full ``horizon`` (label 0). Writes ``train.npz``
    as a flat float32 state array plus per-trajectory offsets and metadata.
    '''
    if controller not in VALID_CONTROLLERS:
        raise ValueError(f'[ERROR] unknown controller {controller!r}; valid: {VALID_CONTROLLERS}')
    os.makedirs(output_dir, exist_ok=True)

    ctrl_freq = pyb_freq = 100  # dt = 0.01, matching the trained-on physics.
    env_config = {'ctrl_freq': ctrl_freq, 'pyb_freq': pyb_freq,
                  'episode_len_sec': math.ceil(horizon / ctrl_freq) + 1,
                  'max_steps': horizon, 'noise': noise, 'invariant': False}

    shards_dir = os.path.join(output_dir, '_shards')
    os.makedirs(shards_dir, exist_ok=True)
    fingerprint = json.dumps({'controller': controller, 'num_trajs': num_trajs,
                              'seed': seed, 'horizon': horizon, 'noise': noise,
                              'batch_size': batch_size}, sort_keys=True)

    init_states = sample_initial_states(num_trajs, True, seed, THETA_DOT_MAX, GRID_RESOLUTION)
    todo = list(enumerate(init_states))
    worker_args = [
        (todo[i:i + batch_size], controller, env_config, seed,
         os.path.join(shards_dir, f'shard_{i // batch_size:06d}.npz'), fingerprint)
        for i in range(0, len(todo), batch_size)]

    if parallel:
        with Pool(processes=num_workers or get_available_cpus()) as pool:
            for _ in tqdm(pool.imap_unordered(_train_worker, worker_args),
                          total=len(worker_args), desc='Train', disable=not verbose):
                pass
    else:
        for args in tqdm(worker_args, desc='Train', disable=not verbose):
            _train_worker(args)

    stats = merge_train_shards(output_dir, [a[4] for a in worker_args], init_states)
    stats['controller'] = controller
    atomic_write_text(os.path.join(output_dir, 'dataset_description.json'), json.dumps({
        'dataset_name': 'Inverted Pendulum Trajectories (train split)',
        'split': 'train',
        'controller': controller,
        'noise': noise,
        'state_order': ['theta', 'theta_dot'],
        'ctrl_freq': ctrl_freq, 'pyb_freq': pyb_freq, 'dt': 1.0 / pyb_freq,
        'horizon_steps': horizon,
        'u_sat': U_SAT,
        'theta_dot_max': THETA_DOT_MAX,
        'seed': seed,
        'sampling': {'type': 'uniform random over the full state space',
                     'theta_range': [-math.pi, math.pi],
                     'theta_dot_range': [-THETA_DOT_MAX, THETA_DOT_MAX]},
        'label_semantics': ('1 = the trajectory was cut at (and includes) the first state '
                            'inside the 0.075 goal ball; 0 = it ran the full horizon. Under '
                            'noise a rollout can enter and drift back out, so cutting at '
                            'entry keeps the label a function of the terminal state.'),
        'data_format': {
            'file': 'train.npz',
            'states_dtype': 'float32',
            'keys': {'states': 'flat (M, 2) array of all trajectories concatenated',
                     'offsets': 'int64 (N+1,); trajectory i is states[offsets[i]:offsets[i+1]]',
                     'starts': 'float64 (N, 2) sampled initial states',
                     'labels': 'uint8 (N,)',
                     'seeds': 'int64 (N,) per-rollout seed; replays the rollout exactly'},
            'note': ('float32 costs 2.4e-7, three orders of magnitude below the smallest '
                     'per-step state change (p1 = 1.7e-4), and halves the size.'),
        },
        **stats,
    }, indent=2))
    return stats


def merge_train_shards(output_dir, shard_paths, init_states):
    '''Concatenate shards (already in index order) into ``train.npz``.'''
    states, lengths, labels, seeds, indices = [], [], [], [], []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as shard:
            states.append(shard['states'])
            lengths.append(shard['lengths'])
            labels.append(shard['labels'])
            seeds.append(shard['seeds'])
            indices.append(shard['indices'])
    indices = np.concatenate(indices)
    if not np.array_equal(indices, np.arange(len(init_states))):
        raise RuntimeError('[ERROR] shards do not cover 0..N-1 exactly; delete _shards and re-run')
    lengths = np.concatenate(lengths)
    labels = np.concatenate(labels)
    atomic_savez(os.path.join(output_dir, 'train.npz'),
                 states=np.concatenate(states, axis=0),
                 offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
                 starts=init_states,
                 labels=labels,
                 seeds=np.concatenate(seeds))
    n = len(init_states)
    return {'num_trajectories': n,
            'success_count': int(labels.sum()),
            'success_rate': float(labels.sum()) / n if n else 0.0,
            'mean_length': float(lengths.mean()) if n else 0.0}


# --- eval split -------------------------------------------------------------

def _eval_worker(args_tuple):
    '''Worker: roll out one chunk of grid cells for one batch.

    Only the outcome is kept -- eval stores probabilities, not trajectories.
    '''
    chunk, controller, env_config, base_seed, batch_no = args_tuple
    env_func = make_env_func(env_config)
    ctrl = make_controller(controller, env_func)
    env = env_func()
    indices, outcomes = [], []
    for idx, init_state in chunk:
        seed = rollout_seed(base_seed, EVAL_SPLIT_ID, idx, batch_no)
        _, success, _ = run_trajectory(
            env, ctrl, init_state, env_config['max_steps'], seed=seed)
        indices.append(idx)
        outcomes.append(int(bool(success)))
    env.close()
    ctrl.close()
    return np.array(indices, dtype=np.int64), np.array(outcomes, dtype=np.int64)


def mean_standard_error(successes, trials):
    '''Mean per-cell uncertainty of the success probability.

    Near-monotone in the batch count, so -- unlike a drift statistic, which is
    itself a noisy sample of the movement it measures -- it cannot trip early
    by chance.

    Uses the posterior SD under a Jeffreys Beta(1/2, 1/2) prior rather than the
    plug-in ``p(1-p)/n``. The plug-in form is **degenerate at the extremes**: a
    cell that came back 10/10 has p-hat = 1 and contributes exactly 0, so at low
    noise -- where most cells really are near 0 or 1 -- the mean would collapse
    to ~0 and stop the run at ``min_batches``, leaving each probability resolved
    only to 1/min_batches. The smoothed form keeps an honest ~1/n floor.
    '''
    trials = np.maximum(trials, 1)
    p = (successes + 0.5) / (trials + 1.0)
    return float(np.mean(np.sqrt(p * (1.0 - p) / (trials + 2.0))))


def load_eval_state(output_dir, n_cells):
    '''Resume from the published dataset; the dataset *is* the checkpoint.'''
    path = os.path.join(output_dir, 'eval_success_prob.npz')
    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as data:
                if len(data['successes']) == n_cells:
                    return (data['successes'].astype(np.int64),
                            data['trials'].astype(np.int64),
                            int(data['n_batches']))
        except (OSError, ValueError, KeyError):
            pass  # truncated or foreign file: start over
    return np.zeros(n_cells, np.int64), np.zeros(n_cells, np.int64), 0


def publish_eval(output_dir, grid, theta_axis, theta_dot_axis, successes, trials,
                 n_batches, description=None, converged=False):
    '''Atomically publish the complete eval dataset.

    Every write goes through a temp file plus ``os.replace``, so the directory
    always holds a loadable, self-consistent dataset -- the run can be killed
    at any moment. Only whole batches reach here, so every cell always carries
    the same ``trials``.

    All three files are staged first and committed with the npz **last**. The
    npz is the authoritative artifact and its presence is the commit point:
    because the mirrors are moved into place before it, an npz on disk always
    has its ``success_probabilities.txt`` and description beside it. A kill
    inside the commit sequence can only leave the mirrors one batch *ahead* of
    the npz, never the npz orphaned.
    '''
    p_success = successes / np.maximum(trials, 1)
    staged = []
    if description is not None:
        staged.append(stage_text(
            os.path.join(output_dir, 'dataset_description.json'),
            json.dumps({**description,
                        'n_batches': int(n_batches),
                        'converged': bool(converged),
                        'mean_se': mean_standard_error(successes, trials),
                        'success_rate': float(p_success.mean())}, indent=2)))
    staged.append(stage_text(
        os.path.join(output_dir, 'success_probabilities.txt'),
        ''.join(f'{t:.6f},{d:.6f},{p:.6f}\n' for (t, d), p in zip(grid, p_success))))
    staged.append(stage_npz(
        os.path.join(output_dir, 'eval_success_prob.npz'),
        starts=grid,
        successes=successes.astype(np.int32),
        trials=trials.astype(np.int32),
        p_success=p_success,
        grid_theta=theta_axis,
        grid_theta_dot=theta_dot_axis,
        grid_shape=np.array([len(theta_axis), len(theta_dot_axis)], dtype=np.int64),
        n_batches=np.int64(n_batches)))
    commit_staged(staged)
    return p_success


def collect_eval(controller, output_dir, seed=42, horizon=1000, noise=None,
                 resolution=GRID_RESOLUTION, se_tol=0.01, min_batches=10,
                 max_batches=500, check_every=10, parallel=False,
                 num_workers=None, chunk_size=512, verbose=False):
    '''Collect the eval split: batches over the grid until the estimate settles.

    One batch is one rollout from every grid state. Only per-cell success
    counts are kept, and the complete dataset is republished after every batch.
    '''
    if controller not in VALID_CONTROLLERS:
        raise ValueError(f'[ERROR] unknown controller {controller!r}; valid: {VALID_CONTROLLERS}')
    os.makedirs(output_dir, exist_ok=True)

    ctrl_freq = pyb_freq = 100  # dt = 0.01, matching the trained-on physics.
    env_config = {'ctrl_freq': ctrl_freq, 'pyb_freq': pyb_freq,
                  'episode_len_sec': math.ceil(horizon / ctrl_freq) + 1,
                  'max_steps': horizon, 'noise': noise, 'invariant': False}

    theta_axis = grid_axis(-math.pi, math.pi, resolution)
    theta_dot_axis = grid_axis(-THETA_DOT_MAX, THETA_DOT_MAX, resolution)
    grid = sample_initial_states(0, False, seed, THETA_DOT_MAX, resolution)
    successes, trials, n_batches = load_eval_state(output_dir, len(grid))

    description = {
        'dataset_name': 'Inverted Pendulum Success Probabilities (eval split)',
        'split': 'eval',
        'controller': controller,
        'noise': noise,
        'num_cells': len(grid),
        'state_order': ['theta', 'theta_dot'],
        'ctrl_freq': ctrl_freq, 'pyb_freq': pyb_freq, 'dt': 1.0 / pyb_freq,
        'horizon_steps': horizon,
        'u_sat': U_SAT,
        'theta_dot_max': THETA_DOT_MAX,
        'seed': seed,
        'grid': {'resolution': resolution,
                 'shape': [len(theta_axis), len(theta_dot_axis)],
                 'theta_range': [float(theta_axis[0]), float(theta_axis[-1])],
                 'theta_dot_range': [float(theta_dot_axis[0]), float(theta_dot_axis[-1])],
                 'note': 'half-open in both coordinates; theta is periodic, so -pi and '
                         '+pi are the same state and only one is sampled'},
        'label_semantics': ('p_success is the fraction of rollouts from that cell that ever '
                            'entered the 0.075 goal ball within horizon_steps'),
        'stopping_rule': {'statistic': 'mean per-cell Jeffreys posterior SD of p_success',
                          'se_tol': se_tol, 'min_batches': min_batches,
                          'max_batches': max_batches, 'check_every': check_every},
        'data_format': {'file': 'eval_success_prob.npz',
                        'note': 'no trajectories are stored; one batch is one rollout from '
                                'every grid cell, and the dataset is republished atomically '
                                'after each batch',
                        'mirror': 'success_probabilities.txt (theta,theta_dot,p_success)'},
    }

    cells = list(enumerate(grid))
    workers = num_workers or get_available_cpus()
    if parallel:
        # Aim for several chunks per worker. A chunk is the unit of parallelism,
        # so too few leaves workers idle, and rollout cost varies a lot per cell
        # (successes stop early, failures run the full horizon) -- oversubscribing
        # lets imap_unordered even that out.
        chunk_size = max(1, min(chunk_size, math.ceil(len(cells) / (4 * workers))))
    chunks = [cells[i:i + chunk_size] for i in range(0, len(cells), chunk_size)]
    converged = False

    # One pool for the whole run: the batch loop can run for hundreds of
    # iterations, and re-forking the workers each time would repay the
    # per-process casadi warmup every batch.
    pool = Pool(processes=workers) if parallel else None
    try:
        while n_batches < max_batches:
            args = [(c, controller, env_config, seed, n_batches) for c in chunks]
            outcomes = np.zeros(len(grid), dtype=np.int64)
            results = pool.imap_unordered(_eval_worker, args) if pool else map(_eval_worker, args)
            for indices, values in results:
                outcomes[indices] = values

            successes += outcomes
            trials += 1
            n_batches += 1

            standard_error = mean_standard_error(successes, trials)
            due_for_check = n_batches >= min_batches and n_batches % check_every == 0
            converged = due_for_check and standard_error < se_tol
            # Publish after the flag is known, so the description on disk is
            # honest about whether this converged or was merely stopped.
            publish_eval(output_dir, grid, theta_axis, theta_dot_axis,
                         successes, trials, n_batches, description, converged)
            if verbose:
                print(f'[eval] batch {n_batches}  mean_se={standard_error:.5f}')
            if converged:
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return {'controller': controller, 'n_batches': n_batches,
            'num_cells': len(grid), 'converged': converged,
            'mean_se': mean_standard_error(successes, trials),
            'success_rate': float((successes / np.maximum(trials, 1)).mean())}


def _load_cache(output_dir):
    path = os.path.join(output_dir, CACHE_NAME)
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def generate(controller, output_dir, num_trajs=100000, random_init=True, seed=42,
             parallel=False, num_workers=None, horizon=None, resolution=0.1,
             theta_dot_max=THETA_DOT_MAX, skip_save=False, overwrite=False,
             batch_size=256, noise=None, invariant=False, verbose=False):
    '''Generate a dataset and return aggregate statistics.

    Resumable: trajectories whose ``sequence_<idx>.txt`` already exists (and whose
    label is cached) are skipped unless ``overwrite`` is set.
    '''
    if controller not in VALID_CONTROLLERS:
        raise ValueError(f'[ERROR] unknown controller {controller!r}; valid: {VALID_CONTROLLERS}')

    trajectories_dir = os.path.join(output_dir, 'trajectories')
    os.makedirs(trajectories_dir if not skip_save else output_dir, exist_ok=True)

    ctrl_freq = pyb_freq = 100  # dt = 0.01, matching the trained-on physics.
    if horizon is None:
        # Default (first-entry termination): 10 s horizon as before. Invariant
        # mode: fixed horizon = old max success length + settle buffer.
        horizon = DEFAULT_HORIZON['lqr' if controller == 'lqr' else 'rl'] if invariant else 1000
    # episode_len_sec only needs to admit `horizon` steps (loop runs exactly that many).
    episode_len_sec = math.ceil(horizon / ctrl_freq) + 1
    env_config = {'ctrl_freq': ctrl_freq, 'pyb_freq': pyb_freq,
                  'episode_len_sec': episode_len_sec, 'max_steps': horizon,
                  'noise': noise, 'invariant': invariant}

    init_states = sample_initial_states(num_trajs, random_init, seed, theta_dot_max, resolution)
    n = len(init_states)

    cache = _load_cache(output_dir)
    todo = []
    for idx in range(n):
        seq = os.path.join(trajectories_dir, f'sequence_{idx}.txt')
        done_already = (not overwrite) and (str(idx) in cache) and (skip_save or os.path.exists(seq))
        if not done_already:
            todo.append((idx, init_states[idx]))

    if verbose:
        print(f'[generate] controller={controller} total={n} to_process={len(todo)} '
              f'(resuming {n - len(todo)})')

    if todo:
        batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
        worker_args = [(b, controller, env_config, trajectories_dir, skip_save) for b in batches]
        if parallel:
            workers = num_workers or get_available_cpus()
            with Pool(processes=workers) as pool:
                for records in tqdm(pool.imap_unordered(_process_batch, worker_args),
                                    total=len(worker_args), desc='Generating', disable=not verbose):
                    for idx, rec in records:
                        cache[str(idx)] = rec
        else:
            for wa in tqdm(worker_args, desc='Generating', disable=not verbose):
                for idx, rec in _process_batch(wa):
                    cache[str(idx)] = rec

    # Persist cache + write ordered roa labels + dataset description.
    with open(os.path.join(output_dir, CACHE_NAME), 'w') as f:
        json.dump(cache, f)

    success_count = timeout_count = 0
    success_v = []
    with open(os.path.join(output_dir, 'roa_labels.txt'), 'w') as f:
        for idx in range(n):
            rec = cache[str(idx)]
            theta, thetadot = rec['init_state']
            f.write(f'{theta:.6f},{thetadot:.6f},{rec["label"]}\n')
            success_count += int(rec['success'])
            timeout_count += int(rec.get('timeout', False))
            if rec['success'] and rec.get('terminal_v_over_c') is not None:
                success_v.append(rec['terminal_v_over_c'])

    stats = {
        'controller': controller,
        'num_trajectories': n,
        'total_count': n,
        'success_count': success_count,
        'timeout_count': timeout_count,
        'success_rate': success_count / n if n else 0.0,
    }
    description = {
        **stats,
        'state_order': ['theta', 'theta_dot'],
        'ctrl_freq': ctrl_freq, 'pyb_freq': pyb_freq, 'dt': 1.0 / pyb_freq,
        'episode_len_sec': episode_len_sec, 'horizon_steps': horizon,
        'u_sat': U_SAT,
        'theta_dot_max': theta_dot_max,
        'noise': noise if isinstance(noise, (str, type(None))) else str(noise),
        'random_init': random_init, 'seed': seed,
        'invariant_terminal_sets': invariant,
    }
    if invariant:
        P, center, c = load_invariant_set()
        success_v = np.array(success_v) if success_v else np.array([np.nan])
        # margin diagnostic: should be << 1; a tail near 1 means late arrivals
        # that entered the ellipsoid shortly before the horizon (consider a
        # larger horizon and resume).
        stats['terminal_v_over_c'] = {
            'p50': float(np.nanpercentile(success_v, 50)),
            'p95': float(np.nanpercentile(success_v, 95)),
            'max': float(np.nanmax(success_v)),
        }
        description['terminal_v_over_c'] = stats['terminal_v_over_c']
        description['termination'] = ('none: every trajectory runs exactly horizon_steps '
                                      '(theta wrapped, theta_dot clipped -> no out-of-bounds failure)')
        description['label_semantics'] = ('1 iff the terminal state lies in the invariant success '
                                          'ellipsoid (equivalent, by invariance, to ever entering it)')
        description['invariant_set'] = {
            'definition': "(s - center)' P (s - center) <= c",
            'P': P.tolist(),
            'center': center.tolist(),
            'c': c,
            'artifact': os.path.relpath(INVARIANT_SET_PATH,
                                        os.path.dirname(os.path.abspath(__file__))),
            'reference': 'plans/invariant-terminal-sets-recollection.md',
        }
    else:
        description['termination'] = ('trajectories are cut at (and include) the first state '
                                      'within the 0.075 goal threshold; non-successful '
                                      'trajectories run to the full horizon')
        description['label_semantics'] = ('1 = reached upright goal within horizon, 0 = timeout '
                                          '(theta wrapped, theta_dot clipped -> no out-of-bounds failure)')
    with open(os.path.join(output_dir, 'dataset_description.json'), 'w') as f:
        json.dump(description, f, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser(description='Generate inverted-pendulum trajectory dataset')
    parser.add_argument('--controller', required=True, choices=VALID_CONTROLLERS,
                        help='lqr, or a standalone RL policy vX_{strong,weak}')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: .../genMoPlan/data_trajectories/inverted_pendulum_<controller>)')
    parser.add_argument('--split', choices=['train', 'eval'], default=None,
                        help='train: num_trajs rollouts from random starts, full trajectories '
                             'stored. eval: batches over the grid storing only the per-cell '
                             'success probability, until it settles. Omit for the legacy '
                             'sequence_<i>.txt output.')
    parser.add_argument('--num_trajs', type=int, default=None,
                        help=f'Number of random trajectories (default: {DEFAULT_NUM_TRAJS} '
                             'with --split train, else 100000)')
    parser.add_argument('--random_init', action='store_true', help='Random sampling (else discretized grid)')
    parser.add_argument('--resolution', type=float, default=None,
                        help=f'Grid resolution (default: {GRID_RESOLUTION} with --split eval, '
                             'else 0.1)')
    parser.add_argument('--se_tol', type=float, default=0.01,
                        help='--split eval: stop once the mean per-cell uncertainty is below this')
    parser.add_argument('--min_batches', type=int, default=10, help='--split eval: floor on batches')
    parser.add_argument('--max_batches', type=int, default=500, help='--split eval: cap on batches')
    parser.add_argument('--check_every', type=int, default=10,
                        help='--split eval: test the stopping rule every N batches')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None,
                        help='Trajectory horizon in steps (default: 1000; with '
                             '--invariant_terminal_sets: 600 for lqr, 1100 for RL)')
    parser.add_argument('--noise', type=str, default=None, choices=sorted(NOISE_PRESETS),
                        help='Noise preset (default: none/deterministic). See envs/gym_control/pendulum_noise.py')
    parser.add_argument('--invariant_terminal_sets', action='store_true',
                        help='Disable goal termination: run every trajectory for a fixed horizon '
                             'and label by terminal-state membership in the invariant ellipsoid '
                             '(plans/invariant-terminal-sets-recollection.md). Default: terminate '
                             'at (and include) the first state within the 0.075 goal threshold.')
    parser.add_argument('--skip_save', action='store_true', help='Compute labels/stats without writing sequences')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate even if sequence files exist')
    args = parser.parse_args()

    noise = None if args.noise in (None, 'none') else args.noise

    if args.split is not None:
        output_dir = args.output_dir or default_output_dir(args.controller, noise)
        if args.split == 'train':
            stats = collect_train(args.controller, output_dir,
                                  num_trajs=args.num_trajs or DEFAULT_NUM_TRAJS,
                                  seed=args.seed, horizon=args.horizon or 1000,
                                  noise=noise, parallel=args.parallel,
                                  num_workers=args.num_workers, verbose=True)
            print(f"\n{'=' * 70}")
            print(f"Split:        train ({stats['controller']}, noise={noise})")
            print(f"Trajectories: {stats['num_trajectories']}")
            print(f"Successful:   {stats['success_count']} ({stats['success_rate'] * 100:.2f}%)")
            print(f"Mean length:  {stats['mean_length']:.1f} states")
        else:
            stats = collect_eval(args.controller, output_dir, seed=args.seed,
                                 horizon=args.horizon or 1000, noise=noise,
                                 resolution=args.resolution or GRID_RESOLUTION,
                                 se_tol=args.se_tol, min_batches=args.min_batches,
                                 max_batches=args.max_batches, check_every=args.check_every,
                                 parallel=args.parallel, num_workers=args.num_workers,
                                 verbose=True)
            print(f"\n{'=' * 70}")
            print(f"Split:        eval ({stats['controller']}, noise={noise})")
            print(f"Grid cells:   {stats['num_cells']}")
            print(f"Batches:      {stats['n_batches']} "
                  f"({'converged' if stats['converged'] else 'STOPPED AT CAP'})")
            print(f"Mean SE:      {stats['mean_se']:.5f}")
            print(f"Mean p:       {stats['success_rate']:.4f}")
        print(f'Output:       {output_dir}')
        print(f"{'=' * 70}")
        return

    suffix = f'_{args.noise}' if noise else ''
    suffix += '_invariant' if args.invariant_terminal_sets else ''
    output_dir = args.output_dir or (
        f'/common/users/shared/pracsys/genMoPlan/data_trajectories/inverted_pendulum_{args.controller}{suffix}')

    stats = generate(args.controller, output_dir, num_trajs=args.num_trajs or 100000,
                     random_init=args.random_init, seed=args.seed, parallel=args.parallel,
                     num_workers=args.num_workers, horizon=args.horizon,
                     resolution=args.resolution or 0.1, skip_save=args.skip_save,
                     overwrite=args.overwrite, noise=noise,
                     invariant=args.invariant_terminal_sets, verbose=True)

    print(f"\n{'=' * 70}")
    print(f"Controller:   {stats['controller']}")
    print(f"Trajectories: {stats['num_trajectories']}")
    print(f"Successful:   {stats['success_count']} ({stats['success_rate'] * 100:.2f}%)")
    print(f"Timed out:    {stats['timeout_count']}")
    if 'terminal_v_over_c' in stats:
        v = stats['terminal_v_over_c']
        print(f"Terminal V/c: p50={v['p50']:.4f} p95={v['p95']:.4f} max={v['max']:.4f}")
    print(f'Output:       {output_dir}')
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
