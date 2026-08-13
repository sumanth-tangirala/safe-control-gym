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
# Torque-noise success rule: the FIRST state with every channel inside the box,
# with the trajectory cut there. The 0.05 tolerances replace the shipped L2 ball,
# which adds radians to rad/s with equal weight.
#
# BOX_HOLD = 1 means no dwell [user, 2026-08-06]. A dwell was insurance against
# the state-additive teleport, where one noise draw could place the state in the
# goal set; a torque disturbance cannot do that, so it bought nothing here and
# cost the invariant the entry-cut exists to protect. With a dwell, a rollout
# that visited the box without holding it ran on to the horizon and could be
# stored ending INSIDE the box with label 0 -- measured at tau=0.50, 9,863 of
# 100,000 trajectories, so the same terminal state carried both labels. Stopping
# at first entry makes `terminal state in the box` and `label 1` the same
# statement again.
BOX_TOL = np.array([0.05, 0.05])
BOX_HOLD = 1
INVARIANT_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'invariant_sets', 'pendulum.npz')
# Fixed horizons (steps): old max success length + settle buffer
# (plans/invariant-terminal-sets-recollection.md).
DEFAULT_HORIZON = {'lqr': 600, 'rl': 1100}


def validate_timing(ctrl_freq, pyb_freq):
    '''Validate and normalize the control/integration frequencies.'''
    ctrl_freq, pyb_freq = int(ctrl_freq), int(pyb_freq)
    if ctrl_freq <= 0 or pyb_freq <= 0:
        raise ValueError('ctrl_freq and pyb_freq must be positive integers')
    if pyb_freq % ctrl_freq != 0:
        raise ValueError('pyb_freq must be an integer multiple of ctrl_freq')
    return ctrl_freq, pyb_freq


def timing_description(ctrl_freq, pyb_freq):
    '''Self-describing timing metadata shared by every collection path.'''
    ctrl_freq, pyb_freq = validate_timing(ctrl_freq, pyb_freq)
    return {
        'ctrl_freq': ctrl_freq,
        'pyb_freq': pyb_freq,
        # Keep ``dt`` as the numerical integration step for compatibility.
        'dt': 1.0 / pyb_freq,
        'control_dt': 1.0 / ctrl_freq,
        'integration_dt': 1.0 / pyb_freq,
        'substeps_per_control': pyb_freq // ctrl_freq,
    }


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


def default_output_dir(controller, noise, torque_noise=None):
    '''Dataset location, following the ``<family>/pendulum/<controller>/`` layout.

    Torque-noise datasets get their own family rather than a new level under
    ``noisy/``. They share no vocabulary with the preset levels and are not
    comparable to them -- different mechanism, different units -- so filing
    ``tau_0.10`` beside ``high`` would invite exactly the wrong comparison.
    ``tau = 0.0`` still lands in the torque family, so all levels of a sweep come
    from one pipeline.
    '''
    if torque_noise is not None:
        return os.path.join(DATA_ROOT, 'noisy_torque', 'pendulum', controller,
                            f'tau_{torque_noise:.2f}')
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
    tau = env_config.get('torque_noise')
    if tau is not None:
        # Uniform noise on the commanded torque. Applied in _preprocess_control,
        # i.e. before the u_sat clip, so a saturated actuator cannot be pushed
        # further -- physically right, and it biases p slightly up relative to a
        # disturbance acting on the shaft.
        kwargs['disturbances'] = {'action': [{'disturbance_func': 'uniform',
                                              'low': -tau, 'high': tau}]}
        # The box+dwell rule owns termination; the env's own L2 test must not
        # fire first, or it would cut trajectories on the criterion we replaced.
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


def run_trajectory(env, ctrl, init_state, max_steps, invariant=False, seed=None,
                   box_rule=False):
    '''Roll out one trajectory from ``init_state``.

    Default: terminate at (and include) the first state within the goal
    threshold; returns ``(trajectory, success, timeout)``.

    ``invariant=True``: no early termination; roll exactly ``max_steps`` steps
    and return ``(trajectory, None, False)`` -- the success label is decided
    afterwards from the terminal state.

    ``box_rule=True``: success is ``|theta| < 0.05 and |theta_dot| < 0.05``, and
    the rollout STOPS at the first state satisfying it, which is also the last
    state stored. With ``BOX_HOLD = 1`` (no dwell) this makes the two statements
    `terminal state is in the box` and `label is 1` equivalent in both
    directions: a success ends there by construction, and a failure can never end
    there because reaching the box would have terminated it. Requires
    ``goal_threshold=0`` on the env so its L2 test cannot fire first
    (``make_env_func`` sets this).

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
    run = 0  # consecutive steps inside the box, for box_rule
    for _ in range(max_steps):
        obs_in = ctrl.obs_normalizer(obs) if hasattr(ctrl, 'obs_normalizer') else obs
        action = ctrl.select_action(obs_in, info)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        trajectory.append([normalize_angle(env.state[0]), float(env.state[1])])
        if box_rule:
            run = run + 1 if np.all(np.abs(env.state) < BOX_TOL) else 0
            if run >= BOX_HOLD:
                # Cut back to the state that entered the window. At BOX_HOLD = 1
                # that state is the one just appended and the slice is empty.
                del trajectory[len(trajectory) - run + 1:]
                success = True
                break
            if done:
                break
        elif not invariant and done:
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
    box_rule = env_config.get('torque_noise') is not None
    for idx, init_state in batch:
        trajectory, success, timeout = run_trajectory(
            env, ctrl, init_state, env_config['max_steps'], invariant=invariant,
            box_rule=box_rule)
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
    box_rule = env_config.get('torque_noise') is not None
    for idx, init_state in batch:
        seed = rollout_seed(base_seed, TRAIN_SPLIT_ID, idx)
        trajectory, success, _ = run_trajectory(
            env, ctrl, init_state, env_config['max_steps'], seed=seed,
            box_rule=box_rule)
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
                  batch_size=256, verbose=False, torque_noise=None,
                  ctrl_freq=100, pyb_freq=100):
    '''Collect the training split: ``num_trajs`` rollouts from random starts.

    Each trajectory is cut at (and includes) the first state inside the goal
    ball (label 1) or runs the full ``horizon`` (label 0). Writes ``train.npz``
    as a flat float32 state array plus per-trajectory offsets and metadata.
    '''
    if controller not in VALID_CONTROLLERS:
        raise ValueError(f'[ERROR] unknown controller {controller!r}; valid: {VALID_CONTROLLERS}')
    os.makedirs(output_dir, exist_ok=True)

    ctrl_freq, pyb_freq = validate_timing(ctrl_freq, pyb_freq)
    env_config = {'ctrl_freq': ctrl_freq, 'pyb_freq': pyb_freq,
                  'episode_len_sec': math.ceil(horizon / ctrl_freq) + 1,
                  'max_steps': horizon, 'noise': noise, 'invariant': False,
                  'torque_noise': torque_noise}

    shards_dir = os.path.join(output_dir, '_shards')
    os.makedirs(shards_dir, exist_ok=True)
    fingerprint = json.dumps({'controller': controller, 'num_trajs': num_trajs,
                              'seed': seed, 'horizon': horizon, 'noise': noise,
                              'torque_noise': torque_noise,
                              'ctrl_freq': ctrl_freq, 'pyb_freq': pyb_freq,
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
    atomic_write_text(os.path.join(output_dir, 'train_description.json'), json.dumps({
        'dataset_name': 'Inverted Pendulum Trajectories (train split)',
        'split': 'train',
        'controller': controller,
        'noise': noise,
        'torque_noise': torque_noise,
        'noise_mechanism': ('uniform on commanded torque, pre-saturation'
                            if torque_noise is not None else
                            ('state-additive preset' if noise else 'none')),
        'state_order': ['theta', 'theta_dot'],
        **timing_description(ctrl_freq, pyb_freq),
        'horizon_steps': horizon,
        'u_sat': U_SAT,
        'fraction_of_u_sat': (None if torque_noise is None else torque_noise / U_SAT),
        'theta_dot_max': THETA_DOT_MAX,
        'seed': seed,
        'sampling': {'type': 'uniform random over the full state space',
                     'theta_range': [-math.pi, math.pi],
                     'theta_dot_range': [-THETA_DOT_MAX, THETA_DOT_MAX]},
        'label_semantics': (
            f'1 = the trajectory was cut at (and includes) the first state with '
            f'|theta| < {BOX_TOL[0]} and |theta_dot| < {BOX_TOL[1]}; 0 = it ran the full '
            'horizon. The rollout STOPS on entry, so a label-0 trajectory can never end '
            'inside the box and the label is a function of the terminal state.'
            if torque_noise is not None else
            '1 = the trajectory was cut at (and includes) the first state '
            'inside the 0.075 goal ball; 0 = it ran the full horizon. Under '
            'noise a rollout can enter and drift back out, so cutting at '
            'entry keeps the label a function of the terminal state.'),
        'success_rule': ({'kind': ('per_channel_box_entry' if BOX_HOLD == 1
                                   else 'per_channel_box_with_dwell'),
                          'tol': BOX_TOL.tolist(), 'hold_steps': BOX_HOLD}
                         if torque_noise is not None else
                         {'kind': 'l2_ball', 'radius': 0.075}),
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
    box_rule = env_config.get('torque_noise') is not None
    for idx, init_state in chunk:
        seed = rollout_seed(base_seed, EVAL_SPLIT_ID, idx, batch_no)
        _, success, _ = run_trajectory(
            env, ctrl, init_state, env_config['max_steps'], seed=seed,
            box_rule=box_rule)
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


def _load_shard_state(path, n_cells, batch_lo):
    """Resume a shard from itself. Mirrors load_eval_state for the whole dataset."""
    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as d:
                if len(d['successes']) == n_cells and int(d['batch_lo']) == batch_lo:
                    return (d['successes'].astype(np.int64),
                            d['trials'].astype(np.int64), int(d['batch_hi']))
        except (OSError, ValueError, KeyError):
            pass  # truncated or foreign file: start this shard over
    return np.zeros(n_cells, np.int64), np.zeros(n_cells, np.int64), batch_lo


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
            os.path.join(output_dir, 'eval_description.json'),
            json.dumps({**description,
                        'n_batches': int(n_batches),
                        'converged': bool(converged),
                        'mean_se': mean_standard_error(successes, trials),
                        'success_rate': float(p_success.mean())}, indent=2)))
    # State dimension is read from the grid, not assumed: the cartpole collector
    # shares this writer and its state is 4-D. For a 2-D grid the output is
    # byte-identical to the previous pendulum-only form.
    staged.append(stage_text(
        os.path.join(output_dir, 'success_probabilities.txt'),
        ''.join(','.join(f'{v:.6f}' for v in row) + f',{p:.6f}\n'
                for row, p in zip(np.atleast_2d(grid), p_success))))
    arrays = dict(starts=grid,
                  successes=successes.astype(np.int32),
                  trials=trials.astype(np.int32),
                  p_success=p_success,
                  n_batches=np.int64(n_batches))
    # The per-axis arrays only exist for a system whose grid was built from axes.
    if len(theta_axis) and len(theta_dot_axis):
        arrays.update(grid_theta=theta_axis, grid_theta_dot=theta_dot_axis,
                      grid_shape=np.array([len(theta_axis), len(theta_dot_axis)],
                                          dtype=np.int64))
    staged.append(stage_npz(os.path.join(output_dir, 'eval_success_prob.npz'), **arrays))
    commit_staged(staged)
    return p_success


def eval_description(controller, noise, torque_noise, grid, theta_axis, theta_dot_axis,
                     ctrl_freq, pyb_freq, horizon, seed, resolution,
                     se_tol, min_batches, max_batches, check_every):
    '''The eval dataset's provenance block.

    Shared by the collector and by --merge_eval_shards. Merging used to pass
    description=None, so every merged dataset landed with no
    eval_description.json beside it -- no tau, no success rule, no seed.
    '''
    return {
        'dataset_name': 'Inverted Pendulum Success Probabilities (eval split)',
        'split': 'eval',
        'controller': controller,
        'noise': noise,
        'torque_noise': torque_noise,
        'noise_mechanism': ('uniform on commanded torque, pre-saturation'
                            if torque_noise is not None else
                            ('state-additive preset' if noise else 'none')),
        'num_cells': len(grid),
        'state_order': ['theta', 'theta_dot'],
        **timing_description(ctrl_freq, pyb_freq),
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
        'label_semantics': (
            'p_success is the fraction of rollouts from that cell that held '
            f'|theta| < {BOX_TOL[0]} and |theta_dot| < {BOX_TOL[1]} for {BOX_HOLD} '
            'consecutive control steps within horizon_steps'
            if torque_noise is not None else
            'p_success is the fraction of rollouts from that cell that ever '
            'entered the 0.075 goal ball within horizon_steps'),
        'success_rule': ({'kind': ('per_channel_box_entry' if BOX_HOLD == 1
                                   else 'per_channel_box_with_dwell'),
                          'tol': BOX_TOL.tolist(), 'hold_steps': BOX_HOLD,
                          'cut': 'rollout stops at, and stores, the first state inside the box'}
                         if torque_noise is not None else
                         {'kind': 'l2_ball', 'radius': 0.075, 'cut': 'first entry'}),
        'stopping_rule': {'statistic': 'mean per-cell Jeffreys posterior SD of p_success',
                          'se_tol': se_tol, 'min_batches': min_batches,
                          'max_batches': max_batches, 'check_every': check_every},
        'data_format': {'file': 'eval_success_prob.npz',
                        'note': 'no trajectories are stored; one batch is one rollout from '
                                'every grid cell, and the dataset is republished atomically '
                                'after each batch',
                        'mirror': 'success_probabilities.txt (theta,theta_dot,p_success)'},
    }


def shard_path(output_dir, batch_lo, batch_hi):
    return os.path.join(output_dir, f'eval_shard_{batch_lo:05d}_{batch_hi:05d}.npz')


def merge_eval_shards(output_dir, grid, theta_axis, theta_dot_axis, description=None):
    """Sum the per-batch shards into the canonical whole-grid dataset.

    Sharding is sound because eval batches are INDEPENDENT: batch b's outcome
    does not depend on batch b-1, and rollout_seed(base, EVAL_SPLIT_ID, cell,
    batch) is a pure function of its coordinates. So the same rollouts happen
    whichever node runs them, and successes/trials are counters that add. The
    merged result is bit-identical to the sequential run -- there is a test.

    Refuses to merge a batch range that is not exactly tiled: an overlap would
    double-count and a gap would silently under-report `trials`, and both look
    like a perfectly ordinary dataset afterwards.
    """
    paths = sorted(f for f in os.listdir(output_dir)
                   if f.startswith('eval_shard_') and f.endswith('.npz'))
    if not paths:
        raise FileNotFoundError(f'no eval_shard_*.npz in {output_dir}')
    successes = np.zeros(len(grid), np.int64)
    trials = np.zeros(len(grid), np.int64)
    covered = []
    for name in paths:
        with np.load(os.path.join(output_dir, name), allow_pickle=False) as d:
            lo, hi = int(d['batch_lo']), int(d['batch_hi'])
            if len(d['successes']) != len(grid):
                raise ValueError(f'{name}: grid size {len(d["successes"])} != {len(grid)}')
            successes += d['successes'].astype(np.int64)
            trials += d['trials'].astype(np.int64)
            covered.append((lo, hi))
    covered.sort()
    expected = 0
    for lo, hi in covered:
        if lo != expected:
            raise ValueError(f'batch coverage is not contiguous: expected batch {expected}, '
                             f'shard starts at {lo}. Ranges: {covered}')
        expected = hi
    n_batches = expected
    if not np.all(trials == n_batches):
        raise ValueError('per-cell trials disagree with the batch count; a shard is partial')
    # `converged` is the substantive claim -- did the estimate settle -- so it is
    # decided by the ACHIEVED uncertainty, not by which loop stopped. A shard
    # budget cannot fire the stopping rule, but it can still land under it, and a
    # reader wants to know that rather than which code path ran.
    se_tol = (description or {}).get('stopping_rule', {}).get('se_tol', 0.01)
    settled = mean_standard_error(successes, trials) < se_tol
    if description is not None:
        description = {**description, 'stopped_by': 'batch_budget'}
    publish_eval(output_dir, grid, theta_axis, theta_dot_axis, successes, trials,
                 n_batches, description, converged=settled)
    return {'n_batches': n_batches, 'shards': len(paths), 'num_cells': len(grid),
            'success_rate': float((successes / np.maximum(trials, 1)).mean()),
            'mean_se': mean_standard_error(successes, trials)}


def collect_eval(controller, output_dir, seed=42, horizon=1000, noise=None,
                 torque_noise=None,
                 resolution=GRID_RESOLUTION, se_tol=0.01, min_batches=10,
                 max_batches=500, check_every=10, parallel=False,
                 num_workers=None, chunk_size=512, verbose=False,
                 batch_offset=None, batch_count=None,
                 ctrl_freq=100, pyb_freq=100):
    '''Collect the eval split: batches over the grid until the estimate settles.

    One batch is one rollout from every grid state. Only per-cell success
    counts are kept, and the complete dataset is republished after every batch.

    ``batch_offset``/``batch_count``: run only batches
    ``[offset, offset + count)`` and write them to their own shard file, for
    fanning one grid across many nodes. The stopping rule does not apply --
    a shard runs its assigned batches and stops -- so the caller picks the
    batch budget up front and `merge_eval_shards` combines them. The default
    (both None) is unchanged: one process, whole-grid atomic publication
    after every batch, stopping when the estimate settles.
    '''
    if controller not in VALID_CONTROLLERS:
        raise ValueError(f'[ERROR] unknown controller {controller!r}; valid: {VALID_CONTROLLERS}')
    os.makedirs(output_dir, exist_ok=True)

    ctrl_freq, pyb_freq = validate_timing(ctrl_freq, pyb_freq)
    env_config = {'ctrl_freq': ctrl_freq, 'pyb_freq': pyb_freq,
                  'episode_len_sec': math.ceil(horizon / ctrl_freq) + 1,
                  'max_steps': horizon, 'noise': noise, 'invariant': False,
                  'torque_noise': torque_noise}

    theta_axis = grid_axis(-math.pi, math.pi, resolution)
    theta_dot_axis = grid_axis(-THETA_DOT_MAX, THETA_DOT_MAX, resolution)
    grid = sample_initial_states(0, False, seed, THETA_DOT_MAX, resolution)
    sharded = batch_offset is not None
    if sharded:
        if batch_count is None or batch_count < 1:
            raise ValueError('batch_offset requires a positive batch_count')
        batch_lo, batch_hi = int(batch_offset), int(batch_offset) + int(batch_count)
        spath = shard_path(output_dir, batch_lo, batch_hi)
        successes, trials, n_batches = _load_shard_state(spath, len(grid), batch_lo)
    else:
        successes, trials, n_batches = load_eval_state(output_dir, len(grid))

    description = eval_description(
        controller, noise, torque_noise, grid, theta_axis, theta_dot_axis,
        ctrl_freq, pyb_freq, horizon, seed, resolution, se_tol, min_batches,
        max_batches, check_every)

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
    stop_at = batch_hi if sharded else max_batches
    try:
        while n_batches < stop_at:
            args = [(c, controller, env_config, seed, n_batches) for c in chunks]
            # n_batches is the GLOBAL batch index in both modes, so a cell's
            # noise draw does not depend on which shard ran it.
            outcomes = np.zeros(len(grid), dtype=np.int64)
            results = pool.imap_unordered(_eval_worker, args) if pool else map(_eval_worker, args)
            for indices, values in results:
                outcomes[indices] = values

            successes += outcomes
            trials += 1
            n_batches += 1

            if sharded:
                # The shard is its own checkpoint, atomically replaced each
                # batch, so a killed shard resumes where it stopped.
                atomic_savez(spath,
                             successes=successes.astype(np.int32),
                             trials=trials.astype(np.int32),
                             batch_lo=np.int64(batch_lo),
                             batch_hi=np.int64(n_batches))
                if verbose:
                    print(f'[eval shard {batch_lo}-{batch_hi}] batch {n_batches}', flush=True)
                continue

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

    if sharded:
        return {'controller': controller, 'n_batches': n_batches,
                'num_cells': len(grid), 'converged': False, 'shard': [batch_lo, batch_hi],
                'mean_se': mean_standard_error(successes, trials),
                'success_rate': float((successes / np.maximum(trials, 1)).mean())}
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
             batch_size=256, noise=None, invariant=False, verbose=False,
             torque_noise=None, ctrl_freq=100, pyb_freq=100):
    '''Generate a dataset and return aggregate statistics.

    Resumable: trajectories whose ``sequence_<idx>.txt`` already exists (and whose
    label is cached) are skipped unless ``overwrite`` is set.
    '''
    if controller not in VALID_CONTROLLERS:
        raise ValueError(f'[ERROR] unknown controller {controller!r}; valid: {VALID_CONTROLLERS}')

    trajectories_dir = os.path.join(output_dir, 'trajectories')
    os.makedirs(trajectories_dir if not skip_save else output_dir, exist_ok=True)

    ctrl_freq, pyb_freq = validate_timing(ctrl_freq, pyb_freq)
    if horizon is None:
        # Default (first-entry termination): 10 s horizon as before. Invariant
        # mode: fixed horizon = old max success length + settle buffer.
        horizon = DEFAULT_HORIZON['lqr' if controller == 'lqr' else 'rl'] if invariant else 1000
    # episode_len_sec only needs to admit `horizon` steps (loop runs exactly that many).
    episode_len_sec = math.ceil(horizon / ctrl_freq) + 1
    env_config = {'ctrl_freq': ctrl_freq, 'pyb_freq': pyb_freq,
                  'episode_len_sec': episode_len_sec, 'max_steps': horizon,
                  'noise': noise, 'invariant': invariant,
                  'torque_noise': torque_noise}

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
        **timing_description(ctrl_freq, pyb_freq),
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


def _noise_desc(noise, torque_noise):
    '''One-line mechanism description for the run banner.'''
    if torque_noise is not None:
        return f'torque tau={torque_noise:g} ({100 * torque_noise / U_SAT:.1f}% of u_sat)'
    return f'noise={noise}'


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
    parser.add_argument('--ctrl_freq', type=int, default=100,
                        help='Control/update frequency in Hz (default: 100).')
    parser.add_argument('--pyb_freq', type=int, default=100,
                        help='Explicit-Euler integration frequency in Hz; must be an integer '
                             'multiple of --ctrl_freq (default: 100).')
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
    parser.add_argument('--batch_offset', type=int, default=None,
                        help='--split eval: run only batches [OFFSET, OFFSET+--batch_count) '
                             'into their own shard file, to fan one grid across many nodes. '
                             'The stopping rule does not apply to a shard.')
    parser.add_argument('--batch_count', type=int, default=None,
                        help='--split eval: number of batches this shard runs.')
    parser.add_argument('--merge_eval_shards', action='store_true',
                        help='--split eval: combine eval_shard_*.npz in --output_dir into the '
                             'canonical dataset and exit. Refuses a range with gaps or overlaps.')
    parser.add_argument('--torque_noise', type=float, default=None,
                        help='Uniform noise on the commanded torque, U(-TAU, TAU) per control '
                             'step, applied before the u_sat clip. The physically admissible '
                             'channel: it enters the acceleration row only. Switches the success '
                             f'rule to |theta| < {BOX_TOL[0]} and |theta_dot| < {BOX_TOL[1]} held '
                             f'for {BOX_HOLD} steps, and writes to the noisy_torque/ family. '
                             'Mutually exclusive with --noise.')
    parser.add_argument('--invariant_terminal_sets', action='store_true',
                        help='Disable goal termination: run every trajectory for a fixed horizon '
                             'and label by terminal-state membership in the invariant ellipsoid '
                             '(plans/invariant-terminal-sets-recollection.md). Default: terminate '
                             'at (and include) the first state within the 0.075 goal threshold.')
    parser.add_argument('--skip_save', action='store_true', help='Compute labels/stats without writing sequences')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate even if sequence files exist')
    args = parser.parse_args()

    noise = None if args.noise in (None, 'none') else args.noise
    if noise is not None and args.torque_noise is not None:
        # Rejecting the combination beats defining a precedence nobody remembers.
        parser.error('--noise and --torque_noise are different mechanisms and cannot be '
                     'combined; pick one.')
    if args.torque_noise is not None and args.torque_noise < 0:
        parser.error('--torque_noise must be non-negative (it is a half-width).')
    try:
        validate_timing(args.ctrl_freq, args.pyb_freq)
    except ValueError as exc:
        parser.error(str(exc))

    if args.split is not None:
        output_dir = args.output_dir or default_output_dir(
            args.controller, noise, args.torque_noise)
        if args.split == 'train':
            stats = collect_train(args.controller, output_dir,
                                  num_trajs=args.num_trajs or DEFAULT_NUM_TRAJS,
                                  seed=args.seed, horizon=args.horizon or 1000,
                                  noise=noise, parallel=args.parallel,
                                  num_workers=args.num_workers, verbose=True,
                                  torque_noise=args.torque_noise,
                                  ctrl_freq=args.ctrl_freq, pyb_freq=args.pyb_freq)
            print(f"\n{'=' * 70}")
            print(f"Split:        train ({stats['controller']}, {_noise_desc(noise, args.torque_noise)})")
            print(f"Trajectories: {stats['num_trajectories']}")
            print(f"Successful:   {stats['success_count']} ({stats['success_rate'] * 100:.2f}%)")
            print(f"Mean length:  {stats['mean_length']:.1f} states")
        else:
            if args.merge_eval_shards:
                theta_axis = grid_axis(-math.pi, math.pi, args.resolution or GRID_RESOLUTION)
                theta_dot_axis = grid_axis(-THETA_DOT_MAX, THETA_DOT_MAX,
                                           args.resolution or GRID_RESOLUTION)
                grid = sample_initial_states(0, False, args.seed, THETA_DOT_MAX,
                                             args.resolution or GRID_RESOLUTION)
                desc = eval_description(
                    args.controller, noise, args.torque_noise, grid, theta_axis,
                    theta_dot_axis, args.ctrl_freq, args.pyb_freq,
                    args.horizon or 1000, args.seed,
                    args.resolution or GRID_RESOLUTION, args.se_tol,
                    args.min_batches, args.max_batches, args.check_every)
                stats = merge_eval_shards(output_dir, grid, theta_axis, theta_dot_axis,
                                          desc)
                print(f"\n{'=' * 70}")
                print(f"Merged:       {stats['shards']} shards, {stats['n_batches']} batches")
                print(f"Grid cells:   {stats['num_cells']}")
                print(f"Mean SE:      {stats['mean_se']:.5f}")
                print(f"Mean p:       {stats['success_rate']:.4f}")
                print(f'Output:       {output_dir}')
                print(f"{'=' * 70}")
                return
            stats = collect_eval(args.controller, output_dir, seed=args.seed,
                                 horizon=args.horizon or 1000, noise=noise,
                                 torque_noise=args.torque_noise,
                                 resolution=args.resolution or GRID_RESOLUTION,
                                 se_tol=args.se_tol, min_batches=args.min_batches,
                                 max_batches=args.max_batches, check_every=args.check_every,
                                 parallel=args.parallel, num_workers=args.num_workers,
                                 verbose=True, batch_offset=args.batch_offset,
                                 batch_count=args.batch_count,
                                 ctrl_freq=args.ctrl_freq, pyb_freq=args.pyb_freq)
            print(f"\n{'=' * 70}")
            print(f"Split:        eval ({stats['controller']}, {_noise_desc(noise, args.torque_noise)})")
            print(f"Grid cells:   {stats['num_cells']}")
            print(f"Batches:      {stats['n_batches']} "
                  f"({'converged' if stats['converged'] else 'STOPPED AT CAP'})")
            print(f"Mean SE:      {stats['mean_se']:.5f}")
            print(f"Mean p:       {stats['success_rate']:.4f}")
        print(f'Output:       {output_dir}')
        print(f"{'=' * 70}")
        return

    if args.torque_noise is not None:
        parser.error('--torque_noise requires --split train or --split eval; the legacy '
                     'single-pass path does not carry it.')

    suffix = f'_{args.noise}' if noise else ''
    suffix += '_invariant' if args.invariant_terminal_sets else ''
    output_dir = args.output_dir or (
        f'/common/users/shared/pracsys/genMoPlan/data_trajectories/inverted_pendulum_{args.controller}{suffix}')

    stats = generate(args.controller, output_dir, num_trajs=args.num_trajs or 100000,
                     random_init=args.random_init, seed=args.seed, parallel=args.parallel,
                     num_workers=args.num_workers, horizon=args.horizon,
                     resolution=args.resolution or 0.1, skip_save=args.skip_save,
                     overwrite=args.overwrite, noise=noise,
                     invariant=args.invariant_terminal_sets, verbose=True,
                     ctrl_freq=args.ctrl_freq, pyb_freq=args.pyb_freq)

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
