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
(label 0).

Examples:
    python generate_inverted_pendulum_trajectories.py --controller lqr \
        --random_init --num_trajs 100000 --parallel --seed 42
    python generate_inverted_pendulum_trajectories.py --controller v3_strong \
        --random_init --num_trajs 50000 --parallel --seed 42
'''

import argparse
import json
import math
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from safe_control_gym.utils.registration import make

THETA_DOT_MAX = 2 * math.pi
U_SAT = 0.6371781908344007
VARIANTS = ['v1', 'v2', 'v3', 'v4']
STRENGTHS = ['strong', 'weak']
VALID_CONTROLLERS = ['lqr'] + [f'{v}_{s}' for v in VARIANTS for s in STRENGTHS]
CACHE_NAME = '_results_cache.json'


def normalize_angle(angle):
    '''Normalize an angle to [-pi, pi].'''
    return math.atan2(math.sin(angle), math.cos(angle))


def get_available_cpus():
    '''Number of CPUs available to this process (respects affinity/taskset).'''
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return cpu_count()


def sample_initial_states(num_trajs, random_init, seed, theta_dot_max, resolution):
    '''Sample initial ``[theta, theta_dot]`` states.

    ``random_init``: ``num_trajs`` uniform samples over the full state space.
    Otherwise: a discretized grid at ``resolution`` (``num_trajs`` ignored).
    '''
    if random_init:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-math.pi, math.pi, size=num_trajs)
        thetadot = rng.uniform(-theta_dot_max, theta_dot_max, size=num_trajs)
        return np.stack([theta, thetadot], axis=1)
    thetas = np.arange(-math.pi, math.pi + resolution, resolution)
    thetadots = np.arange(-theta_dot_max, theta_dot_max + resolution, resolution)
    return np.array([[t, d] for t in thetas for d in thetadots], dtype=np.float64)


def make_env_func(env_config):
    return partial(make, 'inverted_pendulum',
                   ctrl_freq=env_config['ctrl_freq'],
                   pyb_freq=env_config['pyb_freq'],
                   episode_len_sec=env_config['episode_len_sec'],
                   cost='quadratic',
                   gui=False,
                   randomized_init=False)


def make_controller(controller, env_func):
    '''Instantiate the LQR or the requested standalone RL policy.'''
    if controller == 'lqr':
        return make('pendulum_lqr', env_func, q_lqr=[1, 1], r_lqr=[1])
    ctrl = make('pendulum_rl', env_func, model_path=controller)
    ctrl.obs_normalizer.set_read_only()
    return ctrl


def run_trajectory(env, ctrl, init_state, max_steps):
    '''Roll out one trajectory from ``init_state``.

    Returns ``(trajectory, success, timeout)`` with ``trajectory`` a list of
    ``[theta_wrapped, theta_dot]`` states (including the initial state).
    '''
    env.reset()
    env.state = np.array(init_state, dtype=np.float64)
    obs = env._get_observation()
    info = None
    if hasattr(ctrl, 'reset'):
        ctrl.reset()

    trajectory = [[normalize_angle(env.state[0]), float(env.state[1])]]
    success = timeout = False
    for _ in range(max_steps):
        obs_in = ctrl.obs_normalizer(obs) if hasattr(ctrl, 'obs_normalizer') else obs
        action = ctrl.select_action(obs_in, info)
        obs, _, done, info = env.step(action)
        trajectory.append([normalize_angle(env.state[0]), float(env.state[1])])
        if done:
            success = bool(info.get('goal_reached', False))
            break
    else:
        timeout = True
    return trajectory, success, timeout


def save_trajectory(trajectory, filepath):
    '''Save a trajectory (one comma-separated state per line, 6 decimals).'''
    with open(filepath, 'w') as f:
        for state in trajectory:
            f.write(','.join(f'{v:.6f}' for v in state) + '\n')


def _process_batch(args_tuple):
    '''Worker: roll out a batch of trajectories with a shared env + controller.'''
    batch, controller, env_config, trajectories_dir, skip_save = args_tuple
    env_func = make_env_func(env_config)
    ctrl = make_controller(controller, env_func)
    env = env_func()
    records = []
    for idx, init_state in batch:
        trajectory, success, timeout = run_trajectory(env, ctrl, init_state, env_config['max_steps'])
        if not skip_save:
            save_trajectory(trajectory, os.path.join(trajectories_dir, f'sequence_{idx}.txt'))
        records.append((idx, {
            'init_state': [normalize_angle(init_state[0]), float(init_state[1])],
            'label': 1 if success else 0,
            'success': bool(success),
            'timeout': bool(timeout),
            'length': len(trajectory),
        }))
    env.close()
    ctrl.close()
    return records


def _load_cache(output_dir):
    path = os.path.join(output_dir, CACHE_NAME)
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def generate(controller, output_dir, num_trajs=100000, random_init=True, seed=42,
             parallel=False, num_workers=None, episode_len_sec=10, resolution=0.1,
             theta_dot_max=THETA_DOT_MAX, skip_save=False, overwrite=False, max_steps=None,
             batch_size=256, verbose=False):
    '''Generate a dataset and return aggregate statistics.

    Resumable: trajectories whose ``sequence_<idx>.txt`` already exists (and whose
    label is cached) are skipped unless ``overwrite`` is set.
    '''
    if controller not in VALID_CONTROLLERS:
        raise ValueError(f'[ERROR] unknown controller {controller!r}; valid: {VALID_CONTROLLERS}')

    trajectories_dir = os.path.join(output_dir, 'trajectories')
    os.makedirs(trajectories_dir if not skip_save else output_dir, exist_ok=True)

    ctrl_freq = pyb_freq = 100  # dt = 0.01, matching the trained-on physics.
    if max_steps is None:
        max_steps = episode_len_sec * ctrl_freq
    env_config = {'ctrl_freq': ctrl_freq, 'pyb_freq': pyb_freq,
                  'episode_len_sec': episode_len_sec, 'max_steps': max_steps}

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
    with open(os.path.join(output_dir, 'roa_labels.txt'), 'w') as f:
        for idx in range(n):
            rec = cache[str(idx)]
            theta, thetadot = rec['init_state']
            f.write(f'{theta:.6f},{thetadot:.6f},{rec["label"]}\n')
            success_count += int(rec['success'])
            timeout_count += int(rec['timeout'])

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
        'episode_len_sec': episode_len_sec, 'max_steps': max_steps,
        'u_sat': U_SAT,
        'theta_dot_max': theta_dot_max,
        'random_init': random_init, 'seed': seed,
        'label_semantics': '1 = reached upright goal within horizon, 0 = timeout '
                           '(theta wrapped, theta_dot clipped -> no out-of-bounds failure)',
    }
    with open(os.path.join(output_dir, 'dataset_description.json'), 'w') as f:
        json.dump(description, f, indent=2)
    return stats


def main():
    parser = argparse.ArgumentParser(description='Generate inverted-pendulum trajectory dataset')
    parser.add_argument('--controller', required=True, choices=VALID_CONTROLLERS,
                        help='lqr, or a standalone RL policy vX_{strong,weak}')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: .../genMoPlan/data_trajectories/inverted_pendulum_<controller>)')
    parser.add_argument('--num_trajs', type=int, default=100000, help='Number of random trajectories')
    parser.add_argument('--random_init', action='store_true', help='Random sampling (else discretized grid)')
    parser.add_argument('--resolution', type=float, default=0.1, help='Grid resolution (grid mode)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--episode_len_sec', type=int, default=10, help='Horizon (default 10 s = 1000 steps)')
    parser.add_argument('--skip_save', action='store_true', help='Compute labels/stats without writing sequences')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate even if sequence files exist')
    args = parser.parse_args()

    output_dir = args.output_dir or (
        f'/common/users/shared/pracsys/genMoPlan/data_trajectories/inverted_pendulum_{args.controller}')

    stats = generate(args.controller, output_dir, num_trajs=args.num_trajs,
                     random_init=args.random_init, seed=args.seed, parallel=args.parallel,
                     num_workers=args.num_workers, episode_len_sec=args.episode_len_sec,
                     resolution=args.resolution, skip_save=args.skip_save,
                     overwrite=args.overwrite, verbose=True)

    print(f"\n{'=' * 70}")
    print(f"Controller:   {stats['controller']}")
    print(f"Trajectories: {stats['num_trajectories']}")
    print(f"Successful:   {stats['success_count']} ({stats['success_rate'] * 100:.2f}%)")
    print(f"Timed out:    {stats['timeout_count']}")
    print(f"Output:       {output_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
