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

from safe_control_gym.envs.gym_control.pendulum_noise import NOISE_PRESETS
from safe_control_gym.utils.registration import make

THETA_DOT_MAX = 2 * math.pi
U_SAT = 0.6371781908344007
VARIANTS = ['v1', 'v2', 'v3', 'v4']
STRENGTHS = ['strong', 'weak']
VALID_CONTROLLERS = ['lqr'] + [f'{v}_{s}' for v in VARIANTS for s in STRENGTHS]
CACHE_NAME = '_results_cache.json'
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


def run_trajectory(env, ctrl, init_state, max_steps, invariant=False):
    '''Roll out one trajectory from ``init_state``.

    Default: terminate at (and include) the first state within the goal
    threshold; returns ``(trajectory, success, timeout)``.

    ``invariant=True``: no early termination; roll exactly ``max_steps`` steps
    and return ``(trajectory, None, False)`` -- the success label is decided
    afterwards from the terminal state.
    '''
    env.reset()
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
    parser.add_argument('--num_trajs', type=int, default=100000, help='Number of random trajectories')
    parser.add_argument('--random_init', action='store_true', help='Random sampling (else discretized grid)')
    parser.add_argument('--resolution', type=float, default=0.1, help='Grid resolution (grid mode)')
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
    suffix = f'_{args.noise}' if noise else ''
    suffix += '_invariant' if args.invariant_terminal_sets else ''
    output_dir = args.output_dir or (
        f'/common/users/shared/pracsys/genMoPlan/data_trajectories/inverted_pendulum_{args.controller}{suffix}')

    stats = generate(args.controller, output_dir, num_trajs=args.num_trajs,
                     random_init=args.random_init, seed=args.seed, parallel=args.parallel,
                     num_workers=args.num_workers, horizon=args.horizon,
                     resolution=args.resolution, skip_save=args.skip_save,
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
    print(f"Output:       {output_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
