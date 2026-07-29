#!/usr/bin/env python3
"""
Generate cartpole ROA visualization data and plots for two 2D slices of the state space.

Slice 1 (--slice x_xdot):          x vs x_dot       (theta=0, theta_dot=0)
Slice 2 (--slice theta_thetadot):  theta vs theta_dot (x=0, x_dot=0)
Slice 3 (--slice x_xdot_b):       x vs x_dot       (theta=-0.31, theta_dot=-4.29)
Slice 4 (--slice theta_thetadot_b): theta vs theta_dot (x=-1.51, x_dot=0.00)

Each slice is a 400x400 grid. Outputs CSV files with initial states and labels,
plus visualization images.

Usage:
    python generate_cartpole_visualization.py --slice x_xdot
    python generate_cartpole_visualization.py --slice theta_thetadot
    python generate_cartpole_visualization.py --slice x_xdot_b
    python generate_cartpole_visualization.py --slice theta_thetadot_b
    python generate_cartpole_visualization.py --slice both
    python generate_cartpole_visualization.py --slice all
"""

import argparse
import os
import time
from functools import partial
from multiprocessing import Pool, shared_memory

import numpy as np
from tqdm import tqdm

from safe_control_gym.utils.registration import make


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def get_available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        from multiprocessing import cpu_count
        return cpu_count()


def run_trajectory(env, ctrl, init_state, max_steps):
    """Run a single trajectory. init_state is [x, x_dot, theta, theta_dot] (internal order)."""
    import pybullet as p

    obs, info = env.reset()
    x, x_dot, theta, theta_dot = init_state

    p.resetJointState(env.CARTPOLE_ID, jointIndex=0,
                      targetValue=x, targetVelocity=x_dot,
                      physicsClientId=env.PYB_CLIENT)
    p.resetJointState(env.CARTPOLE_ID, jointIndex=1,
                      targetValue=theta, targetVelocity=theta_dot,
                      physicsClientId=env.PYB_CLIENT)

    env.state = np.array([x, x_dot, theta, theta_dot])
    obs = env._get_observation()

    for step in range(max_steps):
        action = ctrl.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            return info.get('goal_reached', False)

    return False


def _worker_init(shm_name, shm_shape, shm_dtype, env_config):
    """Initialize per-worker env/ctrl once, attach to shared memory for results."""
    global _worker_env, _worker_ctrl, _worker_shm, _worker_labels, _worker_config

    _worker_config = env_config

    env_func = partial(make,
                       'cartpole',
                       task='stabilization',
                       ctrl_freq=env_config['ctrl_freq'],
                       pyb_freq=env_config['pyb_freq'],
                       episode_len_sec=env_config['episode_len_sec'],
                       done_on_out_of_bound=True,
                       cost='quadratic',
                       gui=False,
                       randomized_init=False,
                       obs_wrap_angle=True,
                       x_dot_limit=float('inf'),
                       theta_dot_limit=float('inf'),
                       action_scale=env_config['action_scale'])

    _worker_ctrl = make('lqr', env_func,
                        q_lqr=[1, 1, 1, 1],
                        r_lqr=[0.1],
                        discrete_dynamics=True)

    _worker_env = env_func()
    _worker_env.x_threshold = env_config['x_threshold']
    _worker_env.x_dot_threshold = env_config['x_dot_threshold']
    _worker_env.theta_threshold_radians = env_config['theta_threshold_radians']
    _worker_env.theta_dot_threshold = env_config['theta_dot_threshold']

    _worker_shm = shared_memory.SharedMemory(name=shm_name)
    _worker_labels = np.ndarray(shm_shape, dtype=shm_dtype, buffer=_worker_shm.buf)


def _worker_process_chunk(chunk_info):
    """Process a chunk of indices. Writes labels directly to shared memory."""
    indices, states = chunk_info
    cfg = _worker_config
    env = _worker_env
    ctrl = _worker_ctrl
    max_steps = cfg['max_steps']
    x_th = cfg['x_threshold']
    xd_th = cfg['x_dot_threshold']
    thd_th = cfg['theta_dot_threshold']

    for i, (idx, state) in enumerate(zip(indices, states)):
        x, x_dot, theta, theta_dot = state

        # Fast-path: immediate termination without simulation
        if (abs(x) >= x_th or abs(x_dot) >= xd_th or abs(theta_dot) >= thd_th):
            _worker_labels[idx] = 0
        else:
            success = run_trajectory(env, ctrl, state, max_steps)
            _worker_labels[idx] = 1 if success else 0

    return len(indices)


def generate_slice(slice_name, grid_states, env_config, num_workers):
    """Generate labels for a grid of initial states using parallel workers with shared memory."""
    n_states = len(grid_states)
    print(f'\nGenerating {slice_name}: {n_states} states using {num_workers} workers...')

    # Create shared memory for labels
    labels = np.zeros(n_states, dtype=np.int8)
    shm = shared_memory.SharedMemory(create=True, size=labels.nbytes)
    shm_labels = np.ndarray(labels.shape, dtype=labels.dtype, buffer=shm.buf)
    shm_labels[:] = -1  # sentinel

    # Split into chunks — many small chunks for good load balancing
    # (some trajectories are short failures, others are long successes)
    chunk_size = max(1, n_states // (num_workers * 20))
    chunks = []
    for i in range(0, n_states, chunk_size):
        end = min(i + chunk_size, n_states)
        chunks.append((list(range(i, end)), grid_states[i:end]))

    t0 = time.time()

    with Pool(processes=num_workers,
              initializer=_worker_init,
              initargs=(shm.name, labels.shape, labels.dtype, env_config)) as pool:
        done_count = 0
        with tqdm(total=n_states, desc=slice_name, unit='states') as pbar:
            for count in pool.imap_unordered(_worker_process_chunk, chunks):
                done_count += count
                pbar.update(count)

    elapsed = time.time() - t0
    print(f'  Completed in {elapsed:.1f}s ({n_states / elapsed:.0f} states/s)')

    # Copy results from shared memory
    result_labels = np.array(shm_labels)
    shm.close()
    shm.unlink()

    return result_labels


def save_csv(grid_states, labels, filepath):
    """Save results as CSV: x,theta,x_dot,theta_dot,label"""
    with open(filepath, 'w') as f:
        f.write('x,theta,x_dot,theta_dot,label\n')
        for state, label in zip(grid_states, labels):
            x, x_dot, theta, theta_dot = state
            f.write(f'{x:.6f},{normalize_angle(theta):.6f},{x_dot:.6f},{theta_dot:.6f},{label}\n')
    print(f'  Saved {len(labels)} states to {filepath}')


def plot_slice(labels, axis1_vals, axis2_vals, axis1_label, axis2_label,
               title, filepath, N):
    """Plot a 2D heatmap of ROA labels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    label_grid = labels.reshape(N, N)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    cmap = mcolors.ListedColormap(['#d62728', '#2ca02c'])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    im = ax.imshow(label_grid,
                   extent=[axis2_vals[0], axis2_vals[-1],
                           axis1_vals[0], axis1_vals[-1]],
                   origin='lower', aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')

    ax.set_xlabel(axis2_label, fontsize=13)
    ax.set_ylabel(axis1_label, fontsize=13)
    ax.set_title(title, fontsize=14)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], shrink=0.8)
    cbar.ax.set_yticklabels(['Fail (0)', 'Success (1)'])

    n_success = label_grid.sum()
    n_total = label_grid.size
    ax.text(0.02, 0.98, f'Success: {int(n_success)}/{n_total} ({100 * n_success / n_total:.1f}%)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved plot to {filepath}')


def main():
    parser = argparse.ArgumentParser(description='Generate cartpole ROA visualization slices')
    parser.add_argument('--slice', type=str, required=True,
                        choices=['x_xdot', 'theta_thetadot', 'x_xdot_b', 'theta_thetadot_b',
                                 'x_xdot_z', 'theta_thetadot_z', 'both', 'zeros', 'all'],
                        help='Which slice to generate. "both" = _b slices. "zeros" = _z slices. "all" = all 6 slices.')
    parser.add_argument('--output_dir', type=str,
                        default='/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_pybullet',
                        help='Output directory for CSV and plot files')
    parser.add_argument('--resolution', type=int, default=400,
                        help='Number of grid points per dimension (default: 400)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: all available CPUs)')
    args = parser.parse_args()

    num_workers = args.num_workers or get_available_cpus()
    N = args.resolution

    env_config = {
        'ctrl_freq': 100,
        'pyb_freq': 5000,
        'episode_len_sec': 10,
        'action_scale': 2000.0,
        'x_threshold': 6.0,
        'x_dot_threshold': 5.0,
        'theta_threshold_radians': float('inf'),
        'theta_dot_threshold': 5.0,
        'max_steps': 1000,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    if args.slice == 'both':
        slices_to_run = ['x_xdot_b', 'theta_thetadot_b']
    elif args.slice == 'zeros':
        slices_to_run = ['x_xdot_z', 'theta_thetadot_z']
    elif args.slice == 'all':
        slices_to_run = ['x_xdot', 'theta_thetadot', 'x_xdot_b', 'theta_thetadot_b', 'x_xdot_z', 'theta_thetadot_z']
    else:
        slices_to_run = [args.slice]

    # Precompute shared grid axes (exclusive endpoints)
    x_half = 12.0 / (2 * N)
    xd_half = 10.0 / (2 * N)
    th_half = (2 * np.pi) / (2 * N)
    thd_half = 10.0 / (2 * N)

    x_vals = np.linspace(-6.0 + x_half, 6.0 - x_half, N)
    x_dot_vals = np.linspace(-5.0 + xd_half, 5.0 - xd_half, N)
    theta_vals = np.linspace(-np.pi + th_half, np.pi - th_half, N)
    theta_dot_vals = np.linspace(-5.0 + thd_half, 5.0 - thd_half, N)

    # Slice definitions: (name, row_vals, col_vals, fixed_vars, file_suffix, plot_labels)
    # fixed_vars: dict mapping internal-order index to fixed value
    #   internal order: [x, x_dot, theta, theta_dot]
    slice_defs = {
        'x_xdot': {
            'row_var': ('x', x_vals, 0),           # internal idx 0
            'col_var': ('x_dot', x_dot_vals, 1),   # internal idx 1
            'fixed': {2: 0.0, 3: 0.0},             # theta=0, theta_dot=0
            'suffix': 'x_vs_xdot',
            'row_label': 'x (m)',
            'col_label': 'x_dot (m/s)',
            'title': r'Cartpole ROA: $x$ vs $\dot{x}$ ($\theta=0$, $\dot{\theta}=0$)',
        },
        'theta_thetadot': {
            'row_var': ('theta', theta_vals, 2),
            'col_var': ('theta_dot', theta_dot_vals, 3),
            'fixed': {0: 0.0, 1: 0.0},             # x=0, x_dot=0
            'suffix': 'theta_vs_thetadot',
            'row_label': r'$\theta$ (rad)',
            'col_label': r'$\dot{\theta}$ (rad/s)',
            'title': r'Cartpole ROA: $\theta$ vs $\dot{\theta}$ ($x=0$, $\dot{x}=0$)',
        },
        'x_xdot_b': {
            'row_var': ('x', x_vals, 0),
            'col_var': ('x_dot', x_dot_vals, 1),
            'fixed': {2: -0.31, 3: -4.29},         # theta=-0.31, theta_dot=-4.29
            'suffix': 'x_vs_xdot_b',
            'row_label': 'x (m)',
            'col_label': 'x_dot (m/s)',
            'title': r'Cartpole ROA: $x$ vs $\dot{x}$ ($\theta=-0.31$, $\dot{\theta}=-4.29$)',
        },
        'theta_thetadot_b': {
            'row_var': ('theta', theta_vals, 2),
            'col_var': ('theta_dot', theta_dot_vals, 3),
            'fixed': {0: -1.51, 1: 0.0},           # x=-1.51, x_dot=0.0
            'suffix': 'theta_vs_thetadot_b',
            'row_label': r'$\theta$ (rad)',
            'col_label': r'$\dot{\theta}$ (rad/s)',
            'title': r'Cartpole ROA: $\theta$ vs $\dot{\theta}$ ($x=-1.51$, $\dot{x}=0$)',
        },
        'x_xdot_z': {
            'row_var': ('x', x_vals, 0),
            'col_var': ('x_dot', x_dot_vals, 1),
            'fixed': {2: 0.0, 3: 0.0},             # theta=0, theta_dot=0
            'suffix': 'x_vs_xdot_z',
            'row_label': 'x (m)',
            'col_label': 'x_dot (m/s)',
            'title': r'Cartpole ROA: $x$ vs $\dot{x}$ ($\theta=0$, $\dot{\theta}=0$)',
        },
        'theta_thetadot_z': {
            'row_var': ('theta', theta_vals, 2),
            'col_var': ('theta_dot', theta_dot_vals, 3),
            'fixed': {0: 0.0, 1: 0.0},             # x=0, x_dot=0
            'suffix': 'theta_vs_thetadot_z',
            'row_label': r'$\theta$ (rad)',
            'col_label': r'$\dot{\theta}$ (rad/s)',
            'title': r'Cartpole ROA: $\theta$ vs $\dot{\theta}$ ($x=0$, $\dot{x}=0$)',
        },
    }

    for slice_name in slices_to_run:
        sd = slice_defs[slice_name]
        row_name, row_vals, row_idx = sd['row_var']
        col_name, col_vals, col_idx = sd['col_var']
        fixed = sd['fixed']

        # Build grid states in internal order [x, x_dot, theta, theta_dot]
        grid_states = []
        for rv in row_vals:
            for cv in col_vals:
                state = [0.0, 0.0, 0.0, 0.0]
                state[row_idx] = rv
                state[col_idx] = cv
                for fix_idx, fix_val in fixed.items():
                    state[fix_idx] = fix_val
                grid_states.append(state)

        labels = generate_slice(sd['suffix'], grid_states, env_config, num_workers)

        csv_path = os.path.join(args.output_dir, f"viz_{sd['suffix']}.csv")
        save_csv(grid_states, labels, csv_path)

        plot_slice(labels, row_vals, col_vals,
                   sd['row_label'], sd['col_label'], sd['title'],
                   os.path.join(args.output_dir, f"viz_{sd['suffix']}.png"), N)

    print('\nDone!')


if __name__ == '__main__':
    main()
