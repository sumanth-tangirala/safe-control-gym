#!/usr/bin/env python3
"""
Generate 3D quadrotor ROA visualization data and plots for phase-plane slices.

State (Euler): [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r]
Goal:          [0, 0, 1, 0,   0,     0,   0,     0,     0,     0, 0, 0]
Controller: LQR

CSV format: x,y,z,qw,qx,qy,qz,x_dot,y_dot,z_dot,p,q,r,phi,theta_euler,psi,label

Slices (all fixed dims set to goal values):
  --slice x_xdot:     x vs x_dot
  --slice y_ydot:     y vs y_dot
  --slice z_zdot:     z vs z_dot
  --slice phi_p:      phi (roll) vs p (roll rate)
  --slice theta_q:    theta (pitch) vs q (pitch rate)
  --slice psi_r:      psi (yaw) vs r (yaw rate)

Usage:
    python generate_quadrotor_3d_visualization.py --slice x_xdot
    python generate_quadrotor_3d_visualization.py --slice all
"""

import argparse
import os
import time
import numpy as np
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, shared_memory

import pybullet as pb

from safe_control_gym.utils.registration import make
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType

try:
    import scipy.linalg
except ImportError:
    pass


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def euler_to_quat(phi, theta, psi):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (qw, qx, qy, qz).
    Uses scipy Rotation with extrinsic X-Y-Z convention (matches PyBullet)."""
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler('xyz', [phi, theta, psi], degrees=False)
    qx, qy, qz, qw = r.as_quat()  # scipy returns (x, y, z, w)
    # Canonicalize: qw >= 0
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    return qw, qx, qy, qz


def get_available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        from multiprocessing import cpu_count
        return cpu_count()


def run_trajectory(env, ctrl, init_state, max_steps):
    """Run a single trajectory. init_state is [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r]."""
    obs, info = env.reset()
    x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p_body, q_body, r_body = init_state

    pb.resetBasePositionAndOrientation(
        env.DRONE_ID,
        [x, y, z],
        pb.getQuaternionFromEuler([phi, theta, psi]),
        physicsClientId=env.PYB_CLIENT)

    pb.resetBaseVelocity(
        env.DRONE_ID,
        [x_dot, y_dot, z_dot],
        [p_body, q_body, r_body],
        physicsClientId=env.PYB_CLIENT)

    env._update_and_store_kinematic_information()
    obs = env._get_observation()

    for step in range(max_steps):
        action = ctrl.select_action(obs, info)
        obs, reward, done, info = env.step(action)
        if done:
            return info.get('goal_reached', False)

    return False


def _worker_init(shm_name, shm_shape, shm_dtype, env_config):
    """Initialize per-worker env/ctrl once."""
    global _worker_env, _worker_ctrl, _worker_shm, _worker_labels, _worker_config

    _worker_config = env_config

    task_info = {
        'stabilization_goal': [0, 0, 1],
        'stabilization_goal_tolerance': 0.05,
    }

    env_func = partial(make,
                       'quadrotor',
                       quad_type=QuadType.THREE_D,
                       task='stabilization',
                       task_info=task_info,
                       ctrl_freq=env_config['ctrl_freq'],
                       pyb_freq=env_config['pyb_freq'],
                       episode_len_sec=env_config['episode_len_sec'],
                       done_on_out_of_bound=True,
                       cost='quadratic',
                       gui=False,
                       randomized_init=False)

    _worker_ctrl = make('lqr', env_func,
                        q_lqr=env_config['q_lqr'],
                        r_lqr=env_config['r_lqr'],
                        discrete_dynamics=True)

    _worker_env = env_func()

    # Set termination bounds (env state order: x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r)
    e = _worker_env
    e.state_space.low[0] = -env_config['x_termination']
    e.state_space.high[0] = env_config['x_termination']
    e.state_space.low[2] = -env_config['y_termination']
    e.state_space.high[2] = env_config['y_termination']
    e.state_space.low[4] = env_config['z_min_termination']
    e.state_space.high[4] = env_config['z_max_termination']
    e.state_space.low[6] = -env_config['phi_termination']
    e.state_space.high[6] = env_config['phi_termination']
    e.state_space.low[7] = -env_config['theta_termination']
    e.state_space.high[7] = env_config['theta_termination']
    e.state_space.low[8] = -env_config['psi_termination']
    e.state_space.high[8] = env_config['psi_termination']
    e.state_space.low[1] = -env_config['x_dot_termination']
    e.state_space.high[1] = env_config['x_dot_termination']
    e.state_space.low[3] = -env_config['y_dot_termination']
    e.state_space.high[3] = env_config['y_dot_termination']
    e.state_space.low[5] = -env_config['z_dot_termination']
    e.state_space.high[5] = env_config['z_dot_termination']
    e.state_space.low[9] = -env_config['p_termination']
    e.state_space.high[9] = env_config['p_termination']
    e.state_space.low[10] = -env_config['q_termination']
    e.state_space.high[10] = env_config['q_termination']
    e.state_space.low[11] = -env_config['r_termination']
    e.state_space.high[11] = env_config['r_termination']

    _worker_shm = shared_memory.SharedMemory(name=shm_name)
    _worker_labels = np.ndarray(shm_shape, dtype=shm_dtype, buffer=_worker_shm.buf)


def _worker_process_chunk(chunk_info):
    """Process a chunk of indices."""
    indices, states = chunk_info
    cfg = _worker_config
    max_steps = cfg['max_steps']

    for idx, state in zip(indices, states):
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p_body, q_body, r_body = state

        # Fast-path: immediate termination on Euclidean dims
        if (abs(x) >= cfg['x_termination'] or
            abs(y) >= cfg['y_termination'] or
            z <= cfg['z_min_termination'] or z >= cfg['z_max_termination'] or
            abs(x_dot) >= cfg['x_dot_termination'] or
            abs(y_dot) >= cfg['y_dot_termination'] or
            abs(z_dot) >= cfg['z_dot_termination'] or
            abs(p_body) >= cfg['p_termination'] or
            abs(q_body) >= cfg['q_termination'] or
            abs(r_body) >= cfg['r_termination']):
            _worker_labels[idx] = 0
        else:
            success = run_trajectory(_worker_env, _worker_ctrl, state, max_steps)
            _worker_labels[idx] = 1 if success else 0

    return len(indices)


def generate_slice(slice_name, grid_states, env_config, num_workers):
    """Generate labels for a grid of initial states."""
    n_states = len(grid_states)
    print(f"\nGenerating {slice_name}: {n_states} states using {num_workers} workers...")

    labels = np.zeros(n_states, dtype=np.int8)
    shm = shared_memory.SharedMemory(create=True, size=labels.nbytes)
    shm_labels = np.ndarray(labels.shape, dtype=labels.dtype, buffer=shm.buf)
    shm_labels[:] = -1

    chunk_size = max(1, n_states // (num_workers * 20))
    chunks = []
    for i in range(0, n_states, chunk_size):
        end = min(i + chunk_size, n_states)
        chunks.append((list(range(i, end)), grid_states[i:end]))

    t0 = time.time()

    with Pool(processes=num_workers,
              initializer=_worker_init,
              initargs=(shm.name, labels.shape, labels.dtype, env_config)) as pool:
        with tqdm(total=n_states, desc=slice_name, unit='states') as pbar:
            for count in pool.imap_unordered(_worker_process_chunk, chunks):
                pbar.update(count)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s ({n_states / elapsed:.0f} states/s)")

    result_labels = np.array(shm_labels)
    shm.close()
    shm.unlink()
    return result_labels


def save_csv(grid_states, labels, filepath):
    """Save CSV: x,y,z,qw,qx,qy,qz,x_dot,y_dot,z_dot,p,q,r,phi,theta_euler,psi,label"""
    with open(filepath, 'w') as f:
        f.write('x,y,z,qw,qx,qy,qz,x_dot,y_dot,z_dot,p,q,r,phi,theta_euler,psi,label\n')
        for state, label in zip(grid_states, labels):
            x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p_body, q_body, r_body = state
            phi_n = normalize_angle(phi)
            theta_n = normalize_angle(theta)
            psi_n = normalize_angle(psi)
            qw, qx, qy, qz = euler_to_quat(phi_n, theta_n, psi_n)
            f.write(f'{x:.6f},{y:.6f},{z:.6f},'
                    f'{qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f},'
                    f'{x_dot:.6f},{y_dot:.6f},{z_dot:.6f},'
                    f'{p_body:.6f},{q_body:.6f},{r_body:.6f},'
                    f'{phi_n:.6f},{theta_n:.6f},{psi_n:.6f},'
                    f'{label}\n')
    print(f"  Saved {len(labels)} states to {filepath}")


def plot_slice(labels, axis1_vals, axis2_vals, axis1_label, axis2_label,
               title, filepath, N):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

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
    print(f"  Saved plot to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Generate 3D quadrotor ROA visualization slices')
    parser.add_argument('--slice', type=str, required=True,
                        choices=['x_xdot', 'y_ydot', 'z_zdot', 'phi_p', 'theta_q', 'psi_r', 'all'],
                        help='Which slice to generate')
    parser.add_argument('--output_dir', type=str,
                        default='/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor3D_lqr',
                        help='Output directory')
    parser.add_argument('--resolution', type=int, default=400,
                        help='Grid points per dimension (default: 400)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: all available CPUs)')
    args = parser.parse_args()

    num_workers = args.num_workers or get_available_cpus()
    N = args.resolution

    # Environment config matching existing dataset (quadrotor3D_lqr)
    env_config = {
        'ctrl_freq': 100,
        'pyb_freq': 5000,
        'episode_len_sec': 1000,
        'max_steps': 100000,
        'q_lqr': [1] * 12,
        'r_lqr': [0.1] * 4,
        # Termination thresholds (from dataset_description.json)
        'x_termination': 1.8,
        'y_termination': 1.8,
        'z_min_termination': 0.1,
        'z_max_termination': 3.0,
        'phi_termination': float('inf'),
        'theta_termination': float('inf'),
        'psi_termination': float('inf'),
        'x_dot_termination': 3.0,
        'y_dot_termination': 3.0,
        'z_dot_termination': 3.0,
        'p_termination': 24.0,
        'q_termination': 24.0,
        'r_termination': 24.0,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    if args.slice == 'all':
        slices_to_run = ['x_xdot', 'y_ydot', 'z_zdot', 'phi_p', 'theta_q', 'psi_r']
    else:
        slices_to_run = [args.slice]

    # Grid axes (exclusive endpoints)
    def make_axis(lo, hi, n):
        half = (hi - lo) / (2 * n)
        return np.linspace(lo + half, hi - half, n)

    x_vals = make_axis(-1.8, 1.8, N)
    y_vals = make_axis(-1.8, 1.8, N)
    z_vals = make_axis(0.1, 3.0, N)
    phi_vals = make_axis(-np.pi, np.pi, N)
    theta_vals = make_axis(-np.pi, np.pi, N)
    psi_vals = make_axis(-np.pi, np.pi, N)
    x_dot_vals = make_axis(-3.0, 3.0, N)
    y_dot_vals = make_axis(-3.0, 3.0, N)
    z_dot_vals = make_axis(-3.0, 3.0, N)
    p_vals = make_axis(-24.0, 24.0, N)
    q_vals = make_axis(-24.0, 24.0, N)
    r_vals = make_axis(-24.0, 24.0, N)

    # Goal state: [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r]
    goal = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Slice definitions
    # Euler state indices: x=0, y=1, z=2, phi=3, theta=4, psi=5, x_dot=6, y_dot=7, z_dot=8, p=9, q=10, r=11
    slice_defs = {
        'x_xdot': {
            'row_var': ('x', x_vals, 0),
            'col_var': ('x_dot', x_dot_vals, 6),
            'suffix': 'x_vs_xdot',
            'row_label': 'x (m)',
            'col_label': 'x_dot (m/s)',
            'title': r'3D Quad ROA: $x$ vs $\dot{x}$',
        },
        'y_ydot': {
            'row_var': ('y', y_vals, 1),
            'col_var': ('y_dot', y_dot_vals, 7),
            'suffix': 'y_vs_ydot',
            'row_label': 'y (m)',
            'col_label': 'y_dot (m/s)',
            'title': r'3D Quad ROA: $y$ vs $\dot{y}$',
        },
        'z_zdot': {
            'row_var': ('z', z_vals, 2),
            'col_var': ('z_dot', z_dot_vals, 8),
            'suffix': 'z_vs_zdot',
            'row_label': 'z (m)',
            'col_label': 'z_dot (m/s)',
            'title': r'3D Quad ROA: $z$ vs $\dot{z}$',
        },
        'phi_p': {
            'row_var': ('phi', phi_vals, 3),
            'col_var': ('p', p_vals, 9),
            'suffix': 'phi_vs_p',
            'row_label': r'$\phi$ (rad)',
            'col_label': 'p (rad/s)',
            'title': r'3D Quad ROA: $\phi$ (roll) vs $p$ (roll rate)',
        },
        'theta_q': {
            'row_var': ('theta', theta_vals, 4),
            'col_var': ('q', q_vals, 10),
            'suffix': 'theta_vs_q',
            'row_label': r'$\theta$ (rad)',
            'col_label': 'q (rad/s)',
            'title': r'3D Quad ROA: $\theta$ (pitch) vs $q$ (pitch rate)',
        },
        'psi_r': {
            'row_var': ('psi', psi_vals, 5),
            'col_var': ('r', r_vals, 11),
            'suffix': 'psi_vs_r',
            'row_label': r'$\psi$ (rad)',
            'col_label': 'r (rad/s)',
            'title': r'3D Quad ROA: $\psi$ (yaw) vs $r$ (yaw rate)',
        },
    }

    for slice_name in slices_to_run:
        sd = slice_defs[slice_name]
        row_name, row_vals, row_idx = sd['row_var']
        col_name, col_vals, col_idx = sd['col_var']

        grid_states = []
        for rv in row_vals:
            for cv in col_vals:
                state = list(goal)
                state[row_idx] = rv
                state[col_idx] = cv
                grid_states.append(state)

        labels = generate_slice(sd['suffix'], grid_states, env_config, num_workers)

        csv_path = os.path.join(args.output_dir, f"viz_{sd['suffix']}.csv")
        save_csv(grid_states, labels, csv_path)

        plot_slice(labels, row_vals, col_vals,
                   sd['row_label'], sd['col_label'], sd['title'],
                   os.path.join(args.output_dir, f"viz_{sd['suffix']}.png"), N)

    print("\nDone!")


if __name__ == '__main__':
    main()
