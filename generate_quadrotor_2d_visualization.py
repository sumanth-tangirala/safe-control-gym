#!/usr/bin/env python3
"""
Generate 2D quadrotor ROA visualization data and plots for phase-plane slices.

State: [x, z, theta, x_dot, z_dot, theta_dot]
Goal:  [0, 1, 0, 0, 0, 0]  (hover at x=0, z=1)
Controller: Safe Explorer PPO (RL)

Slices (all fixed dims set to goal values):
  --slice x_xdot:         x vs x_dot         (z=1, theta=0, z_dot=0, theta_dot=0)
  --slice z_zdot:         z vs z_dot         (x=0, theta=0, x_dot=0, theta_dot=0)
  --slice theta_thetadot: theta vs theta_dot (x=0, z=1, x_dot=0, z_dot=0)

Usage:
    python generate_quadrotor_2d_visualization.py --slice x_xdot
    python generate_quadrotor_2d_visualization.py --slice z_zdot
    python generate_quadrotor_2d_visualization.py --slice theta_thetadot
    python generate_quadrotor_2d_visualization.py --slice all
"""

import argparse
import os
import tempfile
import time
from functools import partial
from multiprocessing import Pool, shared_memory

import numpy as np
import pybullet as p
from tqdm import tqdm

from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
from safe_control_gym.utils.registration import make

try:
    import scipy.linalg  # noqa: F401 (side-effecting pre-import, not referenced directly)
except ImportError:
    pass

# Safe Explorer PPO config (from generate_quadrotor_2d_trajectories_rl.py)
ALGO_CONFIG = {
    'hidden_dim': 128,
    'norm_obs': False,
    'norm_reward': False,
    'clip_obs': 10.0,
    'clip_reward': 10.0,
    'pretraining': False,
    'pretrained': None,
    'constraint_hidden_dim': 150,
    'constraint_lr': 0.0001,
    'constraint_batch_size': 256,
    'constraint_steps_per_epoch': 6000,
    'constraint_epochs': 25,
    'constraint_eval_steps': 1500,
    'constraint_eval_interval': 5,
    'constraint_buffer_size': 1000000,
    'constraint_slack': [0.05, 0.05, 0.05, 0.05, 0.01, 0.01,
                         0.05, 0.05, 0.05, 0.05, 0.01, 0.01],
    'gamma': 0.99,
    'use_gae': True,
    'gae_lambda': 0.95,
    'use_clipped_value': False,
    'clip_param': 0.2,
    'target_kl': 0.01,
    'entropy_coef': 0.01,
    'opt_epochs': 20,
    'mini_batch_size': 250,
    'actor_lr': 0.001,
    'critic_lr': 0.001,
    'max_grad_norm': 0.5,
    'max_env_steps': 500000,
    'num_workers': 1,
    'rollout_batch_size': 4,
    'rollout_steps': 250,
    'deque_size': 10,
    'eval_batch_size': 10,
    'log_interval': 0,
    'save_interval': 0,
    'num_checkpoints': 0,
    'eval_interval': 0,
    'eval_save_best': False,
    'tensorboard': False,
    'training': False,
}

SAFE_EXPLORER_CONSTRAINTS = [
    {
        'constraint_form': 'default_constraint',
        'constrained_variable': 'state',
        'upper_bounds': [2, 1, 2, 1, 0.2, 1.5],
        'lower_bounds': [-2, -1, 0, -1, -0.2, -1.5],
    },
    {
        'constraint_form': 'default_constraint',
        'constrained_variable': 'input',
        'upper_bounds': [0.29, 0.29],
        'lower_bounds': [0.06, 0.06],
    }
]


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def get_available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        from multiprocessing import cpu_count
        return cpu_count()


def run_trajectory(env, ctrl, init_state, max_steps):
    """Run a single trajectory. init_state is [x, z, theta, x_dot, z_dot, theta_dot]."""
    import torch

    obs, info = env.reset()
    x, z, theta, x_dot, z_dot, theta_dot = init_state

    p.resetBasePositionAndOrientation(
        env.DRONE_ID,
        [x, 0, z],
        p.getQuaternionFromEuler([0, theta, 0]),
        physicsClientId=env.PYB_CLIENT)

    p.resetBaseVelocity(
        env.DRONE_ID,
        [x_dot, 0, z_dot],
        [0, theta_dot, 0],
        physicsClientId=env.PYB_CLIENT)

    env._update_and_store_kinematic_information()
    obs = env._get_observation()

    if hasattr(env, 'constraints') and env.constraints is not None:
        info['constraint_values'] = env.constraints.get_values(env, only_state=True)

    with torch.no_grad():
        for step in range(max_steps):
            normalized_obs = ctrl.obs_normalizer(obs)
            action = ctrl.select_action(normalized_obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                return info.get('goal_reached', False)

    return False


def _patch_safe_ppo_act():
    """Monkey-patch SafePPO act() to work around torch/numpy incompatibility."""
    import torch

    from safe_control_gym.controllers.safe_explorer import safe_ppo_utils

    def patched_act(self, obs, c=None):
        with torch.no_grad():
            dist, _ = self.actor(obs, c=c)
            a = dist.mode()
            return np.array(a.detach().cpu().tolist())

    safe_ppo_utils.MLPActorCritic.act = patched_act


# Module-level globals set by parent before fork
_shared_ctrl = None
_shared_env_config = None
_shared_env_kwargs = None


def _worker_init(shm_name, shm_shape, shm_dtype):
    """Initialize per-worker env (PyBullet needs separate instances). Controller inherited from parent via fork."""
    global _worker_env, _worker_ctrl, _worker_shm, _worker_labels, _worker_config

    _worker_config = _shared_env_config
    _worker_ctrl = _shared_ctrl

    env_func = partial(make, 'quadrotor', **_shared_env_kwargs)
    _worker_env = env_func()

    # Set termination bounds (env state order: x, x_dot, z, z_dot, theta, theta_dot)
    cfg = _worker_config
    _worker_env.state_space.low[0] = -cfg['x_termination']
    _worker_env.state_space.high[0] = cfg['x_termination']
    _worker_env.state_space.low[2] = cfg['z_min_termination']
    _worker_env.state_space.high[2] = cfg['z_max_termination']
    _worker_env.state_space.low[4] = -cfg['theta_termination']
    _worker_env.state_space.high[4] = cfg['theta_termination']
    _worker_env.state_space.low[1] = -cfg['x_dot_termination']
    _worker_env.state_space.high[1] = cfg['x_dot_termination']
    _worker_env.state_space.low[3] = -cfg['z_dot_termination']
    _worker_env.state_space.high[3] = cfg['z_dot_termination']
    _worker_env.state_space.low[5] = -cfg['theta_dot_termination']
    _worker_env.state_space.high[5] = cfg['theta_dot_termination']

    _worker_shm = shared_memory.SharedMemory(name=shm_name)
    _worker_labels = np.ndarray(shm_shape, dtype=shm_dtype, buffer=_worker_shm.buf)


def _worker_process_chunk(chunk_info):
    """Process a chunk of indices."""
    indices, states = chunk_info
    cfg = _worker_config
    max_steps = cfg['max_steps']

    for idx, state in zip(indices, states):
        x, z, theta, x_dot, z_dot, theta_dot = state

        # Fast-path: immediate termination
        if (abs(x) >= cfg['x_termination'] or
            z <= cfg['z_min_termination'] or z >= cfg['z_max_termination'] or
            abs(x_dot) >= cfg['x_dot_termination'] or
            abs(z_dot) >= cfg['z_dot_termination'] or
                abs(theta_dot) >= cfg['theta_dot_termination']):
            _worker_labels[idx] = 0
        else:
            success = run_trajectory(_worker_env, _worker_ctrl, state, max_steps)
            _worker_labels[idx] = 1 if success else 0

    return len(indices)


def generate_slice(slice_name, grid_states, num_workers):
    """Generate labels for a grid of initial states."""
    n_states = len(grid_states)
    print(f'\nGenerating {slice_name}: {n_states} states using {num_workers} workers...')

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
              initargs=(shm.name, labels.shape, labels.dtype)) as pool:
        with tqdm(total=n_states, desc=slice_name, unit='states') as pbar:
            for count in pool.imap_unordered(_worker_process_chunk, chunks):
                pbar.update(count)

    elapsed = time.time() - t0
    print(f'  Completed in {elapsed:.1f}s ({n_states / elapsed:.0f} states/s)')

    result_labels = np.array(shm_labels)
    shm.close()
    shm.unlink()
    return result_labels


def save_csv(grid_states, labels, filepath):
    """Save as CSV: x,z,theta,x_dot,z_dot,theta_dot,label"""
    with open(filepath, 'w') as f:
        f.write('x,z,theta,x_dot,z_dot,theta_dot,label\n')
        for state, label in zip(grid_states, labels):
            x, z, theta, x_dot, z_dot, theta_dot = state
            f.write(f'{x:.6f},{z:.6f},{normalize_angle(theta):.6f},{x_dot:.6f},{z_dot:.6f},{theta_dot:.6f},{label}\n')
    print(f'  Saved {len(labels)} states to {filepath}')


def plot_slice(labels, axis1_vals, axis2_vals, axis1_label, axis2_label,
               title, filepath, N):
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
    parser = argparse.ArgumentParser(description='Generate 2D quadrotor ROA visualization slices')
    parser.add_argument('--slice', type=str, required=True,
                        choices=['x_xdot', 'z_zdot', 'theta_thetadot', 'all'],
                        help='Which slice to generate')
    parser.add_argument('--output_dir', type=str,
                        default='/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor2D_rl',
                        help='Output directory')
    parser.add_argument('--model_path', type=str,
                        default='examples/rl/models/safe_explorer_ppo/safe_explorer_ppo_model_quadrotor_2D_stab.pt',
                        help='Path to trained Safe Explorer PPO model')
    parser.add_argument('--resolution', type=int, default=400,
                        help='Grid points per dimension (default: 400)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: all available CPUs)')
    args = parser.parse_args()

    global _shared_ctrl, _shared_env_config, _shared_env_kwargs

    # Patch before any torch model construction so .act() never calls .numpy()
    _patch_safe_ppo_act()

    num_workers = args.num_workers or get_available_cpus()
    N = args.resolution

    # Environment config matching existing dataset (quadrotor2D_rl)
    env_config = {
        'ctrl_freq': 100,
        'pyb_freq': 5000,
        'episode_len_sec': 1000,
        'max_steps': 1200,
        'goal_tolerance': 0.2,
        # Termination thresholds
        'x_termination': 1.0,
        'z_min_termination': 0.1,
        'z_max_termination': 1.5,
        'theta_termination': float('inf'),
        'x_dot_termination': 1.0,
        'z_dot_termination': 1.0,
        'theta_dot_termination': 8.0,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # Build controller ONCE in parent process (avoids torch/numpy crash in forked children)
    env_kwargs = {
        'quad_type': QuadType.TWO_D,
        'task': 'stabilization',
        'ctrl_freq': env_config['ctrl_freq'],
        'pyb_freq': env_config['pyb_freq'],
        'episode_len_sec': env_config['episode_len_sec'],
        'done_on_out_of_bound': True,
        'cost': 'quadratic',
        'normalized_rl_action_space': True,
        'gui': False,
        'randomized_init': False,
        'task_info': {
            'stabilization_goal': [0, 1],
            'stabilization_goal_tolerance': env_config['goal_tolerance'],
        },
        'constraints': SAFE_EXPLORER_CONSTRAINTS,
        'done_on_violation': False,
    }
    env_func = partial(make, 'quadrotor', **env_kwargs)

    print('Building controller in parent process...')
    temp_dir = tempfile.mkdtemp()
    algo_config = ALGO_CONFIG.copy()
    ctrl = make('safe_explorer_ppo', env_func, **algo_config, output_dir=temp_dir)
    ctrl.load(args.model_path)
    ctrl.obs_normalizer.set_read_only()

    # Set module-level globals that forked children will inherit
    _shared_ctrl = ctrl
    _shared_env_config = env_config
    _shared_env_kwargs = env_kwargs

    if args.slice == 'all':
        slices_to_run = ['x_xdot', 'z_zdot', 'theta_thetadot']
    else:
        slices_to_run = [args.slice]

    # Grid axes (exclusive endpoints)
    # x: [-1.0, 1.0]
    x_half = 2.0 / (2 * N)
    x_vals = np.linspace(-1.0 + x_half, 1.0 - x_half, N)
    # z: [0.1, 1.5] (asymmetric)
    z_half = 1.4 / (2 * N)
    z_vals = np.linspace(0.1 + z_half, 1.5 - z_half, N)
    # theta: [-pi, pi]
    th_half = (2 * np.pi) / (2 * N)
    theta_vals = np.linspace(-np.pi + th_half, np.pi - th_half, N)
    # x_dot: [-1.0, 1.0]
    xd_half = 2.0 / (2 * N)
    x_dot_vals = np.linspace(-1.0 + xd_half, 1.0 - xd_half, N)
    # z_dot: [-1.0, 1.0]
    zd_half = 2.0 / (2 * N)
    z_dot_vals = np.linspace(-1.0 + zd_half, 1.0 - zd_half, N)
    # theta_dot: [-8.0, 8.0]
    thd_half = 16.0 / (2 * N)
    theta_dot_vals = np.linspace(-8.0 + thd_half, 8.0 - thd_half, N)

    # Goal state: [x=0, z=1, theta=0, x_dot=0, z_dot=0, theta_dot=0]
    goal = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    # Slice definitions
    # State order: [x, z, theta, x_dot, z_dot, theta_dot] (indices 0-5)
    slice_defs = {
        'x_xdot': {
            'row_var': ('x', x_vals, 0),
            'col_var': ('x_dot', x_dot_vals, 3),
            'suffix': 'x_vs_xdot',
            'row_label': 'x (m)',
            'col_label': 'x_dot (m/s)',
            'title': r'2D Quad ROA: $x$ vs $\dot{x}$ (at goal $z{=}1$, $\theta{=}0$)',
        },
        'z_zdot': {
            'row_var': ('z', z_vals, 1),
            'col_var': ('z_dot', z_dot_vals, 4),
            'suffix': 'z_vs_zdot',
            'row_label': 'z (m)',
            'col_label': 'z_dot (m/s)',
            'title': r'2D Quad ROA: $z$ vs $\dot{z}$ (at goal $x{=}0$, $\theta{=}0$)',
        },
        'theta_thetadot': {
            'row_var': ('theta', theta_vals, 2),
            'col_var': ('theta_dot', theta_dot_vals, 5),
            'suffix': 'theta_vs_thetadot',
            'row_label': r'$\theta$ (rad)',
            'col_label': r'$\dot{\theta}$ (rad/s)',
            'title': r'2D Quad ROA: $\theta$ vs $\dot{\theta}$ (at goal $x{=}0$, $z{=}1$)',
        },
    }

    for slice_name in slices_to_run:
        sd = slice_defs[slice_name]
        row_name, row_vals, row_idx = sd['row_var']
        col_name, col_vals, col_idx = sd['col_var']

        grid_states = []
        for rv in row_vals:
            for cv in col_vals:
                state = list(goal)  # start from goal
                state[row_idx] = rv
                state[col_idx] = cv
                grid_states.append(state)

        labels = generate_slice(sd['suffix'], grid_states, num_workers)

        csv_path = os.path.join(args.output_dir, f"viz_{sd['suffix']}.csv")
        save_csv(grid_states, labels, csv_path)

        plot_slice(labels, row_vals, col_vals,
                   sd['row_label'], sd['col_label'], sd['title'],
                   os.path.join(args.output_dir, f"viz_{sd['suffix']}.png"), N)

    print('\nDone!')


if __name__ == '__main__':
    main()
