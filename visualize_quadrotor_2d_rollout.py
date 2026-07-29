#!/usr/bin/env python3
"""
Visualize a 2D quadrotor rollout from a given initial state.
Generates a video (MP4) and an x-z state-space trajectory plot.

Phase 1: Run physics with the normal small drone, record poses.
Phase 2: Replay poses with a scaled-up drone for video capture.

Usage:
    python visualize_quadrotor_2d_rollout.py
    python visualize_quadrotor_2d_rollout.py --output_dir /path/to/output
"""

import argparse
import os
import tempfile
from functools import partial

import numpy as np
import pybullet as p
import pybullet_data

from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
from safe_control_gym.utils.registration import make

# ============ CONFIGURATION ============
# Initial state: [x, z, theta, x_dot, z_dot, theta_dot] (output order)
INIT_STATE = [-0.72, 0.33, -1.491593, 0.96, 0.96, 7.3]

# Environment parameters (matching dataset)
CTRL_FREQ = 100
PYB_FREQ = 5000
EPISODE_LEN_SEC = 1000
MAX_STEPS = 1200
GOAL_TOLERANCE = 0.2

# Termination thresholds
X_TERMINATION = 1.0
Z_MIN_TERMINATION = 0.1
Z_MAX_TERMINATION = 1.5
THETA_TERMINATION = float('inf')
X_DOT_TERMINATION = 1.0
Z_DOT_TERMINATION = 1.0
THETA_DOT_TERMINATION = 8.0

# Rendering
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
FPS = 30
FRAME_SKIP = max(1, CTRL_FREQ // FPS)
DRONE_VISUAL_SCALE = 5.0
# =======================================

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


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


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


def _draw_goal_circle(client, center, radius, color=[0.2, 0.75, 0.2], n_points=48, bead_radius=0.005):
    """Draw goal region border as a circle of small spheres in the X-Z plane."""
    cx, _, cz = center
    rgba = color + [1.0]
    bead_vis = p.createVisualShape(p.GEOM_SPHERE, radius=bead_radius,
                                   rgbaColor=rgba, physicsClientId=client)
    for j in range(n_points):
        th = 2 * np.pi * j / n_points
        pos = [cx + radius * np.cos(th), 0, cz + radius * np.sin(th)]
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=bead_vis,
                          basePosition=pos, physicsClientId=client)


def run_and_record(output_dir, model_path):
    import torch
    os.makedirs(output_dir, exist_ok=True)

    _patch_safe_ppo_act()

    # ---- Phase 1: Run physics, record poses ----
    env_kwargs = {
        'quad_type': QuadType.TWO_D,
        'task': 'stabilization',
        'ctrl_freq': CTRL_FREQ,
        'pyb_freq': PYB_FREQ,
        'episode_len_sec': EPISODE_LEN_SEC,
        'done_on_out_of_bound': True,
        'cost': 'quadratic',
        'normalized_rl_action_space': True,
        'gui': False,
        'randomized_init': False,
        'task_info': {
            'stabilization_goal': [0, 1],
            'stabilization_goal_tolerance': GOAL_TOLERANCE,
        },
        'constraints': SAFE_EXPLORER_CONSTRAINTS,
        'done_on_violation': False,
    }
    env_func = partial(make, 'quadrotor', **env_kwargs)

    # Create controller
    temp_dir = tempfile.mkdtemp()
    ctrl = make('safe_explorer_ppo', env_func, **ALGO_CONFIG, output_dir=temp_dir)
    ctrl.load(model_path)
    ctrl.obs_normalizer.set_read_only()

    env = env_func()

    # Set termination bounds (env internal order: x, x_dot, z, z_dot, theta, theta_dot)
    env.state_space.low[0] = -X_TERMINATION
    env.state_space.high[0] = X_TERMINATION
    env.state_space.low[2] = Z_MIN_TERMINATION
    env.state_space.high[2] = Z_MAX_TERMINATION
    env.state_space.low[4] = -THETA_TERMINATION
    env.state_space.high[4] = THETA_TERMINATION
    env.state_space.low[1] = -X_DOT_TERMINATION
    env.state_space.high[1] = X_DOT_TERMINATION
    env.state_space.low[3] = -Z_DOT_TERMINATION
    env.state_space.high[3] = Z_DOT_TERMINATION
    env.state_space.low[5] = -THETA_DOT_TERMINATION
    env.state_space.high[5] = THETA_DOT_TERMINATION

    # Reset and set initial state
    obs, info = env.reset()
    x, z, theta, x_dot, z_dot, theta_dot = INIT_STATE

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

    # Record poses and states
    poses = []
    state_history = [[x, z, normalize_angle(theta), x_dot, z_dot, theta_dot]]
    # Use custom 2D quadrotor visual for rendering (not the 3D cf2x model)
    urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'safe_control_gym', 'envs', 'gym_pybullet_drones',
                             'assets', 'quad2d_visual.urdf')

    pos0, orn0 = p.getBasePositionAndOrientation(env.DRONE_ID, physicsClientId=env.PYB_CLIENT)
    poses.append((list(pos0), list(orn0)))

    print(f'Initial state: x={x:.3f}, z={z:.3f}, theta={theta:.3f}, x_dot={x_dot:.3f}, z_dot={z_dot:.3f}, theta_dot={theta_dot:.3f}')
    print(f'Running physics rollout (max {MAX_STEPS} steps)...')

    success = False
    with torch.no_grad():
        for step in range(MAX_STEPS):
            normalized_obs = ctrl.obs_normalizer(obs)
            action = ctrl.select_action(normalized_obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record pose
            pos_i, orn_i = p.getBasePositionAndOrientation(env.DRONE_ID, physicsClientId=env.PYB_CLIENT)
            poses.append((list(pos_i), list(orn_i)))

            # Record state in output order
            s = env.state
            state_history.append([s[0], s[2], normalize_angle(s[4]), s[1], s[3], s[5]])

            if done:
                success = info.get('goal_reached', False)
                break

    n_steps = step + 1 if done else MAX_STEPS
    print(f"Physics: {n_steps} steps, {'SUCCESS' if success else 'FAIL'}")

    env.close()

    # ---- Phase 2: Replay with scaled drone, capture video ----
    print('Rendering video with scaled drone...')

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.resetSimulation(physicsClientId=client)

    # Scene setup
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)

    # Scaled drone
    drone_id = p.loadURDF(urdf_path, poses[0][0], poses[0][1],
                          globalScaling=DRONE_VISUAL_SCALE,
                          physicsClientId=client)
    p.changeVisualShape(drone_id, -1,
                        rgbaColor=[0.1, 0.1, 0.1, 1.0],
                        physicsClientId=client)

    # Goal region border - ring of small spheres
    _draw_goal_circle(client, [0, 0, 1], GOAL_TOLERANCE, color=[0.2, 0.75, 0.2])

    # Camera - angled so the 3D shape of the quad is visible
    cam_center_z = (Z_MIN_TERMINATION + Z_MAX_TERMINATION) / 2.0
    cam_view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, cam_center_z],
        distance=1.8,
        yaw=0,
        pitch=0,
        roll=0,
        upAxisIndex=2,
        physicsClientId=client
    )
    cam_pro = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
        nearVal=0.01,
        farVal=1000.0
    )

    # Render frames at FRAME_SKIP interval
    frames = []
    for i, (pos_i, orn_i) in enumerate(poses):
        if i == 0 or (i - 1) % FRAME_SKIP == 0:
            p.resetBasePositionAndOrientation(drone_id, pos_i, orn_i, physicsClientId=client)
            (w, h, rgb, _, _) = p.getCameraImage(
                width=RENDER_WIDTH, height=RENDER_HEIGHT,
                shadow=1,
                renderer=p.ER_TINY_RENDERER,
                viewMatrix=cam_view, projectionMatrix=cam_pro,
                physicsClientId=client
            )
            frames.append(np.reshape(rgb, (h, w, 4))[:, :, :3])

    p.disconnect(client)

    # Save video
    video_path = os.path.join(output_dir, 'quadrotor_2d_rollout.mp4')
    save_video(frames, video_path, FPS)

    # Save x-z trajectory plot
    state_history = np.array(state_history)
    plot_path = os.path.join(output_dir, 'quadrotor_2d_rollout_xz.png')
    plot_xz_trajectory(state_history, plot_path, success)


def save_video(frames, path, fps):
    import imageio
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f'Saved video: {path} ({len(frames)} frames, {fps} fps, {len(frames)/fps:.1f}s)')


def plot_xz_trajectory(states, path, success):
    """Plot x vs z trajectory in the state space."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    x_vals = states[:, 0]
    z_vals = states[:, 1]

    # Plot trajectory with color gradient for time
    n = len(x_vals)
    for i in range(n - 1):
        t = i / max(n - 1, 1)
        color = (0.2 + 0.6 * t, 0.3 * (1 - t), 0.8 * (1 - t))
        ax.plot(x_vals[i:i + 2], z_vals[i:i + 2], color=color, linewidth=1.5)

    # Mark start and end
    ax.plot(x_vals[0], z_vals[0], 'bo', markersize=10, label='Start', zorder=5)
    ax.plot(x_vals[-1], z_vals[-1], 'rs', markersize=10, label='End', zorder=5)

    # Mark goal with tolerance circle
    goal_circle = plt.Circle((0, 1), GOAL_TOLERANCE, color='green', alpha=0.2, label=f'Goal (r={GOAL_TOLERANCE})')
    ax.add_patch(goal_circle)

    # State space bounds
    ax.axvline(-X_TERMINATION, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(X_TERMINATION, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(Z_MIN_TERMINATION, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(Z_MAX_TERMINATION, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlim(-X_TERMINATION, X_TERMINATION)
    ax.set_ylim(Z_MIN_TERMINATION, Z_MAX_TERMINATION)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('z (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    result_str = 'SUCCESS' if success else 'FAIL'
    ax.set_title(f'2D Quadrotor Trajectory ({result_str}, {len(states)-1} steps)', fontsize=13)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot: {path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize 2D quadrotor rollout')
    parser.add_argument('--output_dir', type=str, default='rollout_visualizations',
                        help='Output directory')
    parser.add_argument('--model_path', type=str,
                        default='examples/rl/models/safe_explorer_ppo/safe_explorer_ppo_model_quadrotor_2D_stab.pt',
                        help='Path to trained Safe Explorer PPO model')
    args = parser.parse_args()

    run_and_record(args.output_dir, args.model_path)
    print('\nDone!')


if __name__ == '__main__':
    main()
