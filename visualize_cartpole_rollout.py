#!/usr/bin/env python3
"""
Visualize a cartpole rollout from a given initial state.
Generates a video (MP4) and an x-theta state-space trajectory plot.

Phase 1: Run physics with the normal cartpole, record joint states.
Phase 2: Replay joint states with a refined visual cartpole for video capture.

Usage:
    python visualize_cartpole_rollout.py
    python visualize_cartpole_rollout.py --output_dir /path/to/output
"""

import argparse
import os
from functools import partial

import numpy as np
import pybullet as p
import pybullet_data

from safe_control_gym.utils.registration import make

# ============ CONFIGURATION ============
# Initial state: [x, x_dot, theta, theta_dot] (internal order)
INIT_STATE = [-3.0, -3.5, 1.858407, -4.5]

# Environment parameters (matching dataset)
CTRL_FREQ = 100
PYB_FREQ = 5000
EPISODE_LEN_SEC = 10
ACTION_SCALE = 2000.0
MAX_STEPS = 1000

# Termination thresholds
X_THRESHOLD = 6.0
X_DOT_THRESHOLD = 5.0
THETA_THRESHOLD = float('inf')
THETA_DOT_THRESHOLD = 5.0

# Rendering
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
FPS = 30
FRAME_SKIP = max(1, CTRL_FREQ // FPS)
CARTPOLE_VISUAL_SCALE = 2.6
# =======================================


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


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


def run_and_record(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Phase 1: Run physics, record joint states ----
    env_func = partial(make,
                       'cartpole',
                       task='stabilization',
                       ctrl_freq=CTRL_FREQ,
                       pyb_freq=PYB_FREQ,
                       episode_len_sec=EPISODE_LEN_SEC,
                       done_on_out_of_bound=True,
                       cost='quadratic',
                       gui=False,
                       randomized_init=False,
                       obs_wrap_angle=True,
                       x_dot_limit=float('inf'),
                       theta_dot_limit=float('inf'),
                       action_scale=ACTION_SCALE)

    ctrl = make('lqr', env_func,
                q_lqr=[1, 1, 1, 1],
                r_lqr=[0.1],
                discrete_dynamics=True)

    env = env_func()

    # Set termination thresholds
    env.x_threshold = X_THRESHOLD
    env.x_dot_threshold = X_DOT_THRESHOLD
    env.theta_threshold_radians = THETA_THRESHOLD
    env.theta_dot_threshold = THETA_DOT_THRESHOLD

    # Reset and set initial state
    obs, info = env.reset()
    x, x_dot, theta, theta_dot = INIT_STATE

    p.resetJointState(env.CARTPOLE_ID, jointIndex=0,
                      targetValue=x, targetVelocity=x_dot,
                      physicsClientId=env.PYB_CLIENT)
    p.resetJointState(env.CARTPOLE_ID, jointIndex=1,
                      targetValue=theta, targetVelocity=theta_dot,
                      physicsClientId=env.PYB_CLIENT)

    env.state = np.array([x, x_dot, theta, theta_dot])
    obs = env._get_observation()

    # Record joint states: [(x, theta), ...]
    joint_states = [(x, theta)]
    state_history = [[x, normalize_angle(theta), x_dot, theta_dot]]

    print(f'Initial state: x={x:.3f}, x_dot={x_dot:.3f}, theta={theta:.3f}, theta_dot={theta_dot:.3f}')
    print(f'Running physics rollout (max {MAX_STEPS} steps)...')

    success = False
    for step in range(MAX_STEPS):
        action = ctrl.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        s = env.state  # internal: [x, x_dot, theta, theta_dot]
        joint_states.append((s[0], s[2]))
        state_history.append([s[0], normalize_angle(s[2]), s[1], s[3]])

        if done:
            success = info.get('goal_reached', False)
            break

    n_steps = step + 1 if done else MAX_STEPS
    print(f"Physics: {n_steps} steps, {'SUCCESS' if success else 'FAIL'}")

    env.close()

    # ---- Phase 2: Replay with visual cartpole, capture video ----
    print('Rendering video with visual cartpole...')

    visual_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'safe_control_gym', 'envs', 'gym_control',
                               'assets', 'cartpole_visual.urdf')

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.resetSimulation(physicsClientId=client)

    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)

    # Load visual cartpole (scaled up to compensate for zoomed-out camera)
    cartpole_id = p.loadURDF(visual_urdf, [0, 0, 0],
                             globalScaling=CARTPOLE_VISUAL_SCALE,
                             physicsClientId=client)

    # Goal circle at pole tip when upright at x=0
    pole_tip_z = 0.65 * CARTPOLE_VISUAL_SCALE  # visual pole length * scale
    _draw_goal_circle(client, [0, 0, pole_tip_z], 0.15, color=[0.2, 0.75, 0.2], bead_radius=0.01)

    # Set initial joint state (prismatic joint scaled by globalScaling)
    p.resetJointState(cartpole_id, jointIndex=0,
                      targetValue=joint_states[0][0] / CARTPOLE_VISUAL_SCALE,
                      physicsClientId=client)
    p.resetJointState(cartpole_id, jointIndex=1,
                      targetValue=joint_states[0][1],
                      physicsClientId=client)

    # Camera
    cam_view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.3],
        distance=4.0,
        yaw=0,
        pitch=-5,
        roll=0,
        upAxisIndex=2,
        physicsClientId=client
    )
    cam_pro = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
        nearVal=0.01,
        farVal=1000.0
    )

    # Render frames
    frames = []
    for i, (x_i, theta_i) in enumerate(joint_states):
        if i == 0 or (i - 1) % FRAME_SKIP == 0:
            p.resetJointState(cartpole_id, jointIndex=0,
                              targetValue=x_i / CARTPOLE_VISUAL_SCALE,
                              physicsClientId=client)
            p.resetJointState(cartpole_id, jointIndex=1,
                              targetValue=theta_i, physicsClientId=client)
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
    video_path = os.path.join(output_dir, 'cartpole_rollout.mp4')
    save_video(frames, video_path, FPS)

    # Save x-theta trajectory plot
    state_history = np.array(state_history)
    plot_path = os.path.join(output_dir, 'cartpole_rollout_xtheta.png')
    plot_xtheta_trajectory(state_history, plot_path, success)


def save_video(frames, path, fps):
    """Save frames as MP4 using imageio."""
    import imageio
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f'Saved video: {path} ({len(frames)} frames, {fps} fps, {len(frames)/fps:.1f}s)')


def plot_xtheta_trajectory(states, path, success):
    """Plot x vs theta trajectory in the state space."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    x_vals = states[:, 0]
    theta_vals = states[:, 1]

    # Plot trajectory
    ax.plot(x_vals, theta_vals, color='tab:blue', linewidth=1.5)

    # Mark start and end
    ax.plot(x_vals[0], theta_vals[0], 'bo', markersize=10, label='Start', zorder=5)
    ax.plot(x_vals[-1], theta_vals[-1], 'rs', markersize=10, label='End', zorder=5)

    # Mark goal with tolerance circle
    goal_tolerance = 0.05
    goal_circle = plt.Circle((0, 0), goal_tolerance, color='green', alpha=0.2, label=f'Goal (r={goal_tolerance})')
    ax.add_patch(goal_circle)

    # State space bounds
    ax.axvline(-X_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(X_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(-np.pi, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(np.pi, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlim(-X_THRESHOLD, X_THRESHOLD)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel(r'$\theta$ (rad)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    result_str = 'SUCCESS' if success else 'FAIL'
    ax.set_title(f'Cartpole Trajectory ({result_str}, {len(states)-1} steps)', fontsize=13)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot: {path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize cartpole rollout')
    parser.add_argument('--output_dir', type=str, default='rollout_visualizations',
                        help='Output directory')
    args = parser.parse_args()

    run_and_record(args.output_dir)
    print('\nDone!')


if __name__ == '__main__':
    main()
