#!/usr/bin/env python3
"""
Visualize a 3D quadrotor rollout from a given initial state.
Generates a video (MP4) and a 3D x-y-z trajectory plot.

Phase 1: Run physics with the normal small drone, record poses.
Phase 2: Replay poses with a scaled-up drone for video capture.

Usage:
    python visualize_quadrotor_3d_rollout.py
    python visualize_quadrotor_3d_rollout.py --output_dir /path/to/output
"""

import argparse
import os
import numpy as np
import pybullet as p
import pybullet_data
from functools import partial
from PIL import Image

from safe_control_gym.utils.registration import make
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType

# ============ CONFIGURATION ============
# Initial state: [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r] (Euler order)
INIT_STATE = [0.673787, -1.531676, 2.681568, 1.116336, -0.232252, -0.608311, 1.147831, 1.465307, -1.566223, -23.81415, 17.591969, 23.091495]

# Environment parameters
CTRL_FREQ = 50
PYB_FREQ = 500
EPISODE_LEN_SEC = 1000
MAX_STEPS = 1200

# LQR controller gains
Q_LQR = [1] * 12
R_LQR = [0.1] * 4

# Termination thresholds
X_TERMINATION = 1.8
Y_TERMINATION = 1.8
Z_MIN_TERMINATION = 0.1
Z_MAX_TERMINATION = 3.0
X_DOT_TERMINATION = 3.0
Y_DOT_TERMINATION = 3.0
Z_DOT_TERMINATION = 3.0
P_TERMINATION = 24.0
Q_TERMINATION = 24.0
R_TERMINATION = 24.0

# Rendering
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
FPS = 30
FRAME_SKIP = max(1, CTRL_FREQ // FPS)
DRONE_VISUAL_SCALE = 5.0
# =======================================


def _draw_goal_ring(client, center, radius, color=[0.2, 0.75, 0.2], n_points=48, bead_radius=0.005, n_rings=3):
    """Draw goal region border as rings of small spheres (visible in TinyRenderer)."""
    cx, cy, cz = center
    rgba = color + [1.0]
    bead_vis = p.createVisualShape(p.GEOM_SPHERE, radius=bead_radius,
                                   rgbaColor=rgba, physicsClientId=client)
    # Draw rings at different latitudes (equator + above/below)
    for ring_i in range(n_rings):
        phi = np.pi * (ring_i + 1) / (n_rings + 1)
        r = radius * np.sin(phi)
        z = cz + radius * np.cos(phi)
        for j in range(n_points):
            th = 2 * np.pi * j / n_points
            pos = [cx + r * np.cos(th), cy + r * np.sin(th), z]
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=bead_vis,
                              basePosition=pos, physicsClientId=client)


def run_and_record(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Phase 1: Run physics, record poses ----
    env_kwargs = {
        'quad_type': QuadType.THREE_D,
        'task': 'stabilization',
        'ctrl_freq': CTRL_FREQ,
        'pyb_freq': PYB_FREQ,
        'episode_len_sec': EPISODE_LEN_SEC,
        'done_on_out_of_bound': True,
        'cost': 'quadratic',
        'gui': False,
        'randomized_init': False,
        'task_info': {
            'stabilization_goal': [0, 0, 1],
            'stabilization_goal_tolerance': 0.1,
        },
        'normalized_rl_action_space': False,
    }

    env_func = partial(make, 'quadrotor', **env_kwargs)

    ctrl = make('lqr', env_func,
                q_lqr=Q_LQR,
                r_lqr=R_LQR,
                discrete_dynamics=True)

    env = env_func()

    # Set termination bounds
    # 3D env internal order: [x, x_dot, y, y_dot, z, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot]
    env.state_space.low[0] = -X_TERMINATION
    env.state_space.high[0] = X_TERMINATION
    env.state_space.low[1] = -X_DOT_TERMINATION
    env.state_space.high[1] = X_DOT_TERMINATION
    env.state_space.low[2] = -Y_TERMINATION
    env.state_space.high[2] = Y_TERMINATION
    env.state_space.low[3] = -Y_DOT_TERMINATION
    env.state_space.high[3] = Y_DOT_TERMINATION
    env.state_space.low[4] = Z_MIN_TERMINATION
    env.state_space.high[4] = Z_MAX_TERMINATION
    env.state_space.low[5] = -Z_DOT_TERMINATION
    env.state_space.high[5] = Z_DOT_TERMINATION
    env.state_space.low[6] = -float('inf')   # phi
    env.state_space.high[6] = float('inf')
    env.state_space.low[7] = -P_TERMINATION
    env.state_space.high[7] = P_TERMINATION
    env.state_space.low[8] = -float('inf')   # theta
    env.state_space.high[8] = float('inf')
    env.state_space.low[9] = -Q_TERMINATION
    env.state_space.high[9] = Q_TERMINATION
    env.state_space.low[10] = -float('inf')  # psi
    env.state_space.high[10] = float('inf')
    env.state_space.low[11] = -R_TERMINATION
    env.state_space.high[11] = R_TERMINATION

    # Reset and set initial state
    obs, info = env.reset()
    x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p_rate, q_rate, r_rate = INIT_STATE

    p.resetBasePositionAndOrientation(
        env.DRONE_ID,
        [x, y, z],
        p.getQuaternionFromEuler([phi, theta, psi]),
        physicsClientId=env.PYB_CLIENT)

    p.resetBaseVelocity(
        env.DRONE_ID,
        [x_dot, y_dot, z_dot],
        [p_rate, q_rate, r_rate],
        physicsClientId=env.PYB_CLIENT)

    env._update_and_store_kinematic_information()
    obs = env._get_observation()

    # Record poses (pos, quaternion) and positions
    poses = []  # [(pos, orn), ...]
    pos_history = [[x, y, z]]
    urdf_path = env.URDF_PATH

    pos0, orn0 = p.getBasePositionAndOrientation(env.DRONE_ID, physicsClientId=env.PYB_CLIENT)
    poses.append((list(pos0), list(orn0)))

    print(f"Initial state: x={x:.3f}, y={y:.3f}, z={z:.3f}, phi={phi:.3f}, theta={theta:.3f}, psi={psi:.3f}")
    print(f"Running physics rollout (max {MAX_STEPS} steps)...")

    success = False
    for step in range(MAX_STEPS):
        action = ctrl.select_action(obs, info)
        obs, reward, done, info = env.step(action)

        # Record pose
        pos_i, orn_i = p.getBasePositionAndOrientation(env.DRONE_ID, physicsClientId=env.PYB_CLIENT)
        poses.append((list(pos_i), list(orn_i)))

        # Record position: env internal [x, x_dot, y, y_dot, z, z_dot, ...]
        s = env.state
        pos_history.append([s[0], s[2], s[4]])

        if done:
            success = info.get('goal_reached', False)
            break

    n_steps = step + 1 if done else MAX_STEPS
    print(f"Physics: {n_steps} steps, {'SUCCESS' if success else 'FAIL'}")

    env.close()

    # ---- Phase 2: Replay with scaled drone, capture video ----
    print("Rendering video with scaled drone...")

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.resetSimulation(physicsClientId=client)

    # Scene setup
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)

    # Ground plane with grey/white checker
    plane_id = p.loadURDF('plane.urdf', [0, 0, 0], physicsClientId=client)
    tex_data = np.ones((64, 64, 3), dtype=np.uint8) * 255
    tex_data[32:, :32] = [210, 210, 210]
    tex_data[:32, 32:] = [210, 210, 210]
    Image.fromarray(tex_data).save('/tmp/_checker_grey.png')
    tex_id = p.loadTexture('/tmp/_checker_grey.png', physicsClientId=client)
    p.changeVisualShape(plane_id, -1, textureUniqueId=tex_id,
                        rgbaColor=[1, 1, 1, 1], physicsClientId=client)

    # Scaled drone
    drone_id = p.loadURDF(urdf_path, poses[0][0], poses[0][1],
                          globalScaling=DRONE_VISUAL_SCALE,
                          physicsClientId=client)
    p.changeVisualShape(drone_id, -1,
                        rgbaColor=[0.1, 0.1, 0.1, 1.0],
                        physicsClientId=client)

    # Goal region border - rings of small spheres
    _draw_goal_ring(client, [0, 0, 1], 0.1, color=[0.2, 0.75, 0.2])

    # Camera
    cam_center_z = (Z_MIN_TERMINATION + Z_MAX_TERMINATION) / 2.0
    cam_view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, cam_center_z],
        distance=3.5,
        yaw=-45,
        pitch=-25,
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
    video_path = os.path.join(output_dir, "quadrotor_3d_rollout.mp4")
    save_video(frames, video_path, FPS)

    # Save 3D trajectory plot
    pos_history = np.array(pos_history)
    plot_path = os.path.join(output_dir, "quadrotor_3d_rollout_xyz.png")
    plot_xyz_trajectory(pos_history, plot_path, success)


def save_video(frames, path, fps):
    import imageio
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Saved video: {path} ({len(frames)} frames, {fps} fps, {len(frames)/fps:.1f}s)")


def plot_xyz_trajectory(positions, path, success):
    """Plot 3D x-y-z trajectory in the state space."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_vals = positions[:, 0]
    y_vals = positions[:, 1]
    z_vals = positions[:, 2]

    # Plot trajectory with color gradient for time
    n = len(x_vals)
    for i in range(n - 1):
        t = i / max(n - 1, 1)
        color = (0.2 + 0.6 * t, 0.3 * (1 - t), 0.8 * (1 - t))
        ax.plot(x_vals[i:i+2], y_vals[i:i+2], z_vals[i:i+2], color=color, linewidth=1.5)

    # Mark start and end
    ax.scatter(*positions[0], color='blue', s=100, label='Start', zorder=5)
    ax.scatter(*positions[-1], color='red', s=100, marker='s', label='End', zorder=5)

    # Mark goal with tolerance sphere
    goal_tolerance = 0.1
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sx = goal_tolerance * np.outer(np.cos(u), np.sin(v))
    sy = goal_tolerance * np.outer(np.sin(u), np.sin(v))
    sz = 1.0 + goal_tolerance * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(sx, sy, sz, color='green', alpha=0.15)

    # State space bounds as wireframe box
    xl, xh = -X_TERMINATION, X_TERMINATION
    yl, yh = -Y_TERMINATION, Y_TERMINATION
    zl, zh = Z_MIN_TERMINATION, Z_MAX_TERMINATION
    for z_edge in [zl, zh]:
        ax.plot([xl, xh], [yl, yl], [z_edge, z_edge], 'gray', alpha=0.3, linewidth=0.5)
        ax.plot([xl, xh], [yh, yh], [z_edge, z_edge], 'gray', alpha=0.3, linewidth=0.5)
        ax.plot([xl, xl], [yl, yh], [z_edge, z_edge], 'gray', alpha=0.3, linewidth=0.5)
        ax.plot([xh, xh], [yl, yh], [z_edge, z_edge], 'gray', alpha=0.3, linewidth=0.5)
    for x_edge in [xl, xh]:
        for y_edge in [yl, yh]:
            ax.plot([x_edge, x_edge], [y_edge, y_edge], [zl, zh], 'gray', alpha=0.3, linewidth=0.5)

    ax.set_xlim(xl, xh)
    ax.set_ylim(yl, yh)
    ax.set_zlim(zl, zh)
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.set_zlabel('z (m)', fontsize=11)
    ax.legend(fontsize=10)

    result_str = 'SUCCESS' if success else 'FAIL'
    ax.set_title(f'3D Quadrotor Trajectory ({result_str}, {len(positions)-1} steps)', fontsize=13)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D quadrotor rollout')
    parser.add_argument('--output_dir', type=str, default='rollout_visualizations',
                        help='Output directory')
    args = parser.parse_args()

    run_and_record(args.output_dir)
    print("\nDone!")


if __name__ == '__main__':
    main()
