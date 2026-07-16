"""Generate visualization images for CartPole, Quadrotor 2D, and Quadrotor 3D environments."""

import os
import numpy as np
from PIL import Image
import pybullet as p

from safe_control_gym.utils.registration import make
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import QuadType

# High resolution settings
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080


def save_frame(frame, filename):
    """Save a rendered frame as PNG image."""
    # Frame is RGBA, convert to RGB for saving
    img = Image.fromarray(frame[:, :, :3])
    img.save(filename)
    print(f"Saved: {filename} ({frame.shape[1]}x{frame.shape[0]})")


def visualize_cartpole(output_dir):
    """Generate visualization of CartPole environment."""
    print("\n=== CartPole Visualization ===")

    # Create environment (headless mode for rendering)
    env = make('cartpole', gui=False, normalized_rl_action_space=False)

    # Reset to get initial state
    obs, info = env.reset()

    # Set upright pose (stabilization goal)
    # State: [x, x_dot, theta, theta_dot]
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])  # Cart centered, pole upright

    # Set state in PyBullet
    cart_pos = [initial_state[0], 0, 0]
    p.resetBasePositionAndOrientation(
        env.CARTPOLE_ID,
        cart_pos,
        p.getQuaternionFromEuler([0, initial_state[2], 0]),
        physicsClientId=env.PYB_CLIENT
    )

    # Custom camera view for better close-up
    cam_view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.3],
        distance=1.5,  # Closer camera
        yaw=0,  # Side view
        pitch=-5,
        roll=0,
        upAxisIndex=2,
        physicsClientId=env.PYB_CLIENT
    )
    cam_pro = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
        nearVal=0.1,
        farVal=100.0
    )

    # Render with custom camera at high resolution
    (w, h, rgb, _, _) = p.getCameraImage(
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        viewMatrix=cam_view,
        projectionMatrix=cam_pro,
        physicsClientId=env.PYB_CLIENT
    )
    frame = np.reshape(rgb, (h, w, 4))
    save_frame(frame, os.path.join(output_dir, "cartpole.png"))

    env.close()


def visualize_quadrotor_2d(output_dir):
    """Generate visualization of 2D Quadrotor environment."""
    print("\n=== Quadrotor 2D Visualization ===")

    # Create 2D quadrotor environment
    env = make('quadrotor',
               gui=False,
               quad_type=QuadType.TWO_D,
               task='stabilization',
               normalized_rl_action_space=False)

    # Reset environment
    obs, info = env.reset()

    # Set level pose for clean visualization
    # 2D state: [x, x_dot, z, z_dot, theta, theta_dot]
    pos = [0.0, 0, 0.5]  # Centered position
    theta = 0.0  # Level (no tilt)
    orn = p.getQuaternionFromEuler([0, theta, 0])  # No rotation

    p.resetBasePositionAndOrientation(
        env.DRONE_ID,
        pos,
        orn,
        physicsClientId=env.PYB_CLIENT
    )

    # Custom camera view for better close-up
    cam_view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.5],
        distance=0.25,  # Very close camera
        yaw=0,  # Side view for 2D
        pitch=-10,
        roll=0,
        upAxisIndex=2,
        physicsClientId=env.PYB_CLIENT
    )
    cam_pro = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=RENDER_WIDTH / RENDER_HEIGHT,
        nearVal=0.1,
        farVal=1000.0
    )

    # Render with custom camera at high resolution
    [w, h, rgb, _, _] = p.getCameraImage(
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        shadow=1,
        viewMatrix=cam_view,
        projectionMatrix=cam_pro,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=env.PYB_CLIENT
    )
    frame = np.reshape(rgb, (h, w, 4))
    save_frame(frame, os.path.join(output_dir, "quadrotor_2d.png"))

    env.close()


def visualize_quadrotor_3d(output_dir):
    """Generate visualization of 3D Quadrotor environment."""
    print("\n=== Quadrotor 3D Visualization ===")

    # Create 3D quadrotor environment
    # Need to provide task_info with 3D stabilization goal (x, y, z)
    env = make('quadrotor',
               gui=False,
               quad_type=QuadType.THREE_D,
               task='stabilization',
               task_info={'stabilization_goal': [0, 0, 1]},  # x=0, y=0, z=1
               normalized_rl_action_space=False)

    # Reset environment
    obs, info = env.reset()

    # Set level pose for clean visualization
    pos = [0.0, 0.0, 0.5]  # Centered
    roll, pitch, yaw = 0.0, 0.0, 0.0  # Level (no rotation)
    orn = p.getQuaternionFromEuler([roll, pitch, yaw])

    p.resetBasePositionAndOrientation(
        env.DRONE_ID,
        pos,
        orn,
        physicsClientId=env.PYB_CLIENT
    )

    # Custom camera view for better close-up with 3D perspective
    cam_view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0.5],
        distance=0.25,  # Very close camera
        yaw=-45,  # Angled view to show 3D
        pitch=-15,
        roll=0,
        upAxisIndex=2,
        physicsClientId=env.PYB_CLIENT
    )
    cam_pro = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=RENDER_WIDTH / RENDER_HEIGHT,
        nearVal=0.1,
        farVal=1000.0
    )

    # Render with custom camera at high resolution
    [w, h, rgb, _, _] = p.getCameraImage(
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        shadow=1,
        viewMatrix=cam_view,
        projectionMatrix=cam_pro,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=env.PYB_CLIENT
    )
    frame = np.reshape(rgb, (h, w, 4))
    save_frame(frame, os.path.join(output_dir, "quadrotor_3d.png"))

    env.close()


def main():
    # Output directory for images
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating environment visualizations...")
    print(f"Output directory: {output_dir}")

    # Generate visualizations
    visualize_cartpole(output_dir)
    visualize_quadrotor_2d(output_dir)
    visualize_quadrotor_3d(output_dir)

    print("\n=== Done! ===")
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
