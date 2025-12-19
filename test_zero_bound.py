#!/usr/bin/env python3
"""Test with zero control bound - system should barely move."""

import numpy as np
from functools import partial
from safe_control_gym.utils.registration import make
import pybullet as p


def test_zero_bound():
    """Test that control_bound=0.0 produces minimal state changes."""

    init_state = [2.0, 0.0, 1.0, 0.0]  # x=2, x_dot=0, theta=1rad, theta_dot=0

    print(f"Initial state: x={init_state[0]}, x_dot={init_state[1]}, theta={init_state[2]}, theta_dot={init_state[3]}")
    print("="*80)

    env_func = partial(make,
                      'cartpole',
                      task='stabilization',
                      ctrl_freq=15,
                      pyb_freq=750,
                      episode_len_sec=10,
                      done_on_out_of_bound=False,
                      cost='quadratic',
                      gui=False,
                      randomized_init=False,
                      x_dot_limit=10.0,
                      theta_dot_limit=10.0,
                      action_scale=0.0)

    ctrl = make('lqr',
                env_func,
                q_lqr=[1, 1, 1, 1],
                r_lqr=[0.1],
                discrete_dynamics=True)

    env = env_func()
    env.reset()

    print(f"Environment action_scale: {env.action_scale}")
    print(f"Environment physical_action_bounds: {env.physical_action_bounds}")
    print(f"Velocity limits: x_dot=±{env.x_dot_limit}, theta_dot=±{env.theta_dot_limit}")

    # Set initial state
    x, x_dot, theta, theta_dot = init_state
    p.resetJointState(env.CARTPOLE_ID, jointIndex=0, targetValue=x, targetVelocity=x_dot, physicsClientId=env.PYB_CLIENT)
    p.resetJointState(env.CARTPOLE_ID, jointIndex=1, targetValue=theta, targetVelocity=theta_dot, physicsClientId=env.PYB_CLIENT)
    env.state = np.array([x, x_dot, theta, theta_dot])

    print(f"\nInitial state (after setting): {env.state}")

    trajectory = [env.state.copy()]

    # Run for 20 steps
    for step in range(20):
        obs = env._get_observation()
        action = ctrl.select_action(obs, None)
        obs, reward, done, info = env.step(action)

        if step < 3:
            print(f"\nStep {step+1}:")
            print(f"  LQR action: {action}")
            print(f"  Clipped action: {env.current_clipped_action}")
            print(f"  State: {env.state}")

        trajectory.append(env.state.copy())

    trajectory = np.array(trajectory)

    print(f"\n{'='*80}")
    print(f"Final state (step 20): {env.state}")
    print(f"\nState variable ranges over trajectory:")
    print(f"  x: [{trajectory[:, 0].min():.6f}, {trajectory[:, 0].max():.6f}]")
    print(f"  x_dot: [{trajectory[:, 1].min():.6f}, {trajectory[:, 1].max():.6f}]")
    print(f"  theta: [{trajectory[:, 2].min():.6f}, {trajectory[:, 2].max():.6f}]")
    print(f"  theta_dot: [{trajectory[:, 3].min():.6f}, {trajectory[:, 3].max():.6f}]")

    env.close()
    ctrl.close()


if __name__ == '__main__':
    test_zero_bound()
