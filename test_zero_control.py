#!/usr/bin/env python3
"""Test with zero control bounds to verify clipping."""

import numpy as np
from functools import partial
from safe_control_gym.utils.registration import make


def test_zero_control():
    """Test that control_bound=0.0 actually clips actions to zero."""

    print("Testing with control bound = 0.0 N")
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
                      action_scale=0.0)

    ctrl = make('lqr',
                env_func,
                q_lqr=[1, 1, 1, 1],
                r_lqr=[0.1],
                discrete_dynamics=True)

    env = env_func()

    print(f"Environment action_scale: {env.action_scale}")
    print(f"Environment physical_action_bounds: {env.physical_action_bounds}")

    # Reset and set a non-zero initial state
    env.reset()
    env.state = np.array([1.0, 0.0, 0.5, 0.0])  # x=1, x_dot=0, theta=0.5, theta_dot=0

    print(f"\nInitial state: {env.state}")

    # Get observation and action from LQR
    obs = env._get_observation()
    action = ctrl.select_action(obs, None)

    print(f"LQR action (before clipping): {action}")

    # Step through environment
    for i in range(5):
        obs, rew, done, info = env.step(action)
        action = ctrl.select_action(obs, None)
        print(f"Step {i+1}:")
        print(f"  LQR action: {action}")
        print(f"  Physical action: {env.current_physical_action}")
        print(f"  Noisy action: {env.current_noisy_physical_action}")
        print(f"  Clipped action: {env.current_clipped_action}")
        print(f"  State: {env.state}")

    env.close()
    ctrl.close()


if __name__ == '__main__':
    test_zero_control()
