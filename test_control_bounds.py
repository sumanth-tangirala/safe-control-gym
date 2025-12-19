#!/usr/bin/env python3
"""Test script to verify control bounds are being applied correctly."""

import numpy as np
from functools import partial
from safe_control_gym.utils.registration import make


def test_control_bounds():
    """Test that different control bounds produce different results."""

    # Test with control bound 5.0
    print("Testing with control bound = 5.0 N")
    env_func_5 = partial(make,
                        'cartpole',
                        task='stabilization',
                        ctrl_freq=15,
                        pyb_freq=750,
                        episode_len_sec=10,
                        done_on_out_of_bound=True,
                        cost='quadratic',
                        gui=False,
                        randomized_init=False,
                        action_scale=5.0)

    ctrl_5 = make('lqr',
                  env_func_5,
                  q_lqr=[1, 1, 1, 1],
                  r_lqr=[0.1],
                  discrete_dynamics=True)

    env_5 = env_func_5()

    # Reset and set initial state
    env_5.reset()
    env_5.state = np.array([1.0, 0.0, 0.5, 0.0])  # x=1, x_dot=0, theta=0.5, theta_dot=0

    # Get observation and action
    obs_5 = env_5._get_observation()
    action_5 = ctrl_5.select_action(obs_5, None)

    print(f"  LQR action (before clipping): {action_5}")
    print(f"  Physical action bounds: {env_5.physical_action_bounds}")

    # Step through environment to see clipped action
    obs_5_next, rew_5, done_5, info_5 = env_5.step(action_5)
    print(f"  Clipped action: {env_5.current_clipped_action}")
    print(f"  Next state: {env_5.state}")

    env_5.close()
    ctrl_5.close()

    print("\n" + "="*80 + "\n")

    # Test with control bound 20.0
    print("Testing with control bound = 20.0 N")
    env_func_20 = partial(make,
                         'cartpole',
                         task='stabilization',
                         ctrl_freq=15,
                         pyb_freq=750,
                         episode_len_sec=10,
                         done_on_out_of_bound=True,
                         cost='quadratic',
                         gui=False,
                         randomized_init=False,
                         action_scale=20.0)

    ctrl_20 = make('lqr',
                   env_func_20,
                   q_lqr=[1, 1, 1, 1],
                   r_lqr=[0.1],
                   discrete_dynamics=True)

    env_20 = env_func_20()

    # Reset and set initial state (same as before)
    env_20.reset()
    env_20.state = np.array([1.0, 0.0, 0.5, 0.0])

    # Get observation and action
    obs_20 = env_20._get_observation()
    action_20 = ctrl_20.select_action(obs_20, None)

    print(f"  LQR action (before clipping): {action_20}")
    print(f"  Physical action bounds: {env_20.physical_action_bounds}")

    # Step through environment to see clipped action
    obs_20_next, rew_20, done_20, info_20 = env_20.step(action_20)
    print(f"  Clipped action: {env_20.current_clipped_action}")
    print(f"  Next state: {env_20.state}")

    env_20.close()
    ctrl_20.close()

    print("\n" + "="*80 + "\n")
    print("Comparison:")
    print(f"  State difference: {np.linalg.norm(env_5.state - env_20.state)}")
    print(f"  Clipped action difference: {np.linalg.norm(env_5.current_clipped_action - env_20.current_clipped_action)}")

    if np.linalg.norm(env_5.state - env_20.state) < 1e-6:
        print("\n  WARNING: States are identical! Control bounds may not be working.")
    else:
        print("\n  States are different. Control bounds are working correctly.")


if __name__ == '__main__':
    test_control_bounds()
