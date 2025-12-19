#!/usr/bin/env python3
"""Debug LQR gain computation with different control bounds."""

import numpy as np
from functools import partial
from safe_control_gym.utils.registration import make


def debug_lqr_gain(control_bound):
    """Check LQR gain with specific control bound."""

    print(f"\n{'='*80}")
    print(f"Testing with control bound = {control_bound} N")
    print(f"{'='*80}\n")

    env_func = partial(make,
                      'cartpole',
                      task='stabilization',
                      ctrl_freq=15,
                      pyb_freq=750,
                      episode_len_sec=10,
                      done_on_out_of_bound=True,
                      cost='quadratic',
                      gui=False,
                      randomized_init=False,
                      action_scale=control_bound)

    # Create LQR controller
    ctrl = make('lqr',
                env_func,
                q_lqr=[1, 1, 1, 1],
                r_lqr=[0.1],
                discrete_dynamics=True)

    # Check the controller's environment
    print(f"LQR controller's env action_scale: {ctrl.env.action_scale}")
    print(f"LQR controller's env physical_action_bounds: {ctrl.env.physical_action_bounds}")
    print(f"LQR gain matrix:\n{ctrl.gain}")
    print(f"LQR model U_EQ: {ctrl.model.U_EQ}")

    # Test with a sample state
    test_state = np.array([1.0, 0.0, 0.5, 0.0])  # x=1, x_dot=0, theta=0.5, theta_dot=0
    test_obs = test_state  # For stabilization task, obs = state

    action = ctrl.select_action(test_obs, None)
    print(f"\nFor test state {test_state}:")
    print(f"  LQR action: {action}")
    print(f"  Action components: -K @ (x - x_goal) + u_eq")
    print(f"    -K @ (x - x_goal) = {-ctrl.gain @ (test_obs - ctrl.env.X_GOAL)}")
    print(f"    u_eq = {ctrl.model.U_EQ}")

    ctrl.close()


if __name__ == '__main__':
    debug_lqr_gain(5.0)
    debug_lqr_gain(10.0)
    debug_lqr_gain(20.0)
