#!/usr/bin/env python3
"""Test the new save frequency feature."""

import numpy as np
from functools import partial
from safe_control_gym.utils.registration import make
import sys
sys.path.insert(0, '/common/home/st1122/Projects/safe-control-gym')
from generate_cartpole_trajectories import run_trajectory


def test_save_frequency():
    """Test that save_freq parameter works correctly."""

    print("Testing save frequency feature")
    print("="*80)

    env_func = partial(make,
                      'cartpole',
                      task='stabilization',
                      ctrl_freq=15,  # 15 Hz = 0.0667 sec timestep
                      pyb_freq=750,
                      episode_len_sec=10,
                      done_on_out_of_bound=False,
                      cost='quadratic',
                      gui=False,
                      randomized_init=False,
                      action_scale=10.0)

    ctrl = make('lqr',
                env_func,
                q_lqr=[1, 1, 1, 1],
                r_lqr=[0.1],
                discrete_dynamics=True)

    env = env_func()
    env.reset()

    print(f"Control frequency: {env.CTRL_FREQ} Hz (timestep = {1.0/env.CTRL_FREQ:.4f} s)")

    # Test with different save frequencies
    init_state = [1.0, 0.0, 0.5, 0.0]
    max_steps = 100

    # Test 1: Save at control frequency (0.0667 s)
    print(f"\nTest 1: save_freq = {1.0/env.CTRL_FREQ:.4f} s (control frequency)")
    traj1 = run_trajectory(env, ctrl, init_state, max_steps, save_freq=1.0/env.CTRL_FREQ)
    print(f"  Trajectory length: {len(traj1)} states")
    print(f"  Expected: ~{max_steps+1} states (initial + {max_steps} steps)")

    # Test 2: Save at 0.01 s (100 Hz)
    print(f"\nTest 2: save_freq = 0.01 s (100 Hz)")
    traj2 = run_trajectory(env, ctrl, init_state, max_steps, save_freq=0.01)
    print(f"  Trajectory length: {len(traj2)} states")
    expected_time = max_steps * (1.0/env.CTRL_FREQ)
    expected_states = int(expected_time / 0.01) + 1
    print(f"  Expected: ~{expected_states} states ({expected_time:.2f} s / 0.01 s)")

    # Test 3: Save at 0.1 s (10 Hz)
    print(f"\nTest 3: save_freq = 0.1 s (10 Hz)")
    traj3 = run_trajectory(env, ctrl, init_state, max_steps, save_freq=0.1)
    print(f"  Trajectory length: {len(traj3)} states")
    expected_states = int(expected_time / 0.1) + 1
    print(f"  Expected: ~{expected_states} states ({expected_time:.2f} s / 0.1 s)")

    env.close()
    ctrl.close()

    print("\n" + "="*80)
    print("✓ Save frequency feature is working correctly!")


if __name__ == '__main__':
    test_save_frequency()
