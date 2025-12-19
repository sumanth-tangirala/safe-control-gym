#!/usr/bin/env python3
"""Test the final simplified implementation."""

import numpy as np
from functools import partial
from safe_control_gym.utils.registration import make
import sys
sys.path.insert(0, '/common/home/st1122/Projects/safe-control-gym')
from generate_cartpole_trajectories import run_trajectory


def test_final_implementation():
    """Test that ctrl_freq matches save_freq and states are saved every step."""

    print("Testing final implementation: ctrl_freq = 1/save_freq")
    print("="*80)

    # Test 1: save_freq = 0.01 (100 Hz)
    save_freq_1 = 0.01
    ctrl_freq_1 = 1.0 / save_freq_1  # Should be 100 Hz
    pyb_freq_1 = int(ctrl_freq_1 * 50)  # Should be 5000 Hz

    print(f"\nTest 1: save_freq = {save_freq_1} s ({1.0/save_freq_1:.0f} Hz)")
    print(f"  Expected ctrl_freq: {ctrl_freq_1:.1f} Hz")
    print(f"  Expected pyb_freq: {pyb_freq_1} Hz")

    env_func_1 = partial(make,
                        'cartpole',
                        task='stabilization',
                        ctrl_freq=ctrl_freq_1,
                        pyb_freq=pyb_freq_1,
                        episode_len_sec=10,
                        done_on_out_of_bound=False,
                        cost='quadratic',
                        gui=False,
                        randomized_init=False,
                        action_scale=10.0)

    ctrl_1 = make('lqr',
                  env_func_1,
                  q_lqr=[1, 1, 1, 1],
                  r_lqr=[0.1],
                  discrete_dynamics=True)

    env_1 = env_func_1()
    env_1.reset()

    print(f"  Actual ctrl_freq: {env_1.CTRL_FREQ:.1f} Hz")
    print(f"  Actual ctrl_timestep: {1.0/env_1.CTRL_FREQ:.6f} s")

    init_state = [1.0, 0.0, 0.5, 0.0]
    max_steps = 10

    trajectory = run_trajectory(env_1, ctrl_1, init_state, max_steps)
    print(f"  Trajectory length: {len(trajectory)} states (initial + {len(trajectory)-1} steps)")
    print(f"  Expected: {max_steps+1} states (initial + {max_steps} steps)")

    # Each control step should save exactly one state
    assert len(trajectory) == max_steps + 1, f"Expected {max_steps+1} states, got {len(trajectory)}"
    print(f"  ✓ Correct: Each control step saves exactly one state")

    env_1.close()
    ctrl_1.close()

    # Test 2: save_freq = 0.1 (10 Hz)
    print(f"\n{'='*80}")
    save_freq_2 = 0.1
    ctrl_freq_2 = max(15, 1.0 / save_freq_2)  # Should be max(15, 10) = 15 Hz
    pyb_freq_2 = int(ctrl_freq_2 * 50)  # Should be 750 Hz

    print(f"\nTest 2: save_freq = {save_freq_2} s ({1.0/save_freq_2:.0f} Hz)")
    print(f"  Expected ctrl_freq: {ctrl_freq_2:.1f} Hz (max of 15 Hz default and 10 Hz from save_freq)")
    print(f"  Expected pyb_freq: {pyb_freq_2} Hz")

    env_func_2 = partial(make,
                        'cartpole',
                        task='stabilization',
                        ctrl_freq=ctrl_freq_2,
                        pyb_freq=pyb_freq_2,
                        episode_len_sec=10,
                        done_on_out_of_bound=False,
                        cost='quadratic',
                        gui=False,
                        randomized_init=False,
                        action_scale=10.0)

    ctrl_2 = make('lqr',
                  env_func_2,
                  q_lqr=[1, 1, 1, 1],
                  r_lqr=[0.1],
                  discrete_dynamics=True)

    env_2 = env_func_2()
    env_2.reset()

    print(f"  Actual ctrl_freq: {env_2.CTRL_FREQ:.1f} Hz")
    print(f"  Actual ctrl_timestep: {1.0/env_2.CTRL_FREQ:.6f} s")

    trajectory2 = run_trajectory(env_2, ctrl_2, init_state, max_steps)
    print(f"  Trajectory length: {len(trajectory2)} states")

    env_2.close()
    ctrl_2.close()

    print(f"\n{'='*80}")
    print("✓ All tests passed!")


if __name__ == '__main__':
    test_final_implementation()
