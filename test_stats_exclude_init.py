#!/usr/bin/env python3
"""Test that statistics exclude the initial state."""

import numpy as np
from functools import partial
from safe_control_gym.utils.registration import make
import sys
sys.path.insert(0, '/common/home/st1122/Projects/safe-control-gym')
from generate_cartpole_trajectories import run_trajectory


def test_stats_exclude_init():
    """Test statistics calculation excluding initial state."""

    print("Testing statistics calculation (excluding initial state)")
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
                      action_scale=10.0)

    ctrl = make('lqr',
                env_func,
                q_lqr=[1, 1, 1, 1],
                r_lqr=[0.1],
                discrete_dynamics=True)

    env = env_func()
    env.reset()

    # Run a trajectory with a known initial state
    init_state = [2.0, 0.0, 1.0, 0.0]  # x=2, x_dot=0, theta=1rad, theta_dot=0
    max_steps = 20

    print(f"Initial state: x={init_state[0]}, x_dot={init_state[1]}, theta={init_state[2]}, theta_dot={init_state[3]}")

    trajectory = run_trajectory(env, ctrl, init_state, max_steps, save_freq=0.01)
    traj_array = np.array(trajectory)

    print(f"\nTrajectory length: {len(trajectory)} states")
    print(f"\nFirst state (initial): {traj_array[0]}")
    print(f"Second state: {traj_array[1]}")

    # Calculate statistics INCLUDING initial state
    print(f"\n{'='*80}")
    print("Statistics INCLUDING initial state:")
    print(f"{'='*80}")
    for var_name, col_idx in [('x', 0), ('theta', 1), ('x_dot', 2), ('theta_dot', 3)]:
        print(f"{var_name:>10}: min={traj_array[:, col_idx].min():>10.6f}, max={traj_array[:, col_idx].max():>10.6f}")

    # Calculate statistics EXCLUDING initial state
    print(f"\n{'='*80}")
    print("Statistics EXCLUDING initial state:")
    print(f"{'='*80}")
    traj_array_no_init = traj_array[1:]
    stats = {}
    for var_name, col_idx in [('x', 0), ('theta', 1), ('x_dot', 2), ('theta_dot', 3)]:
        min_val = traj_array_no_init[:, col_idx].min()
        max_val = traj_array_no_init[:, col_idx].max()
        min_idx_no_init = traj_array_no_init[:, col_idx].argmin()
        max_idx_no_init = traj_array_no_init[:, col_idx].argmax()
        min_idx = min_idx_no_init + 1  # Adjust for skipped initial state
        max_idx = max_idx_no_init + 1

        stats[var_name] = {
            'min': min_val,
            'max': max_val,
            'min_idx': min_idx,
            'max_idx': max_idx,
            'prev_at_min': traj_array[min_idx - 1],
            'prev_at_max': traj_array[max_idx - 1]
        }

        print(f"{var_name:>10}: min={min_val:>10.6f} (at index {min_idx}), max={max_val:>10.6f} (at index {max_idx})")

    # Verify that initial state values are excluded
    print(f"\n{'='*80}")
    print("Verification:")
    print(f"{'='*80}")

    # Check if initial x=2.0 is in the stats
    initial_x = init_state[0]
    if abs(stats['x']['min'] - initial_x) > 1e-6 or abs(stats['x']['max'] - initial_x) > 1e-6:
        print(f"✓ Initial x={initial_x} is NOT the min or max (correctly excluded)")
    else:
        print(f"⚠ Initial x={initial_x} appears to be the min or max (may not be excluded)")

    # Check if initial theta=1.0 is in the stats
    initial_theta = init_state[2]
    if abs(stats['theta']['min'] - initial_theta) > 1e-6 or abs(stats['theta']['max'] - initial_theta) > 1e-6:
        print(f"✓ Initial theta={initial_theta} is NOT the min or max (correctly excluded)")
    else:
        print(f"⚠ Initial theta={initial_theta} appears to be the min or max (may not be excluded)")

    env.close()
    ctrl.close()


if __name__ == '__main__':
    test_stats_exclude_init()
