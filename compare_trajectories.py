#!/usr/bin/env python3
"""Compare trajectories with different control bounds."""

import numpy as np
from functools import partial
from safe_control_gym.utils.registration import make


def run_single_trajectory(control_bound, init_state, max_steps=20):
    """Run a single trajectory with given control bound."""

    env_func = partial(make,
                      'cartpole',
                      task='stabilization',
                      ctrl_freq=15,
                      pyb_freq=750,
                      episode_len_sec=10,
                      done_on_out_of_bound=False,  # Don't terminate early
                      cost='quadratic',
                      gui=False,
                      randomized_init=False,
                      action_scale=control_bound)

    ctrl = make('lqr',
                env_func,
                q_lqr=[1, 1, 1, 1],
                r_lqr=[0.1],
                discrete_dynamics=True)

    env = env_func()
    env.reset()

    # Set initial state
    import pybullet as p
    x, x_dot, theta, theta_dot = init_state
    p.resetJointState(env.CARTPOLE_ID, jointIndex=0, targetValue=x, targetVelocity=x_dot, physicsClientId=env.PYB_CLIENT)
    p.resetJointState(env.CARTPOLE_ID, jointIndex=1, targetValue=theta, targetVelocity=theta_dot, physicsClientId=env.PYB_CLIENT)
    env.state = np.array([x, x_dot, theta, theta_dot])

    trajectory = [env.state.copy()]
    actions_taken = []

    for step in range(max_steps):
        obs = env._get_observation()
        action = ctrl.select_action(obs, None)
        obs, reward, done, info = env.step(action)
        trajectory.append(env.state.copy())
        actions_taken.append(env.current_clipped_action.copy())

    env.close()
    ctrl.close()

    return np.array(trajectory), np.array(actions_taken)


def main():
    # Test with a challenging initial state
    init_state = [1.0, 0.0, 1.0, 0.0]  # x=1, x_dot=0, theta=1rad, theta_dot=0

    print(f"Initial state: x={init_state[0]}, x_dot={init_state[1]}, theta={init_state[2]}, theta_dot={init_state[3]}")
    print("="*80)

    # Run with different control bounds
    traj_5, actions_5 = run_single_trajectory(5.0, init_state)
    traj_10, actions_10 = run_single_trajectory(10.0, init_state)
    traj_20, actions_20 = run_single_trajectory(20.0, init_state)

    print(f"\nControl bound = 5.0 N")
    print(f"  Final state: {traj_5[-1]}")
    print(f"  Actions (first 5): {actions_5[:5].flatten()}")
    print(f"  Actions (last 5): {actions_5[-5:].flatten()}")

    print(f"\nControl bound = 10.0 N")
    print(f"  Final state: {traj_10[-1]}")
    print(f"  Actions (first 5): {actions_10[:5].flatten()}")
    print(f"  Actions (last 5): {actions_10[-5:].flatten()}")

    print(f"\nControl bound = 20.0 N")
    print(f"  Final state: {traj_20[-1]}")
    print(f"  Actions (first 5): {actions_20[:5].flatten()}")
    print(f"  Actions (last 5): {actions_20[-5:].flatten()}")

    print("\n" + "="*80)
    print("Trajectory Differences:")
    print(f"  ||traj_5 - traj_10||: {np.linalg.norm(traj_5 - traj_10):.6f}")
    print(f"  ||traj_5 - traj_20||: {np.linalg.norm(traj_5 - traj_20):.6f}")
    print(f"  ||traj_10 - traj_20||: {np.linalg.norm(traj_10 - traj_20):.6f}")

    print("\nAction Differences:")
    print(f"  ||actions_5 - actions_10||: {np.linalg.norm(actions_5 - actions_10):.6f}")
    print(f"  ||actions_5 - actions_20||: {np.linalg.norm(actions_5 - actions_20):.6f}")
    print(f"  ||actions_10 - actions_20||: {np.linalg.norm(actions_10 - actions_20):.6f}")

    if np.linalg.norm(traj_5 - traj_10) < 1e-6:
        print("\n⚠ WARNING: Trajectories are identical! Control bounds may not be working.")
    else:
        print("\n✓ Trajectories are different. Control bounds are working correctly.")


if __name__ == '__main__':
    main()
