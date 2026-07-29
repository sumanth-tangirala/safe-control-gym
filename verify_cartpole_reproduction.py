#!/usr/bin/env python3
"""
Verify that we can reproduce cartpole trajectories from the existing dataset.
Samples a few trajectories, re-runs them, and compares results.
"""

import os
from functools import partial

import numpy as np

from safe_control_gym.utils.registration import make


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def run_trajectory(env, ctrl, init_state, max_steps=1000):
    """Run a single trajectory. init_state is [x, x_dot, theta, theta_dot] (internal order)."""
    import pybullet as p

    obs, info = env.reset()

    x, x_dot, theta, theta_dot = init_state

    p.resetJointState(env.CARTPOLE_ID, jointIndex=0,
                      targetValue=x, targetVelocity=x_dot,
                      physicsClientId=env.PYB_CLIENT)
    p.resetJointState(env.CARTPOLE_ID, jointIndex=1,
                      targetValue=theta, targetVelocity=theta_dot,
                      physicsClientId=env.PYB_CLIENT)

    env.state = np.array([x, x_dot, theta, theta_dot])
    obs = env._get_observation()

    # Output order: [x, theta, x_dot, theta_dot]
    trajectory = [[x, normalize_angle(theta), x_dot, theta_dot]]

    success = False
    timeout = False

    for step in range(max_steps):
        action = ctrl.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        x, x_dot, theta, theta_dot = obs[:4]
        trajectory.append([x, normalize_angle(theta), x_dot, theta_dot])

        if done:
            success = info.get('goal_reached', False)
            break
    else:
        timeout = True

    return trajectory, success, timeout


def main():
    # Parameters from dataset_description.json
    ctrl_freq = 100
    pyb_freq = 5000
    episode_len_sec = 10
    max_steps = 1000
    control_bound = 2000.0
    x_termination = 6.0
    x_dot_termination = 5.0
    theta_termination = float('inf')
    theta_dot_termination = 5.0

    env_func = partial(make,
                       'cartpole',
                       task='stabilization',
                       ctrl_freq=ctrl_freq,
                       pyb_freq=pyb_freq,
                       episode_len_sec=episode_len_sec,
                       done_on_out_of_bound=True,
                       cost='quadratic',
                       gui=False,
                       randomized_init=False,
                       obs_wrap_angle=True,
                       x_dot_limit=float('inf'),
                       theta_dot_limit=float('inf'),
                       action_scale=control_bound)

    ctrl = make('lqr',
                env_func,
                q_lqr=[1, 1, 1, 1],
                r_lqr=[0.1],
                discrete_dynamics=True)

    env = env_func()

    # Set termination thresholds
    env.x_threshold = x_termination
    env.x_dot_threshold = x_dot_termination
    env.theta_threshold_radians = theta_termination
    env.theta_dot_threshold = theta_dot_termination

    dataset_dir = '/common/users/shared/pracsys/genMoPlan/data_trajectories/cartpole_pybullet'
    traj_dir = os.path.join(dataset_dir, 'trajectories')

    # Sample trajectory indices to verify (mix of various positions in the dataset)
    np.random.seed(42)
    random_indices = sorted(np.random.choice(116242, size=80, replace=False).tolist())
    test_indices = random_indices

    print('Verifying trajectory reproduction...')
    print('=' * 80)

    match_count = 0
    pi_edge_count = 0
    mismatch_count = 0
    tested_count = 0

    for idx in test_indices:
        filepath = os.path.join(traj_dir, f'sequence_{idx}.txt')
        if not os.path.exists(filepath):
            print(f'  sequence_{idx}.txt not found, skipping')
            continue

        # Read existing trajectory
        existing_traj = []
        with open(filepath, 'r') as f:
            for line in f:
                vals = [float(v) for v in line.strip().split(',')]
                existing_traj.append(vals)
        existing_traj = np.array(existing_traj)

        # Extract initial state in internal order: [x, x_dot, theta, theta_dot]
        # File format is output order: [x, theta, x_dot, theta_dot]
        init_x = existing_traj[0, 0]
        init_theta = existing_traj[0, 1]
        init_x_dot = existing_traj[0, 2]
        init_theta_dot = existing_traj[0, 3]
        init_state = [init_x, init_x_dot, init_theta, init_theta_dot]

        # Check if this is a ±π edge case
        is_pi_edge = abs(abs(init_theta) - np.pi) < 1e-3

        # Run trajectory
        new_traj, success, timeout = run_trajectory(env, ctrl, init_state, max_steps)
        new_traj = np.array(new_traj)

        tested_count += 1

        # Compare
        len_match = len(existing_traj) == len(new_traj)
        if len_match:
            max_diff = np.abs(existing_traj - new_traj).max()
            match = max_diff < 1e-4
        else:
            max_diff = float('nan')
            match = False

        if match:
            match_count += 1
            status = 'MATCH'
        elif is_pi_edge:
            pi_edge_count += 1
            status = 'PI_EDGE (expected)'
        else:
            mismatch_count += 1
            status = 'MISMATCH'

        print(f'  sequence_{idx}: len={len(existing_traj)} vs {len(new_traj)}, '
              f'max_diff={max_diff:.8f}, init_theta={init_theta:.4f}, {status}')

        if not match and not is_pi_edge and len(existing_traj) > 0:
            print(f'    Existing first: {existing_traj[0]}')
            print(f'    New first:      {new_traj[0]}')
            if len(existing_traj) > 1 and len(new_traj) > 1:
                print(f'    Existing second: {existing_traj[1]}')
                print(f'    New second:      {new_traj[1]}')

    print('=' * 80)
    print(f'Results: {match_count} match, {pi_edge_count} pi-edge (expected), {mismatch_count} unexpected mismatch (out of {tested_count})')
    if mismatch_count == 0:
        print('SUCCESS: All non-edge-case trajectories match!')
    else:
        print('FAILURE: Unexpected mismatches found.')

    # Also verify ROA labels for the sampled trajectories
    print('\nVerifying ROA labels...')

    # Read roa_labels to build a lookup by init state
    roa_lookup = {}
    with open(os.path.join(dataset_dir, 'roa_labels.txt'), 'r') as f:
        for line in f:
            vals = line.strip().split(',')
            state = tuple(float(v) for v in vals[:4])
            label = int(float(vals[4]))
            roa_lookup[state] = label

    for idx in test_indices:
        filepath = os.path.join(traj_dir, f'sequence_{idx}.txt')
        if not os.path.exists(filepath):
            continue

        # Read initial state from trajectory
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        vals = [float(v) for v in first_line.split(',')]
        init_state_internal = [vals[0], vals[2], vals[1], vals[3]]  # x, x_dot, theta, theta_dot

        # Run trajectory
        _, success, timeout = run_trajectory(env, ctrl, init_state_internal, max_steps)
        new_label = 1 if success else 0

        # Look up existing label
        state_key = tuple(round(v, 6) for v in vals)
        existing_label = roa_lookup.get(state_key, None)

        if existing_label is not None:
            match = new_label == existing_label
            status = 'MATCH' if match else 'MISMATCH'
            print(f'  sequence_{idx}: existing={existing_label}, new={new_label}, {status}')
        else:
            print(f'  sequence_{idx}: init state not found in roa_labels (state={state_key})')

    env.close()
    ctrl.close()


if __name__ == '__main__':
    main()
