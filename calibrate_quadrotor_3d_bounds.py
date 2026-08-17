#!/usr/bin/env python3
"""
Calibration script to find 3D quadrotor state bounds that yield ~50% success rate.
Uses random sampling and binary search to efficiently find appropriate bounds.
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pybullet as pb
from tqdm import tqdm

from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
from safe_control_gym.utils.registration import make


def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def get_available_cpus():
    """Get number of available CPUs respecting affinity."""
    try:
        affinity = os.sched_getaffinity(0)
        return len(affinity)
    except AttributeError:
        # Fallback to cpu_count if sched_getaffinity is not available
        return cpu_count()


def run_trajectory(env, ctrl, init_state, max_steps):
    """
    Run a single trajectory from the given initial state.

    Args:
        env: The quadrotor environment
        ctrl: The LQR controller
        init_state: Initial state [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r]
        max_steps: Maximum number of steps

    Returns:
        tuple: (success, timeout, steps)
    """
    # Reset environment first
    obs, info = env.reset()
    ctrl.reset()

    # Set initial state using PyBullet
    x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p_body, q_body, r_body = init_state

    # Set position and orientation
    pb.resetBasePositionAndOrientation(
        env.DRONE_ID,
        [x, y, z],  # Position in 3D
        pb.getQuaternionFromEuler([phi, theta, psi]),  # Orientation: [roll, pitch, yaw]
        physicsClientId=env.PYB_CLIENT)

    # Set velocities
    pb.resetBaseVelocity(
        env.DRONE_ID,
        [x_dot, y_dot, z_dot],  # Linear velocity
        [p_body, q_body, r_body],  # Angular velocity in body frame
        physicsClientId=env.PYB_CLIENT)

    # Update environment's internal state
    env._update_and_store_kinematic_information()
    obs = env._get_observation()

    done = False
    steps = 0

    while not done and steps < max_steps:
        action = ctrl.select_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        # Check for goal reached
        if info.get('goal_reached', False):
            return True, False, steps

    # Check if timeout (reached max_steps without termination)
    timeout = (steps >= max_steps and not done)
    return False, timeout, steps


def evaluate_single_trajectory(args):
    """
    Worker function to evaluate a single trajectory in parallel.

    Args:
        args: Tuple of (init_state, bounds, max_steps)

    Returns:
        tuple: (success, timeout)
    """
    init_state, bounds, max_steps = args

    # Create environment and controller for this worker
    ctrl_freq = 100  # Hz
    pyb_freq = ctrl_freq * 50  # 5000 Hz

    # For 3D quadrotor, need to set stabilization_goal with 3 elements [x, y, z]
    task_info = {
        'stabilization_goal': [0, 0, 1],  # Stabilize at x=0, y=0, z=1
        'stabilization_goal_tolerance': 0.05
    }

    env_func = partial(make,
                       'quadrotor',
                       quad_type=QuadType.THREE_D,
                       task='stabilization',
                       task_info=task_info,
                       ctrl_freq=ctrl_freq,
                       pyb_freq=pyb_freq,
                       episode_len_sec=10,
                       done_on_out_of_bound=True,
                       cost='quadratic',
                       gui=False,
                       randomized_init=False)

    ctrl = make('lqr',
                env_func,
                q_lqr=[1.0] * 12,
                r_lqr=[0.1] * 4,
                discrete_dynamics=True)

    env = env_func()

    # Configure environment's state_space bounds to match termination thresholds
    # Env state order: [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
    env.state_space.low[0] = -bounds['x']
    env.state_space.high[0] = bounds['x']
    env.state_space.low[2] = -bounds['y']
    env.state_space.high[2] = bounds['y']
    env.state_space.low[4] = bounds['z_min']
    env.state_space.high[4] = bounds['z_max']
    env.state_space.low[6] = -bounds['phi']
    env.state_space.high[6] = bounds['phi']
    env.state_space.low[7] = -bounds['theta']
    env.state_space.high[7] = bounds['theta']
    env.state_space.low[8] = -bounds['psi']
    env.state_space.high[8] = bounds['psi']
    # Velocities
    env.state_space.low[1] = -bounds['x_dot']
    env.state_space.high[1] = bounds['x_dot']
    env.state_space.low[3] = -bounds['y_dot']
    env.state_space.high[3] = bounds['y_dot']
    env.state_space.low[5] = -bounds['z_dot']
    env.state_space.high[5] = bounds['z_dot']
    env.state_space.low[9] = -bounds['p_body']
    env.state_space.high[9] = bounds['p_body']
    env.state_space.low[10] = -bounds['q_body']
    env.state_space.high[10] = bounds['q_body']
    env.state_space.low[11] = -bounds['r_body']
    env.state_space.high[11] = bounds['r_body']

    # Run trajectory
    success, timeout, _ = run_trajectory(env, ctrl, init_state, max_steps)

    env.close()

    return success, timeout


def evaluate_bounds(bounds, n_samples, seed=None, parallel=True, num_workers=None):
    """
    Evaluate success rate for given bounds using random sampling.

    Args:
        bounds: dict with keys for all state dimensions
        n_samples: Number of random trajectories to test
        seed: Random seed
        parallel: Use multiprocessing for parallel evaluation
        num_workers: Number of worker processes (default: number of available CPUs)

    Returns:
        dict: Statistics including success_rate, success_count, total_count
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate all initial states upfront
    initial_states = []
    for _ in range(n_samples):
        x = np.random.uniform(-bounds['x'], bounds['x'])
        y = np.random.uniform(-bounds['y'], bounds['y'])
        z = np.random.uniform(bounds['z_min'], bounds['z_max'])
        phi = np.random.uniform(-bounds['phi'], bounds['phi'])
        theta = np.random.uniform(-bounds['theta'], bounds['theta'])
        psi = np.random.uniform(-bounds['psi'], bounds['psi'])
        x_dot = np.random.uniform(-bounds['x_dot'], bounds['x_dot'])
        y_dot = np.random.uniform(-bounds['y_dot'], bounds['y_dot'])
        z_dot = np.random.uniform(-bounds['z_dot'], bounds['z_dot'])
        p = np.random.uniform(-bounds['p_body'], bounds['p_body'])
        q = np.random.uniform(-bounds['q_body'], bounds['q_body'])
        r = np.random.uniform(-bounds['r_body'], bounds['r_body'])
        init_state = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r]
        initial_states.append(init_state)

    max_steps = 500

    # Prepare arguments for workers
    worker_args = [(init_state, bounds, max_steps) for init_state in initial_states]

    if parallel:
        # Determine number of workers
        if num_workers is None:
            num_workers = get_available_cpus()

        # Run trajectories in parallel
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(evaluate_single_trajectory, worker_args),
                total=n_samples,
                desc='Evaluating',
                leave=False
            ))
    else:
        # Run sequentially
        results = []
        for args in tqdm(worker_args, desc='Evaluating', leave=False):
            results.append(evaluate_single_trajectory(args))

    # Count successes and timeouts
    success_count = sum(1 for success, _ in results if success)
    timeout_count = sum(1 for _, timeout in results if timeout)

    return {
        'success_count': success_count,
        'timeout_count': timeout_count,
        'total_count': n_samples,
        'success_rate': success_count / n_samples,
        'timeout_rate': timeout_count / n_samples
    }


def scale_bounds(base_bounds, scale_factor):
    """Scale position/velocity bounds by a factor. Angles stay fixed at pi."""
    return {
        'x': base_bounds['x'] * scale_factor,
        'y': base_bounds['y'] * scale_factor,
        'z_min': base_bounds['z_min'],  # Keep minimum altitude fixed
        'z_max': base_bounds['z_max'] * scale_factor,
        'phi': np.pi,  # Angles always span full range [-pi, pi]
        'theta': np.pi,
        'psi': np.pi,
        'x_dot': base_bounds['x_dot'] * scale_factor,
        'y_dot': base_bounds['y_dot'] * scale_factor,
        'z_dot': base_bounds['z_dot'] * scale_factor,
        'p_body': base_bounds['p_body'] * scale_factor,
        'q_body': base_bounds['q_body'] * scale_factor,
        'r_body': base_bounds['r_body'] * scale_factor
    }


def scale_bounds_by_group(base_bounds, pos_scale, angle_scale, vel_scale, ang_vel_scale):
    """Scale bounds by groups. Angles always stay fixed at pi (angle_scale is ignored)."""
    return {
        'x': base_bounds['x'] * pos_scale,
        'y': base_bounds['y'] * pos_scale,
        'z_min': base_bounds['z_min'],  # Keep minimum altitude fixed
        'z_max': base_bounds['z_max'] * pos_scale,
        'phi': np.pi,  # Angles always span full range [-pi, pi]
        'theta': np.pi,
        'psi': np.pi,
        'x_dot': base_bounds['x_dot'] * vel_scale,
        'y_dot': base_bounds['y_dot'] * vel_scale,
        'z_dot': base_bounds['z_dot'] * vel_scale,
        'p_body': base_bounds['p_body'] * ang_vel_scale,
        'q_body': base_bounds['q_body'] * ang_vel_scale,
        'r_body': base_bounds['r_body'] * ang_vel_scale
    }


def binary_search_scale(base_bounds, target_success_rate, n_samples, tolerance=0.05,
                        min_scale=0.1, max_scale=3.0, max_iterations=10, seed=None):
    """
    Binary search for the scale factor that gives target success rate.
    """
    print(f'\nBinary search for {target_success_rate*100:.0f}% success rate...')
    print(f'Scale range: [{min_scale}, {max_scale}]')
    print('-' * 70)

    low, high = min_scale, max_scale
    best_scale = None
    best_bounds = None
    best_stats = None
    best_diff = float('inf')

    for i in range(max_iterations):
        mid = (low + high) / 2
        bounds = scale_bounds(base_bounds, mid)

        stats = evaluate_bounds(bounds, n_samples, seed=seed)
        success_rate = stats['success_rate']
        diff = abs(success_rate - target_success_rate)

        print(f'Iter {i+1}: scale={mid:.3f}, success_rate={success_rate*100:.1f}%')

        if diff < best_diff:
            best_diff = diff
            best_scale = mid
            best_bounds = bounds
            best_stats = stats

        if diff <= tolerance:
            print('  -> Within tolerance! Found good scale.')
            break

        # Higher scale = harder = lower success rate
        if success_rate > target_success_rate:
            low = mid  # Need harder (larger bounds)
        else:
            high = mid  # Need easier (smaller bounds)

    return best_scale, best_bounds, best_stats


def grid_search_groups(base_bounds, n_samples, seed=None):
    """
    Grid search over grouped scale factors to understand which group has most impact.
    """
    print('\n' + '=' * 70)
    print('GROUP SENSITIVITY ANALYSIS')
    print('=' * 70)
    print('Testing impact of scaling each group of variables:')
    print('  - positions (x, y, z_max)')
    print('  - angles (phi, theta, psi)')
    print('  - linear velocities (x_dot, y_dot, z_dot)')
    print('  - angular velocities (p, q, r)')

    groups = ['positions', 'angles', 'linear_vel', 'angular_vel']
    scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]

    results = {}

    for group in groups:
        print(f'\nScaling {group}:')
        results[group] = []

        for sf in scale_factors:
            if group == 'positions':
                bounds = scale_bounds_by_group(base_bounds, sf, 1.0, 1.0, 1.0)
            elif group == 'angles':
                bounds = scale_bounds_by_group(base_bounds, 1.0, sf, 1.0, 1.0)
            elif group == 'linear_vel':
                bounds = scale_bounds_by_group(base_bounds, 1.0, 1.0, sf, 1.0)
            else:  # angular_vel
                bounds = scale_bounds_by_group(base_bounds, 1.0, 1.0, 1.0, sf)

            stats = evaluate_bounds(bounds, n_samples, seed=seed)
            results[group].append((sf, stats['success_rate']))
            print(f"  scale={sf:.2f} -> {stats['success_rate']*100:.1f}%")

    return results


def maximize_success_rate(base_bounds, n_samples, seed=None):
    """
    Find the scale factor that maximizes success rate.
    Tests a comprehensive range and reports detailed statistics.
    """
    print('\n' + '=' * 70)
    print('MAXIMIZE SUCCESS RATE')
    print('=' * 70)
    print('Testing comprehensive range of scale factors to find maximum achievable success rate.')
    print(f'Using {n_samples} samples per configuration for reliable statistics.')

    # Test more granular range
    scale_factors = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                     1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]

    results = []
    best_success_rate = 0
    best_scale = None
    best_bounds = None
    best_stats = None

    print(f"\n{'Scale':<10} {'Success Rate':<15} {'Timeout Rate':<15} {'Status':<20}")
    print('-' * 70)

    for sf in tqdm(scale_factors, desc='Testing scale factors'):
        bounds = scale_bounds(base_bounds, sf)
        stats = evaluate_bounds(bounds, n_samples, seed=seed)
        results.append((sf, stats))

        status = ''
        if stats['success_rate'] > best_success_rate:
            best_success_rate = stats['success_rate']
            best_scale = sf
            best_bounds = bounds
            best_stats = stats
            status = '← NEW BEST'

        print(f"{sf:<10.2f} {stats['success_rate']*100:<15.1f}% {stats['timeout_rate']*100:<15.1f}% {status:<20}")

    print('\n' + '=' * 70)
    print('BEST CONFIGURATION FOUND')
    print('=' * 70)
    print(f'Scale factor: {best_scale:.3f}')
    print(f"Success rate: {best_stats['success_rate']*100:.1f}%")
    print(f"Timeout rate: {best_stats['timeout_rate']*100:.1f}%")
    print('\nBounds:')
    for k, v in best_bounds.items():
        print(f'  {k}: {v:.4f}')

    # Also find configurations near common target rates
    print('\n' + '=' * 70)
    print('CONFIGURATIONS NEAR TARGET RATES')
    print('=' * 70)

    target_rates = [0.40, 0.50, 0.60, 0.70]
    for target in target_rates:
        closest = min(results, key=lambda x: abs(x[1]['success_rate'] - target))
        scale, stats = closest
        if abs(stats['success_rate'] - target) < 0.15:  # Within 15%
            print(f"\nTarget {target*100:.0f}%: scale={scale:.2f}, actual={stats['success_rate']*100:.1f}%")

    return best_scale, best_bounds, best_stats, results


def main():
    parser = argparse.ArgumentParser(description='Calibrate 3D quadrotor bounds for target success rate')

    # Base bounds (starting point for search)
    parser.add_argument('--base_x', type=float, default=0.5, help='Base x bound (m)')
    parser.add_argument('--base_y', type=float, default=0.5, help='Base y bound (m)')
    parser.add_argument('--base_z_min', type=float, default=0.1, help='Base z minimum (m)')
    parser.add_argument('--base_z_max', type=float, default=1.0, help='Base z maximum (m)')
    parser.add_argument('--base_phi', type=float, default=np.pi, help='Base phi bound (rad) - angles span full range')
    parser.add_argument('--base_theta', type=float, default=np.pi, help='Base theta bound (rad) - angles span full range')
    parser.add_argument('--base_psi', type=float, default=np.pi, help='Base psi bound (rad) - angles span full range')
    parser.add_argument('--base_x_dot', type=float, default=0.5, help='Base x_dot bound (m/s)')
    parser.add_argument('--base_y_dot', type=float, default=0.5, help='Base y_dot bound (m/s)')
    parser.add_argument('--base_z_dot', type=float, default=0.5, help='Base z_dot bound (m/s)')
    parser.add_argument('--base_p_body', type=float, default=0.5, help='Base p_body bound (rad/s)')
    parser.add_argument('--base_q_body', type=float, default=0.5, help='Base q_body bound (rad/s)')
    parser.add_argument('--base_r_body', type=float, default=0.5, help='Base r_body bound (rad/s)')

    # Search parameters
    parser.add_argument('--target_success_rate', type=float, default=0.5, help='Target success rate')
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Samples per evaluation (default: 500, use more for final calibration)')
    parser.add_argument('--tolerance', type=float, default=0.05, help='Acceptable deviation from target')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Mode
    parser.add_argument('--mode', type=str, default='binary',
                        choices=['binary', 'grid', 'groups', 'all', 'maximize'],
                        help='Search mode: binary, grid, groups, all, or maximize')

    args = parser.parse_args()

    base_bounds = {
        'x': args.base_x,
        'y': args.base_y,
        'z_min': args.base_z_min,
        'z_max': args.base_z_max,
        'phi': args.base_phi,
        'theta': args.base_theta,
        'psi': args.base_psi,
        'x_dot': args.base_x_dot,
        'y_dot': args.base_y_dot,
        'z_dot': args.base_z_dot,
        'p_body': args.base_p_body,
        'q_body': args.base_q_body,
        'r_body': args.base_r_body
    }

    print('=' * 70)
    print('3D QUADROTOR BOUNDS CALIBRATION')
    print('=' * 70)
    print('\nBase bounds:')
    for k, v in base_bounds.items():
        print(f'  {k}: {v}')
    print(f'\nTarget success rate: {args.target_success_rate*100:.0f}%')
    print(f'Samples per evaluation: {args.n_samples}')

    # First, evaluate base bounds
    print('\n' + '-' * 70)
    print('Evaluating base bounds...')
    base_stats = evaluate_bounds(base_bounds, args.n_samples, seed=args.seed)
    print(f"Base bounds success rate: {base_stats['success_rate']*100:.1f}%")
    print(f"  (success: {base_stats['success_count']}, timeout: {base_stats['timeout_count']}, total: {base_stats['total_count']})")

    if args.mode in ['maximize', 'all']:
        # Find maximum achievable success rate
        n_samples_maximize = max(args.n_samples, 500)

        best_scale, best_bounds, best_stats, all_results = maximize_success_rate(
            base_bounds, n_samples_maximize, seed=args.seed
        )

        print('\n' + '-' * 70)
        print('RECOMMENDED BOUNDS FOR MAXIMUM SUCCESS RATE')
        print('-' * 70)
        print(f"  --x_bound {best_bounds['x']:.4f} \\")
        print(f"  --y_bound {best_bounds['y']:.4f} \\")
        print(f"  --z_min {best_bounds['z_min']:.4f} \\")
        print(f"  --z_max {best_bounds['z_max']:.4f} \\")
        print(f"  --phi_bound {best_bounds['phi']:.4f} \\")
        print(f"  --theta_bound {best_bounds['theta']:.4f} \\")
        print(f"  --psi_bound {best_bounds['psi']:.4f} \\")
        print(f"  --x_dot_bound {best_bounds['x_dot']:.4f} \\")
        print(f"  --y_dot_bound {best_bounds['y_dot']:.4f} \\")
        print(f"  --z_dot_bound {best_bounds['z_dot']:.4f} \\")
        print(f"  --p_body_bound {best_bounds['p_body']:.4f} \\")
        print(f"  --q_body_bound {best_bounds['q_body']:.4f} \\")
        print(f"  --r_body_bound {best_bounds['r_body']:.4f}")

    if args.mode in ['binary', 'all']:
        # Binary search for optimal scale
        best_scale, best_bounds, best_stats = binary_search_scale(
            base_bounds, args.target_success_rate, args.n_samples,
            args.tolerance, seed=args.seed
        )

        print('\n' + '=' * 70)
        print('RECOMMENDED BOUNDS (from binary search)')
        print('=' * 70)
        print(f'Scale factor: {best_scale:.3f}')
        print(f"Success rate: {best_stats['success_rate']*100:.1f}%")
        print('\nBounds:')
        for k, v in best_bounds.items():
            print(f'  {k}: {v:.4f}')

        print('\n' + '-' * 70)
        print('Command line arguments for generate_quadrotor_3d_trajectories.py:')
        print('-' * 70)
        print(f"  --x_bound {best_bounds['x']:.4f} \\")
        print(f"  --y_bound {best_bounds['y']:.4f} \\")
        print(f"  --z_min {best_bounds['z_min']:.4f} \\")
        print(f"  --z_max {best_bounds['z_max']:.4f} \\")
        print(f"  --phi_bound {best_bounds['phi']:.4f} \\")
        print(f"  --theta_bound {best_bounds['theta']:.4f} \\")
        print(f"  --psi_bound {best_bounds['psi']:.4f} \\")
        print(f"  --x_dot_bound {best_bounds['x_dot']:.4f} \\")
        print(f"  --y_dot_bound {best_bounds['y_dot']:.4f} \\")
        print(f"  --z_dot_bound {best_bounds['z_dot']:.4f} \\")
        print(f"  --p_body_bound {best_bounds['p_body']:.4f} \\")
        print(f"  --q_body_bound {best_bounds['q_body']:.4f} \\")
        print(f"  --r_body_bound {best_bounds['r_body']:.4f}")

    if args.mode in ['grid', 'all']:
        # Grid search over scale factors
        print('\n' + '=' * 70)
        print('GRID SEARCH OVER SCALE FACTORS')
        print('=' * 70)

        scale_factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

        print(f"\n{'Scale':<10} {'Success Rate':<15} {'Timeout Rate':<15}")
        print('-' * 40)

        for sf in scale_factors:
            bounds = scale_bounds(base_bounds, sf)
            stats = evaluate_bounds(bounds, args.n_samples, seed=args.seed)
            print(f"{sf:<10.2f} {stats['success_rate']*100:<15.1f}% {stats['timeout_rate']*100:<15.1f}%")

    if args.mode in ['groups', 'all']:
        grid_search_groups(base_bounds, args.n_samples, seed=args.seed)


if __name__ == '__main__':
    main()
