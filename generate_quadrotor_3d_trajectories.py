#!/usr/bin/env python3
"""
Script to generate 3D quadrotor trajectory dataset with LQR controller.
Discretizes the initial state space and saves trajectories.

State: [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body]
"""

import argparse
import os
import numpy as np
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pybullet as p  # Import at module level to avoid multiprocessing issues

from safe_control_gym.utils.registration import make
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType

# Pre-import scipy to avoid multiprocessing import errors
try:
    import scipy.linalg
except ImportError:
    pass


def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi] range.

    Args:
        angle (float): Angle in radians

    Returns:
        float: Normalized angle in [-pi, pi]
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def get_available_cpus():
    """
    Get the number of CPUs available to this process, respecting taskset/affinity.

    Returns:
        int: Number of available CPUs
    """
    try:
        # Try to get CPU affinity (respects taskset, cgroups, etc.)
        import os
        affinity = os.sched_getaffinity(0)
        return len(affinity)
    except (AttributeError, OSError):
        # Fallback to cpu_count if sched_getaffinity is not available
        return cpu_count()


def generate_discretized_initial_states(bounds, resolution=0.05, termination_thresholds=None):
    """
    Generate discretized initial states within given bounds.

    Args:
        bounds (dict): Dictionary with keys for all 12 state dimensions:
                      'x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                      'phi', 'theta', 'psi', 'p_body', 'q_body', 'r_body'
                      Values are symmetric bound magnitudes (states go from -value to +value)
        resolution (float): Discretization resolution
        termination_thresholds (dict, optional): Dictionary with same keys as bounds,
                                                 specifying termination thresholds.
                                                 States at or beyond these thresholds
                                                 will be excluded from initial states.

    Returns:
        list: List of initial states as [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
    """
    states = []

    # Create discretized ranges for each dimension (symmetric around zero)
    x_vals = np.arange(-bounds['x'], bounds['x'] + resolution, resolution)
    x_dot_vals = np.arange(-bounds['x_dot'], bounds['x_dot'] + resolution, resolution)
    y_vals = np.arange(-bounds['y'], bounds['y'] + resolution, resolution)
    y_dot_vals = np.arange(-bounds['y_dot'], bounds['y_dot'] + resolution, resolution)
    z_vals = np.arange(-bounds['z'], bounds['z'] + resolution, resolution)
    z_dot_vals = np.arange(-bounds['z_dot'], bounds['z_dot'] + resolution, resolution)
    phi_vals = np.arange(-bounds['phi'], bounds['phi'] + resolution, resolution)
    theta_vals = np.arange(-bounds['theta'], bounds['theta'] + resolution, resolution)
    psi_vals = np.arange(-bounds['psi'], bounds['psi'] + resolution, resolution)
    p_vals = np.arange(-bounds['p_body'], bounds['p_body'] + resolution, resolution)
    q_vals = np.arange(-bounds['q_body'], bounds['q_body'] + resolution, resolution)
    r_vals = np.arange(-bounds['r_body'], bounds['r_body'] + resolution, resolution)

    # Generate all combinations
    for x in x_vals:
        for x_dot in x_dot_vals:
            for y in y_vals:
                for y_dot in y_dot_vals:
                    for z in z_vals:
                        for z_dot in z_dot_vals:
                            for phi in phi_vals:
                                for theta in theta_vals:
                                    for psi in psi_vals:
                                        for p in p_vals:
                                            for q in q_vals:
                                                for r in r_vals:
                                                    # Check if state violates termination thresholds
                                                    if termination_thresholds is not None:
                                                        # Skip states that would immediately trigger termination
                                                        if (abs(x) >= termination_thresholds['x'] or
                                                            abs(x_dot) >= termination_thresholds['x_dot'] or
                                                            abs(y) >= termination_thresholds['y'] or
                                                            abs(y_dot) >= termination_thresholds['y_dot'] or
                                                            abs(z) >= termination_thresholds['z'] or
                                                            abs(z_dot) >= termination_thresholds['z_dot'] or
                                                            abs(phi) >= termination_thresholds['phi'] or
                                                            abs(theta) >= termination_thresholds['theta'] or
                                                            abs(psi) >= termination_thresholds['psi'] or
                                                            abs(p) >= termination_thresholds['p_body'] or
                                                            abs(q) >= termination_thresholds['q_body'] or
                                                            abs(r) >= termination_thresholds['r_body']):
                                                            continue

                                                    states.append([x, x_dot, y, y_dot, z, z_dot,
                                                                 phi, theta, psi, p, q, r])

    return states


def generate_random_initial_states(bounds, num_samples, termination_thresholds=None, seed=None):
    """
    Generate random initial states within given bounds.

    Args:
        bounds (dict): Dictionary with keys for all 12 state dimensions:
                      'x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                      'phi', 'theta', 'psi', 'p_body', 'q_body', 'r_body'
                      Values are symmetric bound magnitudes (states go from -value to +value)
        num_samples (int): Number of random initial states to generate
        termination_thresholds (dict, optional): Dictionary with same keys as bounds,
                                                 specifying termination thresholds.
                                                 States at or beyond these thresholds
                                                 will be excluded from initial states.
        seed (int, optional): Random seed for reproducibility

    Returns:
        list: List of initial states as [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
    """
    if seed is not None:
        np.random.seed(seed)

    states = []
    attempts = 0
    max_attempts = num_samples * 100  # Prevent infinite loop

    while len(states) < num_samples and attempts < max_attempts:
        attempts += 1

        # Randomly sample each dimension from uniform distribution
        x = np.random.uniform(-bounds['x'], bounds['x'])
        x_dot = np.random.uniform(-bounds['x_dot'], bounds['x_dot'])
        y = np.random.uniform(-bounds['y'], bounds['y'])
        y_dot = np.random.uniform(-bounds['y_dot'], bounds['y_dot'])
        z = np.random.uniform(-bounds['z'], bounds['z'])
        z_dot = np.random.uniform(-bounds['z_dot'], bounds['z_dot'])
        phi = np.random.uniform(-bounds['phi'], bounds['phi'])
        theta = np.random.uniform(-bounds['theta'], bounds['theta'])
        psi = np.random.uniform(-bounds['psi'], bounds['psi'])
        p = np.random.uniform(-bounds['p_body'], bounds['p_body'])
        q = np.random.uniform(-bounds['q_body'], bounds['q_body'])
        r = np.random.uniform(-bounds['r_body'], bounds['r_body'])

        # Check if state violates termination thresholds
        if termination_thresholds is not None:
            if (abs(x) >= termination_thresholds['x'] or
                abs(x_dot) >= termination_thresholds['x_dot'] or
                abs(y) >= termination_thresholds['y'] or
                abs(y_dot) >= termination_thresholds['y_dot'] or
                abs(z) >= termination_thresholds['z'] or
                abs(z_dot) >= termination_thresholds['z_dot'] or
                abs(phi) >= termination_thresholds['phi'] or
                abs(theta) >= termination_thresholds['theta'] or
                abs(psi) >= termination_thresholds['psi'] or
                abs(p) >= termination_thresholds['p_body'] or
                abs(q) >= termination_thresholds['q_body'] or
                abs(r) >= termination_thresholds['r_body']):
                continue

        states.append([x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r])

    if len(states) < num_samples:
        print(f"Warning: Could only generate {len(states)} valid states out of {num_samples} requested")

    return states


def run_trajectory(env, ctrl, init_state, max_steps=1000):
    """
    Run a single trajectory with given initial state.

    Args:
        env: Environment instance
        ctrl: Controller instance
        init_state: Initial state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
        max_steps: Maximum number of steps

    Returns:
        tuple: (trajectory, success, timeout)
            - trajectory: List of states
            - success: Boolean indicating if goal was reached (True) or terminated due to bounds (False)
            - timeout: Boolean indicating if trajectory reached max_steps without terminating
    """
    # Reset environment first
    obs, info = env.reset()

    # Now properly set the initial state in PyBullet simulation
    x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body = init_state

    # Set position and orientation
    p.resetBasePositionAndOrientation(
        env.DRONE_ID,
        [x, y, z],  # Position in 3D
        p.getQuaternionFromEuler([phi, theta, psi]),  # Orientation: [roll, pitch, yaw]
        physicsClientId=env.PYB_CLIENT)

    # Set velocities
    p.resetBaseVelocity(
        env.DRONE_ID,
        [x_dot, y_dot, z_dot],  # Linear velocity
        [p_body, q_body, r_body],  # Angular velocity in body frame
        physicsClientId=env.PYB_CLIENT)

    # Update environment's internal state to match
    env._update_and_store_kinematic_information()
    obs = env._get_observation()

    # Store initial state with normalized angles
    trajectory = [[x, x_dot, y, y_dot, z, z_dot,
                  normalize_angle(phi), normalize_angle(theta), normalize_angle(psi),
                  p_body, q_body, r_body]]

    success = False
    timeout = False

    for step in range(max_steps):
        # Get action from LQR controller
        action = ctrl.select_action(obs, info)

        # Take step in environment (old Gym API returns 4 values)
        obs, reward, done, info = env.step(action)

        # Extract state (obs order: x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body)
        x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body = obs[:12]

        # Store with normalized angles
        current_state = [x, x_dot, y, y_dot, z, z_dot,
                        normalize_angle(phi), normalize_angle(theta), normalize_angle(psi),
                        p_body, q_body, r_body]
        trajectory.append(current_state)

        # Check if episode naturally ends (goal reached or out of bounds)
        if done:
            # Check if goal was reached (success) or out of bounds (failure)
            success = info.get('goal_reached', False)
            break
    else:
        # Loop completed without breaking - trajectory timed out
        timeout = True

    return trajectory, success, timeout


def save_trajectory(trajectory, filepath):
    """
    Save trajectory to file in required format.

    Args:
        trajectory: List of states
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        for state in trajectory:
            # Format each state as comma-separated values (no spaces)
            line = ','.join([f'{val:.6f}' for val in state])
            f.write(line + '\n')


def process_single_trajectory(args_tuple):
    """
    Worker function to process a single trajectory in parallel.

    Args:
        args_tuple: Tuple containing (idx, init_state, env_config, output_dir, skip_save)
            - idx: trajectory index
            - init_state: initial state for the trajectory
            - env_config: dict with environment configuration
            - output_dir: directory to save trajectories
            - skip_save: boolean flag to skip saving files

    Returns:
        dict: Statistics for this trajectory including ROA label
    """
    idx, init_state, env_config, output_dir, skip_save = args_tuple

    # Create environment and controller for this worker
    env_func = partial(make,
                      'quadrotor',
                      quad_type=QuadType.THREE_D,
                      task=env_config['task'],
                      ctrl_freq=env_config['ctrl_freq'],
                      pyb_freq=env_config['pyb_freq'],
                      episode_len_sec=env_config['episode_len_sec'],
                      done_on_out_of_bound=env_config['done_on_out_of_bound'],
                      cost=env_config['cost'],
                      gui=False,
                      randomized_init=False)

    ctrl = make('lqr',
                env_func,
                q_lqr=env_config['q_lqr'],
                r_lqr=env_config['r_lqr'],
                discrete_dynamics=True)

    env = env_func()

    # Initialize statistics for this trajectory
    state_vars = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                  'phi', 'theta', 'psi', 'p_body', 'q_body', 'r_body']
    traj_stats = {
        var: {'min': float('inf'), 'max': float('-inf'),
              'prev_at_min': None, 'prev_at_max': None}
        for var in state_vars
    }
    traj_stats.update({
        'success_count': 0,
        'total_count': 0,
        'timeout_count': 0,
        'max_traj_length': 0,
        'roa_label': None,
        'init_state': None
    })

    # Run trajectory
    trajectory, success, timeout = run_trajectory(env, ctrl, init_state, env_config['max_steps'])

    # Update success and timeout tracking
    traj_stats['total_count'] = 1
    if success:
        traj_stats['success_count'] = 1
    if timeout:
        traj_stats['timeout_count'] = 1

    # Store ROA label: 1 for success, 0 for failure (timeout or out of bounds)
    # Store initial state with normalized angles
    x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r = init_state
    traj_stats['init_state'] = [x, x_dot, y, y_dot, z, z_dot,
                                normalize_angle(phi), normalize_angle(theta), normalize_angle(psi),
                                p, q, r]
    traj_stats['roa_label'] = 1 if success else 0

    # Update trajectory length tracking
    traj_stats['max_traj_length'] = len(trajectory)

    # Update statistics with previous state tracking
    traj_array = np.array(trajectory)

    # Include all states (including initial state) in statistics
    if len(traj_array) > 0:
        # For each state variable, track min/max and the previous state
        state_var_indices = [
            ('x', 0), ('x_dot', 1), ('y', 2), ('y_dot', 3), ('z', 4), ('z_dot', 5),
            ('phi', 6), ('theta', 7), ('psi', 8), ('p_body', 9), ('q_body', 10), ('r_body', 11)
        ]
        for var_name, col_idx in state_var_indices:
            # Find min value and its index
            traj_min = traj_array[:, col_idx].min()
            traj_stats[var_name]['min'] = traj_min
            min_idx = traj_array[:, col_idx].argmin()
            # Store previous state (None if this is the initial state)
            traj_stats[var_name]['prev_at_min'] = traj_array[min_idx - 1].tolist() if min_idx > 0 else None

            # Find max value and its index
            traj_max = traj_array[:, col_idx].max()
            traj_stats[var_name]['max'] = traj_max
            max_idx = traj_array[:, col_idx].argmax()
            # Store previous state (None if this is the initial state)
            traj_stats[var_name]['prev_at_max'] = traj_array[max_idx - 1].tolist() if max_idx > 0 else None

    # Save trajectory (only if skip_save is False)
    if not skip_save:
        filepath = os.path.join(output_dir, f'sequence_{idx}.txt')
        save_trajectory(trajectory, filepath)

    # Clean up
    env.close()
    ctrl.close()

    return traj_stats


def main():
    parser = argparse.ArgumentParser(description='Generate 3D quadrotor trajectory dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save trajectory files')
    parser.add_argument('--resolution', type=float, default=0.1,
                        help='Discretization resolution (default: 0.1)')

    # Position bounds
    parser.add_argument('--x_bound', type=float, default=1.0,
                        help='Symmetric bound for x position (default: 1.0)')
    parser.add_argument('--y_bound', type=float, default=1.0,
                        help='Symmetric bound for y position (default: 1.0)')
    parser.add_argument('--z_bound', type=float, default=1.0,
                        help='Symmetric bound for z position (default: 1.0)')

    # Velocity bounds
    parser.add_argument('--x_dot_bound', type=float, default=1.0,
                        help='Symmetric bound for x velocity (default: 1.0)')
    parser.add_argument('--y_dot_bound', type=float, default=1.0,
                        help='Symmetric bound for y velocity (default: 1.0)')
    parser.add_argument('--z_dot_bound', type=float, default=1.0,
                        help='Symmetric bound for z velocity (default: 1.0)')

    # Orientation bounds (Euler angles)
    parser.add_argument('--phi_bound', type=float, default=0.5,
                        help='Symmetric bound for roll angle (default: 0.5 rad)')
    parser.add_argument('--theta_bound', type=float, default=0.5,
                        help='Symmetric bound for pitch angle (default: 0.5 rad)')
    parser.add_argument('--psi_bound', type=float, default=0.5,
                        help='Symmetric bound for yaw angle (default: 0.5 rad)')

    # Angular velocity bounds (body frame)
    parser.add_argument('--p_body_bound', type=float, default=1.0,
                        help='Symmetric bound for roll rate (default: 1.0 rad/s)')
    parser.add_argument('--q_body_bound', type=float, default=1.0,
                        help='Symmetric bound for pitch rate (default: 1.0 rad/s)')
    parser.add_argument('--r_body_bound', type=float, default=1.0,
                        help='Symmetric bound for yaw rate (default: 1.0 rad/s)')

    # Termination thresholds
    parser.add_argument('--x_termination', type=float, default=None,
                        help='Termination threshold for x (default: copies x_bound)')
    parser.add_argument('--y_termination', type=float, default=None,
                        help='Termination threshold for y (default: copies y_bound)')
    parser.add_argument('--z_termination', type=float, default=None,
                        help='Termination threshold for z (default: copies z_bound)')
    parser.add_argument('--x_dot_termination', type=float, default=float('inf'),
                        help='Termination threshold for x velocity (default: inf)')
    parser.add_argument('--y_dot_termination', type=float, default=float('inf'),
                        help='Termination threshold for y velocity (default: inf)')
    parser.add_argument('--z_dot_termination', type=float, default=float('inf'),
                        help='Termination threshold for z velocity (default: inf)')
    parser.add_argument('--phi_termination', type=float, default=float('inf'),
                        help='Termination threshold for roll (default: inf)')
    parser.add_argument('--theta_termination', type=float, default=float('inf'),
                        help='Termination threshold for pitch (default: inf)')
    parser.add_argument('--psi_termination', type=float, default=float('inf'),
                        help='Termination threshold for yaw (default: inf)')
    parser.add_argument('--p_body_termination', type=float, default=float('inf'),
                        help='Termination threshold for roll rate (default: inf)')
    parser.add_argument('--q_body_termination', type=float, default=float('inf'),
                        help='Termination threshold for pitch rate (default: inf)')
    parser.add_argument('--r_body_termination', type=float, default=float('inf'),
                        help='Termination threshold for yaw rate (default: inf)')

    # Simulation parameters
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per trajectory (default: 500)')
    parser.add_argument('--episode_len_sec', type=int, default=10,
                        help='Episode length in seconds (default: 10)')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel processing using multiple CPU cores (default: False)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes for parallel execution (default: all available CPUs)')
    parser.add_argument('--save_freq', type=float, default=0.01,
                        help='Frequency in seconds at which to save trajectory states (default: 0.01 = 100 Hz)')
    parser.add_argument('--skip_save', action='store_true',
                        help='Skip saving trajectory files to disk (default: False)')
    parser.add_argument('--random_init', action='store_true',
                        help='Use random sampling instead of discretized grid for initial states (default: False)')
    parser.add_argument('--num_trajs', type=int, default=1000,
                        help='Number of trajectories to generate when using --random_init (default: 1000)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility when using --random_init (default: None)')

    args = parser.parse_args()

    # Set default termination thresholds
    if args.x_termination is None:
        args.x_termination = args.x_bound
    if args.y_termination is None:
        args.y_termination = args.y_bound
    if args.z_termination is None:
        args.z_termination = args.z_bound

    # Create output directory structure
    trajectories_dir = os.path.join(args.output_dir, 'trajectories')
    if not args.skip_save:
        os.makedirs(trajectories_dir, exist_ok=True)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # Define bounds (symmetric around zero)
    bounds = {
        'x': args.x_bound,
        'x_dot': args.x_dot_bound,
        'y': args.y_bound,
        'y_dot': args.y_dot_bound,
        'z': args.z_bound,
        'z_dot': args.z_dot_bound,
        'phi': args.phi_bound,
        'theta': args.theta_bound,
        'psi': args.psi_bound,
        'p_body': args.p_body_bound,
        'q_body': args.q_body_bound,
        'r_body': args.r_body_bound
    }

    # Define termination thresholds
    termination_thresholds = {
        'x': args.x_termination,
        'x_dot': args.x_dot_termination,
        'y': args.y_termination,
        'y_dot': args.y_dot_termination,
        'z': args.z_termination,
        'z_dot': args.z_dot_termination,
        'phi': args.phi_termination,
        'theta': args.theta_termination,
        'psi': args.psi_termination,
        'p_body': args.p_body_termination,
        'q_body': args.q_body_termination,
        'r_body': args.r_body_termination
    }

    # Generate initial states (either discretized or random)
    if args.random_init:
        print(f"Generating {args.num_trajs} random initial states...")
        initial_states = generate_random_initial_states(bounds, args.num_trajs, termination_thresholds, args.seed)
        print(f"Generated {len(initial_states)} random initial states (excluding those that violate termination bounds)")
    else:
        print("Generating discretized initial states...")
        initial_states = generate_discretized_initial_states(bounds, args.resolution, termination_thresholds)
        print(f"Generated {len(initial_states)} discretized initial states (excluding those that violate termination bounds)")

    # Calculate control frequency based on save_freq
    default_ctrl_freq = 30  # Hz (quadrotor default)
    min_ctrl_freq_for_save = 1.0 / args.save_freq
    ctrl_freq = max(default_ctrl_freq, min_ctrl_freq_for_save)
    ctrl_timestep = 1.0 / ctrl_freq

    # PyBullet frequency should be high enough for accurate physics
    pyb_freq = int(ctrl_freq * 50)

    # Prepare environment configuration for workers
    env_config = {
        'task': 'stabilization',
        'ctrl_freq': ctrl_freq,
        'pyb_freq': pyb_freq,
        'episode_len_sec': args.episode_len_sec,
        'done_on_out_of_bound': True,
        'cost': 'quadratic',
        'q_lqr': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 12 state dimensions
        'r_lqr': [0.1, 0.1, 0.1, 0.1],  # 4 control dimensions (4 rotors)
        'max_steps': args.max_steps
    }

    print(f"Termination thresholds:")
    print(f"  Position: x=±{args.x_termination}, y=±{args.y_termination}, z=±{args.z_termination}")
    print(f"  Velocity: x_dot=±{args.x_dot_termination}, y_dot=±{args.y_dot_termination}, z_dot=±{args.z_dot_termination}")
    print(f"  Angles: phi=±{args.phi_termination}, theta=±{args.theta_termination}, psi=±{args.psi_termination}")
    print(f"  Rates: p=±{args.p_body_termination}, q=±{args.q_body_termination}, r=±{args.r_body_termination}")
    print(f"Save frequency: {args.save_freq} s ({1.0/args.save_freq:.1f} Hz)")
    print(f"Control frequency: {ctrl_freq:.1f} Hz (timestep: {ctrl_timestep:.6f} s)")
    print(f"Physics frequency: {pyb_freq} Hz (timestep: {1.0/pyb_freq:.6f} s)")

    # Initialize statistics tracking
    state_vars = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                  'phi', 'theta', 'psi', 'p_body', 'q_body', 'r_body']
    stats = {
        var: {'min': float('inf'), 'max': float('-inf'),
              'prev_at_min': None, 'prev_at_max': None}
        for var in state_vars
    }
    stats.update({
        'success_count': 0,
        'total_count': 0,
        'timeout_count': 0,
        'max_traj_length': 0
    })

    # Collect ROA labels for all trajectories
    roa_labels = []

    if args.parallel:
        # Parallel execution
        num_workers = args.num_workers if args.num_workers else get_available_cpus()
        print(f"Generating trajectories using {num_workers} CPU cores (parallel mode)...")

        # Create arguments for each individual trajectory
        trajectory_args = [
            (idx, initial_states[idx], env_config, trajectories_dir, args.skip_save)
            for idx in range(len(initial_states))
        ]

        # Process trajectories in parallel with progress bar
        with Pool(processes=num_workers) as pool:
            traj_results = list(tqdm(
                pool.imap_unordered(process_single_trajectory, trajectory_args),
                total=len(initial_states),
                desc="Generating trajectories"
            ))

        # Aggregate statistics from all trajectories
        for traj_stats in traj_results:
            for key in state_vars:
                # Update min and its previous state
                if traj_stats[key]['min'] < stats[key]['min']:
                    stats[key]['min'] = traj_stats[key]['min']
                    stats[key]['prev_at_min'] = traj_stats[key]['prev_at_min']
                # Update max and its previous state
                if traj_stats[key]['max'] > stats[key]['max']:
                    stats[key]['max'] = traj_stats[key]['max']
                    stats[key]['prev_at_max'] = traj_stats[key]['prev_at_max']

            # Aggregate success and timeout counts
            stats['success_count'] += traj_stats['success_count']
            stats['total_count'] += traj_stats['total_count']
            stats['timeout_count'] += traj_stats['timeout_count']
            stats['max_traj_length'] = max(stats['max_traj_length'], traj_stats['max_traj_length'])

            # Collect ROA label
            roa_labels.append((traj_stats['init_state'], traj_stats['roa_label']))

    else:
        # Sequential execution
        print(f"Generating trajectories sequentially (single core)...")

        # Create environment and controller once for sequential execution
        env_func = partial(make,
                          'quadrotor',
                          quad_type=QuadType.THREE_D,
                          task=env_config['task'],
                          ctrl_freq=env_config['ctrl_freq'],
                          pyb_freq=env_config['pyb_freq'],
                          episode_len_sec=env_config['episode_len_sec'],
                          done_on_out_of_bound=env_config['done_on_out_of_bound'],
                          cost=env_config['cost'],
                          gui=False,
                          randomized_init=False)

        ctrl = make('lqr',
                    env_func,
                    q_lqr=env_config['q_lqr'],
                    r_lqr=env_config['r_lqr'],
                    discrete_dynamics=True)

        env = env_func()

        # Process trajectories sequentially
        for i, init_state in enumerate(tqdm(initial_states, desc="Generating trajectories")):
            # Run trajectory
            trajectory, success, timeout = run_trajectory(env, ctrl, init_state, env_config['max_steps'])

            # Update success and timeout tracking
            stats['total_count'] += 1
            if success:
                stats['success_count'] += 1
            if timeout:
                stats['timeout_count'] += 1

            # Store ROA label with normalized angles
            x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r = init_state
            init_state_normalized = [x, x_dot, y, y_dot, z, z_dot,
                                    normalize_angle(phi), normalize_angle(theta), normalize_angle(psi),
                                    p, q, r]
            roa_label = 1 if success else 0
            roa_labels.append((init_state_normalized, roa_label))

            # Update trajectory length tracking
            stats['max_traj_length'] = max(stats['max_traj_length'], len(trajectory))

            # Update statistics with previous state tracking
            traj_array = np.array(trajectory)

            if len(traj_array) > 0:
                state_var_indices = [
                    ('x', 0), ('x_dot', 1), ('y', 2), ('y_dot', 3), ('z', 4), ('z_dot', 5),
                    ('phi', 6), ('theta', 7), ('psi', 8), ('p_body', 9), ('q_body', 10), ('r_body', 11)
                ]
                for var_name, col_idx in state_var_indices:
                    traj_min = traj_array[:, col_idx].min()
                    if traj_min < stats[var_name]['min']:
                        stats[var_name]['min'] = traj_min
                        min_idx = traj_array[:, col_idx].argmin()
                        stats[var_name]['prev_at_min'] = traj_array[min_idx - 1].tolist() if min_idx > 0 else None

                    traj_max = traj_array[:, col_idx].max()
                    if traj_max > stats[var_name]['max']:
                        stats[var_name]['max'] = traj_max
                        max_idx = traj_array[:, col_idx].argmax()
                        stats[var_name]['prev_at_max'] = traj_array[max_idx - 1].tolist() if max_idx > 0 else None

            # Save trajectory
            if not args.skip_save:
                filepath = os.path.join(trajectories_dir, f'sequence_{i}.txt')
                save_trajectory(trajectory, filepath)

        # Clean up
        env.close()
        ctrl.close()

    # Write ROA labels to file
    roa_labels_path = os.path.join(args.output_dir, 'roa_labels.txt')
    with open(roa_labels_path, 'w') as f:
        for init_state, label in roa_labels:
            # Format: x,x_dot,y,y_dot,z,z_dot,phi,theta,psi,p,q,r,label
            line = ','.join([f'{val:.6f}' for val in init_state] + [str(label)])
            f.write(line + '\n')

    if args.skip_save:
        print(f"\nSuccessfully generated {len(initial_states)} trajectories (files not saved)")
        print(f"ROA labels saved to: {roa_labels_path}")
    else:
        print(f"\nSuccessfully generated {len(initial_states)} trajectories in {trajectories_dir}")
        print(f"Each file contains states: x,x_dot,y,y_dot,z,z_dot,phi,theta,psi,p,q,r")
        print(f"ROA labels saved to: {roa_labels_path}")

    # Print success rate and trajectory statistics
    success_rate = (stats['success_count'] / stats['total_count'] * 100) if stats['total_count'] > 0 else 0
    timeout_rate = (stats['timeout_count'] / stats['total_count'] * 100) if stats['total_count'] > 0 else 0
    failed_count = stats['total_count'] - stats['success_count'] - stats['timeout_count']
    failed_rate = (failed_count / stats['total_count'] * 100) if stats['total_count'] > 0 else 0

    print(f"\n{'='*80}")
    print(f"Trajectory Statistics:")
    print(f"{'='*80}")
    print(f"  Total trajectories:     {stats['total_count']}")
    print(f"  Successful (goal):      {stats['success_count']} ({success_rate:.2f}%)")
    print(f"  Failed (out of bounds): {failed_count} ({failed_rate:.2f}%)")
    print(f"  Timeout (max steps):    {stats['timeout_count']} ({timeout_rate:.2f}%)")
    print(f"  Max trajectory length:  {stats['max_traj_length']} states")

    # Print actual achieved bounds statistics
    print(f"\n{'='*80}")
    print(f"Actual Achieved Bounds Across All Trajectories:")
    print(f"{'='*80}")

    # Helper function to format state
    def format_state(state):
        if state is None:
            return "N/A (initial state)"
        return (f"[x={state[0]:>7.3f}, ẋ={state[1]:>7.3f}, y={state[2]:>7.3f}, ẏ={state[3]:>7.3f}, "
                f"z={state[4]:>7.3f}, ż={state[5]:>7.3f}, φ={state[6]:>7.3f}, θ={state[7]:>7.3f}, "
                f"ψ={state[8]:>7.3f}, p={state[9]:>7.3f}, q={state[10]:>7.3f}, r={state[11]:>7.3f}]")

    var_labels = [
        ('x', 'x (position)'), ('x_dot', 'x_dot (velocity)'),
        ('y', 'y (position)'), ('y_dot', 'y_dot (velocity)'),
        ('z', 'z (position)'), ('z_dot', 'z_dot (velocity)'),
        ('phi', 'phi (roll)'), ('theta', 'theta (pitch)'), ('psi', 'psi (yaw)'),
        ('p_body', 'p (roll rate)'), ('q_body', 'q (pitch rate)'), ('r_body', 'r (yaw rate)')
    ]

    for var_name, var_label in var_labels:
        print(f"\n  {var_label}:")
        print(f"    Min: {stats[var_name]['min']:>10.6f}")
        if stats[var_name]['prev_at_min'] is not None:
            print(f"         Previous state: {format_state(stats[var_name]['prev_at_min'])}")
        print(f"    Max: {stats[var_name]['max']:>10.6f}")
        if stats[var_name]['prev_at_max'] is not None:
            print(f"         Previous state: {format_state(stats[var_name]['prev_at_max'])}")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
