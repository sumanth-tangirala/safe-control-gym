#!/usr/bin/env python3
"""
Script to generate quadrotor trajectory dataset with LQR controller.
Discretizes the initial state space with 0.05 resolution and saves trajectories.
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


def count_existing_trajectories(trajectories_dir):
    """
    Count existing trajectory files and find the next available index.

    Args:
        trajectories_dir: Directory containing trajectory files (sequence_*.txt)

    Returns:
        tuple: (count, next_index) - number of existing files and next available index
    """
    import glob
    import re

    if not os.path.exists(trajectories_dir):
        return 0, 0

    traj_files = glob.glob(os.path.join(trajectories_dir, 'sequence_*.txt'))

    if not traj_files:
        return 0, 0

    # Extract indices from filenames and find the maximum
    indices = []
    for f in traj_files:
        match = re.search(r'sequence_(\d+)\.txt$', f)
        if match:
            indices.append(int(match.group(1)))

    if not indices:
        return 0, 0

    count = len(indices)
    next_index = max(indices) + 1

    return count, next_index


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
        bounds (dict): Dictionary with keys 'x', 'z', 'theta', 'x_dot', 'z_dot', 'theta_dot'
                      Values are bound magnitudes. For x and theta, bounds are symmetric around zero.
                      For z (altitude), bounds are [z_min, z_max] (positive range above ground).
        resolution (float): Discretization resolution
        termination_thresholds (dict, optional): Dictionary with same keys as bounds,
                                                 specifying termination thresholds. States at or beyond
                                                 these thresholds will be excluded from initial states.

    Returns:
        list: List of initial states as [x, z, theta, x_dot, z_dot, theta_dot]
    """
    states = []

    # Create discretized ranges for each dimension
    # x is symmetric around zero
    x_vals = np.arange(-bounds['x'], bounds['x'] + resolution, resolution)
    # z is altitude (positive), range from z_min to z_max
    z_vals = np.arange(bounds['z_min'], bounds['z_max'] + resolution, resolution)
    # theta is symmetric around zero
    theta_vals = np.arange(-bounds['theta'], bounds['theta'] + resolution, resolution)
    # Velocities are symmetric around zero
    x_dot_vals = np.arange(-bounds['x_dot'], bounds['x_dot'] + resolution, resolution)
    z_dot_vals = np.arange(-bounds['z_dot'], bounds['z_dot'] + resolution, resolution)
    theta_dot_vals = np.arange(-bounds['theta_dot'], bounds['theta_dot'] + resolution, resolution)

    # Generate all combinations in output order: [x, z, theta, x_dot, z_dot, theta_dot]
    for x in x_vals:
        for z in z_vals:
            for theta in theta_vals:
                for x_dot in x_dot_vals:
                    for z_dot in z_dot_vals:
                        for theta_dot in theta_dot_vals:
                            # Check if state violates termination thresholds
                            if termination_thresholds is not None:
                                # Skip states that would immediately trigger termination
                                if (abs(x) >= termination_thresholds['x'] or
                                    z < termination_thresholds['z_min'] or
                                    z >= termination_thresholds['z_max'] or
                                    abs(theta) >= termination_thresholds['theta'] or
                                    abs(x_dot) >= termination_thresholds['x_dot'] or
                                    abs(z_dot) >= termination_thresholds['z_dot'] or
                                    abs(theta_dot) >= termination_thresholds['theta_dot']):
                                    continue

                            states.append([x, z, theta, x_dot, z_dot, theta_dot])

    return states


def generate_random_initial_states(bounds, num_samples, termination_thresholds=None, seed=None):
    """
    Generate random initial states within given bounds.

    Args:
        bounds (dict): Dictionary with keys 'x', 'z_min', 'z_max', 'theta', 'x_dot', 'z_dot', 'theta_dot'
                      Values are bound magnitudes. For x and theta, bounds are symmetric around zero.
                      For z (altitude), bounds are [z_min, z_max] (positive range above ground).
        num_samples (int): Number of random initial states to generate
        termination_thresholds (dict, optional): Dictionary with same keys as bounds,
                                                 specifying termination thresholds. States at or beyond
                                                 these thresholds will be excluded from initial states.
        seed (int, optional): Random seed for reproducibility

    Returns:
        list: List of initial states as [x, z, theta, x_dot, z_dot, theta_dot]
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
        z = np.random.uniform(bounds['z_min'], bounds['z_max'])
        theta = np.random.uniform(-bounds['theta'], bounds['theta'])
        x_dot = np.random.uniform(-bounds['x_dot'], bounds['x_dot'])
        z_dot = np.random.uniform(-bounds['z_dot'], bounds['z_dot'])
        theta_dot = np.random.uniform(-bounds['theta_dot'], bounds['theta_dot'])

        # Check if state violates termination thresholds
        if termination_thresholds is not None:
            if (abs(x) >= termination_thresholds['x'] or
                z < termination_thresholds['z_min'] or
                z >= termination_thresholds['z_max'] or
                abs(theta) >= termination_thresholds['theta'] or
                abs(x_dot) >= termination_thresholds['x_dot'] or
                abs(z_dot) >= termination_thresholds['z_dot'] or
                abs(theta_dot) >= termination_thresholds['theta_dot']):
                continue

        states.append([x, z, theta, x_dot, z_dot, theta_dot])

    if len(states) < num_samples:
        print(f"Warning: Could only generate {len(states)} valid states out of {num_samples} requested")

    return states


def generate_stratified_initial_states(bounds, discretizations, termination_thresholds=None):
    """
    Generate initial states on a fixed grid with per-dimension discretizations.

    Args:
        bounds (dict): Dictionary with keys 'x', 'z_min', 'z_max', 'theta', 'x_dot', 'z_dot', 'theta_dot'
                      Values are bound magnitudes. For x and theta, bounds are symmetric around zero.
                      For z (altitude), bounds are [z_min, z_max] (positive range above ground).
        discretizations (dict): Dictionary with keys 'x', 'z', 'theta', 'x_dot', 'z_dot', 'theta_dot'
                               specifying the grid spacing for each dimension.
        termination_thresholds (dict, optional): Dictionary with same keys as bounds,
                                                 specifying termination thresholds. States at or beyond
                                                 these thresholds will be excluded from initial states.

    Returns:
        list: List of initial states as [x, z, theta, x_dot, z_dot, theta_dot]
    """
    # Compute grid points for each dimension
    # x is symmetric: [-x_bound, x_bound]
    x_vals = np.arange(-bounds['x'], bounds['x'] + discretizations['x'] / 2, discretizations['x'])
    # z is asymmetric: [z_min, z_max]
    z_vals = np.arange(bounds['z_min'], bounds['z_max'] + discretizations['z'] / 2, discretizations['z'])
    # theta is symmetric: [-theta_bound, theta_bound]
    theta_vals = np.arange(-bounds['theta'], bounds['theta'] + discretizations['theta'] / 2, discretizations['theta'])
    # Velocities are symmetric
    x_dot_vals = np.arange(-bounds['x_dot'], bounds['x_dot'] + discretizations['x_dot'] / 2, discretizations['x_dot'])
    z_dot_vals = np.arange(-bounds['z_dot'], bounds['z_dot'] + discretizations['z_dot'] / 2, discretizations['z_dot'])
    theta_dot_vals = np.arange(-bounds['theta_dot'], bounds['theta_dot'] + discretizations['theta_dot'] / 2, discretizations['theta_dot'])

    # Print grid dimensions
    n_x, n_z, n_theta = len(x_vals), len(z_vals), len(theta_vals)
    n_x_dot, n_z_dot, n_theta_dot = len(x_dot_vals), len(z_dot_vals), len(theta_dot_vals)
    total_points = n_x * n_z * n_theta * n_x_dot * n_z_dot * n_theta_dot
    print(f"Grid dimensions: {n_x} x {n_z} x {n_theta} x {n_x_dot} x {n_z_dot} x {n_theta_dot} = {total_points} points")

    # Vectorized generation using meshgrid
    grid = np.meshgrid(x_vals, z_vals, theta_vals, x_dot_vals, z_dot_vals, theta_dot_vals, indexing='ij')

    # Stack and reshape to (N, 6) array
    states = np.column_stack([g.ravel() for g in grid])

    # Apply termination threshold filtering (vectorized)
    if termination_thresholds is not None:
        valid_mask = (
            (np.abs(states[:, 0]) < termination_thresholds['x']) &
            (states[:, 1] >= termination_thresholds['z_min']) &
            (states[:, 1] < termination_thresholds['z_max']) &
            (np.abs(states[:, 2]) < termination_thresholds['theta']) &
            (np.abs(states[:, 3]) < termination_thresholds['x_dot']) &
            (np.abs(states[:, 4]) < termination_thresholds['z_dot']) &
            (np.abs(states[:, 5]) < termination_thresholds['theta_dot'])
        )
        states = states[valid_mask]
        print(f"After filtering: {len(states)} valid states")

    return states.tolist()


def run_trajectory(env, ctrl, init_state, max_steps=1000):
    """
    Run a single trajectory with given initial state.

    Args:
        env: Environment instance
        ctrl: Controller instance
        init_state: Initial state [x, z, theta, x_dot, z_dot, theta_dot]
        max_steps: Maximum number of steps

    Returns:
        tuple: (trajectory, success, timeout)
            - trajectory: List of states in order [x, z, theta, x_dot, z_dot, theta_dot]
            - success: Boolean indicating if goal was reached (True) or terminated due to bounds (False)
            - timeout: Boolean indicating if trajectory reached max_steps without terminating
    """
    # Reset environment first
    obs, info = env.reset()

    # Now properly set the initial state in PyBullet simulation
    # Input order: [x, z, theta, x_dot, z_dot, theta_dot]
    x, z, theta, x_dot, z_dot, theta_dot = init_state

    # Set position and orientation
    p.resetBasePositionAndOrientation(
        env.DRONE_ID,
        [x, 0, z],  # Position: [x, y=0, z] for 2D quadrotor
        p.getQuaternionFromEuler([0, theta, 0]),  # Orientation: [roll=0, pitch=theta, yaw=0]
        physicsClientId=env.PYB_CLIENT)

    # Set velocities
    p.resetBaseVelocity(
        env.DRONE_ID,
        [x_dot, 0, z_dot],  # Linear velocity: [x_dot, y_dot=0, z_dot]
        [0, theta_dot, 0],  # Angular velocity: [p=0, q=theta_dot, r=0]
        physicsClientId=env.PYB_CLIENT)

    # Update environment's internal state to match
    env._update_and_store_kinematic_information()
    obs = env._get_observation()

    # Store initial state in output order: [x, z, theta, x_dot, z_dot, theta_dot]
    # Normalize theta to [-pi, pi] range
    trajectory = [[x, z, normalize_angle(theta), x_dot, z_dot, theta_dot]]

    success = False
    timeout = False

    for step in range(max_steps):
        # Get action from LQR controller
        action = ctrl.select_action(obs, info)

        # Take step in environment (old Gym API returns 4 values)
        obs, reward, done, info = env.step(action)

        # Extract state (env obs order: x, x_dot, z, z_dot, theta, theta_dot)
        x, x_dot, z, z_dot, theta, theta_dot = obs[:6]

        # Store in output order: [x, z, theta, x_dot, z_dot, theta_dot]
        # Normalize theta to [-pi, pi] range
        current_state = [x, z, normalize_angle(theta), x_dot, z_dot, theta_dot]
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
            - init_state: initial state for the trajectory [x, z, theta, x_dot, z_dot, theta_dot]
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
                      quad_type=QuadType.TWO_D,
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

    # Configure environment's state_space bounds to match termination thresholds
    # Env state order: [x, x_dot, z, z_dot, theta, theta_dot]
    # Positions
    env.state_space.low[0] = -env_config['x_termination']
    env.state_space.high[0] = env_config['x_termination']
    env.state_space.low[2] = env_config['z_min_termination']
    env.state_space.high[2] = env_config['z_max_termination']
    env.state_space.low[4] = -env_config['theta_termination']
    env.state_space.high[4] = env_config['theta_termination']
    # Velocities (for closed state space)
    env.state_space.low[1] = -env_config['x_dot_termination']
    env.state_space.high[1] = env_config['x_dot_termination']
    env.state_space.low[3] = -env_config['z_dot_termination']
    env.state_space.high[3] = env_config['z_dot_termination']
    env.state_space.low[5] = -env_config['theta_dot_termination']
    env.state_space.high[5] = env_config['theta_dot_termination']

    # Initialize statistics for this trajectory
    # State order: [x, z, theta, x_dot, z_dot, theta_dot]
    traj_stats = {
        'x': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'z': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'theta': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'x_dot': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'z_dot': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'theta_dot': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'success_count': 0,
        'total_count': 0,
        'timeout_count': 0,
        'max_traj_length': 0,
        'roa_label': None,
        'init_state': None
    }

    # Run trajectory
    trajectory, success, timeout = run_trajectory(env, ctrl, init_state, env_config['max_steps'])

    # Update success and timeout tracking
    traj_stats['total_count'] = 1
    if success:
        traj_stats['success_count'] = 1
    if timeout:
        traj_stats['timeout_count'] = 1

    # Store ROA label: 1 for success, 0 for failure (timeout or out of bounds)
    # Store initial state with normalized theta
    # Input order: [x, z, theta, x_dot, z_dot, theta_dot]
    x, z, theta, x_dot, z_dot, theta_dot = init_state
    traj_stats['init_state'] = [x, z, normalize_angle(theta), x_dot, z_dot, theta_dot]
    traj_stats['roa_label'] = 1 if success else 0

    # Update trajectory length tracking
    traj_stats['max_traj_length'] = len(trajectory)

    # Update statistics with previous state tracking
    traj_array = np.array(trajectory)

    # Include all states (including initial state) in statistics
    # Output state order: [x, z, theta, x_dot, z_dot, theta_dot]
    if len(traj_array) > 0:
        # For each state variable, track min/max and the previous state
        state_vars = [('x', 0), ('z', 1), ('theta', 2), ('x_dot', 3), ('z_dot', 4), ('theta_dot', 5)]
        for var_name, col_idx in state_vars:
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


def generate_roa_labels_from_trajectories(trajectories_dir, output_path):
    """
    Generate ROA labels from saved trajectory files.

    For each trajectory, all states except the termination state are labeled
    with the trajectory's success label (1 for success, 0 for failure).

    Args:
        trajectories_dir: Directory containing trajectory files (sequence_*.txt)
        output_path: Path to write roa_labels.txt

    Returns:
        tuple: (total_states, success_count, failure_count)
    """
    import glob

    # Find all trajectory files
    traj_files = sorted(glob.glob(os.path.join(trajectories_dir, 'sequence_*.txt')))

    if not traj_files:
        print(f"Warning: No trajectory files found in {trajectories_dir}")
        return 0, 0, 0

    total_states = 0
    success_traj_count = 0
    failure_traj_count = 0

    with open(output_path, 'w') as f_out:
        for traj_file in tqdm(traj_files, desc="Generating ROA labels"):
            # Read trajectory
            states = []
            with open(traj_file, 'r') as f_in:
                for line in f_in:
                    line = line.strip()
                    if line:
                        values = [float(v) for v in line.split(',')]
                        states.append(values)

            if len(states) < 2:
                continue  # Skip trajectories with less than 2 states

            # Determine if trajectory was successful
            # Success: final state is near goal [0, 1, 0, 0, 0, 0] (x=0, z=1, theta=0, velocities=0)
            # State order: [x, z, theta, x_dot, z_dot, theta_dot]
            final_state = states[-1]
            goal_state = [0, 1, 0, 0, 0, 0]
            tolerance = 0.05

            is_success = np.linalg.norm(np.array(final_state) - np.array(goal_state)) < tolerance
            label = 1 if is_success else 0

            if is_success:
                success_traj_count += 1
            else:
                failure_traj_count += 1

            # Write all states except the termination state with the trajectory's label
            for state in states[:-1]:
                line = ','.join([f'{val:.6f}' for val in state] + [str(label)])
                f_out.write(line + '\n')
                total_states += 1

    return total_states, success_traj_count, failure_traj_count


def main():
    parser = argparse.ArgumentParser(description='Generate 2D quadrotor trajectory dataset')
    parser.add_argument('--output_dir', type=str,
                        default='/common/users/shared/pracsys/genMoPlan/data_trajectories/quadrotor2D_lqr',
                        help='Directory to save trajectory files')
    parser.add_argument('--resolution', type=float, default=0.05,
                        help='Discretization resolution (default: 0.05)')
    # Position bounds
    parser.add_argument('--x_bound', type=float, default=1.0,
                        help='Symmetric bound for x position, range: [-x_bound, +x_bound] (default: 1.0)')
    parser.add_argument('--z_min', type=float, default=0.1,
                        help='Minimum z (altitude) position (default: 0.1, above ground)')
    parser.add_argument('--z_max', type=float, default=1.5,
                        help='Maximum z (altitude) position (default: 1.5)')
    parser.add_argument('--theta_bound', type=float, default=np.pi,
                        help='Symmetric bound for theta (pitch) angle, range: [-theta_bound, +theta_bound] (default: pi)')
    # Velocity bounds
    parser.add_argument('--x_dot_bound', type=float, default=1.0,
                        help='Symmetric bound for x velocity, range: [-x_dot_bound, +x_dot_bound] (default: 1.0)')
    parser.add_argument('--z_dot_bound', type=float, default=1.0,
                        help='Symmetric bound for z velocity, range: [-z_dot_bound, +z_dot_bound] (default: 1.0)')
    parser.add_argument('--theta_dot_bound', type=float, default=8.0,
                        help='Symmetric bound for theta velocity, range: [-theta_dot_bound, +theta_dot_bound] (default: 8.0)')
    # Simulation parameters (high max_steps to avoid timeouts - trajectories run until success or out-of-bounds)
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum steps per trajectory (default: 100000, effectively no timeout)')
    parser.add_argument('--episode_len_sec', type=int, default=1000,
                        help='Episode length in seconds (default: 1000, effectively no timeout)')
    # Termination thresholds
    parser.add_argument('--x_termination', type=float, default=None,
                        help='Termination threshold for x position (default: copies x_bound)')
    parser.add_argument('--z_min_termination', type=float, default=None,
                        help='Termination threshold for minimum z position (default: copies z_min)')
    parser.add_argument('--z_max_termination', type=float, default=None,
                        help='Termination threshold for maximum z position (default: copies z_max)')
    parser.add_argument('--theta_termination', type=float, default=float('inf'),
                        help='Termination threshold for theta angle (default: inf)')
    parser.add_argument('--x_dot_termination', type=float, default=None,
                        help='Termination threshold for x velocity (default: copies x_dot_bound)')
    parser.add_argument('--z_dot_termination', type=float, default=None,
                        help='Termination threshold for z velocity (default: copies z_dot_bound)')
    parser.add_argument('--theta_dot_termination', type=float, default=None,
                        help='Termination threshold for theta velocity (default: copies theta_dot_bound)')
    # Execution options
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel processing using multiple CPU cores (default: False, sequential)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes for parallel execution (default: all available CPUs)')
    parser.add_argument('--save_freq', type=float, default=0.01,
                        help='Frequency in seconds at which to save trajectory states. '
                             'The control and physics integration frequencies will be automatically adjusted '
                             'to match or exceed this frequency for accurate state computation. (default: 0.01 = 100 Hz)')
    parser.add_argument('--skip_save', action='store_true',
                        help='Skip saving trajectory files to disk. Trajectories will still be generated and statistics computed. (default: False)')
    parser.add_argument('--random_init', action='store_true',
                        help='Use random sampling instead of discretized grid for initial states (default: False)')
    parser.add_argument('--num_trajs', type=int, default=1000,
                        help='Number of trajectories to generate when using --random_init (default: 1000)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility when using --random_init or --stratified (default: None)')
    # Stratified sampling options
    parser.add_argument('--stratified', action='store_true',
                        help='Use stratified grid sampling: create a grid and sample uniformly within each cell (default: False)')
    parser.add_argument('--x_disc', type=float, default=0.28,
                        help='Grid spacing for x dimension in stratified sampling (default: 0.28)')
    parser.add_argument('--z_disc', type=float, default=0.23,
                        help='Grid spacing for z dimension in stratified sampling (default: 0.23)')
    parser.add_argument('--theta_disc', type=float, default=0.55,
                        help='Grid spacing for theta dimension in stratified sampling (default: 0.55)')
    parser.add_argument('--x_dot_disc', type=float, default=0.28,
                        help='Grid spacing for x_dot dimension in stratified sampling (default: 0.28)')
    parser.add_argument('--z_dot_disc', type=float, default=0.28,
                        help='Grid spacing for z_dot dimension in stratified sampling (default: 0.28)')
    parser.add_argument('--theta_dot_disc', type=float, default=0.9,
                        help='Grid spacing for theta_dot dimension in stratified sampling (default: 0.9)')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt for stratified mode')
    parser.add_argument('--generate_roa_only', action='store_true',
                        help='Only generate ROA labels from existing trajectory files. '
                             'Skips trajectory generation. Use this after terminating a run early '
                             'or to regenerate ROA labels from all existing trajectories.')

    args = parser.parse_args()

    # Set up directories
    trajectories_dir = os.path.join(args.output_dir, 'trajectories')
    roa_labels_path = os.path.join(args.output_dir, 'roa_labels.txt')

    # Handle --generate_roa_only mode
    if args.generate_roa_only:
        print(f"Generating ROA labels from existing trajectory files...")
        print(f"Trajectories directory: {trajectories_dir}")

        total_states, success_trajs, failure_trajs = generate_roa_labels_from_trajectories(
            trajectories_dir, roa_labels_path
        )

        print(f"\nROA labels saved to: {roa_labels_path}")
        print(f"  Total trajectories processed: {success_trajs + failure_trajs}")
        print(f"  Successful trajectories: {success_trajs}")
        print(f"  Failed trajectories: {failure_trajs}")
        print(f"  Total state-label pairs: {total_states}")
        return

    # Check existing trajectories and calculate how many more are needed
    existing_count, start_idx = count_existing_trajectories(trajectories_dir)

    if existing_count > 0:
        print(f"Found {existing_count} existing trajectories in {trajectories_dir}")
        print(f"Next trajectory index: {start_idx}")

    if args.random_init:
        # For random init, check if we already have enough trajectories
        if existing_count >= args.num_trajs:
            print(f"Target of {args.num_trajs} trajectories already reached ({existing_count} exist).")
            print(f"Skipping trajectory generation, generating ROA labels from existing files...")

            total_states, success_trajs, failure_trajs = generate_roa_labels_from_trajectories(
                trajectories_dir, roa_labels_path
            )

            print(f"\nROA labels saved to: {roa_labels_path}")
            print(f"  Total trajectories processed: {success_trajs + failure_trajs}")
            print(f"  Successful trajectories: {success_trajs}")
            print(f"  Failed trajectories: {failure_trajs}")
            print(f"  Total state-label pairs: {total_states}")
            return

        # Calculate how many more trajectories to generate
        num_to_generate = args.num_trajs - existing_count
        print(f"Need to generate {num_to_generate} more trajectories to reach target of {args.num_trajs}")

    # Set default termination thresholds
    # Position termination
    if args.x_termination is None:
        args.x_termination = args.x_bound
    if args.z_min_termination is None:
        args.z_min_termination = args.z_min
    if args.z_max_termination is None:
        args.z_max_termination = args.z_max
    # Velocity termination (must match bounds for closed state space)
    if args.x_dot_termination is None:
        args.x_dot_termination = args.x_dot_bound
    if args.z_dot_termination is None:
        args.z_dot_termination = args.z_dot_bound
    if args.theta_dot_termination is None:
        args.theta_dot_termination = args.theta_dot_bound

    # Create output directory structure
    if not args.skip_save:
        os.makedirs(trajectories_dir, exist_ok=True)
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # Define bounds
    # State order: [x, z, theta, x_dot, z_dot, theta_dot]
    bounds = {
        'x': args.x_bound,        # Symmetric: [-x_bound, +x_bound]
        'z_min': args.z_min,      # z (altitude) is asymmetric: [z_min, z_max]
        'z_max': args.z_max,
        'theta': args.theta_bound,  # Symmetric: [-theta_bound, +theta_bound]
        'x_dot': args.x_dot_bound,
        'z_dot': args.z_dot_bound,
        'theta_dot': args.theta_dot_bound
    }

    # Define termination thresholds
    termination_thresholds = {
        'x': args.x_termination,
        'z_min': args.z_min_termination,
        'z_max': args.z_max_termination,
        'theta': args.theta_termination,
        'x_dot': args.x_dot_termination,
        'z_dot': args.z_dot_termination,
        'theta_dot': args.theta_dot_termination
    }

    # Generate initial states (either discretized, random, or stratified)
    if args.stratified:
        # Stratified grid sampling
        if existing_count > 0:
            print(f"Warning: {existing_count} trajectories already exist. Stratified mode will add new trajectories starting from index {start_idx}.")
        print("Generating stratified initial states...")
        discretizations = {
            'x': args.x_disc,
            'z': args.z_disc,
            'theta': args.theta_disc,
            'x_dot': args.x_dot_disc,
            'z_dot': args.z_dot_disc,
            'theta_dot': args.theta_dot_disc
        }
        initial_states = generate_stratified_initial_states(
            bounds, discretizations, termination_thresholds
        )
        print(f"Generated {len(initial_states)} stratified initial states (excluding those that violate termination bounds)")
    elif args.random_init:
        print(f"Generating {num_to_generate} random initial states...")
        initial_states = generate_random_initial_states(bounds, num_to_generate, termination_thresholds, args.seed)
        print(f"Generated {len(initial_states)} random initial states (excluding those that violate termination bounds)")
    else:
        # For discretized mode, generate all states (resumption not supported for grid)
        if existing_count > 0:
            print(f"Warning: {existing_count} trajectories already exist. Discretized mode will add new trajectories starting from index {start_idx}.")
        print("Generating discretized initial states...")
        initial_states = generate_discretized_initial_states(bounds, args.resolution, termination_thresholds)
        print(f"Generated {len(initial_states)} discretized initial states (excluding those that violate termination bounds)")

    # Confirmation prompt for stratified mode
    if args.stratified and not args.yes:
        print(f"\n{'='*60}")
        print(f"Will generate {len(initial_states):,} trajectories.")
        print(f"{'='*60}")
        response = input("Proceed? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    # Calculate control frequency based on save_freq
    # Control frequency should be at least as high as save frequency to avoid duplicating states
    default_ctrl_freq = 30  # Hz (quadrotor default)
    min_ctrl_freq_for_save = 1.0 / args.save_freq  # Hz required for save_freq
    ctrl_freq = max(default_ctrl_freq, min_ctrl_freq_for_save)

    # Adjust control timestep to be compatible with save_freq
    ctrl_timestep = 1.0 / ctrl_freq

    # PyBullet frequency should be high enough for accurate physics
    # Use at least 50 steps per control step for accuracy
    pyb_freq = int(ctrl_freq * 50)

    # Prepare environment configuration for workers
    env_config = {
        'task': 'stabilization',
        'ctrl_freq': ctrl_freq,
        'pyb_freq': pyb_freq,
        'episode_len_sec': args.episode_len_sec,
        'done_on_out_of_bound': True,
        'cost': 'quadratic',
        'q_lqr': [1, 1, 1, 1, 1, 1],  # 6 state dimensions for 2D quadrotor
        'r_lqr': [0.1, 0.1],  # 2 control dimensions for 2D quadrotor
        'max_steps': args.max_steps,
        # Termination thresholds for configuring environment state_space bounds
        'x_termination': args.x_termination,
        'z_min_termination': args.z_min_termination,
        'z_max_termination': args.z_max_termination,
        'theta_termination': args.theta_termination,
        'x_dot_termination': args.x_dot_termination,
        'z_dot_termination': args.z_dot_termination,
        'theta_dot_termination': args.theta_dot_termination
    }

    print(f"Termination thresholds: x=±{args.x_termination}, z=[{args.z_min_termination}, {args.z_max_termination}], "
          f"theta=±{args.theta_termination}, x_dot=±{args.x_dot_termination}, "
          f"z_dot=±{args.z_dot_termination}, theta_dot=±{args.theta_dot_termination}")
    print(f"Save frequency: {args.save_freq} s ({1.0/args.save_freq:.1f} Hz)")
    print(f"Control frequency: {ctrl_freq:.1f} Hz (timestep: {ctrl_timestep:.6f} s)")
    print(f"Physics frequency: {pyb_freq} Hz (timestep: {1.0/pyb_freq:.6f} s)")

    # Initialize statistics tracking
    # State order: [x, z, theta, x_dot, z_dot, theta_dot]
    stats = {
        'x': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'z': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'theta': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'x_dot': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'z_dot': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'theta_dot': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'success_count': 0,
        'total_count': 0,
        'timeout_count': 0,
        'max_traj_length': 0
    }

    if args.parallel:
        # Parallel execution
        # Get number of CPUs available (respecting taskset/affinity)
        num_workers = args.num_workers if args.num_workers else get_available_cpus()

        print(f"Generating trajectories using {num_workers} CPU cores (parallel mode)...")

        # Create arguments for each individual trajectory
        # Use start_idx to allow resuming from a previous run
        trajectory_args = [
            (start_idx + idx, initial_states[idx], env_config, trajectories_dir, args.skip_save)
            for idx in range(len(initial_states))
        ]

        # Process trajectories in parallel with progress bar showing trajectory count
        with Pool(processes=num_workers) as pool:
            traj_results = list(tqdm(
                pool.imap_unordered(process_single_trajectory, trajectory_args),
                total=len(initial_states),
                desc="Generating trajectories"
            ))

        # Aggregate statistics from all trajectories
        # State order: [x, z, theta, x_dot, z_dot, theta_dot]
        for traj_stats in traj_results:
            for key in ['x', 'z', 'theta', 'x_dot', 'z_dot', 'theta_dot']:
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

            # Update max trajectory length
            stats['max_traj_length'] = max(stats['max_traj_length'], traj_stats['max_traj_length'])

    else:
        # Sequential execution
        print(f"Generating trajectories sequentially (single core)...")

        # Create environment and controller once for sequential execution
        env_func = partial(make,
                          'quadrotor',
                          quad_type=QuadType.TWO_D,
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

        # Configure environment's state_space bounds to match termination thresholds
        # Env state order: [x, x_dot, z, z_dot, theta, theta_dot]
        env.state_space.low[0] = -env_config['x_termination']
        env.state_space.high[0] = env_config['x_termination']
        env.state_space.low[2] = env_config['z_min_termination']
        env.state_space.high[2] = env_config['z_max_termination']
        env.state_space.low[4] = -env_config['theta_termination']
        env.state_space.high[4] = env_config['theta_termination']

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

            # Update trajectory length tracking
            stats['max_traj_length'] = max(stats['max_traj_length'], len(trajectory))

            # Update statistics with previous state tracking
            traj_array = np.array(trajectory)

            # Include all states (including initial state) in statistics
            # Output state order: [x, z, theta, x_dot, z_dot, theta_dot]
            if len(traj_array) > 0:
                # For each state variable, track min/max and the previous state
                state_vars = [('x', 0), ('z', 1), ('theta', 2), ('x_dot', 3), ('z_dot', 4), ('theta_dot', 5)]
                for var_name, col_idx in state_vars:
                    # Find min value and its index
                    traj_min = traj_array[:, col_idx].min()
                    if traj_min < stats[var_name]['min']:
                        stats[var_name]['min'] = traj_min
                        min_idx = traj_array[:, col_idx].argmin()
                        # Store previous state (None if this is the initial state)
                        stats[var_name]['prev_at_min'] = traj_array[min_idx - 1].tolist() if min_idx > 0 else None

                    # Find max value and its index
                    traj_max = traj_array[:, col_idx].max()
                    if traj_max > stats[var_name]['max']:
                        stats[var_name]['max'] = traj_max
                        max_idx = traj_array[:, col_idx].argmax()
                        # Store previous state (None if this is the initial state)
                        stats[var_name]['prev_at_max'] = traj_array[max_idx - 1].tolist() if max_idx > 0 else None

            # Save trajectory (only if skip_save is False)
            # Use start_idx to allow resuming from a previous run
            if not args.skip_save:
                filepath = os.path.join(trajectories_dir, f'sequence_{start_idx + i}.txt')
                save_trajectory(trajectory, filepath)

        # Clean up
        env.close()
        ctrl.close()

    if args.skip_save:
        print(f"\nSuccessfully generated {len(initial_states)} trajectories (files not saved)")
        print(f"Note: ROA labels cannot be generated when --skip_save is used")
    else:
        print(f"\nSuccessfully generated {len(initial_states)} trajectories in {trajectories_dir}")
        print(f"Each file contains a trajectory with states in format: x,z,theta,x_dot,z_dot,theta_dot")

        # Generate ROA labels from saved trajectory files (post-processing)
        print(f"\nGenerating ROA labels from saved trajectories...")
        roa_labels_path = os.path.join(args.output_dir, 'roa_labels.txt')
        total_states, success_trajs, failure_trajs = generate_roa_labels_from_trajectories(
            trajectories_dir, roa_labels_path
        )
        print(f"ROA labels saved to: {roa_labels_path}")
        print(f"  Total state-label pairs: {total_states}")
        print(f"  From successful trajectories: {success_trajs}")
        print(f"  From failed trajectories: {failure_trajs}")

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
    # State order: [x, z, theta, x_dot, z_dot, theta_dot]
    def format_state(state):
        if state is None:
            return "N/A (initial state)"
        return f"[x={state[0]:>7.3f}, z={state[1]:>7.3f}, θ={state[2]:>7.3f}, ẋ={state[3]:>7.3f}, ż={state[4]:>7.3f}, θ̇={state[5]:>7.3f}]"

    for var_name, var_label in [('x', 'x'), ('z', 'z'), ('theta', 'theta'), ('x_dot', 'x_dot'), ('z_dot', 'z_dot'), ('theta_dot', 'theta_dot')]:
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
