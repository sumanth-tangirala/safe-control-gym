#!/usr/bin/env python3
"""
Script to generate cartpole trajectory dataset with LQR controller.
Discretizes the initial state space with 0.05 resolution and saves trajectories.
"""

import argparse
import os
import numpy as np
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from safe_control_gym.utils.registration import make


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
        bounds (dict): Dictionary with keys 'x', 'x_dot', 'theta', 'theta_dot'
                      and values as symmetric bound magnitudes (states go from -value to +value)
        resolution (float): Discretization resolution
        termination_thresholds (dict, optional): Dictionary with keys 'x', 'x_dot', 'theta', 'theta_dot'
                                                 specifying termination thresholds. States at or beyond
                                                 these thresholds will be excluded from initial states.

    Returns:
        list: List of initial states as [x, x_dot, theta, theta_dot]
    """
    states = []

    # Create discretized ranges for each dimension (symmetric around zero)
    x_vals = np.arange(-bounds['x'], bounds['x'] + resolution, resolution)
    x_dot_vals = np.arange(-bounds['x_dot'], bounds['x_dot'] + resolution, resolution)
    theta_vals = np.arange(-bounds['theta'], bounds['theta'] + resolution, resolution)
    theta_dot_vals = np.arange(-bounds['theta_dot'], bounds['theta_dot'] + resolution, resolution)

    # Generate all combinations
    for x in x_vals:
        for x_dot in x_dot_vals:
            for theta in theta_vals:
                for theta_dot in theta_dot_vals:
                    # Check if state violates termination thresholds
                    if termination_thresholds is not None:
                        # Skip states that would immediately trigger termination
                        if (abs(x) >= termination_thresholds['x'] or
                            abs(x_dot) >= termination_thresholds['x_dot'] or
                            abs(theta) >= termination_thresholds['theta'] or
                            abs(theta_dot) >= termination_thresholds['theta_dot']):
                            continue

                    states.append([x, x_dot, theta, theta_dot])

    return states


def run_trajectory(env, ctrl, init_state, max_steps=1000):
    """
    Run a single trajectory with given initial state.

    Args:
        env: Environment instance
        ctrl: Controller instance
        init_state: Initial state [x, x_dot, theta, theta_dot] (internal order)
        max_steps: Maximum number of steps

    Returns:
        tuple: (trajectory, success, timeout)
            - trajectory: List of states in order [x, theta, x_dot, theta_dot]
            - success: Boolean indicating if goal was reached (True) or terminated due to bounds (False)
            - timeout: Boolean indicating if trajectory reached max_steps without terminating
    """
    import pybullet as p

    # Reset environment first
    obs, info = env.reset()

    # Now properly set the initial state in PyBullet simulation
    x, x_dot, theta, theta_dot = init_state

    # Set cart position and velocity (joint 0)
    p.resetJointState(
        env.CARTPOLE_ID,
        jointIndex=0,
        targetValue=x,
        targetVelocity=x_dot,
        physicsClientId=env.PYB_CLIENT)

    # Set pole angle and angular velocity (joint 1)
    p.resetJointState(
        env.CARTPOLE_ID,
        jointIndex=1,
        targetValue=theta,
        targetVelocity=theta_dot,
        physicsClientId=env.PYB_CLIENT)

    # Update environment's internal state to match
    env.state = np.array([x, x_dot, theta, theta_dot])
    obs = env._get_observation()

    # Store initial state in output order: [x, theta, x_dot, theta_dot]
    # Normalize theta to [-pi, pi] range
    trajectory = [[x, normalize_angle(theta), x_dot, theta_dot]]

    success = False
    timeout = False

    for step in range(max_steps):
        # Get action from LQR controller
        action = ctrl.select_action(obs, info)

        # Take step in environment (old Gym API returns 4 values)
        obs, reward, done, info = env.step(action)

        # Extract state (internal order: x, x_dot, theta, theta_dot)
        x, x_dot, theta, theta_dot = obs[:4]

        # Store in output order: [x, theta, x_dot, theta_dot]
        # Normalize theta to [-pi, pi] range
        current_state = [x, normalize_angle(theta), x_dot, theta_dot]
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
                      'cartpole',
                      task=env_config['task'],
                      ctrl_freq=env_config['ctrl_freq'],
                      pyb_freq=env_config['pyb_freq'],
                      episode_len_sec=env_config['episode_len_sec'],
                      done_on_out_of_bound=env_config['done_on_out_of_bound'],
                      cost=env_config['cost'],
                      gui=False,
                      randomized_init=False,
                      obs_wrap_angle=True,
                      x_dot_limit=env_config['x_dot_limit'],
                      theta_dot_limit=env_config['theta_dot_limit'],
                      action_scale=env_config['action_scale'])

    ctrl = make('lqr',
                env_func,
                q_lqr=env_config['q_lqr'],
                r_lqr=env_config['r_lqr'],
                discrete_dynamics=True)

    env = env_func()

    # Set termination thresholds
    env.x_threshold = env_config['x_threshold']
    env.x_dot_threshold = env_config['x_dot_threshold']
    env.theta_threshold_radians = env_config['theta_threshold_radians']
    env.theta_dot_threshold = env_config['theta_dot_threshold']

    # Initialize statistics for this trajectory
    traj_stats = {
        'x': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'theta': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'x_dot': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
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
    # Store initial state in order: [x, theta, x_dot, theta_dot] with normalized theta
    x, x_dot, theta, theta_dot = init_state
    traj_stats['init_state'] = [x, normalize_angle(theta), x_dot, theta_dot]
    traj_stats['roa_label'] = 1 if success else 0

    # Update trajectory length tracking
    traj_stats['max_traj_length'] = len(trajectory)

    # Update statistics with previous state tracking
    traj_array = np.array(trajectory)

    # Include all states (including initial state) in statistics
    if len(traj_array) > 0:
        # For each state variable, track min/max and the previous state
        state_vars = [('x', 0), ('theta', 1), ('x_dot', 2), ('theta_dot', 3)]
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


def process_trajectory_batch(args_tuple):
    """
    Worker function to process a batch of trajectories in parallel.

    Args:
        args_tuple: Tuple containing (batch_data, env_config, output_dir, skip_save)
            - batch_data: list of tuples [(index, init_state), ...]
            - env_config: dict with environment configuration
            - output_dir: directory to save trajectories
            - skip_save: boolean flag to skip saving files

    Returns:
        dict: Statistics for this batch (min/max values for each state variable)
    """
    batch_data, env_config, output_dir, skip_save = args_tuple

    # Create environment and controller for this worker
    env_func = partial(make,
                      'cartpole',
                      task=env_config['task'],
                      ctrl_freq=env_config['ctrl_freq'],
                      pyb_freq=env_config['pyb_freq'],
                      episode_len_sec=env_config['episode_len_sec'],
                      done_on_out_of_bound=env_config['done_on_out_of_bound'],
                      cost=env_config['cost'],
                      gui=False,
                      randomized_init=False,
                      obs_wrap_angle=True,
                      x_dot_limit=env_config['x_dot_limit'],
                      theta_dot_limit=env_config['theta_dot_limit'],
                      action_scale=env_config['action_scale'])

    ctrl = make('lqr',
                env_func,
                q_lqr=env_config['q_lqr'],
                r_lqr=env_config['r_lqr'],
                discrete_dynamics=True)

    env = env_func()

    # Set termination thresholds
    env.x_threshold = env_config['x_threshold']
    env.x_dot_threshold = env_config['x_dot_threshold']
    env.theta_threshold_radians = env_config['theta_threshold_radians']
    env.theta_dot_threshold = env_config['theta_dot_threshold']

    # Initialize statistics for this batch
    batch_stats = {
        'x': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'theta': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'x_dot': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'theta_dot': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'success_count': 0,
        'total_count': 0,
        'timeout_count': 0,
        'max_traj_length': 0,
        'roa_labels': []  # List of (init_state, label) tuples
    }

    # Process each trajectory in the batch
    for idx, init_state in batch_data:
        # Run trajectory
        trajectory, success, timeout = run_trajectory(env, ctrl, init_state, env_config['max_steps'])

        # Update success and timeout tracking
        batch_stats['total_count'] += 1
        if success:
            batch_stats['success_count'] += 1
        if timeout:
            batch_stats['timeout_count'] += 1

        # Store ROA label: 1 for success, 0 for failure (timeout or out of bounds)
        x, x_dot, theta, theta_dot = init_state
        init_state_normalized = [x, normalize_angle(theta), x_dot, theta_dot]
        roa_label = 1 if success else 0
        batch_stats['roa_labels'].append((init_state_normalized, roa_label))

        # Update trajectory length tracking
        batch_stats['max_traj_length'] = max(batch_stats['max_traj_length'], len(trajectory))

        # Update statistics with previous state tracking
        traj_array = np.array(trajectory)

        # Include all states (including initial state) in statistics
        if len(traj_array) > 0:
            # For each state variable, track min/max and the previous state
            state_vars = [('x', 0), ('theta', 1), ('x_dot', 2), ('theta_dot', 3)]
            for var_name, col_idx in state_vars:
                # Find min value and its index
                traj_min = traj_array[:, col_idx].min()
                if traj_min < batch_stats[var_name]['min']:
                    batch_stats[var_name]['min'] = traj_min
                    min_idx = traj_array[:, col_idx].argmin()
                    # Store previous state (None if this is the initial state)
                    batch_stats[var_name]['prev_at_min'] = traj_array[min_idx - 1].tolist() if min_idx > 0 else None

                # Find max value and its index
                traj_max = traj_array[:, col_idx].max()
                if traj_max > batch_stats[var_name]['max']:
                    batch_stats[var_name]['max'] = traj_max
                    max_idx = traj_array[:, col_idx].argmax()
                    # Store previous state (None if this is the initial state)
                    batch_stats[var_name]['prev_at_max'] = traj_array[max_idx - 1].tolist() if max_idx > 0 else None

        # Save trajectory (only if skip_save is False)
        if not skip_save:
            filepath = os.path.join(output_dir, f'sequence_{idx}.txt')
            save_trajectory(trajectory, filepath)

    # Clean up
    env.close()
    ctrl.close()

    return batch_stats


def main():
    parser = argparse.ArgumentParser(description='Generate cartpole trajectory dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save trajectory files')
    parser.add_argument('--resolution', type=float, default=0.05,
                        help='Discretization resolution (default: 0.05)')
    parser.add_argument('--x_bound', type=float, default=4.8,
                        help='Symmetric bound for x position, range: [-x_bound, +x_bound] (default: 4.8)')
    parser.add_argument('--x_dot_bound', type=float, default=20.0,
                        help='Symmetric bound for x velocity, range: [-x_dot_bound, +x_dot_bound] (default: 20.0)')
    parser.add_argument('--theta_bound', type=float, default=np.pi,
                        help='Symmetric bound for theta angle, range: [-theta_bound, +theta_bound] (default: pi)')
    parser.add_argument('--theta_dot_bound', type=float, default=20.0,
                        help='Symmetric bound for theta velocity, range: [-theta_dot_bound, +theta_dot_bound] (default: 20.0)')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per trajectory (default: 500)')
    parser.add_argument('--episode_len_sec', type=int, default=10,
                        help='Episode length in seconds (default: 10)')
    parser.add_argument('--x_termination', type=float, default=None,
                        help='Termination threshold for x position (default: copies x_bound)')
    parser.add_argument('--x_dot_termination', type=float, default=float('inf'),
                        help='Termination threshold for x velocity (default: inf)')
    parser.add_argument('--theta_termination', type=float, default=float('inf'),
                        help='Termination threshold for theta angle (default: inf)')
    parser.add_argument('--theta_dot_termination', type=float, default=float('inf'),
                        help='Termination threshold for theta velocity (default: inf)')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel processing using multiple CPU cores (default: False, sequential)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes for parallel execution (default: all available CPUs)')
    parser.add_argument('--control_bound', type=float, default=10.0,
                        help='Control force bound in Newtons, action clipped to [-bound, +bound] (default: 10.0 N)')
    parser.add_argument('--save_freq', type=float, default=0.01,
                        help='Frequency in seconds at which to save trajectory states. '
                             'The control and physics integration frequencies will be automatically adjusted '
                             'to match or exceed this frequency for accurate state computation. (default: 0.01 = 100 Hz)')
    parser.add_argument('--skip_save', action='store_true',
                        help='Skip saving trajectory files to disk. Trajectories will still be generated and statistics computed. (default: False)')

    args = parser.parse_args()

    if args.x_termination is None:
        args.x_termination = args.x_bound

    # Create output directory structure
    # Trajectories will be stored in trajectories/ subdirectory
    # ROA labels will be stored in roa_labels.txt in the parent directory
    trajectories_dir = os.path.join(args.output_dir, 'trajectories')
    if not args.skip_save:
        os.makedirs(trajectories_dir, exist_ok=True)
    else:
        # Even if skip_save, we need the output dir for roa_labels.txt
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Define bounds (symmetric around zero)
    bounds = {
        'x': args.x_bound,
        'x_dot': args.x_dot_bound,
        'theta': args.theta_bound,
        'theta_dot': args.theta_dot_bound
    }

    # Define termination thresholds
    termination_thresholds = {
        'x': args.x_termination,
        'x_dot': args.x_dot_termination,
        'theta': args.theta_termination,
        'theta_dot': args.theta_dot_termination
    }

    # Generate discretized initial states
    print("Generating discretized initial states...")
    initial_states = generate_discretized_initial_states(bounds, args.resolution, termination_thresholds)
    print(f"Generated {len(initial_states)} initial states (excluding those that violate termination bounds)")

    # Calculate control frequency based on save_freq
    # Control frequency should be at least as high as save frequency to avoid duplicating states
    default_ctrl_freq = 15  # Hz
    min_ctrl_freq_for_save = 1.0 / args.save_freq  # Hz required for save_freq
    ctrl_freq = max(default_ctrl_freq, min_ctrl_freq_for_save)

    # Adjust control timestep to be compatible with save_freq
    ctrl_timestep = 1.0 / ctrl_freq

    # PyBullet frequency should be high enough for accurate physics
    # Use at least 50 steps per control step for accuracy
    pyb_freq = int(ctrl_freq * 50)

    # Override velocity bounds to remove clipping (set to inf)
    args.x_dot_bound = float('inf')
    args.theta_dot_bound = float('inf')

    # Prepare environment configuration for workers
    env_config = {
        'task': 'stabilization',
        'ctrl_freq': ctrl_freq,
        'pyb_freq': pyb_freq,
        'episode_len_sec': args.episode_len_sec,
        'done_on_out_of_bound': True,
        'cost': 'quadratic',
        'x_dot_limit': args.x_dot_bound,  # Now = inf (no clipping)
        'theta_dot_limit': args.theta_dot_bound,  # Now = inf (no clipping)
        'action_scale': args.control_bound,
        'q_lqr': [1, 1, 1, 1],
        'r_lqr': [0.1],
        'x_threshold': args.x_termination,
        'x_dot_threshold': args.x_dot_termination,
        'theta_threshold_radians': args.theta_termination,
        'theta_dot_threshold': args.theta_dot_termination,
        'max_steps': args.max_steps
    }

    print(f"Termination thresholds: x=±{args.x_termination}, x_dot=±{args.x_dot_termination}, "
          f"theta=±{args.theta_termination}, theta_dot=±{args.theta_dot_termination}")
    print(f"Velocity limits (clipping dynamics): x_dot=±{args.x_dot_bound}, theta_dot=±{args.theta_dot_bound}")
    print(f"Control bounds: u=±{args.control_bound} N")
    print(f"Save frequency: {args.save_freq} s ({1.0/args.save_freq:.1f} Hz)")
    print(f"Control frequency: {ctrl_freq:.1f} Hz (timestep: {ctrl_timestep:.6f} s)")
    print(f"Physics frequency: {pyb_freq} Hz (timestep: {1.0/pyb_freq:.6f} s)")

    # Initialize statistics tracking
    stats = {
        'x': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'theta': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'x_dot': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'theta_dot': {'min': float('inf'), 'max': float('-inf'), 'prev_at_min': None, 'prev_at_max': None},
        'success_count': 0,
        'total_count': 0,
        'timeout_count': 0,
        'max_traj_length': 0
    }

    # Collect ROA labels for all trajectories
    roa_labels = []  # List of (init_state, label) tuples

    if args.parallel:
        # Parallel execution
        # Get number of CPUs available (respecting taskset/affinity)
        num_workers = args.num_workers if args.num_workers else get_available_cpus()

        print(f"Generating trajectories using {num_workers} CPU cores (parallel mode)...")

        # Create arguments for each individual trajectory
        trajectory_args = [
            (idx, initial_states[idx], env_config, trajectories_dir, args.skip_save)
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
        for traj_stats in traj_results:
            for key in ['x', 'theta', 'x_dot', 'theta_dot']:
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

            # Collect ROA label
            roa_labels.append((traj_stats['init_state'], traj_stats['roa_label']))

    else:
        # Sequential execution
        print(f"Generating trajectories sequentially (single core)...")

        # Create environment and controller once for sequential execution
        env_func = partial(make,
                          'cartpole',
                          task=env_config['task'],
                          ctrl_freq=env_config['ctrl_freq'],
                          pyb_freq=env_config['pyb_freq'],
                          episode_len_sec=env_config['episode_len_sec'],
                          done_on_out_of_bound=env_config['done_on_out_of_bound'],
                          cost=env_config['cost'],
                          gui=False,
                          randomized_init=False,
                          x_dot_limit=env_config['x_dot_limit'],
                          theta_dot_limit=env_config['theta_dot_limit'],
                          action_scale=env_config['action_scale'])

        ctrl = make('lqr',
                    env_func,
                    q_lqr=env_config['q_lqr'],
                    r_lqr=env_config['r_lqr'],
                    discrete_dynamics=True)

        env = env_func()

        # Set termination thresholds
        env.x_threshold = env_config['x_threshold']
        env.x_dot_threshold = env_config['x_dot_threshold']
        env.theta_threshold_radians = env_config['theta_threshold_radians']
        env.theta_dot_threshold = env_config['theta_dot_threshold']

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

            # Store ROA label: 1 for success, 0 for failure (timeout or out of bounds)
            x, x_dot, theta, theta_dot = init_state
            init_state_normalized = [x, normalize_angle(theta), x_dot, theta_dot]
            roa_label = 1 if success else 0
            roa_labels.append((init_state_normalized, roa_label))

            # Update trajectory length tracking
            stats['max_traj_length'] = max(stats['max_traj_length'], len(trajectory))

            # Update statistics with previous state tracking
            traj_array = np.array(trajectory)

            # Include all states (including initial state) in statistics
            if len(traj_array) > 0:
                # For each state variable, track min/max and the previous state
                state_vars = [('x', 0), ('theta', 1), ('x_dot', 2), ('theta_dot', 3)]
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
            # Format: x,theta,x_dot,theta_dot,label
            line = ','.join([f'{val:.6f}' for val in init_state] + [str(label)])
            f.write(line + '\n')

    if args.skip_save:
        print(f"\nSuccessfully generated {len(initial_states)} trajectories (files not saved)")
        print(f"ROA labels saved to: {roa_labels_path}")
    else:
        print(f"\nSuccessfully generated {len(initial_states)} trajectories in {trajectories_dir}")
        print(f"Each file contains a trajectory with states in format: x,theta,x_dot,theta_dot")
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
        return f"[x={state[0]:>7.3f}, θ={state[1]:>7.3f}, ẋ={state[2]:>7.3f}, θ̇={state[3]:>7.3f}]"

    for var_name, var_label in [('x', 'x'), ('theta', 'theta'), ('x_dot', 'x_dot'), ('theta_dot', 'theta_dot')]:
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