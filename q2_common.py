'''Shared quad2d (RL) setup: reproduce the deterministic set, then sweep noise.

Everything is transcribed from deterministic/quadrotor2D_rl/dataset_description.json
and generate_quadrotor_2d_trajectories_rl.py.

Differences from quad3d that matter for calibration:
  controller  safe_explorer_ppo, not LQR. Needs obs_normalizer applied manually
              each step and info['constraint_values'] seeded before step 1.
  tolerance   0.2, not 0.05 -- 4x looser, so more wobble is tolerated.
  horizon     max_steps 1200 (real), vs quad3d's effectively unbounded 100k.
              Longest shipped trajectory is 709 and timeouts are 0.
  bounds      much tighter: x +/-1.0, z [0.1,1.5], velocities +/-1.0, theta_dot
              +/-8.0. Less room before ejection.
  rates       for TWO_D the env stores ang_v[1] directly (world), with no body
              conversion, so injection is symmetric -- none of the quad3d
              body/world asymmetry applies here.
  sampling    stratified grid, and eval == train == the same 489,789 states.

The two tighter-bounds / looser-goal effects push success in opposite
directions under noise, which is why the level range has to be measured rather
than scaled from quad3d.
'''
import os
import sys
import tempfile

import numpy as np
import pybullet as pb

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from functools import partial  # noqa: E402

from generate_quadrotor_2d_trajectories_rl import (ALGO_CONFIGS, SAFE_EXPLORER_CONSTRAINTS,  # noqa: E402
                                                   normalize_angle)
from safe_control_gym.utils.registration import make  # noqa: E402

DET = os.environ.get(
    'Q2_DET_DIR',
    '/common/users/shared/pracsys/genMoPlan/data_trajectories/'
    'deterministic/quadrotor2D_rl')
MODEL = os.environ.get(
    'Q2_MODEL',
    'examples/rl/models/safe_explorer_ppo/safe_explorer_ppo_model_quadrotor_2D_stab.pt')

HORIZON = 1200                     # description: simulation_parameters.max_steps
TOL = 0.2                          # description: success_criteria.threshold
GOAL = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])   # file order [x,z,theta,xd,zd,td]

# env state order is [x, x_dot, z, z_dot, theta, theta_dot]
STATE_BOUNDS = {0: (-1.0, 1.0), 1: (-1.0, 1.0), 2: (0.1, 1.5),
                3: (-1.0, 1.0), 5: (-8.0, 8.0)}      # index 4 (theta) stays infinite
TASK_INFO = {'stabilization_goal': [0, 1], 'stabilization_goal_tolerance': TOL}


def rollout_seed(base, index, trial):
    return int((base + index * 7919 + trial * 104_729) % (2 ** 31 - 1))


def build(level, mechanism='dynamics'):
    kw = dict(quad_type=2, task='stabilization', task_info=TASK_INFO,
              ctrl_freq=100, pyb_freq=5000, gui=False, randomized_init=False,
              episode_len_sec=1000, cost='quadratic', done_on_out_of_bound=True,
              normalized_rl_action_space=True,
              constraints=SAFE_EXPLORER_CONSTRAINTS, done_on_violation=False)
    if level > 0:
        # TWO_D dynamics disturbance is 2-D and enters as [Fx, 0, Fz]: a planar
        # world-frame force at the COM, so no torque, exactly as in quad3d.
        kw['disturbances'] = {mechanism: [{'disturbance_func': 'uniform',
                                           'low': -level, 'high': level}]}
    env_func = partial(make, 'quadrotor', **kw)
    cfg = ALGO_CONFIGS['safe_explorer_ppo'].copy()
    tmp = tempfile.mkdtemp(prefix='q2ctrl-')
    ctrl = make('safe_explorer_ppo', env_func, **cfg, output_dir=tmp)
    ctrl.load(MODEL)
    ctrl.obs_normalizer.set_read_only()      # freeze normalizer stats
    env = env_func()
    for i, (lo, hi) in STATE_BOUNDS.items():
        env.state_space.low[i], env.state_space.high[i] = lo, hi
    return env, ctrl


def roll(env, ctrl, state6, seed, keep=False):
    '''state6 is FILE order [x, z, theta, x_dot, z_dot, theta_dot].'''
    x, z, theta, x_dot, z_dot, theta_dot = state6
    obs, info = env.reset(seed=int(seed))
    pb.resetBasePositionAndOrientation(
        env.DRONE_ID, [x, 0, z], pb.getQuaternionFromEuler([0, theta, 0]),
        physicsClientId=env.PYB_CLIENT)
    pb.resetBaseVelocity(env.DRONE_ID, [x_dot, 0, z_dot], [0, theta_dot, 0],
                         physicsClientId=env.PYB_CLIENT)
    env._update_and_store_kinematic_information()
    obs = env._get_observation()
    # safe_explorer_ppo reads this internally; _get_info() can fail pre-step.
    if getattr(env, 'constraints', None) is not None:
        info['constraint_values'] = env.constraints.get_values(env, only_state=True)
    traj = [[x, z, normalize_angle(theta), x_dot, z_dot, theta_dot]] if keep else None
    success = False
    steps = 0
    for steps in range(1, HORIZON + 1):
        action = ctrl.select_action(ctrl.obs_normalizer(obs), info)
        obs, _, terminated, truncated, info = env.step(action)
        if keep:
            xx, xd, zz, zd, th, td = obs[:6]
            traj.append([xx, zz, normalize_angle(th), xd, zd, td])
        if terminated or truncated:
            success = bool(info.get('goal_reached', False))
            break
    return success, steps, traj


def grid_states(lo, hi):
    '''Rows [lo:hi) of roa_labels.txt: 6 state columns plus the shipped label.

    roa_labels is 1:1 with trajectories here -- row i is sequence_i's start --
    unlike quad3d, where eval_states held intermediate states too.
    '''
    rows = np.loadtxt(DET + '/roa_labels.txt', delimiter=',',
                      skiprows=lo, max_rows=hi - lo, ndmin=2)
    return rows[:, 0:6], rows[:, 6].astype(int)
