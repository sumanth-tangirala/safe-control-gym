'''terminated/truncated must agree with the legacy info key, on every step.

The six RL controllers already compensated for time truncation via
info['TimeLimit.truncated']. The new flags formalise that; they must not
disagree with it, or the compensation silently changes meaning.
'''
import numpy as np
import pybullet as p
import pytest

from safe_control_gym.utils.registration import make

# THREE_D needs a 3-element stabilization_goal; the env's own default
# (`TASK_INFO['stabilization_goal'] = [0, 1]`) is 2D-only and raises
# IndexError on X_GOAL construction otherwise. `task_info` replaces the whole
# dict (not a merge), so the rest of the class default is carried over
# explicitly -- same override the golden quadrotor_3d_rollouts.json fixture
# uses.
_QUAD_3D_TASK_INFO = {
    'stabilization_goal': [0, 0, 1],
    'stabilization_goal_tolerance': 0.05,
    'trajectory_type': 'circle',
    'num_cycles': 1,
    'trajectory_plane': 'zx',
    'trajectory_position_offset': [0.5, 0],
    'trajectory_scale': -0.5,
    'proj_point': [0, 0, 0.5],
    'proj_normal': [0, 1, 1],
}

TASKS = [('inverted_pendulum', {}), ('cartpole', {}),
         ('quadrotor', {'quad_type': 2}),
         ('quadrotor', {'quad_type': 3, 'task_info': _QUAD_3D_TASK_INFO})]

# task_info copies of each class's own TASK_INFO default, but with
# stabilization_goal_tolerance forced negative so `goal_reached` in
# _get_done() can never be satisfied (the norm being compared is >= 0). This
# isolates truncation as the only way the episode below can end, independent
# of the random reset draw or of drift under the benign action.
_CARTPOLE_TASK_INFO_NO_GOAL = {
    'stabilization_goal': [0],
    'stabilization_goal_tolerance': -1.0,
    'trajectory_type': 'circle',
    'num_cycles': 1,
    'trajectory_plane': 'zx',
    'trajectory_position_offset': [0, 0],
    'trajectory_scale': 0.2,
}
_QUAD_2D_TASK_INFO_NO_GOAL = {
    'stabilization_goal': [0, 1],
    'stabilization_goal_tolerance': -1.0,
    'trajectory_type': 'circle',
    'num_cycles': 1,
    'trajectory_plane': 'zx',
    'trajectory_position_offset': [0.5, 0],
    'trajectory_scale': -0.5,
    'proj_point': [0, 0, 0.5],
    'proj_normal': [0, 1, 1],
}
_QUAD_3D_TASK_INFO_NO_GOAL = dict(_QUAD_3D_TASK_INFO, stabilization_goal_tolerance=-1.0)

# Configs for the truncation-agreement test below. `done_on_out_of_bound` is
# turned off (cartpole/quadrotor only -- inverted_pendulum has no such flag,
# it only ever ends on goal or time limit) and the goal tolerance is made
# unsatisfiable, so the only way any of these episodes can end is the time
# limit. `episode_len_sec=1` keeps CTRL_STEPS to 50 (at the default 50 Hz
# ctrl_freq) since the quadrotor envs only step at roughly 250/s wall clock.
TRUNCATION_TASKS = [
    ('inverted_pendulum', {'episode_len_sec': 1}),
    ('cartpole', {'episode_len_sec': 1, 'done_on_out_of_bound': False,
                  'task_info': _CARTPOLE_TASK_INFO_NO_GOAL}),
    ('quadrotor', {'quad_type': 2, 'episode_len_sec': 1, 'done_on_out_of_bound': False,
                   'task_info': _QUAD_2D_TASK_INFO_NO_GOAL}),
    ('quadrotor', {'quad_type': 3, 'episode_len_sec': 1, 'done_on_out_of_bound': False,
                   'task_info': _QUAD_3D_TASK_INFO_NO_GOAL}),
]


@pytest.mark.parametrize('task,cfg', TRUNCATION_TASKS)
def test_truncated_agrees_with_legacy_info_key(task, cfg):
    env = make(task, **cfg)
    env.reset(seed=7)
    # A fixed, benign action -- the action-space midpoint, which is zero
    # force for cartpole/inverted_pendulum and near-hover thrust for
    # quadrotor -- rather than a random one, so the episode reliably survives
    # to the time limit instead of a random action tripping goal/bounds
    # termination first.
    action = (env.action_space.low + env.action_space.high) / 2
    saw_truncation = False
    for _ in range(env.CTRL_STEPS + 5):
        _, _, terminated, truncated, info = env.step(action)
        if 'TimeLimit.truncated' in info:
            saw_truncation = True
            assert truncated is True or truncated == 1
            assert info['TimeLimit.truncated'] == (not terminated)
        if terminated or truncated:
            break
    assert saw_truncation, \
        'episode never truncated at CTRL_STEPS -- goal-reached or out-of-bounds masked the time limit'
    env.close()


@pytest.mark.parametrize('task,cfg', TASKS)
def test_flags_are_booleans(task, cfg):
    env = make(task, **cfg)
    env.reset(seed=3)
    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    env.close()


def test_terminated_and_truncated_can_co_occur():
    '''Goal reached on exactly the horizon step must set both flags.

    The two conditions are computed independently -- termination in
    _get_done(), truncation from ctrl_step_counter in after_step -- so neither
    may mask the other. A migration that returns `terminated or truncated` from
    one slot and False from the other passes every other test here.
    '''
    env = make('inverted_pendulum')
    env.reset(seed=11)
    # Park the state inside the goal ball and wind the counter to the horizon,
    # so this step both terminates and truncates.
    env.state = np.array(env.X_GOAL, dtype=np.float64).copy()
    env.ctrl_step_counter = env.CTRL_STEPS - 1
    _, _, terminated, truncated, info = env.step(np.zeros(1))
    assert bool(terminated) is True, 'goal state must terminate'
    assert bool(truncated) is True, 'horizon step must truncate'
    # The legacy key is `not terminated`, so a genuine co-occurrence records False.
    assert info['TimeLimit.truncated'] is False
    env.close()


@pytest.mark.parametrize('task,cfg', [
    ('cartpole', {}),
    ('quadrotor', {'quad_type': 2}),
    ('quadrotor', {'quad_type': 3, 'task_info': _QUAD_3D_TASK_INFO}),
])
def test_terminated_and_truncated_can_co_occur_pybullet(task, cfg):
    '''Same co-occurrence guard as test_terminated_and_truncated_can_co_occur,
    for the PyBullet-backed systems, which have the same two-source
    termination structure (goal-reached or out-of-bounds) racing the horizon.

    Cartpole/quadrotor read `self.state` back from the physics client on
    every step (see test_env_rollouts.py's module docstring), so assigning
    `env.state = ...` -- as the symbolic inverted_pendulum test above does --
    does not seed the next step(). The goal state is instead written directly
    into PyBullet via resetJointState / resetBasePositionAndOrientation,
    mirroring what each env's own reset() does.
    '''
    env = make(task, **cfg)
    env.reset(seed=11)
    if task == 'cartpole':
        # cartpole's _get_done() returns on goal_reached before ever
        # initialising self.out_of_bounds if that happens on the very first
        # call after __init__/reset(); one ordinary step first sidesteps
        # that. Pre-existing, unrelated to this migration -- not fixed here.
        env.step(np.zeros(1))
        p.resetJointState(env.CARTPOLE_ID, jointIndex=0,
                          targetValue=float(env.X_GOAL[0]), targetVelocity=float(env.X_GOAL[1]),
                          physicsClientId=env.PYB_CLIENT)
        p.resetJointState(env.CARTPOLE_ID, jointIndex=1,
                          targetValue=float(env.X_GOAL[2]), targetVelocity=float(env.X_GOAL[3]),
                          physicsClientId=env.PYB_CLIENT)
        action = np.zeros(1)
    else:
        goal = env.X_GOAL
        if len(goal) == 6:  # 2D: x, x_dot, z, z_dot, theta, theta_dot.
            x, x_dot, z, z_dot, theta, theta_dot = goal
            pos, vel = [x, 0, z], [x_dot, 0, z_dot]
            rpy, ang_vel = [0, theta, 0], [0, theta_dot, 0]
        else:  # 3D: x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r.
            x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, pr, qr, rr = goal
            pos, vel = [x, y, z], [x_dot, y_dot, z_dot]
            rpy, ang_vel = [phi, theta, psi], [pr, qr, rr]
        p.resetBasePositionAndOrientation(env.DRONE_ID, pos, p.getQuaternionFromEuler(rpy),
                                          physicsClientId=env.PYB_CLIENT)
        p.resetBaseVelocity(env.DRONE_ID, vel, ang_vel, physicsClientId=env.PYB_CLIENT)
        # Near-exact hover thrust: the box-midpoint action used elsewhere in
        # this file perturbs velocity by more than the 0.05 goal tolerance
        # over even a single control step and would mask the co-occurrence.
        action = np.full(env.action_dim, env.GRAVITY_ACC * env.MASS / env.action_dim)
    env.ctrl_step_counter = env.CTRL_STEPS - 1
    _, _, terminated, truncated, info = env.step(action)
    assert bool(terminated) is True, 'goal state must terminate'
    assert bool(truncated) is True, 'horizon step must truncate'
    assert info['TimeLimit.truncated'] is False
    env.close()
