'''terminated/truncated must agree with the legacy info key, on every step.

The six RL controllers already compensated for time truncation via
info['TimeLimit.truncated']. The new flags formalise that; they must not
disagree with it, or the compensation silently changes meaning.
'''
import numpy as np
import pytest

from safe_control_gym.utils.registration import make

TASKS = [('inverted_pendulum', {}), ('cartpole', {}),
         ('quadrotor', {'quad_type': 2}),
         # THREE_D needs a 3-element stabilization_goal; the env's own default
         # (`TASK_INFO['stabilization_goal'] = [0, 1]`) is 2D-only and raises
         # IndexError on X_GOAL construction otherwise. `task_info` replaces
         # the whole dict (not a merge), so the rest of the class default is
         # carried over explicitly -- same override the golden
         # quadrotor_3d_rollouts.json fixture uses.
         ('quadrotor', {'quad_type': 3, 'task_info': {
             'stabilization_goal': [0, 0, 1],
             'stabilization_goal_tolerance': 0.05,
             'trajectory_type': 'circle',
             'num_cycles': 1,
             'trajectory_plane': 'zx',
             'trajectory_position_offset': [0.5, 0],
             'trajectory_scale': -0.5,
             'proj_point': [0, 0, 0.5],
             'proj_normal': [0, 1, 1],
         }})]


@pytest.mark.parametrize('task,cfg', TASKS)
def test_truncated_agrees_with_legacy_info_key(task, cfg):
    env = make(task, **cfg)
    env.reset(seed=7)
    rng = np.random.default_rng(7)
    saw_truncation = False
    for _ in range(env.CTRL_STEPS + 5):
        act = rng.uniform(env.action_space.low, env.action_space.high)
        _, _, terminated, truncated, info = env.step(act)
        if 'TimeLimit.truncated' in info:
            saw_truncation = True
            assert truncated is True or truncated == 1
            assert info['TimeLimit.truncated'] == (not terminated)
        if terminated or truncated:
            break
    assert saw_truncation or terminated, \
        'episode neither truncated nor terminated within CTRL_STEPS + 5'
    env.close()


@pytest.mark.parametrize('task,cfg', TASKS)
def test_flags_are_booleans(task, cfg):
    env = make(task, **cfg)
    env.reset(seed=3)
    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
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
