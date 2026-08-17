'''Gymnasium API conformance for every registered environment.

SB3's env_checker validates the contract directly -- tuple arity, reset
signature, space conformance, dtypes -- rather than inferring correctness from
tests that happen to pass. It is the primary evidence the migration is correct,
and it covers the systems that have no golden fixtures.
'''
import pytest
from stable_baselines3.common.env_checker import check_env

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
             'stabilization_goal_tolerance': 0.0,
         }})]


@pytest.mark.parametrize('task,cfg', TASKS)
def test_check_env(task, cfg):
    env = make(task, **cfg)
    check_env(env, warn=True, skip_render_check=True)
    env.close()
