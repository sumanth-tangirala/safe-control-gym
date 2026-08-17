'''Episode state flags must exist before anything reads them.

`_get_info()` reads `goal_reached` and `out_of_bounds`, but `_get_done()`
returns early when the goal is reached -- so on a step where the goal is hit
before the out-of-bounds branch runs, `out_of_bounds` was never assigned and
`_get_info()` raised `AttributeError`.

CartPole had this hole; Quadrotor and InvertedPendulum already initialised the
flags in `reset()`. Nothing in the suite exercised the path, which is why it
survived. These tests exercise it directly.
'''
import numpy as np
import pytest

from safe_control_gym.utils.registration import make

# A goal tolerance this large means the goal is 'reached' on the very first
# step, so _get_done() takes its early return before touching out_of_bounds.
GOAL_REACHED_IMMEDIATELY = 1e9


def test_cartpole_info_after_immediate_goal():
    '''Regression: CartPole raised AttributeError on out_of_bounds here.'''
    env = make('cartpole', task='stabilization',
               task_info={'stabilization_goal': [0],
                          'stabilization_goal_tolerance': GOAL_REACHED_IMMEDIATELY})
    try:
        env.reset(seed=0)
        # Must not raise. The assertion is that step() completes at all.
        env.step(np.zeros(1))
    finally:
        env.close()


@pytest.mark.parametrize('task,cfg', [
    ('cartpole', {}),
    ('inverted_pendulum', {}),
    ('quadrotor', {'quad_type': 2}),
    ('quadrotor', {'quad_type': 3,
                   'task_info': {'stabilization_goal': [0, 0, 1],
                                 'stabilization_goal_tolerance': 0.0}}),
])
def test_flags_exist_immediately_after_reset(task, cfg):
    '''The flags must be readable before any step, not only after one.'''
    env = make(task, **cfg)
    try:
        env.reset(seed=0)
        assert hasattr(env, 'goal_reached'), \
            f'{task}: goal_reached unset after reset; _get_info() reads it'
        if getattr(env, 'done_on_out_of_bound', False):
            assert hasattr(env, 'out_of_bounds'), \
                f'{task}: out_of_bounds unset after reset; _get_info() reads it'
    finally:
        env.close()
