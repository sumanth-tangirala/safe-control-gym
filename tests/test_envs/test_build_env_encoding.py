'''Cross-regime evaluation must not change what the policy sees.

The observation encoding -- normalisation scales, angle channels -- is part of
the trained model. `build_env(config, regime=...)` exists so a policy can be
SCORED in a different regime than it trained in; the regime may move where
episodes die and where they start, and nothing else. Before the ordering fix,
the regime's kill box reached state_space ahead of the normalisation wrapper
and re-scaled the velocity channels 20 -> 5: the same physical state produced
a 4x larger x_dot observation than the policy was trained on, and every
cross-regime score was of a policy fed distorted input.
'''
import os

import numpy as np
import pytest
from munch import Munch

from safe_control_gym.experiments.train_sb3 import build_env, load_collection_bounds

REPO = os.path.join(os.path.dirname(__file__), '..', '..')

CONFIG = Munch(
    task='cartpole_stabilization',
    task_config=Munch(cost='shaped_dmc', episode_len_sec=10, terminate_on_goal=False),
    sb3_config=Munch(collection_bounds=os.path.join(REPO, 'configs/physical/cartpole.yaml')),
)
COLLECTION = os.path.join(REPO, 'configs/collection/cartpole.yaml')
STATE = np.array([3.0, 2.0, 0.0, 2.0])


def wrapped_observation(env, state):
    base = env.unwrapped
    base.state = np.array(state)
    obs = base._get_observation()
    stack = []
    e = env
    while hasattr(e, 'env'):
        stack.append(e)
        e = e.env
    for w in reversed(stack):
        if hasattr(w, 'observation'):
            obs = w.observation(obs)
    return obs


@pytest.fixture(scope='module')
def envs():
    train_env = build_env(CONFIG)
    eval_env = build_env(CONFIG, regime=load_collection_bounds(COLLECTION))
    train_env.reset(seed=0)
    eval_env.reset(seed=0)
    yield train_env, eval_env
    train_env.close()
    eval_env.close()


def test_regime_override_keeps_the_training_encoding(envs):
    train_env, eval_env = envs
    train_obs = wrapped_observation(train_env, STATE)
    eval_obs = wrapped_observation(eval_env, STATE)
    np.testing.assert_allclose(eval_obs, train_obs, err_msg=(
        'the evaluation regime changed the observation encoding; '
        'normalisation scales are part of the trained model'))


def test_regime_override_moves_the_kill_box(envs):
    train_env, eval_env = envs
    # The collection regime kills at |x_dot| >= 5; the physical regime does not.
    assert eval_env.unwrapped.x_dot_threshold == 5.0
    assert train_env.unwrapped.x_dot_threshold == 1000000.0
