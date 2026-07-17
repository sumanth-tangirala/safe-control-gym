'''Integration tests: env + controllers must compose through the make() registry
and produce sensible closed-loop behavior.'''

import json
import math
import os
from functools import partial

import numpy as np

from safe_control_gym.utils.registration import make, get_config
from safe_control_gym.envs.gym_control.inverted_pendulum import InvertedPendulum
from safe_control_gym.controllers.pendulum_lqr.pendulum_lqr import PendulumLQR
from safe_control_gym.controllers.pendulum_rl.pendulum_rl import PendulumRL

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')
GOLDEN_K = json.load(open(os.path.join(FIX, 'lqr_gain.json')))['K']


def env_func_factory(**overrides):
    cfg = dict(get_config('inverted_pendulum'))
    cfg.update(dict(randomized_init=False))
    cfg.update(overrides)
    return partial(make, 'inverted_pendulum', **cfg)


def test_env_registered_and_configured():
    env = make('inverted_pendulum')
    assert isinstance(env, InvertedPendulum)
    cfg = get_config('inverted_pendulum')
    assert cfg['ctrl_freq'] == 100 and cfg['pyb_freq'] == 100
    env.close()


def test_make_pendulum_lqr_matches_golden():
    ctrl = make('pendulum_lqr', env_func_factory(), **dict(get_config('pendulum_lqr')))
    assert isinstance(ctrl, PendulumLQR)
    np.testing.assert_allclose(np.asarray(ctrl.K).reshape(-1), GOLDEN_K, atol=1e-6)
    ctrl.close()


def test_make_pendulum_rl_loads_bundled_policy():
    rl_cfg = dict(get_config('pendulum_rl'))
    rl_cfg['model_path'] = 'v1_strong'
    ctrl = make('pendulum_rl', env_func_factory(), **rl_cfg)
    assert isinstance(ctrl, PendulumRL)
    u = ctrl.select_action(np.array([2.0, 0.0]))
    assert np.asarray(u).shape == (1,)
    ctrl.close()


def test_lqr_stabilizes_from_inside_roa():
    env_func = env_func_factory(init_state=[0.15, 0.0])
    ctrl = make('pendulum_lqr', env_func, **dict(get_config('pendulum_lqr')))
    env = env_func()
    obs, info = env.reset()
    done = False
    for _ in range(1000):
        obs, _, done, info = env.step(ctrl.select_action(obs, info))
        if done:
            break
    assert info.get('goal_reached') is True, 'LQR from inside the ROA should reach upright'
    env.close()
    ctrl.close()


def test_rl_swings_up_from_below():
    env_func = env_func_factory(init_state=[2.5, 0.0])  # well away from upright
    rl_cfg = dict(get_config('pendulum_rl'))
    rl_cfg['model_path'] = 'v1_strong'
    ctrl = make('pendulum_rl', env_func, **rl_cfg)
    env = env_func()
    obs, info = env.reset()
    ctrl.reset()
    reached, min_dist = False, np.inf
    for _ in range(1000):
        obs, _, done, info = env.step(ctrl.select_action(obs, info))
        min_dist = min(min_dist, float(np.linalg.norm(env.state)))
        if done and info.get('goal_reached'):
            reached = True
            break
    assert reached, f'v1_strong swing-up should reach upright (min dist to goal was {min_dist:.3f})'
    env.close()
    ctrl.close()
