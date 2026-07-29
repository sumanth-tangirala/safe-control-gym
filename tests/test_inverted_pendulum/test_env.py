'''Fidelity + contract tests for the InvertedPendulum environment.

The golden fixtures in ``fixtures/env_rollouts.json`` were produced by the
source inverted_pendulum system's own env (see
``scripts/extract_pendulum_rl_policies.py``). The safe-control-gym port must
reproduce those trajectories exactly.
'''

import json
import math
import os

import numpy as np
import pytest

from safe_control_gym.envs.gym_control.inverted_pendulum import InvertedPendulum

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')


def load(name):
    with open(os.path.join(FIX, name)) as f:
        return json.load(f)


ROLLOUTS = load('env_rollouts.json')
SCENARIO_NAMES = [s['name'] for s in ROLLOUTS['scenarios']]


def make_env(**kwargs):
    # pyb_freq == ctrl_freq == 100 -> one Euler substep of dt=0.01 per step,
    # matching the source env's per-dt integrate/wrap/clip/goal cadence.
    cfg = dict(ctrl_freq=100, pyb_freq=100, episode_len_sec=10,
               randomized_init=False, cost='quadratic')
    cfg.update(kwargs)
    return InvertedPendulum(**cfg)


@pytest.mark.parametrize('name', SCENARIO_NAMES)
def test_dynamics_matches_source_env(name):
    scenario = next(s for s in ROLLOUTS['scenarios'] if s['name'] == name)
    env = make_env()
    env.reset()
    env.state = np.array(scenario['x0'], dtype=np.float64)
    for u, expected in zip(scenario['actions'], scenario['states']):
        env.step(np.array([u], dtype=np.float64))
        np.testing.assert_allclose(env.state, expected, atol=1e-9,
                                   err_msg=f'{name}: state drifted from source env')
    env.close()


def test_step_returns_gymnasium_five_tuple():
    env = make_env()
    env.reset()
    out = env.step(np.array([0.0]))
    assert len(out) == 5, 'env.step must return (obs, reward, terminated, truncated, info)'
    env.close()


def test_action_space_is_physical_u_sat():
    env = make_env()
    assert np.allclose(env.action_space.low, -ROLLOUTS['params']['u_sat'])
    assert np.allclose(env.action_space.high, ROLLOUTS['params']['u_sat'])
    env.close()


def test_observation_space_bounds():
    env = make_env()
    tdm = ROLLOUTS['params']['theta_dot_max']
    np.testing.assert_allclose(env.observation_space.low, [-math.pi, -tdm], atol=1e-9)
    np.testing.assert_allclose(env.observation_space.high, [math.pi, tdm], atol=1e-9)
    env.close()


def test_reset_sets_deterministic_init_state():
    env = make_env(init_state=[0.3, -0.4])
    obs, _ = env.reset()
    np.testing.assert_allclose(env.state, [0.3, -0.4], atol=1e-12)
    env.close()


def test_theta_wrapped_to_pi():
    env = make_env()
    env.reset()
    # Just below +pi with positive velocity: one step should carry theta over
    # +pi and wrap it back into [-pi, pi].
    env.state = np.array([math.pi - 0.01, 6.0], dtype=np.float64)
    env.step(np.array([0.0]))
    assert -math.pi <= env.state[0] <= math.pi
    assert env.state[0] < 0.0, 'theta should have wrapped to the negative side'
    env.close()


def test_thetadot_clipped_to_bound():
    env = make_env()
    env.reset()
    tdm = ROLLOUTS['params']['theta_dot_max']
    env.state = np.array([1.0, tdm - 0.05], dtype=np.float64)
    env.step(np.array([ROLLOUTS['params']['u_sat']]))  # push past the bound
    assert abs(env.state[1]) <= tdm + 1e-12
    assert np.isclose(env.state[1], tdm), 'theta_dot should ride the clip bound'
    env.close()


def test_done_on_goal_reached():
    env = make_env()
    env.reset()
    env.state = np.array([0.0, 0.0], dtype=np.float64)  # exactly upright at rest
    _, _, terminated, _, info = env.step(np.array([0.0]))
    assert terminated is True
    assert info.get('goal_reached') is True
    env.close()
