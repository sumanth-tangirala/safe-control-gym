'''Tests for PendulumLQR: it must reproduce the source system's
bounds-normalized LQR gain and actions exactly (golden fixtures from
``fixtures/lqr_gain.json``).'''

import json
import os

import numpy as np

from safe_control_gym.controllers.pendulum_lqr.pendulum_lqr import PendulumLQR
from safe_control_gym.envs.gym_control.inverted_pendulum import InvertedPendulum

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')
GOLDEN = json.load(open(os.path.join(FIX, 'lqr_gain.json')))


def env_func(**kwargs):
    cfg = dict(ctrl_freq=100, pyb_freq=100, randomized_init=False, cost='quadratic')
    cfg.update(kwargs)
    return InvertedPendulum(**cfg)


def make_lqr():
    return PendulumLQR(env_func, q_lqr=[1, 1], r_lqr=[1])


def test_gain_matches_source_lqr():
    ctrl = make_lqr()
    np.testing.assert_allclose(np.asarray(ctrl.K).reshape(-1), GOLDEN['K'], atol=1e-6)
    ctrl.close()


def test_action_matches_source_pairs():
    ctrl = make_lqr()
    for pair in GOLDEN['pairs']:
        obs = np.array([pair['theta'], pair['thetadot']], dtype=np.float64)
        u = ctrl.select_action(obs)
        assert np.asarray(u).shape == (1,)
        np.testing.assert_allclose(float(np.asarray(u).reshape(-1)[0]), pair['action'], atol=1e-9)
    ctrl.close()


def test_action_clipped_to_u_sat():
    ctrl = make_lqr()
    # Large error -> unclipped LQR command would exceed the saturation.
    u = float(np.asarray(ctrl.select_action(np.array([3.0, 6.0]))).reshape(-1)[0])
    assert abs(u) <= GOLDEN['u_sat'] + 1e-12
    ctrl.close()
