'''The flip controller's objective must be attitude-only.

Spec: docs/superpowers/specs/2026-08-16-quad-composition-g1-design.md (D2)
'''
import os
import sys

import numpy as np
import pytest

from quad_composition.flip_env2d import (BONUS, G_NOM, SHAPING_GAMMA, potential, sample_uniform_state,
                                         shaped_reward)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_potential_depends_only_on_attitude():
    '''Two states differing only in position and velocity share a potential.'''
    a = np.array([0.0, 1.0, 0.4, 0.0, 0.0, 2.0])
    b = np.array([-0.9, 0.2, 0.4, 0.9, -0.8, 2.0])
    assert potential(a) == pytest.approx(potential(b))


def test_potential_rises_as_attitude_improves():
    upright = np.array([0.0, 1.0, 0.05, 0.0, 0.0, 0.1])
    tilted = np.array([0.0, 1.0, 2.50, 0.0, 0.0, 5.0])
    assert potential(upright) > potential(tilted)


def test_shaping_is_potential_based():
    '''r = gamma * Phi(s') - Phi(s), so cycles accumulate no reward.'''
    s = np.array([0.0, 1.0, 2.0, 0.0, 0.0, 4.0])
    s2 = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 2.0])
    r = shaped_reward(s, s2, in_g_nom=False, out_of_bounds=False)
    assert r == pytest.approx(SHAPING_GAMMA * potential(s2) - potential(s))


def test_entering_g_nom_pays_the_bonus():
    s = np.array([0.0, 1.0, 0.30, 0.0, 0.0, 1.5])
    s2 = np.array([0.0, 1.0, 0.05, 0.0, 0.0, 0.2])
    assert G_NOM.contains(abs(s2[2]), abs(s2[5]))
    r = shaped_reward(s, s2, in_g_nom=True, out_of_bounds=False)
    assert r > BONUS


def test_uniform_sampler_respects_the_closed_state_space():
    rng = np.random.default_rng(0)
    states = np.array([sample_uniform_state(rng) for _ in range(4000)])
    assert np.abs(states[:, 0]).max() < 1.0        # x
    assert states[:, 1].min() > 0.1 and states[:, 1].max() < 1.5   # z
    assert np.abs(states[:, 2]).max() <= np.pi     # theta
    assert np.abs(states[:, 3]).max() < 1.0        # x_dot
    assert np.abs(states[:, 4]).max() < 1.0        # z_dot
    assert np.abs(states[:, 5]).max() < 8.0        # theta_dot
    # full attitude coverage, not just near-upright
    assert np.abs(states[:, 2]).max() > 3.0


# --- FlipTrainingEnv (Ruling D-A/D-B: built here, not in Task 4) ---
#
# These use a fake env (no PyBullet) mimicking the OLD Gym API this codebase
# actually implements: `step()` -> 4-tuple, `reset()` -> 2-tuple. See
# `safe_control_gym/controllers/sac/sac.py` (`obs, info = self.env.reset()`;
# `next_obs, rew, done, info = self.env.step(action)`) and
# `Quadrotor.step`/`Quadrotor.reset` in
# `safe_control_gym/envs/gym_pybullet_drones/quadrotor.py`.

def test_reset_samples_the_closed_state_space_and_delegates_to_set_initial_state(monkeypatch):
    from quad_composition import flip_env2d

    fixed_init = np.array([0.1, 0.9, 0.2, 0.05, -0.05, 0.3])
    monkeypatch.setattr(flip_env2d, 'sample_uniform_state', lambda rng: fixed_init)

    # env order [x, x_dot, z, z_dot, theta, theta_dot]
    fake_obs = np.array([0.1, 0.05, 0.9, -0.05, 0.2, 0.3])
    fake_info = {'k': 'v'}
    calls = {}

    def fake_set_initial_state(env, init_state):
        calls['env'] = env
        calls['init_state'] = init_state
        return fake_obs, fake_info

    monkeypatch.setattr(flip_env2d, 'set_initial_state', fake_set_initial_state)

    class FakeEnv:
        pass

    env = FakeEnv()
    wrapped = flip_env2d.FlipTrainingEnv(env, flip_env2d.G_NOM, seed=0)
    obs, info = wrapped.reset()

    assert calls['env'] is env
    np.testing.assert_array_equal(calls['init_state'], fixed_init)
    assert obs is fake_obs
    assert info is fake_info
    np.testing.assert_allclose(wrapped._state, flip_env2d.state_from_obs(fake_obs))


def test_step_returns_shaped_reward_and_does_not_terminate_while_still_flying():
    from quad_composition import flip_env2d

    class FakeEnv:
        def step(self, action):
            # env order [x, x_dot, z, z_dot, theta, theta_dot]; theta=1.0,
            # theta_dot=3.0 -- well outside G_NOM (tilt_c=0.175, w_c=1.0).
            obs = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 3.0])
            return obs, 0.0, False, {}

    wrapped = flip_env2d.FlipTrainingEnv(FakeEnv(), flip_env2d.G_NOM, seed=0)
    state_before = np.array([0.0, 1.0, 1.5, 0.0, 0.0, 4.0])  # dataset order
    wrapped._state = state_before.copy()

    obs, reward, done, info = wrapped.step(np.zeros(2))

    next_state = np.asarray(flip_env2d.state_from_obs(obs), dtype=float)
    expected = flip_env2d.shaped_reward(state_before, next_state, in_g_nom=False, out_of_bounds=False)
    assert reward == pytest.approx(expected)
    assert done is False
    assert info == {}
    np.testing.assert_allclose(wrapped._state, next_state)


def test_step_terminates_on_g_nom_entry_even_when_the_env_itself_is_not_done():
    from quad_composition import flip_env2d

    class FakeEnv:
        def step(self, action):
            # theta=0.05, theta_dot=0.1 -- inside G_NOM; env reports not done.
            obs = np.array([0.0, 0.0, 1.0, 0.0, 0.05, 0.1])
            return obs, 0.0, False, {}

    wrapped = flip_env2d.FlipTrainingEnv(FakeEnv(), flip_env2d.G_NOM, seed=0)
    state_before = np.array([0.0, 1.0, 1.5, 0.0, 0.0, 4.0])
    wrapped._state = state_before.copy()

    obs, reward, done, info = wrapped.step(np.zeros(2))

    next_state = np.asarray(flip_env2d.state_from_obs(obs), dtype=float)
    expected = flip_env2d.shaped_reward(state_before, next_state, in_g_nom=True, out_of_bounds=False)
    assert done is True, 'must terminate on G_nom entry even though the wrapped env is not done'
    assert reward == pytest.approx(expected)
    assert reward > flip_env2d.BONUS


def test_step_out_of_bounds_applies_the_penalty_and_terminates():
    from quad_composition import flip_env2d

    class FakeEnv:
        def step(self, action):
            # theta=2.5, theta_dot=6.0 -- outside G_NOM; env terminates (OOB).
            obs = np.array([0.0, 0.0, 1.0, 0.0, 2.5, 6.0])
            return obs, 0.0, True, {'out_of_bounds': True}

    wrapped = flip_env2d.FlipTrainingEnv(FakeEnv(), flip_env2d.G_NOM, seed=0)
    state_before = np.array([0.0, 1.0, 1.5, 0.0, 0.0, 4.0])
    wrapped._state = state_before.copy()

    obs, reward, done, info = wrapped.step(np.zeros(2))

    next_state = np.asarray(flip_env2d.state_from_obs(obs), dtype=float)
    expected = flip_env2d.shaped_reward(state_before, next_state, in_g_nom=False, out_of_bounds=True)
    assert done is True
    assert reward == pytest.approx(expected)
    assert reward < flip_env2d.OOB_PENALTY / 2, 'the OOB penalty must dominate the reward'


def test_out_of_bounds_flag_comes_from_env_info_not_inferred_from_done():
    '''`done` can be True because the ORIGINAL stabilization goal (a
    position+attitude goal unrelated to G_nom) was reached, not because the
    env actually went out of bounds. Inferring `out_of_bounds` as `done and
    not in_g_nom` would wrongly apply OOB_PENALTY in that case; the env's own
    `info['out_of_bounds']` (always present since ENV_CONFIG sets
    done_on_out_of_bound=True) must be trusted instead.
    '''
    from quad_composition import flip_env2d

    class FakeEnv:
        def step(self, action):
            # theta=0.19 -- outside G_NOM (tilt_c=0.175), so in_g_nom=False,
            # but done=True here only because the original stabilization
            # goal was reached; the env itself reports out_of_bounds=False.
            obs = np.array([0.0, 0.0, 1.0, 0.0, 0.19, 0.0])
            return obs, 0.0, True, {'goal_reached': True, 'out_of_bounds': False}

    wrapped = flip_env2d.FlipTrainingEnv(FakeEnv(), flip_env2d.G_NOM, seed=0)
    state_before = np.array([0.0, 1.0, 1.5, 0.0, 0.0, 4.0])
    wrapped._state = state_before.copy()

    obs, reward, done, info = wrapped.step(np.zeros(2))

    next_state = np.asarray(flip_env2d.state_from_obs(obs), dtype=float)
    expected = flip_env2d.shaped_reward(state_before, next_state, in_g_nom=False, out_of_bounds=False)
    assert done is True
    assert reward == pytest.approx(expected), 'must not apply OOB_PENALTY when info["out_of_bounds"] is False'


@pytest.mark.slow
def test_flip_training_env_smoke_runs_a_real_episode():
    '''Integration smoke test against the real Quadrotor/PyBullet env, driven
    with the OLD 4-tuple/2-tuple Gym API this codebase's SAC runner uses
    (safe_control_gym/controllers/sac/sac.py). Catches base-class/API
    mismatches that the fake-env unit tests above cannot.
    '''
    from quad_composition.flip_env2d import G_NOM, FlipTrainingEnv
    from quad_composition.rollout2d import ENV_CONFIG, TERMINATION
    from safe_control_gym.utils.registration import make

    env = make('quadrotor', **ENV_CONFIG)
    for idx, (lo, hi) in TERMINATION.items():
        env.state_space.low[idx] = lo
        env.state_space.high[idx] = hi
    wrapped = FlipTrainingEnv(env, G_NOM, seed=0)
    try:
        obs, info = wrapped.reset()
        assert isinstance(info, dict)
        assert obs.shape == wrapped.observation_space.shape

        action = np.zeros(wrapped.action_space.shape, dtype=np.float32)
        saw_done = False
        for _ in range(20):
            obs, reward, done, info = wrapped.step(action)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            if done:
                saw_done = True
                obs, info = wrapped.reset()
        # zero thrust falls straight through the z-bound quickly, so an
        # episode should end (either OOB or, less likely, a G_nom entry)
        # well within 20 steps.
        assert saw_done
    finally:
        env.close()
