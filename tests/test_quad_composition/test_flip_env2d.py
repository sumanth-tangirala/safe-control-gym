'''The flip controller's objective must be attitude-only.

Spec: docs/superpowers/specs/2026-08-16-quad-composition-g1-design.md (D2)
'''
import math
import os
import sys

import numpy as np
import pybullet as p
import pytest

from quad_composition.flip_env2d import (BONUS, G_NOM, SHAPING_GAMMA, potential, sample_uniform_state,
                                         shaped_reward)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _quat(theta):
    '''Body orientation for a pure pitch of `theta` -- what a real env caches
    on `.quat`, and what `rollout2d.true_theta` reads (Finding C1).
    '''
    return p.getQuaternionFromEuler([0.0, float(theta), 0.0])


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
    r = shaped_reward(s, s2, in_g_nom=False)
    assert r == pytest.approx(SHAPING_GAMMA * potential(s2) - potential(s))


def test_entering_g_nom_pays_the_bonus():
    s = np.array([0.0, 1.0, 0.30, 0.0, 0.0, 1.5])
    s2 = np.array([0.0, 1.0, 0.05, 0.0, 0.0, 0.2])
    assert G_NOM.contains(abs(s2[2]), abs(s2[5]))
    r = shaped_reward(s, s2, in_g_nom=True)
    assert r > BONUS


def test_shaped_reward_is_invariant_to_position_and_translational_velocity():
    '''Standing guard on spec D2 (attitude-only reward). Fix round 1
    (Critical) found that routing `info['out_of_bounds']` into the reward
    leaked exactly this dependency, because `Quadrotor._get_done()` computes
    it from a mask over [x, x_dot, z, z_dot, theta_dot] -- position and
    translational velocity, not just attitude. Two (state, next_state) pairs
    that differ ONLY in x, z, x_dot, z_dot (indices 0, 1, 3, 4; theta and
    theta_dot -- indices 2, 5 -- held fixed) must score identically.
    '''
    state_a = np.array([0.0, 1.0, 0.9, 0.0, 0.0, 3.0])
    state_b = np.array([-0.8, 0.3, 0.9, 0.9, -0.7, 3.0])
    next_a = np.array([0.5, 1.4, 0.2, -0.6, 0.4, 0.5])
    next_b = np.array([-0.3, 0.15, 0.2, 0.3, -0.9, 0.5])

    for in_g_nom in (False, True):
        r_a = shaped_reward(state_a, next_a, in_g_nom=in_g_nom)
        r_b = shaped_reward(state_b, next_b, in_g_nom=in_g_nom)
        assert r_a == pytest.approx(r_b)


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
# These use a fake env (no PyBullet physics) mimicking the OLD Gym API this
# codebase actually implements: `step()` -> 4-tuple, `reset()` -> 2-tuple. See
# `safe_control_gym/controllers/sac/sac.py` (`obs, info = self.env.reset()`;
# `next_obs, rew, done, info = self.env.step(action)`) and
# `Quadrotor.step`/`Quadrotor.reset` in
# `safe_control_gym/envs/gym_pybullet_drones/quadrotor.py`.

class _AttitudeEnv:
    '''Fake env that sits at one TRUE (theta, theta_dot) and reports it the way
    a real env does (Finding C1): `.quat` carries the full orientation, while
    the OBSERVATION carries PyBullet's gimbal-folded pitch.

    Folding here is done by PyBullet itself, not faked, so these tests are not
    begging the question -- an inverted `_AttitudeEnv` really does present the
    same observation as an upright one.
    '''

    def __init__(self, theta, theta_dot, done=False, info=None,
                 x=0.0, x_dot=0.0, z=1.0, z_dot=0.0):
        self.quat = _quat(theta)
        folded = p.getEulerFromQuaternion(self.quat)[1]
        # env order [x, x_dot, z, z_dot, theta, theta_dot]
        self._obs = np.array([x, x_dot, z, z_dot, folded, theta_dot], dtype=float)
        self._done = done
        self._info = {} if info is None else info

    def step(self, action):
        return self._obs.copy(), 0.0, self._done, dict(self._info)


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
        quat = _quat(0.2)

    env = FakeEnv()
    wrapped = flip_env2d.FlipTrainingEnv(env, flip_env2d.G_NOM, seed=0)
    obs, info = wrapped.reset()

    assert calls['env'] is env
    np.testing.assert_array_equal(calls['init_state'], fixed_init)
    assert obs is fake_obs
    assert info is fake_info
    np.testing.assert_allclose(wrapped._state, flip_env2d.state_from_env(env, fake_obs))


def test_step_returns_shaped_reward_and_does_not_terminate_while_still_flying():
    from quad_composition import flip_env2d

    # theta=1.0, theta_dot=3.0 -- well outside G_NOM (tilt_c=0.175, w_c=1.0).
    env = _AttitudeEnv(1.0, 3.0)
    wrapped = flip_env2d.FlipTrainingEnv(env, flip_env2d.G_NOM, seed=0)
    state_before = np.array([0.0, 1.0, 1.5, 0.0, 0.0, 4.0])  # dataset order
    wrapped._state = state_before.copy()

    obs, reward, done, info = wrapped.step(np.zeros(2))

    next_state = np.asarray(flip_env2d.state_from_env(env, obs), dtype=float)
    expected = flip_env2d.shaped_reward(state_before, next_state, in_g_nom=False)
    assert reward == pytest.approx(expected)
    assert done is False
    assert info == {}
    np.testing.assert_allclose(wrapped._state, next_state)


def test_step_terminates_on_g_nom_entry_even_when_the_env_itself_is_not_done():
    from quad_composition import flip_env2d

    # theta=0.05, theta_dot=0.1 -- inside G_NOM; env reports not done.
    env = _AttitudeEnv(0.05, 0.1)
    wrapped = flip_env2d.FlipTrainingEnv(env, flip_env2d.G_NOM, seed=0)
    state_before = np.array([0.0, 1.0, 1.5, 0.0, 0.0, 4.0])
    wrapped._state = state_before.copy()

    obs, reward, done, info = wrapped.step(np.zeros(2))

    next_state = np.asarray(flip_env2d.state_from_env(env, obs), dtype=float)
    expected = flip_env2d.shaped_reward(state_before, next_state, in_g_nom=True)
    assert done is True, 'must terminate on G_nom entry even though the wrapped env is not done'
    assert reward == pytest.approx(expected)
    assert reward > flip_env2d.BONUS


def test_step_does_not_pay_the_g_nom_bonus_to_an_inverted_drone():
    '''Finding C1, the consequence that would have shaped controller 1's whole
    policy: an inverted drone (true theta = pi - 0.05) OBSERVES theta = 0.05,
    which is inside G_NOM (tilt_c = 0.175). On the folded quantity `step()`
    would pay the +100 BONUS and terminate the episode there -- i.e. controller
    1 would be trained to reach, and stay at, upside down.
    '''
    from quad_composition import flip_env2d

    env = _AttitudeEnv(math.pi - 0.05, 0.0)

    # The fake really is presenting an upright-looking observation.
    obs_theta = env.step(np.zeros(2))[0][4]
    assert abs(obs_theta) == pytest.approx(0.05, abs=1e-6)
    assert G_NOM.contains(abs(obs_theta), 0.0), 'the folded observation is inside G_NOM'

    wrapped = flip_env2d.FlipTrainingEnv(env, flip_env2d.G_NOM, seed=0)
    wrapped._state = np.array([0.0, 1.0, math.pi - 0.05, 0.0, 0.0, 0.0])

    _, reward, done, _ = wrapped.step(np.zeros(2))

    assert done is False, 'inversion must not count as reaching G_nom'
    assert reward < BONUS, 'the G_nom bonus must not be paid for being upside down'


def test_step_terminates_on_out_of_bounds_but_does_not_score_a_penalty():
    '''Fix round 1 (Critical): out-of-bounds still ENDS the episode (via
    `done`), but must not be separately penalized -- `info['out_of_bounds']`
    is computed from position/translational velocity, and scoring it would
    make the reward depend on those, biasing G1 toward RoA2.
    '''
    from quad_composition import flip_env2d

    # theta=1.4, theta_dot=6.0 -- outside G_NOM; env terminates (OOB).
    env = _AttitudeEnv(1.4, 6.0, done=True, info={'out_of_bounds': True})
    wrapped = flip_env2d.FlipTrainingEnv(env, flip_env2d.G_NOM, seed=0)
    state_before = np.array([0.0, 1.0, 1.5, 0.0, 0.0, 4.0])
    wrapped._state = state_before.copy()

    obs, reward, done, info = wrapped.step(np.zeros(2))

    next_state = np.asarray(flip_env2d.state_from_env(env, obs), dtype=float)
    expected = flip_env2d.shaped_reward(state_before, next_state, in_g_nom=False)
    assert done is True, 'the episode must still end on out-of-bounds'
    assert reward == pytest.approx(expected), 'reward must be pure shaped_reward -- no OOB penalty term exists'
    assert not hasattr(flip_env2d, 'OOB_PENALTY'), 'OOB_PENALTY must be removed, not merely unused'


def test_step_reward_does_not_depend_on_info_or_why_done_fired():
    '''The reward through the full step() path must be a pure function of
    (state_before, next_state, in_g_nom) -- never of `info` or of why `done`
    fired. Two envs that report done for unrelated reasons (an out-of-bounds
    termination vs. the original stabilization task's unrelated goal_reached
    termination), but land at the same attitude, must score identically.
    '''
    from quad_composition import flip_env2d

    # Same theta/theta_dot (0.19, 0.0), different position and a completely
    # different reason for `done`.
    env_oob = _AttitudeEnv(0.19, 0.0, done=True, info={'out_of_bounds': True})
    env_goal = _AttitudeEnv(0.19, 0.0, done=True, x=0.7, x_dot=-0.3, z_dot=0.4,
                            info={'goal_reached': True, 'out_of_bounds': False})

    state_before = np.array([0.0, 1.0, 1.5, 0.0, 0.0, 4.0])

    wrapped_oob = flip_env2d.FlipTrainingEnv(env_oob, flip_env2d.G_NOM, seed=0)
    wrapped_oob._state = state_before.copy()
    _, reward_oob, done_oob, _ = wrapped_oob.step(np.zeros(2))

    wrapped_goal = flip_env2d.FlipTrainingEnv(env_goal, flip_env2d.G_NOM, seed=0)
    wrapped_goal._state = state_before.copy()
    _, reward_goal, done_goal, _ = wrapped_goal.step(np.zeros(2))

    assert done_oob is True and done_goal is True
    assert reward_oob == pytest.approx(reward_goal)


def test_reward_through_step_is_invariant_to_position_and_translational_velocity():
    '''Standing guard on the experiment's central claim (spec D2), exercised
    through the full step() path rather than just shaped_reward directly.
    Fix round 1 (Critical): routing info['out_of_bounds'] into the reward
    leaked a dependency on x, x_dot, z, z_dot (Quadrotor._get_done() computes
    it from a mask over exactly those plus theta_dot). Many random pairs of
    states differing ONLY in x, z, x_dot, z_dot -- with varied `done`/`info`
    causes on top -- must score identically as long as theta/theta_dot match.
    '''
    from quad_composition import flip_env2d

    rng = np.random.default_rng(1)
    state_before = np.array([0.3, 1.1, 0.9, -0.2, 0.1, 3.0])
    theta, theta_dot = 0.4, 2.0  # shared attitude; outside G_NOM

    for trial in range(20):
        x1, z1 = rng.uniform(-1.0, 1.0), rng.uniform(0.1, 1.5)
        xd1, zd1 = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        x2, z2 = rng.uniform(-1.0, 1.0), rng.uniform(0.1, 1.5)
        xd2, zd2 = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        done = bool(trial % 2)  # alternate done/not-done and info contents

        env_a = _AttitudeEnv(theta, theta_dot, done=done, x=x1, x_dot=xd1, z=z1, z_dot=zd1,
                             info={'out_of_bounds': done})
        env_b = _AttitudeEnv(theta, theta_dot, done=done, x=x2, x_dot=xd2, z=z2, z_dot=zd2,
                             info={'goal_reached': done, 'out_of_bounds': False})

        wrapped_a = flip_env2d.FlipTrainingEnv(env_a, flip_env2d.G_NOM, seed=0)
        wrapped_a._state = state_before.copy()
        _, reward_a, _, _ = wrapped_a.step(np.zeros(2))

        wrapped_b = flip_env2d.FlipTrainingEnv(env_b, flip_env2d.G_NOM, seed=0)
        wrapped_b._state = state_before.copy()
        _, reward_b, _, _ = wrapped_b.step(np.zeros(2))

        assert reward_a == pytest.approx(reward_b)


@pytest.mark.slow
def test_potential_scores_an_inverted_state_worse_than_an_upright_one_through_the_real_env():
    '''Finding C1, through real PyBullet rather than a synthetic state vector.

    `potential` is the function controller 1 is trained to climb and the one
    G1 calibration scores exits by, so if it cannot tell upright from inverted,
    nothing downstream can either. On the env's own folded observation the two
    states below score IDENTICALLY -- that equality is asserted here too, as a
    live record of what the bug was.
    '''
    from quad_composition.rollout2d import make_env, set_initial_state, state_from_env, state_from_obs

    env = make_env(seed=0)
    try:
        obs_upright, _ = set_initial_state(env, [0.0, 1.0, 0.05, 0.0, 0.0, 0.0])
        upright_true = potential(state_from_env(env, obs_upright))
        upright_folded = potential(state_from_obs(obs_upright))

        obs_inverted, _ = set_initial_state(env, [0.0, 1.0, math.pi - 0.05, 0.0, 0.0, 0.0])
        inverted_true = potential(state_from_env(env, obs_inverted))
        inverted_folded = potential(state_from_obs(obs_inverted))

        assert inverted_true < upright_true, \
            'an inverted drone must score strictly worse than an upright one'
        assert inverted_folded == pytest.approx(upright_folded, abs=1e-9), \
            'on the folded observation the two are indistinguishable -- the bug'
    finally:
        env.close()


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
