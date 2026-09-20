'''The corridor collector's shared setup.

Pinned here because these constants are transcribed from the shipped
deterministic set and a silent drift in any of them invalidates the whole
comparison.
'''
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    'q2_corridor_common', os.path.join(ROOT, 'q2_corridor_common.py'))
q2c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q2c)


def test_plant_constants_match_the_shipped_set():
    assert q2c.HORIZON == 1200
    assert q2c.TOL == 0.2
    assert q2c.CENTRE == 0.55
    assert q2c.WIDTH == 0.12


def test_sigma_is_zero_at_the_goal_to_four_decimals():
    assert round(q2c.sigma(1.0, 0.05), 4) == 0.0


def test_band_bounds_are_the_one_percent_contour():
    lo, hi = q2c.BAND
    assert abs(q2c.sigma(lo, 1.0) - 0.01) < 1e-9
    assert abs(q2c.sigma(hi, 1.0) - 0.01) < 1e-9


def test_rollout_seed_excludes_the_level():
    '''Levels must be paired: same coordinates, same stream, any f_max.'''
    import inspect
    src = inspect.getsource(q2c.rollout_seed)
    assert 'f_max' not in src and 'level' not in src


def test_build_wires_the_corridor_to_the_dynamics_channel():
    # 'uniform' is falsified and no longer the default (see NOISE_MODELS), but
    # stays reachable by name for reproduction -- explicitly requested here.
    from safe_control_gym.envs.disturbances import AltitudeGatedNoise
    env, ctrl = q2c.build(0.01, model='uniform')
    try:
        assert 'dynamics' in env.disturbances
        assert 'action' not in env.disturbances
        dist = env.disturbances['dynamics'].disturbances[0]
        assert isinstance(dist, AltitudeGatedNoise)
        assert dist.profile == 'gaussian'
        assert list(dist.mask) == [1.0, 0.0]
    finally:
        env.close()


def test_build_with_sine_draw_wires_the_sine_class():
    from safe_control_gym.envs.disturbances import AltitudeGatedSineNoise
    env, ctrl = q2c.build(0.01, draw='sine')
    try:
        dist = env.disturbances['dynamics'].disturbances[0]
        assert isinstance(dist, AltitudeGatedSineNoise)
        assert dist.period == pytest.approx(2.0)
    finally:
        env.close()


def test_build_with_sine_model_is_a_single_sine_disturbance():
    '''model='sine' is the default noise model; unchanged from draw='sine'.'''
    from safe_control_gym.envs.disturbances import AltitudeGatedSineNoise
    env, ctrl = q2c.build(0.13, model='sine')
    try:
        dists = env.disturbances['dynamics'].disturbances
        assert len(dists) == 1
        assert isinstance(dists[0], AltitudeGatedSineNoise)
    finally:
        env.close()


def test_build_deprecated_draw_alias_matches_the_named_model():
    env_a, ctrl_a = q2c.build(0.01, draw='sine')
    env_b, ctrl_b = q2c.build(0.01, model='sine')
    try:
        dist_a = env_a.disturbances['dynamics'].disturbances[0]
        dist_b = env_b.disturbances['dynamics'].disturbances[0]
        assert type(dist_a) is type(dist_b)
        assert dist_a.profile == dist_b.profile
        assert dist_a.period == pytest.approx(dist_b.period)
    finally:
        env_a.close()
        env_b.close()


def test_build_with_sine_plus_ambient_wires_both_in_order():
    from safe_control_gym.envs.disturbances import AltitudeGatedSineNoise, WhiteNoise
    env, ctrl = q2c.build(0.13, model='sine+ambient', ambient=0.06)
    try:
        dists = env.disturbances['dynamics'].disturbances
        assert len(dists) == 2
        assert isinstance(dists[0], AltitudeGatedSineNoise)
        assert isinstance(dists[1], WhiteNoise)
        assert list(dists[0].mask) == [1.0, 0.0]
        assert list(dists[1].mask) == [1.0, 0.0]
        assert dists[1].std[0] == pytest.approx(0.06)
    finally:
        env.close()


def test_build_with_ambient_model_wires_white_noise_only():
    from safe_control_gym.envs.disturbances import WhiteNoise
    env, ctrl = q2c.build(0.0, model='ambient', ambient=0.10)
    try:
        dists = env.disturbances['dynamics'].disturbances
        assert len(dists) == 1
        assert isinstance(dists[0], WhiteNoise)
        assert dists[0].std[0] == pytest.approx(0.10)
    finally:
        env.close()


def test_ambient_model_requires_zero_f_max():
    with pytest.raises(ValueError):
        q2c.build(0.13, model='ambient', ambient=0.10)


def test_sine_plus_ambient_requires_an_ambient_value():
    with pytest.raises(ValueError):
        q2c.build(0.13, model='sine+ambient')


def test_fixed_ambient_model_rejects_an_override():
    with pytest.raises(ValueError):
        q2c.build(0.13, model='sine', ambient=0.05)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        q2c.build(0.13, model='not-a-model')


def test_resolve_noise_model_round_trips_the_sine_plus_ambient_stack():
    stack = q2c.resolve_noise_model('sine+ambient', 0.13, 0.06)
    assert stack['corridor']['disturbance_func'] == 'altitude_gated_sine'
    assert stack['corridor']['f_max'] == pytest.approx(0.13)
    assert stack['corridor']['centre'] == pytest.approx(q2c.CENTRE)
    assert stack['corridor']['width'] == pytest.approx(q2c.WIDTH)
    assert stack['corridor']['period'] == pytest.approx(q2c.SINE_PERIOD)
    assert stack['ambient'] == pytest.approx(0.06)


def test_negative_f_max_raises():
    with pytest.raises(ValueError):
        q2c.build(-0.01, model='sine')


def test_ambient_model_rejects_negative_f_max():
    with pytest.raises(ValueError):
        q2c.build(-0.01, model='ambient', ambient=0.10)


def test_build_at_zero_installs_no_disturbance():
    env, ctrl = q2c.build(0.0)
    try:
        assert env.disturbances == {}
    finally:
        env.close()


def test_state_bounds_are_applied_after_construction():
    env, _ = q2c.build(0.0)
    try:
        assert env.state_space.low[2] == 0.1
        assert env.state_space.high[2] == 1.5
        assert env.state_space.high[5] == 8.0
        # theta (index 4) is absent from STATE_BOUNDS, so it stays at the
        # env's own default -- not re-bounded by the corridor's overrides.
        assert env.state_space.high[4] == np.float32(env.theta_threshold_radians)
    finally:
        env.close()
