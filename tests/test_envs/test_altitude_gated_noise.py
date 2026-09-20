'''AltitudeGatedNoise: a one-sided draw whose bound is a function of altitude.

The load-bearing properties are that the bound tracks the vehicle's altitude,
that the draw is one-sided so the family has a non-zero mean, and that the
number of variates consumed does not depend on f_max -- which is what keeps
two levels paired under the same seed.
'''
import numpy as np
import pytest

from safe_control_gym.envs.disturbances import ALTITUDE_PROFILES, AltitudeGatedNoise


class _FakeEnv:
    '''Minimal stand-in: the class only ever reads `state` and `np_random`.'''

    def __init__(self, z):
        self.state = np.array([0.0, 0.0, z, 0.0, 0.0, 0.0])
        self.np_random = np.random.default_rng(0)


def _make(z, **kwargs):
    env = _FakeEnv(z)
    cfg = dict(f_max=0.05, centre=0.55, width=0.12)
    cfg.update(kwargs)
    dist = AltitudeGatedNoise(env, dim=2, mask=[1, 0], **cfg)
    dist.seed(env)
    return dist, env


def test_sigma_peaks_at_the_centre():
    dist, _ = _make(0.55)
    assert dist.sigma(0.55) == pytest.approx(0.05)


def test_sigma_falls_off_as_a_gaussian():
    dist, _ = _make(0.55)
    # one width out is exp(-0.5) of the peak
    assert dist.sigma(0.67) == pytest.approx(0.05 * np.exp(-0.5), rel=1e-6)


def test_sigma_is_effectively_zero_at_the_goal():
    '''The whole point of placing the corridor below the goal.'''
    dist, _ = _make(0.55)
    assert dist.sigma(1.0) < 1e-4


def test_draw_is_one_sided_and_within_the_bound():
    dist, env = _make(0.55)
    for _ in range(200):
        out = dist.apply(np.zeros(2), env)
        assert 0.0 <= out[0] <= dist.sigma(0.55)


def test_mask_kills_the_z_component():
    dist, env = _make(0.55)
    for _ in range(50):
        out = dist.apply(np.zeros(2), env)
        assert out[1] == 0.0


def test_bound_tracks_the_env_altitude():
    '''Same disturbance object, two altitudes, different support.'''
    dist, env = _make(0.55)
    env.state[2] = 0.55
    hot = max(dist.apply(np.zeros(2), env)[0] for _ in range(400))
    env.state[2] = 0.95
    cold = max(dist.apply(np.zeros(2), env)[0] for _ in range(400))
    assert cold < hot / 10


def test_variate_count_is_independent_of_f_max():
    '''Levels stay paired: the same seed consumes the same stream at any f_max.'''
    outs = []
    for f_max in (0.01, 0.05):
        env = _FakeEnv(0.55)
        dist = AltitudeGatedNoise(env, dim=2, mask=[1, 0], f_max=f_max,
                                  centre=0.55, width=0.12)
        dist.seed(env, np.random.default_rng(7))
        outs.append([dist.apply(np.zeros(2), env)[0] for _ in range(20)])
    ratio = np.array(outs[1]) / np.array(outs[0])
    assert np.allclose(ratio, 5.0)


def test_rejects_a_negative_f_max():
    env = _FakeEnv(0.55)
    with pytest.raises(ValueError, match='f_max'):
        AltitudeGatedNoise(env, dim=2, f_max=-1.0)


def test_rejects_a_non_positive_width():
    env = _FakeEnv(0.55)
    with pytest.raises(ValueError, match='width'):
        AltitudeGatedNoise(env, dim=2, f_max=0.05, width=0.0)


def test_rejects_an_unknown_profile():
    env = _FakeEnv(0.55)
    with pytest.raises(ValueError, match='profile'):
        AltitudeGatedNoise(env, dim=2, f_max=0.05, profile='tanh_wall')


def test_a_registered_profile_is_swappable():
    '''The strategy seam: a new envelope is a registry entry, nothing else.'''
    ALTITUDE_PROFILES['flat_test'] = lambda z, params: 1.0
    try:
        dist, _ = _make(0.55, profile='flat_test')
        assert dist.sigma(0.1) == pytest.approx(0.05)
        assert dist.sigma(1.4) == pytest.approx(0.05)
    finally:
        del ALTITUDE_PROFILES['flat_test']


def test_draw_law_lives_only_in_draw():
    '''A subclass overriding _draw changes the law without touching apply().'''

    class _Frozen(AltitudeGatedNoise):
        def _draw(self, bound):
            return np.full(self.dim, bound)

    env = _FakeEnv(0.55)
    dist = _Frozen(env, dim=2, mask=[1, 0], f_max=0.05, centre=0.55, width=0.12)
    dist.seed(env)
    out = dist.apply(np.zeros(2), env)
    assert out[0] == pytest.approx(dist.sigma(0.55))
    assert out[1] == 0.0
