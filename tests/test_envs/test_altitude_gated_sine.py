'''AltitudeGatedSineNoise: the per-rollout sine draw law.

The sweep found the per-step uniform draw's fraction_interior peaking at
0.025 (see "The sweep falsified the per-step draw" in
docs/superpowers/specs/2026-08-17-quad2d-altitude-corridor-design.md) -- ~150
i.i.d. draws per crossing average out to nearly the same impulse every time.
The fix holds a gust's strength (A) and phase (phi) fixed for the whole
episode, drawn once in reset(), so the force history within an episode is a
deterministic function of (z, t, A, phi) while still varying rollout to
rollout under the seeded stream.
'''
import numpy as np
import pytest

from safe_control_gym.envs.disturbances import AltitudeGatedSineNoise
from safe_control_gym.utils.registration import make

TASK_INFO = {'stabilization_goal': [0, 1], 'stabilization_goal_tolerance': 0.2}


class _FakeEnv:
    '''Minimal stand-in: adds the control-step clock the sine draw reads.'''

    def __init__(self, z, ctrl_step_counter=0, ctrl_freq=100):
        self.state = np.array([0.0, 0.0, z, 0.0, 0.0, 0.0])
        self.np_random = np.random.default_rng(0)
        self.ctrl_step_counter = ctrl_step_counter
        self.CTRL_FREQ = ctrl_freq


def _make(z, **kwargs):
    env = _FakeEnv(z)
    cfg = dict(f_max=0.05, centre=0.55, width=0.12, period=2.0)
    cfg.update(kwargs)
    dist = AltitudeGatedSineNoise(env, dim=2, mask=[1, 0], **cfg)
    dist.seed(env)
    return dist, env


def test_registry_key_resolves_to_the_class():
    env = make('quadrotor', quad_type=2, task='stabilization', task_info=TASK_INFO,
               ctrl_freq=100, pyb_freq=5000, episode_len_sec=10, cost='quadratic',
               done_on_out_of_bound=False, gui=False, randomized_init=False,
               disturbances={'dynamics': [{'disturbance_func': 'altitude_gated_sine',
                                           'f_max': 0.05, 'profile': 'gaussian',
                                           'centre': 0.55, 'width': 0.12,
                                           'mask': [1, 0], 'period': 2.0}]})
    try:
        dist = env.disturbances['dynamics'].disturbances[0]
        assert isinstance(dist, AltitudeGatedSineNoise)
        assert dist.period == pytest.approx(2.0)
    finally:
        env.close()


def test_deterministic_within_one_episode():
    '''Same reset() draw, replayed twice over the same steps -- identical.'''
    dist, env = _make(0.55)
    dist.reset(env)

    def _pass():
        out = []
        for step in range(6):
            env.ctrl_step_counter = step
            out.append(dist.apply(np.zeros(2), env)[0])
        return out

    first = _pass()
    second = _pass()
    assert first == pytest.approx(second)


def test_draw_differs_across_episodes():
    '''Two reset() calls advance the stream -- different (phi, A) and values.'''
    dist, env = _make(0.55)
    env.ctrl_step_counter = 0

    dist.reset(env)
    phi1, a1 = dist.phi, dist.A
    v1 = dist.apply(np.zeros(2), env)[0]

    dist.reset(env)
    phi2, a2 = dist.phi, dist.A
    v2 = dist.apply(np.zeros(2), env)[0]

    assert (phi1, a1) != (phi2, a2)
    assert v1 != pytest.approx(v2)


def test_value_stays_within_the_bound():
    '''A <= 1 keeps 0.5 + 0.5*A*sin(...) in [0, 1], so the draw is in [0, sigma(z)].'''
    dist, env = _make(0.55)
    bound = dist.sigma(0.55)
    for _trial in range(20):
        dist.reset(env)
        for step in range(0, 400, 7):
            env.ctrl_step_counter = step
            val = dist.apply(np.zeros(2), env)[0]
            assert -1e-9 <= val <= bound + 1e-9


def test_sine_oscillates_over_time():
    '''Pin A and phi directly; a quarter period later the value has moved.'''
    dist, env = _make(0.55)
    dist.reset(env)
    dist.phi = 0.0
    dist.A = 1.0

    env.ctrl_step_counter = 0   # t = 0.0 s
    v_t0 = dist.apply(np.zeros(2), env)[0]

    env.ctrl_step_counter = 50   # t = 0.5 s, a quarter of period=2.0 later
    v_t_quarter = dist.apply(np.zeros(2), env)[0]

    assert v_t0 != pytest.approx(v_t_quarter)


def test_seeding_reproduces_phi_and_a():
    '''Identical streams -> identical (phi, A) after reset().'''
    dists = []
    for _ in range(2):
        env = _FakeEnv(0.55)
        dist = AltitudeGatedSineNoise(env, dim=2, mask=[1, 0], f_max=0.05,
                                      centre=0.55, width=0.12, period=2.0)
        dist.seed(env, np.random.default_rng(42))
        dist.reset(env)
        dists.append(dist)
    assert dists[0].phi == pytest.approx(dists[1].phi)
    assert dists[0].A == pytest.approx(dists[1].A)


def test_reset_consumes_exactly_two_variates():
    '''Pairing property: reset() consumes the same number of variates at any f_max.'''
    followups = []
    for f_max in (0.01, 0.05):
        env = _FakeEnv(0.55)
        dist = AltitudeGatedSineNoise(env, dim=2, mask=[1, 0], f_max=f_max,
                                      centre=0.55, width=0.12, period=2.0)
        dist.seed(env, np.random.default_rng(7))
        dist.reset(env)
        followups.append(dist.np_random.uniform())
    assert followups[0] == followups[1]
