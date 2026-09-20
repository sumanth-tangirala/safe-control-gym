'''The quad3d twin-curtain corridor collector's shared setup.

Pinned here because these constants are transcribed from the design memo
(.superpowers/sdd/quad3d-corridor-design-memo.md) and the user's twin-curtain
revision of it, and a silent drift in any of them invalidates the whole
comparison against the shipped deterministic set.
'''
import importlib.util
import math
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    'q3_corridor_common', os.path.join(ROOT, 'q3_corridor_common.py'))
q3c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q3c)


def test_plant_constants_match_the_design_memo():
    assert q3c.X_C == 0.9
    assert q3c.WIDTH == 0.25
    assert q3c.PERIOD == 2.0
    assert q3c.SINE_PERIOD == 2.0
    assert q3c.HORIZON == 2000
    assert q3c.GATE_COORD_INDEX == 0
    assert q3c.TWIN is True


def test_sigma_peaks_at_each_curtain_centre():
    # Cross-curtain contribution at the OTHER curtain's centre is negligible
    # (the two centres are 1.8 apart, ~7.2 widths), so sigma(X_C, 1.0) should
    # read essentially the single-curtain peak of 1.0.
    assert q3c.sigma(q3c.X_C, 1.0) == pytest.approx(1.0, abs=1e-6)
    assert q3c.sigma(-q3c.X_C, 1.0) == pytest.approx(1.0, abs=1e-6)


def test_sigma_is_symmetric_between_the_two_curtains():
    assert q3c.sigma(q3c.X_C, 1.0) == pytest.approx(q3c.sigma(-q3c.X_C, 1.0), rel=1e-9)


def test_sigma_at_the_goal_is_the_sum_of_both_curtains_skirts():
    # Twin curtain: the goal (x=0) sees BOTH curtains' skirts, so the combined
    # envelope there is 2x a single curtain's -- ~3.1e-3 of peak, not the
    # single-curtain design memo's 1.53e-3. Still well below 1e-2, which is
    # what keeps the goal calm (design memo §A).
    single = math.exp(-0.5 * (q3c.X_C / q3c.WIDTH) ** 2)
    gate0 = q3c.sigma(0.0, 1.0)
    assert gate0 == pytest.approx(2 * single, rel=1e-6)
    assert gate0 == pytest.approx(3.06e-3, rel=0.05)
    assert gate0 < 1e-2


def _single_curtain_profile(x, centre):
    return math.exp(-0.5 * ((x - centre) / q3c.WIDTH) ** 2)


def test_band_is_the_union_of_the_two_one_percent_contours():
    pos, neg = q3c.BAND
    assert pos[0] == pytest.approx(0.14, abs=0.01)
    assert pos[1] == pytest.approx(1.66, abs=0.01)
    assert neg[0] == pytest.approx(-1.66, abs=0.01)
    assert neg[1] == pytest.approx(-0.14, abs=0.01)
    # Each edge sits at 1% of ITS OWN curtain's peak (the other curtain's
    # contribution there is negligible).
    assert _single_curtain_profile(pos[0], q3c.X_C) == pytest.approx(0.01, abs=1e-9)
    assert _single_curtain_profile(pos[1], q3c.X_C) == pytest.approx(0.01, abs=1e-9)
    assert _single_curtain_profile(neg[0], -q3c.X_C) == pytest.approx(0.01, abs=1e-9)
    assert _single_curtain_profile(neg[1], -q3c.X_C) == pytest.approx(0.01, abs=1e-9)


def test_rollout_seed_is_imported_not_rederived():
    '''Design memo §E: import rollout_seed from generate_quadrotor_3d_noisy
    rather than re-deriving it -- it was paid for once already.'''
    import generate_quadrotor_3d_noisy as gqn
    assert q3c.rollout_seed is gqn.rollout_seed


def test_rollout_seed_signature_excludes_f_max():
    '''Levels must be paired: same coordinates, same stream, any f_max.'''
    import inspect
    params = list(inspect.signature(q3c.rollout_seed).parameters)
    assert params == ['base', 'split_id', 'index', 'trial']


def test_build_with_sine_plus_ambient_wires_twin_curtain_then_ambient():
    from safe_control_gym.envs.disturbances import AltitudeGatedSineNoise, WhiteNoise
    env, ctrl = q3c.build(0.05, model='sine+ambient', ambient=0.008)
    try:
        dists = env.disturbances['dynamics'].disturbances
        assert len(dists) == 3
        assert isinstance(dists[0], AltitudeGatedSineNoise)
        assert isinstance(dists[1], AltitudeGatedSineNoise)
        assert isinstance(dists[2], WhiteNoise)

        assert dists[0].profile_params['centre'] == pytest.approx(q3c.X_C)
        assert dists[1].profile_params['centre'] == pytest.approx(-q3c.X_C)
        for d in (dists[0], dists[1]):
            assert d.state_index == q3c.GATE_COORD_INDEX
            assert d.period == pytest.approx(q3c.PERIOD)
            assert d.profile == 'gaussian'
            assert d.profile_params['width'] == pytest.approx(q3c.WIDTH)

        assert dists[2].std[0] == pytest.approx(0.008)
        for d in dists:
            assert list(d.mask) == [0.0, 1.0, 0.0]
    finally:
        env.close()


def test_twin_curtains_draw_independent_gusts_within_one_reset():
    env, ctrl = q3c.build(0.05, model='sine')
    try:
        env.reset(seed=7)
        pos, neg = env.disturbances['dynamics'].disturbances
        # Continuous uniform draws from independent child streams: collision
        # probability is effectively zero, so inequality here is a real check
        # of independence, not a fluke of a shared stream.
        assert (pos.phi, pos.A) != (neg.phi, neg.A)
    finally:
        env.close()


def test_same_seed_reproduces_identical_disturbance_draws():
    '''benchmark_env.py's before_reset() draws the episode's disturbance
    values BEFORE applying the newly-passed seed (it reseeds at the very end,
    for the NEXT reset to pick up) -- an existing property of the shared base
    class, out of scope here. So the first reset(seed=X) after construction
    consumes whatever RNG state construction happened to leave behind, and
    only the second and later reset(seed=X) calls draw from a stream that was
    actually seeded to X. One warm-up call makes the comparison meaningful.
    '''
    env, ctrl = q3c.build(0.05, model='sine')
    try:
        env.reset(seed=123)   # warm-up: primes the RNG to seed 123 for next time

        env.reset(seed=123)
        pos1, neg1 = env.disturbances['dynamics'].disturbances
        phi_pos1, a_pos1, phi_neg1, a_neg1 = pos1.phi, pos1.A, neg1.phi, neg1.A

        env.reset(seed=123)
        pos2, neg2 = env.disturbances['dynamics'].disturbances
        assert pos2.phi == pytest.approx(phi_pos1)
        assert pos2.A == pytest.approx(a_pos1)
        assert neg2.phi == pytest.approx(phi_neg1)
        assert neg2.A == pytest.approx(a_neg1)
    finally:
        env.close()


def test_build_with_sine_model_is_two_sine_disturbances_no_ambient():
    from safe_control_gym.envs.disturbances import AltitudeGatedSineNoise
    env, ctrl = q3c.build(0.13, model='sine')
    try:
        dists = env.disturbances['dynamics'].disturbances
        assert len(dists) == 2
        assert all(isinstance(d, AltitudeGatedSineNoise) for d in dists)
    finally:
        env.close()


def test_build_with_uniform_model_wires_altitude_gated_not_sine():
    from safe_control_gym.envs.disturbances import AltitudeGatedNoise, AltitudeGatedSineNoise
    env, ctrl = q3c.build(0.01, model='uniform')
    try:
        dists = env.disturbances['dynamics'].disturbances
        assert len(dists) == 2
        assert all(isinstance(d, AltitudeGatedNoise) for d in dists)
        assert not any(isinstance(d, AltitudeGatedSineNoise) for d in dists)
    finally:
        env.close()


def test_build_with_ambient_model_wires_white_noise_only():
    from safe_control_gym.envs.disturbances import WhiteNoise
    env, ctrl = q3c.build(0.0, model='ambient', ambient=0.01)
    try:
        dists = env.disturbances['dynamics'].disturbances
        assert len(dists) == 1
        assert isinstance(dists[0], WhiteNoise)
        assert dists[0].std[0] == pytest.approx(0.01)
        assert list(dists[0].mask) == [0.0, 1.0, 0.0]
    finally:
        env.close()


def test_build_at_zero_with_sine_installs_no_disturbance():
    '''baseline path, consistent with q2_corridor_common's convention: f_max=0
    with a fixed-zero-ambient model installs nothing at all.'''
    env, ctrl = q3c.build(0.0, model='sine')
    try:
        assert env.disturbances == {}
    finally:
        env.close()


def test_state_bounds_are_applied_after_construction():
    env, _ = q3c.build(0.0, model='sine')
    try:
        assert env.state_space.low[0] == -1.8
        assert env.state_space.high[0] == 1.8
        assert env.state_space.low[4] == 0.1
        assert env.state_space.high[4] == 3.0
    finally:
        env.close()


def test_episode_len_sec_matches_the_deadline_decision():
    env, _ = q3c.build(0.0, model='sine')
    try:
        assert env.EPISODE_LEN_SEC == 20
        assert env.CTRL_STEPS == q3c.HORIZON
    finally:
        env.close()


def test_resolve_noise_model_round_trips_the_twin_sine_plus_ambient_stack():
    stack = q3c.resolve_noise_model('sine+ambient', 0.13, 0.008)
    assert len(stack['corridor']) == 2
    pos, neg = stack['corridor']
    assert pos['disturbance_func'] == 'altitude_gated_sine'
    assert pos['f_max'] == pytest.approx(0.13)
    assert pos['centre'] == pytest.approx(q3c.X_C)
    assert neg['centre'] == pytest.approx(-q3c.X_C)
    assert pos['width'] == pytest.approx(q3c.WIDTH)
    assert pos['period'] == pytest.approx(q3c.SINE_PERIOD)
    assert pos['state_index'] == q3c.GATE_COORD_INDEX
    assert stack['ambient'] == pytest.approx(0.008)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        q3c.build(0.05, model='not-a-model')


def test_sine_plus_ambient_requires_an_ambient_value():
    with pytest.raises(ValueError):
        q3c.build(0.05, model='sine+ambient')


def test_fixed_ambient_model_rejects_an_override():
    with pytest.raises(ValueError):
        q3c.build(0.05, model='sine', ambient=0.01)


def test_ambient_model_requires_zero_f_max():
    with pytest.raises(ValueError):
        q3c.build(0.13, model='ambient', ambient=0.01)


def test_negative_f_max_raises():
    with pytest.raises(ValueError):
        q3c.build(-0.01, model='sine')


def test_ambient_model_rejects_negative_f_max():
    with pytest.raises(ValueError):
        q3c.build(-0.01, model='ambient', ambient=0.01)
