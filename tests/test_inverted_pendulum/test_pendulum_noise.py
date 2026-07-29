'''Tests for the ported pendulum noise models + preset registry.

Fidelity is *statistical* (matching model definitions and level parameters, not
per-step RNG streams), so these check parameters, application semantics, and
sample statistics rather than golden values.
'''

import numpy as np
import pytest

from safe_control_gym.envs.gym_control.inverted_pendulum import InvertedPendulum
from safe_control_gym.envs.gym_control.pendulum_noise import (NOISE_PRESETS, ActuationGaussianNoiseModel,
                                                              ControlProportionalNoiseModel, NoiseModel,
                                                              TruncatedActuationGaussianNoiseModel,
                                                              VelocityProportionalNoiseModel,
                                                              build_noise_model)


def rng():
    return np.random.default_rng(0)


def test_none_is_noop():
    nm = build_noise_model(None)
    assert isinstance(nm, NoiseModel)
    r = rng()
    assert nm.add_act_noise(r, 0.5) == 0.5
    np.testing.assert_array_equal(nm.add_obs_noise(r, np.array([0.1, 0.2])), [0.1, 0.2])
    np.testing.assert_array_equal(nm.add_dynamics_noise(r, np.array([0.1, 0.2]), 0.3), [0.1, 0.2])


def test_string_none_is_noop():
    assert isinstance(build_noise_model('none'), NoiseModel)


def test_preset_gaussian_act_med_params():
    nm = build_noise_model('gaussian_act_med')
    assert isinstance(nm, ActuationGaussianNoiseModel)
    # act_noise_var 0.01 -> std 0.1; observations must be untouched.
    assert nm._act_noise_std == pytest.approx(np.sqrt(0.01))
    np.testing.assert_array_equal(nm.add_obs_noise(rng(), np.array([1.0, 2.0])), [1.0, 2.0])


def test_preset_truncated_params():
    nm = build_noise_model('truncated_gaussian_act_high')
    assert isinstance(nm, TruncatedActuationGaussianNoiseModel)
    assert nm._act_noise_std == pytest.approx(np.sqrt(0.1))
    assert nm._n_sigma == pytest.approx(1.0)


def test_all_25_presets_build():
    assert len(NOISE_PRESETS) == 25
    for name in NOISE_PRESETS:
        assert isinstance(build_noise_model(name), NoiseModel)


def test_dict_spec_builds():
    nm = build_noise_model({'type': 'velocity_proportional', 'sigma0': 0.01, 'gain': 0.05})
    assert isinstance(nm, VelocityProportionalNoiseModel)
    assert nm._sigma0 == pytest.approx(0.01) and nm._gain == pytest.approx(0.05)


def test_instance_passthrough():
    m = ActuationGaussianNoiseModel(act_noise_var=0.02)
    assert build_noise_model(m) is m


def test_act_noise_statistics():
    nm = build_noise_model('gaussian_act_xhigh')  # var 0.5 -> std ~0.707
    r = rng()
    samples = np.array([nm.add_act_noise(r, 0.0) for _ in range(40000)])
    assert samples.std() == pytest.approx(np.sqrt(0.5), rel=0.05)


def test_truncated_never_exceeds_bound():
    nm = build_noise_model('truncated_gaussian_act_med')  # var 0.01 -> std 0.1, n_sigma 3
    bound = 3.0 * np.sqrt(0.01)
    r = rng()
    samples = np.array([nm.add_act_noise(r, 0.0) for _ in range(20000)])
    assert np.all(np.abs(samples) <= bound + 1e-12)


def test_velocity_proportional_std_grows_with_speed():
    nm = build_noise_model('velocity_proportional_high')  # sigma0 0.008, gain 0.04
    r = rng()
    slow = np.array([nm.add_dynamics_noise(r, np.array([0.0, 0.0]), 0.0)[0] for _ in range(20000)])
    fast = np.array([nm.add_dynamics_noise(r, np.array([0.0, 5.0]), 0.0)[0] for _ in range(20000)])
    assert slow.std() == pytest.approx(0.008, rel=0.06)
    assert fast.std() == pytest.approx(0.008 + 0.04 * 5.0, rel=0.06)


def test_control_proportional_std_grows_with_control():
    nm = build_noise_model('control_proportional_high')  # sigma0 0.008, gain 0.04
    r = rng()
    lo = np.array([nm.add_dynamics_noise(r, np.array([0.0, 0.0]), 0.0)[0] for _ in range(20000)])
    hi = np.array([nm.add_dynamics_noise(r, np.array([0.0, 0.0]), 0.6)[0] for _ in range(20000)])
    assert lo.std() == pytest.approx(0.008, rel=0.06)
    assert hi.std() == pytest.approx(0.008 + 0.04 * 0.6, rel=0.06)
    assert isinstance(nm, ControlProportionalNoiseModel)


# --- env integration ---------------------------------------------------------


def make_env(noise=None, **kw):
    cfg = dict(ctrl_freq=100, pyb_freq=100, randomized_init=False, cost='quadratic', noise=noise)
    cfg.update(kw)
    return InvertedPendulum(**cfg)


def rollout(env, x0, actions, seed=0):
    env.reset(seed=seed)
    env.state = np.array(x0, dtype=np.float64)
    states, obss = [], []
    for u in actions:
        obs, _, terminated, truncated, _ = env.step(np.array([u], dtype=np.float64))
        done = terminated or truncated
        states.append(env.state.copy())
        obss.append(np.asarray(obs, dtype=np.float64).copy())
        if done:
            break
    return np.array(states), np.array(obss)


def test_env_noise_none_is_deterministic():
    a = [0.3] * 40
    s1, _ = rollout(make_env(noise=None), [1.0, 0.0], a, seed=1)
    s2, _ = rollout(make_env(noise='none'), [1.0, 0.0], a, seed=999)
    np.testing.assert_allclose(s1, s2, atol=1e-12)


def test_env_dynamics_noise_perturbs_true_state():
    a = [0.0] * 50
    det, _ = rollout(make_env(noise=None), [1.0, 6.0], a)
    noisy, _ = rollout(make_env(noise='velocity_proportional_xxhigh'), [1.0, 6.0], a)
    assert np.abs(det - noisy).max() > 1e-3


def test_env_act_noise_perturbs_trajectory():
    a = [0.2] * 50
    det, _ = rollout(make_env(noise=None), [1.5, 0.0], a)
    noisy, _ = rollout(make_env(noise='gaussian_act_xhigh'), [1.5, 0.0], a)
    assert np.abs(det - noisy).max() > 1e-3


def test_env_obs_noise_perturbs_obs_not_true_state():
    a = [0.1] * 30
    det_state, _ = rollout(make_env(noise=None), [0.8, 0.0], a)
    noisy_state, noisy_obs = rollout(
        make_env(noise={'type': 'gaussian', 'obs_noise_var': [0.05, 0.05]}), [0.8, 0.0], a)
    # obs noise must not touch the (true) state trajectory ...
    np.testing.assert_allclose(det_state, noisy_state, atol=1e-12)
    # ... but the returned observation must differ from the true state.
    assert np.abs(noisy_state - noisy_obs).max() > 1e-3


def test_make_with_noise_preset():
    env = make_env(noise='control_proportional_med')
    assert isinstance(env.noise_model, ControlProportionalNoiseModel)
    env.close()
