'''Noise models for the inverted pendulum, ported from the source system.

Faithful ports of ``inverted_pendulum.noise`` (statistically equivalent: same
model definitions and level parameters, driven by the env's seeded RNG). Each
model exposes three hooks the env calls at the matching points of ``step``:

  * ``add_act_noise(rng, u)``           -- applied to the command before the u_sat clip
  * ``add_dynamics_noise(rng, state, u)`` -- after Euler integrate, before wrap/clip
  * ``add_obs_noise(rng, obs)``         -- on the returned observation only

``NOISE_PRESETS`` mirrors the source's Hydra noise configs by name (e.g.
``truncated_gaussian_act_med``); ``build_noise_model`` accepts a preset name, a
``{'type': ..., **params}`` dict, a ``NoiseModel`` instance, or ``None``.
'''

import numpy as np


def _sample_truncated_normal(rng, std, n_sigma):
    if std == 0.0:
        return 0.0
    bound = n_sigma * std
    while True:
        sample = float(rng.standard_normal() * std)
        if abs(sample) <= bound:
            return sample


def _sample_truncated_normal_vec(rng, stds, n_sigma):
    noise = np.zeros_like(stds, dtype=np.float64)
    for i, std in enumerate(stds):
        if std == 0.0:
            continue
        bound = n_sigma * std
        while True:
            sample = float(rng.standard_normal() * std)
            if abs(sample) <= bound:
                noise[i] = sample
                break
    return noise


class NoiseModel:
    '''No-op base model (no observation, actuation, or dynamics noise).'''

    def add_obs_noise(self, rng, obs):
        return obs

    def add_act_noise(self, rng, u):
        return u

    def add_dynamics_noise(self, rng, state, u):
        return state


class GaussianNoiseModel(NoiseModel):
    '''Gaussian observation + actuation noise.'''

    def __init__(self, obs_noise_var=(0.0, 0.0), act_noise_var=0.0):
        self._obs_noise_std = np.sqrt(np.array(tuple(obs_noise_var), dtype=np.float64))
        self._act_noise_std = float(np.sqrt(act_noise_var))

    def add_obs_noise(self, rng, obs):
        return obs + rng.standard_normal(size=obs.shape) * self._obs_noise_std

    def add_act_noise(self, rng, u):
        return u + float(rng.standard_normal() * self._act_noise_std)


class TruncatedGaussianNoiseModel(GaussianNoiseModel):
    '''Gaussian obs + act noise, rejection-truncated at ``n_sigma`` std.'''

    def __init__(self, obs_noise_var=(0.0, 0.0), act_noise_var=0.0, n_sigma=3.0):
        super().__init__(obs_noise_var=obs_noise_var, act_noise_var=act_noise_var)
        self._n_sigma = float(n_sigma)

    def add_obs_noise(self, rng, obs):
        return obs + _sample_truncated_normal_vec(rng, self._obs_noise_std, self._n_sigma)

    def add_act_noise(self, rng, u):
        return u + _sample_truncated_normal(rng, self._act_noise_std, self._n_sigma)


class ActuationGaussianNoiseModel(NoiseModel):
    '''Gaussian noise on the actuation channel only (observations noiseless).'''

    def __init__(self, act_noise_var=0.0):
        self._act_noise_std = float(np.sqrt(act_noise_var))

    def add_act_noise(self, rng, u):
        return u + float(rng.standard_normal() * self._act_noise_std)


class TruncatedActuationGaussianNoiseModel(ActuationGaussianNoiseModel):
    '''Actuation-only Gaussian noise, rejection-truncated at ``n_sigma`` std.'''

    def __init__(self, act_noise_var=0.0, n_sigma=3.0):
        super().__init__(act_noise_var=act_noise_var)
        self._n_sigma = float(n_sigma)

    def add_act_noise(self, rng, u):
        return u + _sample_truncated_normal(rng, self._act_noise_std, self._n_sigma)


class VelocityProportionalNoiseModel(NoiseModel):
    '''Dynamics noise with ``std = sigma0 + gain * abs(thetadot)`` on both dims.'''

    def __init__(self, sigma0=0.0, gain=0.0):
        self._sigma0 = float(sigma0)
        self._gain = float(gain)

    def add_dynamics_noise(self, rng, state, u):
        std = self._sigma0 + self._gain * abs(float(state[1]))
        if std == 0.0:
            return state
        return state + rng.standard_normal(size=state.shape) * std


class ControlProportionalNoiseModel(NoiseModel):
    '''Dynamics noise with ``std = sigma0 + gain * abs(u)`` on both dims.'''

    def __init__(self, sigma0=0.0, gain=0.0):
        self._sigma0 = float(sigma0)
        self._gain = float(gain)

    def add_dynamics_noise(self, rng, state, u):
        std = self._sigma0 + self._gain * abs(float(u))
        if std == 0.0:
            return state
        return state + rng.standard_normal(size=state.shape) * std


class ControlProportionalActuationNoiseModel(NoiseModel):
    '''Actuation noise with ``std = sigma0 + gain * abs(u)`` (applied post-clip).'''

    def __init__(self, sigma0=0.0, gain=0.0):
        self._sigma0 = float(sigma0)
        self._gain = float(gain)

    def add_act_noise(self, rng, u):
        std = self._sigma0 + self._gain * abs(float(u))
        if std == 0.0:
            return u
        return u + float(rng.standard_normal() * std)


# Preset registry mirroring the source's configs/noise/*.yaml (by name).
_ACT = ActuationGaussianNoiseModel
_TACT = TruncatedActuationGaussianNoiseModel
_VEL = VelocityProportionalNoiseModel
_CTRL = ControlProportionalNoiseModel

NOISE_PRESETS = {
    'none': (NoiseModel, {}),
    # Actuation Gaussian (act_noise_var).
    'gaussian_act': (_ACT, {'act_noise_var': 0.1}),
    'gaussian_act_low': (_ACT, {'act_noise_var': 0.001}),
    'gaussian_act_med': (_ACT, {'act_noise_var': 0.01}),
    'gaussian_act_high': (_ACT, {'act_noise_var': 0.1}),
    'gaussian_act_xhigh': (_ACT, {'act_noise_var': 0.5}),
    'gaussian_act_xxhigh': (_ACT, {'act_noise_var': 1.0}),
    'gaussian_act_ultra': (_ACT, {'act_noise_var': 3.0}),
    'gaussian_act_max': (_ACT, {'act_noise_var': 2.0}),
    # Truncated actuation Gaussian (act_noise_var, n_sigma).
    'truncated_gaussian_act_low': (_TACT, {'act_noise_var': 0.001, 'n_sigma': 3.0}),
    'truncated_gaussian_act_med': (_TACT, {'act_noise_var': 0.01, 'n_sigma': 3.0}),
    'truncated_gaussian_act_high': (_TACT, {'act_noise_var': 0.1, 'n_sigma': 1.0}),
    'truncated_gaussian_act_xhigh': (_TACT, {'act_noise_var': 0.2, 'n_sigma': 1.5}),
    'truncated_gaussian_act_xxhigh': (_TACT, {'act_noise_var': 0.3, 'n_sigma': 2.0}),
    'truncated_gaussian_act_ultra': (_TACT, {'act_noise_var': 2.0, 'n_sigma': 3.0}),
    'truncated_gaussian_act_max': (_TACT, {'act_noise_var': 1.0, 'n_sigma': 2.0}),
    # Velocity-proportional dynamics noise (sigma0, gain).
    'velocity_proportional_low': (_VEL, {'sigma0': 0.002, 'gain': 0.008}),
    'velocity_proportional_med': (_VEL, {'sigma0': 0.004, 'gain': 0.015}),
    'velocity_proportional_high': (_VEL, {'sigma0': 0.008, 'gain': 0.04}),
    'velocity_proportional_xhigh': (_VEL, {'sigma0': 0.011, 'gain': 0.06}),
    'velocity_proportional_xxhigh': (_VEL, {'sigma0': 0.015, 'gain': 0.09}),
    # Control-proportional dynamics noise (sigma0, gain).
    'control_proportional_low': (_CTRL, {'sigma0': 0.002, 'gain': 0.008}),
    'control_proportional_med': (_CTRL, {'sigma0': 0.004, 'gain': 0.015}),
    'control_proportional_high': (_CTRL, {'sigma0': 0.008, 'gain': 0.04}),
    'control_proportional_xhigh': (_CTRL, {'sigma0': 0.011, 'gain': 0.06}),
}

# Model classes selectable by a dict spec's ``type`` key.
NOISE_TYPES = {
    'none': NoiseModel,
    'gaussian': GaussianNoiseModel,
    'truncated_gaussian': TruncatedGaussianNoiseModel,
    'gaussian_act': ActuationGaussianNoiseModel,
    'truncated_gaussian_act': TruncatedActuationGaussianNoiseModel,
    'velocity_proportional': VelocityProportionalNoiseModel,
    'control_proportional': ControlProportionalNoiseModel,
    'control_proportional_act': ControlProportionalActuationNoiseModel,
}


def build_noise_model(spec):
    '''Build a NoiseModel from a preset name, a dict spec, an instance, or None.'''
    if spec is None:
        return NoiseModel()
    if isinstance(spec, NoiseModel):
        return spec
    if isinstance(spec, str):
        if spec not in NOISE_PRESETS:
            raise ValueError(f'[ERROR] unknown noise preset {spec!r}; valid: {sorted(NOISE_PRESETS)}')
        cls, kwargs = NOISE_PRESETS[spec]
        return cls(**kwargs)
    if isinstance(spec, dict):
        params = dict(spec)
        type_name = params.pop('type', None)
        if type_name not in NOISE_TYPES:
            raise ValueError(f'[ERROR] noise dict needs a valid "type"; valid: {sorted(NOISE_TYPES)}')
        return NOISE_TYPES[type_name](**params)
    raise TypeError(f'[ERROR] unsupported noise spec type: {type(spec)!r}')
