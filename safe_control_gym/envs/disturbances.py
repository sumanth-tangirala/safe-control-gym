'''Disturbances.'''

import numpy as np


class Disturbance:
    '''Base class for disturbance or noise applied to inputs or dyanmics.'''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 **kwargs
                 ):
        self.dim = dim
        self.mask = mask
        if mask is not None:
            self.mask = np.asarray(mask, dtype=np.float32)
            assert self.dim == len(self.mask)

    def reset(self,
              env
              ):
        pass

    def apply(self,
              target,
              env
              ):
        '''Default is identity.'''
        return target

    def seed(self, env, stream=None):
        '''Bind the RNG this disturbance draws from.

        `stream` is a child Generator supplied by DisturbanceList. Falling back
        to `env.np_random` keeps the old behaviour for anything constructing a
        Disturbance directly, but the list path always passes a child -- see
        the note there for why sharing the env's generator is wrong.
        '''
        self.np_random = env.np_random if stream is None else stream


class DisturbanceList:
    '''Combine list of disturbances as one.'''

    def __init__(self,
                 disturbances
                 ):
        '''Initialization of the list of disturbances.'''
        self.disturbances = disturbances

    def reset(self,
              env
              ):
        '''Sequentially reset disturbances.'''
        for disturb in self.disturbances:
            disturb.reset(env)

    def apply(self,
              target,
              env
              ):
        '''Sequentially apply disturbances.'''
        disturbed = target
        for disturb in self.disturbances:
            disturbed = disturb.apply(disturbed, env)
        return disturbed

    def seed(self, env):
        '''Give each disturbance its OWN child stream, not the env's generator.

        Sharing `env.np_random` makes every other draw on that generator --
        initial state randomisation, inertial property randomisation -- depend
        on whether noise happens to be enabled and how many samples it consumed
        this step. Two runs with the same seed then differ in their *starting
        conditions* purely because one had a disturbance configured, and a
        resumed run does not draw what an uninterrupted one would. That breaks
        resident invariant 3, which is what makes a killed collection safe to
        restart.

        Children are spawned in list order from the env's seed sequence, so the
        stream a given disturbance gets is still a pure function of the env seed
        -- reproducible, but no longer entangled with anything else.
        '''
        try:
            seed_seq = env.np_random.bit_generator.seed_seq
            children = seed_seq.spawn(len(self.disturbances))
        except AttributeError:
            # Older bit generators expose no seed sequence. Derive a stable
            # fallback from one draw rather than silently sharing the stream.
            root = int(env.np_random.integers(0, 2 ** 32 - 1))
            children = np.random.SeedSequence(root).spawn(len(self.disturbances))
        for disturb, child in zip(self.disturbances, children):
            disturb.seed(env, np.random.default_rng(child))


class ImpulseDisturbance(Disturbance):
    '''Impulse applied during a short time interval.

    Examples:
        * single step, square (duration=1, decay_rate=1): ______|-|_______
        * multiple step, square (duration>1, decay_rate=1): ______|-----|_____
        * multiple step, triangle (duration>1, decay_rate<1): ______/\\_____
    '''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 magnitude=1,
                 step_offset=None,
                 duration=1,
                 decay_rate=1,
                 **kwargs
                 ):
        super().__init__(env, dim, mask)
        self.magnitude = magnitude
        self.step_offset = step_offset
        self.max_step = int(env.EPISODE_LEN_SEC / env.CTRL_TIMESTEP)
        # Specify shape of the impulse.
        assert duration >= 1
        assert decay_rate > 0 and decay_rate <= 1
        self.duration = duration
        self.decay_rate = decay_rate

    def reset(self,
              env
              ):
        if self.step_offset is None:
            self.current_step_offset = self.np_random.integers(self.max_step)
        else:
            self.current_step_offset = self.step_offset
        self.current_peak_step = int(self.current_step_offset + self.duration / 2)

    def apply(self,
              target,
              env
              ):
        noise = 0
        if env.ctrl_step_counter >= self.current_step_offset:
            peak_offset = np.abs(env.ctrl_step_counter - self.current_peak_step)
            if peak_offset < self.duration / 2:
                decay = self.decay_rate**peak_offset
            else:
                decay = 0
            noise = self.magnitude * decay
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed


class StepDisturbance(Disturbance):
    '''Constant disturbance at all time steps (but after offset).

    Applied after offset step (randomized or given): _______|---------
    '''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 magnitude=1,
                 step_offset=None,
                 **kwargs
                 ):
        super().__init__(env, dim, mask)
        self.magnitude = magnitude
        self.step_offset = step_offset
        self.max_step = int(env.EPISODE_LEN_SEC / env.CTRL_TIMESTEP)

    def reset(self,
              env
              ):
        if self.step_offset is None:
            self.current_step_offset = self.np_random.integers(self.max_step)
        else:
            self.current_step_offset = self.step_offset

    def apply(self,
              target,
              env
              ):
        noise = 0
        if env.ctrl_step_counter >= self.current_step_offset:
            noise = self.magnitude
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed


class UniformNoise(Disturbance):
    '''i.i.d uniform noise ~ U(low, high) per time step.'''

    def __init__(self, env, dim, mask=None, low=0.0, high=1.0, **kwargs):
        super().__init__(env, dim, mask)

        # uniform distribution bounds
        if isinstance(low, float):
            self.low = np.asarray([low] * self.dim)
        elif isinstance(low, list):
            self.low = np.asarray(low)
        else:
            raise ValueError('[ERROR] UniformNoise.__init__(): low must be specified as a float or list.')

        if isinstance(high, float):
            self.high = np.asarray([high] * self.dim)
        elif isinstance(low, list):
            self.high = np.asarray(high)
        else:
            raise ValueError('[ERROR] UniformNoise.__init__(): high must be specified as a float or list.')

    def apply(self, target, env):
        noise = self.np_random.uniform(self.low, self.high, size=self.dim)
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed


class WhiteNoise(Disturbance):
    '''I.i.d Gaussian noise per time step.'''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 std=1.0,
                 **kwargs
                 ):
        super().__init__(env, dim, mask)
        # I.i.d gaussian variance.
        if isinstance(std, float):
            self.std = np.asarray([std] * self.dim)
        elif isinstance(std, list):
            self.std = np.asarray(std)
        else:
            raise ValueError('[ERROR] WhiteNoise.__init__(): std must be specified as a float or list.')
        assert self.dim == len(self.std), 'std shape should be the same as dim.'

    def apply(self,
              target,
              env
              ):
        noise = self.np_random.normal(0, self.std, size=self.dim)
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed


class SignalDependentNoise(Disturbance):
    '''Gaussian noise whose scale grows with the magnitude of the signal.

        w ~ Normal(0, alpha + beta * |target|)

    ``alpha + beta * |target|`` is the STANDARD DEVIATION, not the variance --
    the two differ by ~5x at the values this is used with, and reading it as a
    variance would put the family in the middle of the fixed-sigma sweep rather
    than below it.

    ``|target|``, not ``target``: with a signed command the scale goes negative
    (pendulum: 0.008 - 0.04*0.637 = -0.0175), which is not a standard deviation.

    The two constants are separate mechanisms, which is the point of the class:
    ``alpha`` is the noise floor that survives as the command goes to zero -- at
    the goal, where a stabilising controller commands almost nothing -- and
    ``beta`` is the effort-proportional term that only bites during the
    transient. ``WhiteNoise`` cannot express this: its ``std`` is fixed at
    construction, so it necessarily has the same sigma at the goal as far from
    it.

    Draws are i.i.d. per call and unbounded; any saturation clip is the caller's
    (on the action channel the env clips ``u + w``, not ``w``).
    '''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 alpha=0.0,
                 beta=0.0,
                 **kwargs
                 ):
        super().__init__(env, dim, mask)
        self.alpha = self._as_vector(alpha, 'alpha')
        self.beta = self._as_vector(beta, 'beta')

    def _as_vector(self, value, name):
        if isinstance(value, (int, float)):
            vec = np.asarray([float(value)] * self.dim)
        elif isinstance(value, (list, tuple, np.ndarray)):
            vec = np.asarray(value, dtype=float)
        else:
            raise ValueError(f'[ERROR] SignalDependentNoise.__init__(): {name} must be '
                             'a float or a list.')
        if vec.shape != (self.dim,):
            raise ValueError(f'[ERROR] SignalDependentNoise.__init__(): {name} shape '
                             f'{vec.shape} should be ({self.dim},).')
        if np.any(vec < 0):
            raise ValueError(f'[ERROR] SignalDependentNoise.__init__(): {name} must be '
                             'non-negative; it is part of a standard deviation.')
        return vec

    def apply(self,
              target,
              env
              ):
        std = self.alpha + self.beta * np.abs(np.asarray(target, dtype=float))
        noise = self.np_random.normal(0, std, size=self.dim)
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed


def _gaussian_profile(z, params):
    '''Profile D from the corridor spec: a Gaussian bump, in [0, 1].'''
    if params['width'] <= 0:
        raise ValueError('[ERROR] AltitudeGatedNoise: width must be positive.')
    return np.exp(-0.5 * ((z - params['centre']) / params['width']) ** 2)


# The altitude envelope is a strategy. A new shape is a new entry here plus its
# parameters in the disturbance config -- nothing else changes. Entries map
# name -> fn(z, params) returning a value in [0, 1]; f_max supplies the units.
ALTITUDE_PROFILES = {'gaussian': _gaussian_profile}


class AltitudeGatedNoise(Disturbance):
    '''One-sided uniform noise whose bound is a function of altitude.

        F ~ U(0, sigma(z)),   sigma(z) = f_max * profile(z)

    ``sigma`` is the BOUND of the draw, not a scale on a fixed distribution --
    the support itself moves with altitude, so mean (sigma/2) and standard
    deviation (sigma/(2 sqrt(3))) are locked at a ratio of 1/sqrt(3).

    One-sided on purpose. Every other family here is symmetric about zero; this
    one has a non-zero mean, so the vehicle must hold a standing tilt against it
    rather than merely reject jitter.

    ``profile`` names an entry in ``ALTITUDE_PROFILES``; the remaining kwargs
    are handed to it untouched. The draw law lives in ``_draw`` and nowhere
    else, so an alternative law (the spec's banked per-rollout sinusoid) is a
    subclass overriding ``_draw``, not an edit here.

    The altitude is read from ``env.state``, which ``_get_observation`` refreshes
    at the end of each step, so at ``before_step`` time it is the state entering
    this step. It is the TRUE state -- the observation disturbance is applied to
    a copy afterwards -- which is what we want: the corridor is a property of the
    airspace, not of what the vehicle believes.

    The draw is normalised then scaled rather than drawn directly on
    ``[0, sigma]``, so the number of variates consumed does not depend on
    ``f_max``. Two levels sharing a seed then see the same underlying stream,
    which is the pairing property ``rollout_seed`` exists to provide.
    '''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 f_max=0.0,
                 profile='gaussian',
                 state_index=2,
                 **profile_params
                 ):
        super().__init__(env, dim, mask)
        if f_max < 0:
            raise ValueError('[ERROR] AltitudeGatedNoise.__init__(): f_max must be '
                             'non-negative; it is the upper bound of a one-sided draw.')
        if profile not in ALTITUDE_PROFILES:
            raise ValueError(f'[ERROR] AltitudeGatedNoise.__init__(): unknown profile '
                             f'{profile!r}; registered: {sorted(ALTITUDE_PROFILES)}.')
        self.f_max = float(f_max)
        self.profile = profile
        self.profile_params = {k: float(v) for k, v in profile_params.items()}
        self.state_index = int(state_index)
        self.sigma(0.0)   # fail at construction, not mid-rollout, on bad params

    def sigma(self, z):
        '''The draw bound at altitude z, in newtons.'''
        return self.f_max * ALTITUDE_PROFILES[self.profile](z, self.profile_params)

    def _draw(self, bound):
        '''The draw law -- the ONLY place variates are consumed.'''
        return self.np_random.uniform(0.0, 1.0, size=self.dim) * bound

    def apply(self,
              target,
              env
              ):
        z = float(np.asarray(env.state).ravel()[self.state_index])
        noise = self._draw(self.sigma(z))
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed


class AltitudeGatedSineNoise(AltitudeGatedNoise):
    '''Altitude-gated one-sided force with per-rollout coherent randomness.

    F(z, t) = sigma(z) * (0.5 + 0.5 * A * sin(2*pi/period * t + phi))

    phi ~ U(-pi, pi) and A ~ U(0, 1) are drawn ONCE per episode in reset();
    within an episode the force history is a deterministic function of
    (z, t, A, phi). Adopted after the per-step uniform draw measured
    degenerate: ~150 i.i.d. draws integrate to nearly the same impulse on
    every crossing (fraction_interior peaked at 0.025), whereas holding the
    gust's strength and phase for the whole rollout keeps trial-to-trial
    spread at the 30-50% the sweep needs. Mean force is unchanged:
    E[0.5 + 0.5*A*sin] = 0.5, same as U(0, 1).
    '''

    def __init__(self, env, dim, mask=None, f_max=0.0, profile='gaussian',
                 state_index=2, period=2.0, **profile_params):
        super().__init__(env, dim, mask=mask, f_max=f_max, profile=profile,
                         state_index=state_index, **profile_params)
        if period <= 0:
            raise ValueError('[ERROR] AltitudeGatedSineNoise.__init__(): period '
                             'must be positive.')
        self.period = float(period)
        self.phi = 0.0
        self.A = 0.0
        self._t = 0.0

    def reset(self, env):
        # np_random is bound by Disturbance.seed(), called from
        # BenchmarkEnv.__init__ (self._setup_disturbances() then self.seed())
        # before any env.reset() -- and hence before this reset() -- can run.
        # Guarded anyway: a caller that resets a freshly-constructed instance
        # without ever calling seed() should fall back to env.np_random
        # rather than raise, matching the base Disturbance.seed() fallback.
        rng = getattr(self, 'np_random', None)
        if rng is None:
            rng = env.np_random
        self.phi = float(rng.uniform(-np.pi, np.pi))
        self.A = float(rng.uniform(0.0, 1.0))

    def _draw(self, bound):
        wave = 0.5 + 0.5 * self.A * np.sin(2 * np.pi / self.period * self._t + self.phi)
        return np.full(self.dim, bound * wave)

    def apply(self, target, env):
        self._t = env.ctrl_step_counter / env.CTRL_FREQ
        return super().apply(target, env)


class BrownianNoise(Disturbance):
    '''Simple random walk noise.'''

    def __init__(self):
        super().__init__()


class PeriodicNoise(Disturbance):
    '''Sinuisodal noise.'''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 scale=1.0,
                 frequency=1.0,
                 **kwargs
                 ):
        super().__init__(env, dim)
        # Sine function parameters.
        self.scale = scale
        self.frequency = frequency

    def apply(self,
              target,
              env
              ):
        phase = self.np_random.uniform(low=-np.pi, high=np.pi, size=self.dim)
        t = env.pyb_step_counter * env.PYB_TIMESTEP
        noise = self.scale * np.sin(2 * np.pi * self.frequency * t + phase)
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed


class StateDependentDisturbance(Disturbance):
    '''Time varying and state varying, e.g. friction.

    Here to provide an explicit form, can also enable friction in simulator directly.
    '''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 **kwargs
                 ):
        super().__init__()


DISTURBANCE_TYPES = {'impulse': ImpulseDisturbance,
                     'step': StepDisturbance,
                     'uniform': UniformNoise,
                     'white_noise': WhiteNoise,
                     'periodic': PeriodicNoise,
                     'signal_dependent': SignalDependentNoise,
                     'altitude_gated': AltitudeGatedNoise,
                     'altitude_gated_sine': AltitudeGatedSineNoise,
                     }


def create_disturbance_list(disturbance_specs, shared_args, env):
    '''Creates a DisturbanceList from yaml disturbance specification.

    Args:
        disturbance_specs (list): List of dicts defining the disturbances info.
        shared_args (dict): args shared across the disturbances in the list.
        env (BenchmarkEnv): Env for which the constraints will be applied
    '''
    disturb_list = []
    # Each disturbance for the mode.
    for disturb in disturbance_specs:
        assert 'disturbance_func' in disturb.keys(), '[ERROR]: Every distrubance must specify a disturbance_func.'
        disturb_func = disturb['disturbance_func']
        assert disturb_func in DISTURBANCE_TYPES, '[ERROR] in BenchmarkEnv._setup_disturbances(), disturbance type not available.'
        disturb_cls = DISTURBANCE_TYPES[disturb_func]
        cfg = {key: disturb[key] for key in disturb if key != 'disturbance_func'}
        disturb = disturb_cls(env, **shared_args, **cfg)
        disturb_list.append(disturb)
    return DisturbanceList(disturb_list)
