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
