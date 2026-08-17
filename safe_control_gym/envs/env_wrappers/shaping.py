'''Optional, config-selected observation and cadence shaping.

Nothing here is applied by default. These exist because particular systems were
trained with particular conventions -- the pendulum's policies consume
[cos theta, sin theta, theta_dot / theta_dot_max] at an action_repeat of 4 --
and baking those into the trainer would silently mis-train every other system.
'''
import gymnasium as gym
import numpy as np


class AngleObservation(gym.ObservationWrapper):
    '''Re-encode one angular coordinate as (cos, sin) and scale its rate.

    The angle channel is replaced IN PLACE by two channels and every other
    coordinate is passed through untouched, so the wrapper applies to any system
    with an angle somewhere in its state, not only to one whose whole state is
    (angle, rate).

    On the inverted pendulum -- state (theta, theta_dot), angle_index 0,
    rate_index 1 -- that yields [cos, sin, theta_dot / rate_max], which is
    exactly the three-channel encoding the shipped pendulum_rl policies consume.
    An earlier version emitted those three channels directly and discarded
    everything else, which was indistinguishable on the pendulum and would have
    thrown away x and x_dot on the cartpole.

    Why it is worth applying to the cartpole: its theta is not wrapped at all.
    Measured over the collection region, theta reaches +/-4.4 during rollouts,
    so theta and theta - 2*pi are the same physical state presented as different
    observations and the policy cannot generalise across revolutions. The
    quadrotors need nothing here -- PyBullet reports pitch through an Euler
    convention bounded to +/-pi/2 (measured max 1.54), so their angle never
    wraps.
    '''

    def __init__(self, env, angle_index, rate_index, rate_max):
        super().__init__(env)
        self.angle_index = int(angle_index)
        self.rate_index = int(rate_index)
        self.rate_max = float(rate_max)
        # The scaled rate channel is only guaranteed within [-1, 1] when
        # rate_max matches the wrapped env's own rate bound. Configs are free
        # to pass a smaller rate_max (e.g. a training convention narrower than
        # the physical limit), so the declared bound is widened to the actual
        # reachable ratio instead of clipping the emitted value -- clipping
        # would change what the policy observes, which this wrapper must not
        # do.
        low, high = env.observation_space.low, env.observation_space.high
        rate_bound = float(np.abs(high[self.rate_index]))
        rate_scale = max(1.0, rate_bound / self.rate_max)

        new_low, new_high = [], []
        for i in range(len(low)):
            if i == self.angle_index:
                new_low += [-1.0, -1.0]      # cos, sin
                new_high += [1.0, 1.0]
            elif i == self.rate_index:
                new_low.append(-rate_scale)
                new_high.append(rate_scale)
            else:
                new_low.append(float(low[i]))
                new_high.append(float(high[i]))
        self.observation_space = gym.spaces.Box(
            low=np.array(new_low, dtype=np.float64),
            high=np.array(new_high, dtype=np.float64),
            dtype=np.float64)

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        out = []
        for i, value in enumerate(obs):
            if i == self.angle_index:
                out += [np.cos(value), np.sin(value)]
            elif i == self.rate_index:
                out.append(value / self.rate_max)
            else:
                out.append(value)
        return np.array(out, dtype=np.float64)


def rotation_matrix_from_rpy(phi, theta, psi):
    '''Body-to-world rotation for PyBullet's roll-pitch-yaw convention.

    R = Rz(psi) @ Ry(theta) @ Rx(phi). Verified against
    p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rpy)) over 500 random
    orientations: max elementwise difference 6.1e-16. Written out rather than
    called through pybullet so this module keeps no simulator dependency.
    '''
    cr, sr = np.cos(phi), np.sin(phi)
    cp, sp = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


class RotationMatrixObservation(gym.ObservationWrapper):
    '''Replace three Euler angles with the nine entries of the rotation matrix.

    Zhou et al. (2019), "On the Continuity of Rotation Representations in Neural
    Networks", show that NO representation of SO(3) in four or fewer dimensions
    is continuous -- there are always orientations where an arbitrarily small
    rotation produces a large jump in the encoding, which a network cannot fit.
    Euler angles are the worst case: they gimbal-lock, they wrap, and the chart
    is singular at pitch +/-pi/2. Quaternions avoid gimbal lock but double-cover,
    so q and -q are one rotation presented as two different inputs.

    The rotation matrix is continuous and is what the quadrotor RL literature
    feeds -- the sim-to-real and drone-racing lines both use R rather than Euler
    or quaternions.

    Concretely here: quadrotor3d reads its orientation back through an Euler
    convention whose pitch is clamped to +/-pi/2, which is why measured pitch
    topped out at 1.54 while roll ran to +/-pi. Those were symptoms of the
    representation, not of the configuration.

    Observation grows by six: the three angle channels become nine. Every entry
    of R is already in [-1, 1], so a following NormalizeObservation leaves them
    untouched.
    '''

    def __init__(self, env, angle_indices):
        super().__init__(env)
        self.angle_indices = tuple(int(i) for i in angle_indices)
        if len(self.angle_indices) != 3:
            raise ValueError('RotationMatrixObservation needs exactly three angle '
                             f'indices (roll, pitch, yaw); got {angle_indices}.')
        low = np.asarray(env.observation_space.low, dtype=np.float64)
        high = np.asarray(env.observation_space.high, dtype=np.float64)
        new_low, new_high = [], []
        for i in range(len(low)):
            if i == self.angle_indices[0]:
                new_low += [-1.0] * 9
                new_high += [1.0] * 9
            elif i in self.angle_indices:
                continue          # folded into the block above
            else:
                new_low.append(float(low[i]))
                new_high.append(float(high[i]))
        self.observation_space = gym.spaces.Box(
            low=np.array(new_low, dtype=np.float64),
            high=np.array(new_high, dtype=np.float64), dtype=np.float64)

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        roll, pitch, yaw = (obs[i] for i in self.angle_indices)
        matrix = rotation_matrix_from_rpy(roll, pitch, yaw).reshape(-1)
        out = []
        for i, value in enumerate(obs):
            if i == self.angle_indices[0]:
                out.extend(matrix)
            elif i in self.angle_indices:
                continue
            else:
                out.append(value)
        return np.array(out, dtype=np.float64)


class NormalizeObservation(gym.ObservationWrapper):
    '''Map each observation channel from [low, high] onto [-1, 1].

    Bounds are read from the wrapped env's observation_space at construction,
    which is the region the policy is trained and scored in once the collection
    regime has been applied -- so this needs no configuration of its own and
    cannot disagree with the regime.

    Worth doing because the channels are badly commensurate. Measured widths
    over the collection regions: cartpole spans 6.4x between its widest and
    narrowest channel, quadrotor2d 11.4x, quadrotor3d 16.6x -- there the body
    rates run to +/-24 while position runs to +/-1.8. An MLP with one weight
    scale sees the rate channels as dominating the input purely through units.

    Affine, not standardising: the shift and scale are fixed by the bounds
    rather than estimated from data, so training and evaluation apply exactly
    the same transform and nothing has to be saved alongside the weights. That
    is the tradeoff against SB3's VecNormalize, which tracks a running mean and
    std and must be checkpointed with the model to be reproducible.

    Channels already in [-1, 1] -- the (cos, sin) pair from AngleObservation --
    pass through unchanged, since their bounds are exactly [-1, 1].
    '''

    def __init__(self, env, low=None, high=None):
        super().__init__(env)
        # Explicit bounds win, and callers should pass them. observation_space
        # is NOT the region the regime defines: the collection regime moves
        # state_space (and cartpole's threshold attributes), while
        # observation_space keeps the env's class defaults. Reading it divided
        # quadrotor3d's body rates by the default 8.727 while the regime ran
        # them to +/-24 -- states reaching 33.8 normalised to 3.87 instead of
        # 1.0. Only fall back to observation_space when nothing better is given.
        low = np.asarray(env.observation_space.low if low is None else low, dtype=np.float64)
        high = np.asarray(env.observation_space.high if high is None else high, dtype=np.float64)
        if low.shape != env.observation_space.shape:
            raise ValueError(
                f'NormalizeObservation given {low.shape} bounds for a '
                f'{env.observation_space.shape} observation.')
        if not (np.isfinite(low).all() and np.isfinite(high).all()):
            raise ValueError(
                'NormalizeObservation needs finite observation bounds; got '
                f'low={low}, high={high}. Apply the collection regime first.')
        self.centre = (high + low) / 2.0
        # Guard the degenerate case: a channel pinned to a single value would
        # divide by zero. Leave it at its centre rather than emitting inf.
        self.halfspan = np.where((high - low) > 0, (high - low) / 2.0, 1.0)
        self.observation_space = gym.spaces.Box(
            low=-np.ones_like(low), high=np.ones_like(high), dtype=np.float64)

    def observation(self, obs):
        return (np.asarray(obs, dtype=np.float64) - self.centre) / self.halfspan


class ActionRepeat(gym.Wrapper):
    '''Hold each action for `repeat` control steps, as the policy was trained.'''

    def __init__(self, env, repeat):
        super().__init__(env)
        self.repeat = max(1, int(repeat))

    def step(self, action):
        total = 0.0
        terminated = truncated = False
        obs = info = None
        for _ in range(self.repeat):
            obs, rew, terminated, truncated, info = self.env.step(action)
            total += rew
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info
