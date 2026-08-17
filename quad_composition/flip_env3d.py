'''Training environment for controller 1 -- the 3D flip controller.

The 3D analogue of `flip_env2d.py`.  The reward is ATTITUDE ONLY (spec D2):
controller 1 has authority over attitude and essentially none over position or
translational velocity, so rewarding it for those would (a) ask for something
it cannot deliver and (b) pull G1 toward RoA2, which is precisely the
contrivance this experiment exists to avoid.

Attitude here is `(tilt, |omega|)` -- tilt from the ROTATION MATRIX
(`rollout3d.tilt_from_quat`), never from Euler angles.  See rollout3d.py's
docstring for why that distinction is load-bearing.
'''

import gymnasium as gym
import numpy as np

from quad_composition.g1 import G1Region
from quad_composition.rollout3d import (QUAT_SLICE, ctrl1_observation, ctrl1_observation_space,
                                        omega_norm, set_initial_state, state_from_env,
                                        tilt_from_quat_wxyz)

# Nominal training target.  This is NOT the G1 that triggers handoff -- that is
# calibrated later from measured exits (spec D1 step 2) and may be looser.
#
# 10 deg, 4 rad/s.  `G1Region` is dimension-agnostic (it takes tilt and |omega|
# scalars), so it is REUSED verbatim from the 2D branch, not forked.  w_c=4.0
# is the analogue of 2D's 1.0: 3D's per-axis body-rate bound is 24 where 2D's
# was 8, and 4/24 vs 1/8 keeps the goal comfortably inside the achievable set
# (the feasibility probe reached exactly this region from full inversion).
G_NOM_3D = G1Region(tilt_c=0.175, w_c=4.0)

SHAPING_GAMMA = 0.99
BONUS = 100.0

# Normalisers for the potential: full tilt range, and the per-axis rate bound.
TILT_SCALE = np.pi
RATE_SCALE = 24.0


def potential(state):
    '''Phi(s), attitude only.  Higher is closer to upright and still.

    `state` is a dataset-order 13-dim row
    `[x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r]` -- i.e. one
    built by `rollout3d.state_from_env`.  Only the quaternion (indices 3:7)
    and the body rates (indices 10:13) are read; positions and translational
    velocities are ignored BY CONSTRUCTION, which is what makes the reward
    attitude-only rather than merely attitude-dominated.

    Tilt is `arccos(R[2, 2])`, monotone from 0 (upright) to pi (inverted).
    An Euler-derived substitute would rate a fully inverted drone as upright
    -- in 2D that made `FlipTrainingEnv.step` pay the +100 G_nom BONUS to an
    INVERTED drone and terminate there, training controller 1 to STAY
    inverted, the exact opposite of its objective (Finding C1).
    '''
    s = np.asarray(state, dtype=float)
    tilt = tilt_from_quat_wxyz(s[QUAT_SLICE])
    return -(tilt / TILT_SCALE + omega_norm(s) / RATE_SCALE)


def shaped_reward(state, next_state, in_g_nom):
    '''Potential-based shaping plus the G_nom entry bonus.  ATTITUDE ONLY.

    NO out-of-bounds penalty.  `info['out_of_bounds']` is computed by
    `Quadrotor._get_done()` from a mask over
    [x, x_dot, y, y_dot, z, z_dot, p, q, r] -- it is True whenever POSITION or
    TRANSLATIONAL VELOCITY leaves bounds, not just attitude.  Scoring it would
    make the reward depend on exactly the variables G1 must not be defined
    over, biasing the handoff region toward RoA2.  Out-of-bounds still ENDS
    the episode (via `FlipTrainingEnv3D.step`'s own `done`), which already
    costs the policy all future shaping and the bonus; it is simply not scored
    on top of that.

    Potential-based shaping leaves the optimal policy unchanged, so the bonus
    is what the policy actually optimises and the shaping only speeds it up.
    '''
    reward = SHAPING_GAMMA * potential(next_state) - potential(state)
    if in_g_nom:
        reward += BONUS
    return reward


def sampling_bounds_from_env(env):
    '''Derive dataset-order sampling bounds from the env's OWN termination
    bounds (`env.state_space`).

    CLAUDE.md requires a CLOSED state space: the initialized and achieved
    state spaces must be identical.  Hardcoding the box breaks that the moment
    a caller overrides a termination bound -- training would then sample a
    strictly smaller box than it is allowed to reach, and the policy would
    never see the extreme states it must handle.  Reading the bounds back off
    the env makes the two impossible to desynchronise.

    env state order  [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
    dataset order    [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r]

    The four quaternion slots carry [-1, 1] placeholders and are NEVER sampled
    from: attitude covers the whole of SO(3) (see `sample_uniform_state`), and
    the env's phi/theta/psi bounds are infinite anyway.
    '''
    lo, hi = env.state_space.low, env.state_space.high
    low = np.array([lo[0], lo[2], lo[4], -1.0, -1.0, -1.0, -1.0,
                    lo[1], lo[3], lo[5], lo[9], lo[10], lo[11]], dtype=float)
    high = np.array([hi[0], hi[2], hi[4], 1.0, 1.0, 1.0, 1.0,
                     hi[1], hi[3], hi[5], hi[9], hi[10], hi[11]], dtype=float)
    return low, high


def random_rotation_quat(rng):
    '''A uniform random rotation on SO(3), as a scalar-first [qw, qx, qy, qz]
    with qw >= 0 (Shoemake's method).

    Uniform EULER ANGLES are NOT uniform on SO(3) -- they concentrate near the
    gimbal poles and would bias the training distribution toward exactly the
    attitudes the Euler parameterisation degenerates on.
    '''
    u1, u2, u3 = rng.random(3)
    a, b = np.sqrt(1.0 - u1), np.sqrt(u1)
    qx = a * np.sin(2.0 * np.pi * u2)
    qy = a * np.cos(2.0 * np.pi * u2)
    qz = b * np.sin(2.0 * np.pi * u3)
    qw = b * np.cos(2.0 * np.pi * u3)
    q = np.array([qw, qx, qy, qz], dtype=float)
    return -q if q[0] < 0.0 else q


def capped_tilt_quat(rng, max_tilt, min_tilt=0.0):
    '''A random rotation whose TILT is uniform on [min_tilt, max_tilt], with
    uniform azimuth and uniform yaw about the body z axis.

    Used when training is asked to narrow the initial attitude range (the
    uncapped case uses `random_rotation_quat`, true uniform SO(3)), and by
    `evaluate_geometric_flip3d.py` to draw a single tilt BAND -- which is what
    `min_tilt` is for.  Note this is uniform in tilt, NOT uniform on the
    spherical cap: bucketed evaluation wants equal weight per degree of tilt,
    not the cap's sin-weighting toward the equator.
    '''
    polar = float(rng.uniform(min_tilt, max_tilt))
    az = float(rng.uniform(0.0, 2.0 * np.pi))
    yaw = float(rng.uniform(0.0, 2.0 * np.pi))
    # Tilt world-z by `polar` about the horizontal axis at azimuth `az + pi/2`,
    # i.e. rotate by `polar` about (-sin az, cos az, 0), then spin about body z.
    axis = np.array([-np.sin(az), np.cos(az), 0.0])
    q_tilt = np.concatenate([[np.cos(polar / 2.0)], np.sin(polar / 2.0) * axis])
    q_yaw = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    q = _quat_mul(q_tilt, q_yaw)
    return -q if q[0] < 0.0 else q


def _quat_mul(a, b):
    '''Hamilton product of two scalar-first quaternions.'''
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=float)


def sample_uniform_state(rng, bounds, max_init_tilt=None):
    '''Uniform over the closed state space, with attitude uniform on SO(3).

    `bounds` must come from `sampling_bounds_from_env` -- it is required, not
    optional, so no caller can silently fall back to a hardcoded box that has
    drifted from the env's termination bounds.

    `max_init_tilt` (radians) optionally narrows the INITIAL tilt only, for
    curriculum-style runs.  Training only -- evaluation, calibration and
    dataset generation keep the full closed state space.
    '''
    low, high = bounds
    state = rng.uniform(low, high)
    state[QUAT_SLICE] = (random_rotation_quat(rng) if max_init_tilt is None
                         else capped_tilt_quat(rng, max_init_tilt))
    return state


class FlipTrainingEnv3D(gym.Wrapper):
    '''Attitude-only objective over the 3D quadrotor.

    reset() places the drone at a uniform sample of the closed state space
    (positions/velocities/rates uniform in the env's own termination box,
    attitude uniform on SO(3)); step() replaces the env reward with the
    attitude-only shaped reward and terminates on G_nom entry or
    out-of-bounds.  Out-of-bounds termination is UNSCORED: it ends the episode
    via `done`, forfeiting future shaping and the bonus, but adds no separate
    penalty, since out-of-bounds depends on position and translational
    velocity and penalising it would bias G1 toward RoA2.

    Reward and G_nom termination are computed on tilt read from the ROTATION
    MATRIX, never from Euler angles (see `potential`).

    OBSERVATION (spec D6): the policy does NOT see the env's native 12-dim
    observation.  `reset()`/`step()` return `rollout3d.ctrl1_observation`: the
    native obs with the three Euler elements replaced by the nine rotation
    matrix entries --

        [x, x_dot, y, y_dot, z, z_dot, R00..R22, p, q, r]

    18-dim, matching `self.observation_space`.  `rollout3d.rollout_composite`
    applies the SAME transform at inference time (`_act_ctrl1`) and
    `rollout3d.load_ctrl1` sizes controller 1's network to this same space --
    all three must agree, or controller 1 sees a distribution it never trained
    on.  CONTROLLER 2 (LQR) is untouched by any of this
    (`rollout3d._act_ctrl2`).

    Base class note: this codebase has no standalone `gym` package -- only
    `gymnasium`, imported as `gym` by every wrapper in `safe_control_gym`.
    Those wrappers keep the OLD 4-tuple/2-tuple Gym API that `Quadrotor.step`
    /`Quadrotor.reset` actually implement, and so does this class, because the
    SAC runner calls `obs, info = env.reset()` and
    `obs, reward, done, info = env.step(action)`.
    '''

    def __init__(self, env, g_nom, seed=0, max_init_tilt=None):
        super().__init__(env)
        self.g_nom = g_nom
        self.max_init_tilt = max_init_tilt
        # Closed state space: sample exactly what the env permits.  Must be
        # read AFTER the caller has applied its termination bounds to
        # env.state_space (see rollout3d.apply_termination).
        self.sampling_bounds = sampling_bounds_from_env(env)
        self.rng = np.random.default_rng(seed)
        self._state = None
        # 18-dim (spec D6), built from `env` rather than hardcoded.
        self.observation_space = ctrl1_observation_space(env)

    def reset(self, **kwargs):
        init = sample_uniform_state(self.rng, self.sampling_bounds, self.max_init_tilt)
        obs, info = set_initial_state(self.env, init)
        # state_from_env gives the 13-dim reward/termination state;
        # ctrl1_observation is the SEPARATE 18-dim view handed to the policy.
        self._state = np.asarray(state_from_env(self.env, obs), dtype=float)
        return ctrl1_observation(self.env, obs), info

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        next_state = np.asarray(state_from_env(self.env, obs), dtype=float)
        in_g_nom = bool(self.g_nom.contains(
            tilt_from_quat_wxyz(next_state[QUAT_SLICE]), omega_norm(next_state)))
        # Reward depends only on attitude and never on why `done` fired --
        # `done` itself (out-of-bounds, or the underlying stabilization task's
        # unrelated goal_reached) still ends the episode, unscored.
        reward = shaped_reward(self._state, next_state, in_g_nom)
        self._state = next_state
        return ctrl1_observation(self.env, obs), reward, bool(done or in_g_nom), info
