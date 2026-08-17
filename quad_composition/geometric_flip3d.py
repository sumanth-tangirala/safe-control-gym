'''Controller 1 as a HAND-CODED reduced-attitude geometric controller.

WHY THIS EXISTS.  Controller 1's job is to take the 3D quadrotor from an
arbitrary attitude (including full inversion) into G1 and hand off to
controller 2 (LQR).  SAC is one way to produce such a controller; this module
is another, and it is the one that demonstrably works today.  A feasibility
probe of exactly the law implemented below, run against the real PyBullet
dynamics under the `rollout3d.TERMINATION` limits, recovers from every tilt
bucket including 150-180 deg, and reaches G_nom from 100% of isolated
attitude-only inversions.  It is deterministic, stateless and has four
scalars to tune, so it is also a stable baseline the learned controller can be
measured against.

DROP-IN CONTRACT.  `rollout3d.rollout_composite` never mentions SAC; it uses
controller 1 through exactly two attributes (`rollout3d._act_ctrl1`):

    ctrl1.select_action(ctrl1.obs_normalizer(ctrl1_observation(env, obs)), info)

so this class supplies an `obs_normalizer` (identity -- there is nothing to
normalise in a hand-coded law) and a `select_action`.  `close()` is here for
the same reason `calibrate_quad3d_g1.py` calls it on the SAC object.  NOTHING
IN `rollout3d.py` NEEDED TO CHANGE for this controller to be usable as
controller 1.

THE CONTROL LAW.  Reduced-attitude ("tilt-only") geometric PD plus bang-bang
collective:

    b3      = R[:, 2]                     body z axis, in world coordinates
    e_world = cross(b3, z_hat)            rotation axis that takes b3 -> z_hat
    e_world = e_world / |e_world|         ONLY below the horizon (R[2, 2] < 0)
    e_body  = R.T @ e_world               ... expressed in the body frame
    tau_xy  = k_p * e_body[:2] - k_d * [p, q]
    tau_z   = -k_d * r                    yaw is damped, never commanded
    T       = T_up if R[2, 2] > 0 else T_down

THE ONE CHANGE FROM THE FEASIBILITY PROBE is that normalisation: `|e_world|`
is `sin(tilt)`, which DECAYS to zero as the drone approaches full inversion,
exactly where the most torque is needed.  An inverted drone has ~0.21 s before
free fall (thrust points DOWN while inverted, so it accelerates at g + T/m
~ 14 m/s^2) takes `z_dot` through its 3.0 m/s termination bound, and the
un-normalised law spends that budget rolling at a few rad/s.  Rescaling the
error to unit length below the horizon holds the command at full authority all
the way through inversion.  It is CONTINUOUS at the switch -- `sin(tilt) = 1`
at `R[2, 2] = 0` -- it leaves the upper hemisphere (where the drone is
recovering, not flipping) bit-identical to the probe, and it leaves the
closed-loop rate ceiling at `k_p / k_d = 20` rad/s, still inside the 24 rad/s
bound, because the error magnitude is still at most 1.  MEASURED (600
rollouts each, `evaluate_geometric_flip3d.py`): in the isolated attitude-only
setting it lifts the 150-180 deg bucket from 86% to 100% and fixes exact
inversion outright (the un-normalised law free-falls through the 3.0 m/s
`z_dot` bound in 18 ticks while still at 124 deg of tilt; this one is upright
in 22).  In the full closed state space it is neutral -- 50.0% -> 50.5% of
rollouts reach G1 -- because there the binding constraint is the initial
position/velocity draw, not attitude.  Set `saturate_inverted=False` to
recover the probe's exact law.

Yaw is deliberately unregulated: the tilt error is invariant to rotation about
body z, so commanding a yaw setpoint would spend authority on a coordinate G1
does not score (`g1.G1Region` is (tilt, |omega|) only).  The collective is
bang-bang on the SIGN of `R[2, 2]` because thrust acts along +b3: while the
drone is inverted, thrust pushes it DOWN, so the only safe collective is the
smallest one the actuator can produce, and the moment b3 crosses the horizon
the drone needs the largest one it can produce to arrest the fall.  Both
extremes are read off the env's own `physical_action_bounds`.

ATTITUDE COMES FROM THE ROTATION MATRIX, NEVER FROM EULER ANGLES.  This law
reads `R` straight out of controller 1's 18-dim observation (spec D6), where
`rollout3d.ctrl1_observation` has already put the nine entries of
`rollout3d.rotation_matrix_from_quat(env.quat)`.  There is no Euler angle
anywhere on this path -- see `rollout3d.tilt_from_quat` for what folding pitch
into [-pi/2, pi/2] did to the 2D branch (Finding C1: it trained controller 1
to STAY inverted).

SATURATION IS PART OF THE DESIGN, NOT AN AFTERTHOUGHT.  `k_p = 0.1` commands
a roll torque far beyond what four motors bounded by `physical_action_bounds`
can deliver (the achievable maximum is ~0.0135 N m), so the per-motor clip
turns the proportional term into a saturating, near-bang-bang law while
`k_d = 0.005` still sets the closed-loop rate ceiling at `k_p / k_d = 20`
rad/s -- just inside the env's 24 rad/s body-rate termination bound.  Raising
k_p alone therefore does nothing at all; only the RATIO k_p/k_d moves the
closed loop, and it is already near its optimum.  Re-swept over
k_d in {0.004, 0.005, 0.0065, 0.008} on a common 240-rollout closed-box draw,
the shipped 0.005 wins (58.3% of rollouts reach G1, against 53.3%, 53.8% and
47.1%) -- too tight and the drone overshoots into the rate bound, too loose
and it never turns over in the time the fall allows.
'''

import numpy as np

# Layout of controller 1's 18-dim observation, `rollout3d.ctrl1_observation`:
#   [x, x_dot, y, y_dot, z, z_dot, R00..R22, p, q, r]
# Indices 6:15 are the body->world rotation matrix, ROW MAJOR; 15:18 the body
# rates.  Reading them by slice (rather than recomputing from `env.quat`) is
# what keeps `select_action` stateless and env-free.
OBS_DIM = 18
ROT_SLICE = slice(6, 15)
OBS_RATE_SLICE = slice(15, 18)

# Validated gains (feasibility probe, quadrotor3D_lqr limits).
K_P = 0.1
K_D = 0.005
# Fractions of the env's TOTAL actuator range: T_up = frac * 4 * a_high,
# T_down = frac * 4 * a_high, with None meaning "the actuator's own minimum",
# i.e. 4 * a_low.  With the shipped 3D quadrotor that is T_up = 0.4747 N and
# T_down = 0.1126 N.
THRUST_UP_FRAC = 0.8
THRUST_DOWN_FRAC = None

# Rescale the tilt error to unit length while the drone is below the horizon,
# so the command does not decay to zero at full inversion (module docstring).
SATURATE_INVERTED = True

# Collective law above the horizon.  'bangbang' is the feasibility probe's:
# T_up whenever R[2, 2] > 0.  'compensated' instead asks for the thrust whose
# VERTICAL component is hover, `m g / R[2, 2]`, capped at T_up -- identical to
# 'bangbang' while the drone is tilted enough to need everything it has, and
# altitude-holding once it is not.
#
# 'bangbang' REMAINS THE DEFAULT because 'compensated' did not measurably beat
# it: 60.4% vs 58.3% of rollouts reaching G1 on a common 240-rollout draw, a
# 2-point difference against a ~3-point standard error.  It is kept as an
# option because it is the obvious next thing to try at larger N, not because
# it is known to be better.
COLLECTIVE = 'bangbang'

# At EXACTLY pi of tilt, `cross(b3, z_hat)` is exactly zero: inversion is an
# equilibrium of the tilt error, an unstable one, and a purely reactive law
# sits there forever.  Below this norm the law substitutes a fixed body-x
# error of `SINGULAR_KICK` to break the symmetry deterministically.  Magnitude
# 1.0 is the same error the law sees at 90 deg of tilt, so it produces no
# torque the controller does not already command elsewhere (and the same
# actuator saturation bounds it).
SINGULAR_EPS = 1e-9
SINGULAR_KICK = 1.0


def mixer(env):
    '''The X-configuration mixer `[T, tau_x, tau_y, tau_z] = M @ f` for `env`.

    Rows are total thrust, roll, pitch and yaw torque; columns are the four
    motors in the env's own order.  Built from the env's `KF`, `KM` and `L`
    rather than hardcoded, so a different drone URDF cannot silently
    desynchronise it.

    SIGNS ARE VALIDATED AGAINST THE REAL DYNAMICS, not derived on paper: a
    positive commanded tau_x / tau_y / tau_z produces a positive p / q / r
    after one control step from hover, with no cross-coupling (measured
    (1.425, 0, 0), (0, 1.425, 0), (0, 0, 0.230) rad/s for commands of
    (0.002, 0, 0), (0, 0.002, 0), (0, 0, 0.0005) N m).  Do not "fix" a sign
    here without re-running that check.
    '''
    arm_xy = env.L / np.sqrt(2.0)
    gamma = env.KM / env.KF
    return np.array([
        [1.0, 1.0, 1.0, 1.0],
        [arm_xy, arm_xy, -arm_xy, -arm_xy],
        [-arm_xy, arm_xy, arm_xy, -arm_xy],
        [-gamma, gamma, -gamma, gamma],
    ])


class IdentityNormalizer:
    '''Stands in for SAC's `obs_normalizer` on the `_act_ctrl1` path.

    A hand-coded law has no training statistics to normalise against, and
    rescaling `R` would corrupt a rotation matrix, so this is the identity.
    `set_read_only` exists because every caller that freezes a SAC
    normalizer (`rollout3d.load_ctrl1`, `visualize_quad3d_composition`) calls
    it unconditionally.
    '''

    def __call__(self, obs):
        return obs

    def set_read_only(self):
        return None


class GeometricFlipController3D:
    '''Hand-coded controller 1: reduced-attitude geometric PD + bang-bang
    collective, acting on the PHYSICAL per-motor thrusts.

    DETERMINISTIC AND STATELESS BY CONSTRUCTION: `select_action` is a pure
    function of its observation argument.  There is no phase latch, no
    integrator and no RNG, so the same state always produces the same action
    and a rollout can be restarted mid-flight without changing its future.
    (The composition's ONE latch lives in `rollout3d.rollout_composite`, which
    is where the experiment's semantics say it belongs -- spec D3.)

    Gains are constructor arguments so they can be swept; the defaults are the
    validated ones.
    '''

    def __init__(self, env, k_p=K_P, k_d=K_D,
                 thrust_up_frac=THRUST_UP_FRAC, thrust_down_frac=THRUST_DOWN_FRAC,
                 saturate_inverted=SATURATE_INVERTED, collective=COLLECTIVE,
                 singular_eps=SINGULAR_EPS, singular_kick=SINGULAR_KICK):
        '''`env` is read ONCE, at construction, for its actuator geometry and
        `physical_action_bounds`; no reference to it is used at action time.

        The 3D env leaves `normalized_rl_action_space` at its False default
        (rollout3d.py docstring, item 4), so `physical_action_bounds` IS the
        action space this controller must live inside, and the returned action
        is per-motor thrust in newtons.
        '''
        self.k_p = float(k_p)
        self.k_d = float(k_d)
        low, high = env.physical_action_bounds
        self.action_low = np.asarray(low, dtype=float).copy()
        self.action_high = np.asarray(high, dtype=float).copy()
        self.mixer_inv = np.linalg.inv(mixer(env))
        thrust_max = float(np.sum(self.action_high))
        thrust_min = float(np.sum(self.action_low))
        self.thrust_up = thrust_max * float(thrust_up_frac)
        self.thrust_down = (thrust_min if thrust_down_frac is None
                            else thrust_max * float(thrust_down_frac))
        self.saturate_inverted = bool(saturate_inverted)
        if collective not in ('bangbang', 'compensated'):
            raise ValueError(f"collective must be 'bangbang' or 'compensated', got {collective!r}")
        self.collective = collective
        self.hover_thrust = float(env.MASS * env.GRAVITY_ACC)
        self.singular_eps = float(singular_eps)
        self.singular_kick = float(singular_kick)
        # `_act_ctrl1` pipes the observation through this before handing it
        # over; identity here (see IdentityNormalizer).
        self.obs_normalizer = IdentityNormalizer()

    def select_action(self, obs, info=None):
        '''Per-motor thrusts (N, shape (4,)) for controller 1's 18-dim
        observation.

        `obs` must be `rollout3d.ctrl1_observation(env, env_obs)`: the rotation
        matrix at 6:15 and the body rates at 15:18.  `info` is accepted and
        ignored -- it exists only to match SAC's signature, and using it would
        make the law depend on the env's bookkeeping rather than on the state.

        The action is clipped ELEMENTWISE to `physical_action_bounds`.  The
        clip is not cosmetic: the proportional term routinely commands motor
        thrusts several times the actuator's range, and it is the clip that
        turns this into the saturating law that was measured.
        '''
        obs = np.asarray(obs, dtype=float).reshape(-1)
        if obs.shape[0] != OBS_DIM:
            raise ValueError(f'expected the {OBS_DIM}-dim controller-1 observation '
                             f'(rollout3d.ctrl1_observation), got {obs.shape[0]} dims')
        rot = obs[ROT_SLICE].reshape(3, 3)
        rates = obs[OBS_RATE_SLICE]

        # Reduced-attitude error: the axis (in the body frame) of the shortest
        # rotation carrying the body z axis onto world z.  |e| = sin(tilt).
        body_z_world = rot[:, 2]
        e_world = np.cross(body_z_world, np.array([0.0, 0.0, 1.0]))
        e_norm = float(np.linalg.norm(e_world))
        if e_norm < self.singular_eps:
            # `e` vanishes at pi of tilt just as it does at 0, and only the
            # sign of R[2, 2] tells the two apart.  Upright: nothing to
            # correct.  Inverted: roll, in a fixed direction, so the law stays
            # deterministic rather than sitting on the equilibrium forever.
            e_body = (np.zeros(3) if rot[2, 2] > 0.0
                      else np.array([self.singular_kick, 0.0, 0.0]))
        else:
            if self.saturate_inverted and rot[2, 2] < 0.0:
                # Below the horizon, |e| = sin(tilt) decays toward inversion;
                # hold full authority instead.  Continuous at the switch,
                # where sin(tilt) is already 1.  (rot.T is a rotation, so this
                # bounds |e_body| by 1 exactly as the un-normalised law does.)
                e_world = e_world / e_norm
            e_body = rot.T @ e_world

        tau = np.array([
            self.k_p * e_body[0] - self.k_d * rates[0],
            self.k_p * e_body[1] - self.k_d * rates[1],
            -self.k_d * rates[2],
        ])
        if rot[2, 2] <= 0.0:
            # Below the horizon thrust pushes DOWN; give it as little as the
            # actuator allows, whatever the collective mode.
            thrust = self.thrust_down
        elif self.collective == 'compensated':
            # m g / R[2, 2] is the thrust whose VERTICAL component is hover.
            # Capped at T_up, so a steeply tilted drone still gets everything
            # `bangbang` would have given it; once it is nearly upright it
            # holds altitude instead of climbing into the z_dot bound.
            thrust = min(self.thrust_up, self.hover_thrust / rot[2, 2])
        else:
            thrust = self.thrust_up
        motors = self.mixer_inv @ np.concatenate([[thrust], tau])
        return np.clip(motors, self.action_low, self.action_high)

    def reset(self):
        '''No state to clear.  Present so this object can stand in for a
        controller wherever `BaseController.reset()` is called.
        '''
        return None

    def close(self):
        '''No resources held.  Present because the calibration and rollout
        scripts call `ctrl1.close()` in their `finally` blocks.
        '''
        return None


def make_ctrl1(env, **gains):
    '''Build controller 1 as the hand-coded geometric controller.

    The counterpart of `rollout3d.load_ctrl1` (which builds it as SAC from a
    checkpoint): same role, same call site, but no checkpoint, no observation
    space wrapper and no network -- the law needs only the env's actuator
    constants.  Either object may be passed as `ctrl1` to
    `rollout3d.rollout_composite`.
    '''
    return GeometricFlipController3D(env, **gains)
