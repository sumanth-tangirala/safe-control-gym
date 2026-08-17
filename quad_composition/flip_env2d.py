'''Training environment for controller 1 -- the flip controller.

The reward is ATTITUDE ONLY (spec D2).  Controller 1 has authority over attitude
and essentially none over position or translational velocity, so rewarding it for
those would (a) ask for something it cannot deliver and (b) pull G1 toward RoA2,
which is precisely the contrivance this experiment exists to avoid.
'''

import gymnasium as gym
import numpy as np

from quad_composition.g1 import G1Region
from quad_composition.rollout2d import (ctrl1_observation, ctrl1_observation_space, normalize_angle,
                                        set_initial_state, state_from_env)

# Nominal training target.  This is NOT the G1 that triggers handoff -- that is
# calibrated later from measured exits (spec D1 step 2) and may be looser.
G_NOM = G1Region(tilt_c=0.175, w_c=1.0)     # 10 deg, 1 rad/s

SHAPING_GAMMA = 0.99
BONUS = 100.0

# Normalisers for the potential: full tilt range and the theta_dot bound.
TILT_SCALE = np.pi
RATE_SCALE = 8.0

# Closed state space, dataset order [x, z, theta, x_dot, z_dot, theta_dot].
STATE_LOW = np.array([-1.0, 0.1, -np.pi, -1.0, -1.0, -8.0])
STATE_HIGH = np.array([1.0, 1.5, np.pi, 1.0, 1.0, 8.0])


def potential(state):
    '''Phi(s), attitude only.  Higher is closer to upright and still.

    `state[2]` MUST be TRUE attitude on [-pi, pi] -- i.e. a dataset-order row
    built by `rollout2d.state_from_env`, never by `state_from_obs` (Finding
    C1). The env's own observed theta is gimbal-folded into [-pi/2, pi/2], so
    scoring it would rate a fully inverted drone (true theta pi, folded 0) as
    perfectly upright, and `FlipTrainingEnv.step` would pay the +100 G_NOM
    BONUS and terminate there -- training controller 1 to STAY inverted,
    which is the exact opposite of its objective.
    '''
    theta = normalize_angle(float(state[2]))
    theta_dot = float(state[5])
    return -(abs(theta) / TILT_SCALE + abs(theta_dot) / RATE_SCALE)


def shaped_reward(state, next_state, in_g_nom):
    '''Potential-based shaping plus the G_nom entry bonus. ATTITUDE ONLY.

    No out-of-bounds penalty (Fix round 1, Critical): `info['out_of_bounds']`
    is computed by `Quadrotor._get_done()` from a mask over
    [x, x_dot, z, z_dot, theta_dot] -- i.e. it is True whenever POSITION or
    TRANSLATIONAL VELOCITY leaves bounds, not just attitude. Scoring it would
    make the reward depend on exactly the variables `G1` must not be defined
    over, biasing the handoff region toward `RoA2` -- the contrivance this
    experiment exists to avoid. Out-of-bounds still ENDS the episode (via
    `FlipTrainingEnv.step()`'s own `done`), which already costs the policy
    all future shaping and the bonus; it is simply not scored on top of that.

    Potential-based shaping leaves the optimal policy unchanged, so the bonus is
    what the policy actually optimises and the shaping only speeds it up.
    '''
    reward = SHAPING_GAMMA * potential(next_state) - potential(state)
    if in_g_nom:
        reward += BONUS
    return reward


def sampling_bounds_from_env(env):
    '''Derive dataset-order sampling bounds from the env's OWN termination bounds.

    CLAUDE.md requires a CLOSED state space: the initialized and achieved state
    spaces must be identical. Hardcoding STATE_LOW/STATE_HIGH breaks that the
    moment a caller overrides a termination bound (as the relaxed-limits arm
    does for |x_dot|, |z_dot| and |theta_dot|) -- training would then sample a
    strictly smaller box than it is allowed to reach, and the policy would never
    see the fast states it must handle. Reading the bounds back off the env makes
    the two impossible to desynchronise.

    env state order  [x, x_dot, z, z_dot, theta, theta_dot]
    dataset order    [x, z, theta, x_dot, z_dot, theta_dot]

    theta is periodic and has an infinite termination bound, so it always samples
    the full circle rather than tracking the env.
    '''
    space = getattr(env, 'state_space', None)
    if space is None:
        # No state_space (test doubles): fall back to the shipped closed state
        # space. Real envs always have one, so this never silently narrows a
        # relaxed-limits run.
        return STATE_LOW.copy(), STATE_HIGH.copy()
    lo, hi = space.low, space.high
    low = np.array([lo[0], lo[2], -np.pi, lo[1], lo[3], lo[5]], dtype=float)
    high = np.array([hi[0], hi[2], np.pi, hi[1], hi[3], hi[5]], dtype=float)
    return low, high


def sample_uniform_state(rng, max_init_tilt=None, bounds=None):
    '''Uniform over the closed state space (spec: training distribution).

    `max_init_tilt` (radians) optionally narrows the INITIAL theta range only.
    It exists because recovery above ~90 deg is not achievable in this system:
    a hand-coded bang-bang flip reaches G_nom on 0.593 / 0.194 / 0.061 of starts
    at |theta| of 0-30 / 30-60 / 60-90 deg, and 0.000 above 90 deg -- the same
    cliff the safe_explorer_ppo baseline shows, and consistent with the analytic
    budget's ~107 deg ceiling at zdot=0. Sampling the full +-pi range therefore
    spends about two thirds of every episode on physically impossible tasks and
    dilutes the learning signal by the same factor.

    This narrows TRAINING only. Evaluation, calibration and dataset generation
    keep the full closed state space, so nothing downstream is affected.
    '''
    low, high = bounds if bounds is not None else (STATE_LOW, STATE_HIGH)
    state = rng.uniform(low, high)
    if max_init_tilt is not None:
        state[2] = rng.uniform(-max_init_tilt, max_init_tilt)
    return state


class FlipTrainingEnv(gym.Wrapper):
    '''Attitude-only objective over the 2D quadrotor.

    reset() places the drone at a uniform sample of the closed state space;
    step() replaces the env reward with the attitude-only shaped reward
    (see shaped_reward: potential-based shaping on theta/theta_dot, plus a
    bonus on G_nom entry -- nothing else) and terminates on G_nom entry or
    out-of-bounds. Termination on out-of-bounds is UNSCORED: it still ends
    the episode via `done`, forfeiting future shaping and the bonus, but no
    separate penalty is added, since out-of-bounds depends on position and
    translational velocity and penalizing it would bias G1 toward RoA2.

    Reward and G_nom termination are computed on TRUE attitude
    (`rollout2d.state_from_env`), not on the env's gimbal-folded observation
    theta (Finding C1 -- see `potential`).

    OBSERVATION (spec D6, ruling on the open item flagged after the C1/C2 fix
    wave -- see .superpowers/sdd/2026-08-16-quad2d-composition/final-fix-
    report.md's "OPEN ITEM" section): the policy does NOT see the env's
    native folded observation. Folding aliases true pitch t with
    sign(t)*pi - t, so an upright drone and an inverted one would otherwise
    present the identical observation while their dynamics are opposite
    (thrust up vs thrust down) -- an unlearnable POMDP on exactly the
    distinction controller 1 exists to make. Instead, `reset()`/`step()`
    return `rollout2d.ctrl1_observation(env, obs)`: the env's native 6-dim
    obs `[x, x_dot, z, z_dot, theta, theta_dot]` with the folded theta
    element replaced by `(cos(true_theta), sin(true_theta))`, keeping the
    other five elements unchanged and in their existing positions --

        [x, x_dot, z, z_dot, cos(theta_true), sin(theta_true), theta_dot]

    7-dim, matching `self.observation_space` (`rollout2d.
    ctrl1_observation_space`). `rollout2d.rollout_composite` feeds
    controller 1 the SAME transform at inference time via `_act_ctrl1`, and
    `rollout2d.load_ctrl1` sizes controller 1's network to this same 7-dim
    space -- all three must agree, or controller 1 sees a distribution it
    never trained on. CONTROLLER 2's observation is untouched by any of this
    (`rollout2d._act_ctrl2`) -- redefining it would redefine its region of
    attraction and destroy comparability with the shipped baseline.

    Base class note (Ruling D-A/D-B): this codebase has no standalone `gym`
    package installed -- only `gymnasium`, imported as `gym` by every wrapper
    in `safe_control_gym` (e.g. `RecordEpisodeStatistics`). Those wrappers
    override `reset()`/`step()` to preserve the OLD 4-tuple/2-tuple Gym API
    that `Quadrotor.step()`/`Quadrotor.reset()` actually implement, rather
    than gymnasium's newer 5-tuple `step()`. This class follows the same
    pattern: `gymnasium.Wrapper` for the constructor/plumbing, old-style
    `reset()`/`step()` signatures to match the wrapped env and the SAC
    runner (`safe_control_gym/controllers/sac/sac.py`), which calls
    `obs, info = env.reset()` and `obs, reward, done, info = env.step(action)`.
    '''

    def __init__(self, env, g_nom, seed=0, max_init_tilt=None):
        super().__init__(env)
        self.g_nom = g_nom
        self.max_init_tilt = max_init_tilt
        # Closed state space: sample exactly what the env permits (see
        # sampling_bounds_from_env). Must be read AFTER the caller has applied
        # its termination bounds to env.state_space.
        self.sampling_bounds = sampling_bounds_from_env(env)
        self.rng = np.random.default_rng(seed)
        self._state = None
        # 7-dim (spec D6): the env's native observation_space has the folded
        # theta bound in its place. Built from `env`, not hardcoded, so this
        # always matches whatever quadrotor config the caller wrapped.
        self.observation_space = ctrl1_observation_space(env)

    def reset(self, **kwargs):
        init = sample_uniform_state(self.rng, self.max_init_tilt, self.sampling_bounds)
        obs, info = set_initial_state(self.env, init)
        # state_from_env reads TRUE attitude for the reward/termination state
        # (Finding C1); ctrl1_observation is the SEPARATE 7-dim transform
        # (spec D6) returned to the policy -- see this class's docstring.
        self._state = np.asarray(state_from_env(self.env, obs), dtype=float)
        return ctrl1_observation(self.env, obs), info

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        # state_from_env, not state_from_obs: the reward and the G_NOM
        # termination test must see TRUE attitude, not the env's gimbal-folded
        # observation (Finding C1 -- see `potential`).
        next_state = np.asarray(state_from_env(self.env, obs), dtype=float)
        in_g_nom = bool(self.g_nom.contains(abs(next_state[2]), abs(next_state[5])))
        # Reward depends only on attitude (shaped_reward) and never on why
        # `done` fired -- `done` itself (out-of-bounds, or the original
        # stabilization task's unrelated goal_reached) still ends the
        # episode, but is not separately scored (Fix round 1, Critical: see
        # shaped_reward's docstring).
        reward = shaped_reward(self._state, next_state, in_g_nom)
        self._state = next_state
        # The policy receives the 7-dim unfolded observation (spec D6), not
        # the raw `obs` -- see this class's docstring and `ctrl1_observation`.
        return ctrl1_observation(self.env, obs), reward, bool(done or in_g_nom), info
