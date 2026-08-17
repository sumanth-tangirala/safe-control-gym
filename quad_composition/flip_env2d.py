'''Training environment for controller 1 -- the flip controller.

The reward is ATTITUDE ONLY (spec D2).  Controller 1 has authority over attitude
and essentially none over position or translational velocity, so rewarding it for
those would (a) ask for something it cannot deliver and (b) pull G1 toward RoA2,
which is precisely the contrivance this experiment exists to avoid.
'''

import gymnasium as gym
import numpy as np

from quad_composition.g1 import G1Region
from quad_composition.rollout2d import normalize_angle, set_initial_state, state_from_env

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


def sample_uniform_state(rng):
    '''Uniform over the closed state space (spec: training distribution).'''
    return rng.uniform(STATE_LOW, STATE_HIGH)


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

    OPEN ITEM, deliberately NOT changed here: the OBSERVATION handed to the
    policy is still the env's native folded one. Folding aliases true pitch t
    with sign(t)*pi - t, so an upright drone and an inverted one can present
    the identical observation while their dynamics are opposite (thrust up vs
    thrust down). Controller 1 therefore trains on an aliased, partially
    observed attitude. The fix dispatch that introduced `state_from_env`
    ruled explicitly on which four decisions must use true attitude (G1
    membership, the flip reward/potential, G_nom termination, G1 calibration)
    and on leaving CONTROLLER 2's observation alone; it did not rule on
    controller 1's observation, and unfolding it is a real design change (new
    observation semantics for the training env, and a matching change to what
    `rollout2d.rollout_composite` feeds ctrl1 at eval time). No controller-1
    checkpoint has been trained yet, so nothing is invalidated by deferring
    it -- but it must be ruled on before the 1M-step training run is launched.

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

    def __init__(self, env, g_nom, seed=0):
        super().__init__(env)
        self.g_nom = g_nom
        self.rng = np.random.default_rng(seed)
        self._state = None

    def reset(self, **kwargs):
        init = sample_uniform_state(self.rng)
        obs, info = set_initial_state(self.env, init)
        self._state = np.asarray(state_from_env(self.env, obs), dtype=float)
        return obs, info

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        # state_from_env, not state_from_obs: the reward and the G_NOM
        # termination test must see TRUE attitude, not the env's gimbal-folded
        # observation (Finding C1 -- see `potential`). `obs` itself is returned
        # untouched, so the policy keeps its native observation.
        next_state = np.asarray(state_from_env(self.env, obs), dtype=float)
        in_g_nom = bool(self.g_nom.contains(abs(next_state[2]), abs(next_state[5])))
        # Reward depends only on attitude (shaped_reward) and never on why
        # `done` fired -- `done` itself (out-of-bounds, or the original
        # stabilization task's unrelated goal_reached) still ends the
        # episode, but is not separately scored (Fix round 1, Critical: see
        # shaped_reward's docstring).
        reward = shaped_reward(self._state, next_state, in_g_nom)
        self._state = next_state
        return obs, reward, bool(done or in_g_nom), info
