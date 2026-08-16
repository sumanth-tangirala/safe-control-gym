'''Training environment for controller 1 -- the flip controller.

The reward is ATTITUDE ONLY (spec D2).  Controller 1 has authority over attitude
and essentially none over position or translational velocity, so rewarding it for
those would (a) ask for something it cannot deliver and (b) pull G1 toward RoA2,
which is precisely the contrivance this experiment exists to avoid.
'''

import gymnasium as gym
import numpy as np

from quad_composition.g1 import G1Region
from quad_composition.rollout2d import normalize_angle, set_initial_state, state_from_obs

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
    '''Phi(s), attitude only.  Higher is closer to upright and still.'''
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
        self._state = np.asarray(state_from_obs(obs), dtype=float)
        return obs, info

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        next_state = np.asarray(state_from_obs(obs), dtype=float)
        in_g_nom = bool(self.g_nom.contains(abs(next_state[2]), abs(next_state[5])))
        # Reward depends only on attitude (shaped_reward) and never on why
        # `done` fired -- `done` itself (out-of-bounds, or the original
        # stabilization task's unrelated goal_reached) still ends the
        # episode, but is not separately scored (Fix round 1, Critical: see
        # shaped_reward's docstring).
        reward = shaped_reward(self._state, next_state, in_g_nom)
        self._state = next_state
        return obs, reward, bool(done or in_g_nom), info
