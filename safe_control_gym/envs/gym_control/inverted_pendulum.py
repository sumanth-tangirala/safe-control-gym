'''Inverted pendulum environment (a torque-limited, single-link pendulum).

Ported from the standalone ``inverted_pendulum`` system so that its trained LQR
and SAC controllers reproduce faithfully inside safe-control-gym. The state is
``[theta, theta_dot]`` with ``theta = 0`` the **upright unstable** equilibrium.
Dynamics ``theta_ddot = (g/l) sin(theta) + u/I - (b/I) theta_dot`` (``I = m l^2``)
are integrated with **explicit Euler**, after which ``theta`` is wrapped to
``[-pi, pi]`` and ``theta_dot`` is clipped to ``+/- theta_dot_max``. These
details (integrator, wrap, clip, ``u_sat``) are load-bearing: the trained
policies were trained under them.

Unlike cartpole/quadrotor, ``theta_dot`` is **clipped** (not terminated) at its
bound and ``theta`` is periodic, so there is no out-of-bounds failure; an episode
ends only on reaching the upright goal or on the time limit.
'''

import math
from copy import deepcopy

import casadi as cs
import numpy as np
from gymnasium import spaces

from safe_control_gym.controllers.lqr.lqr_utils import get_cost_weight_matrix
from safe_control_gym.envs.benchmark_env import BenchmarkEnv, Cost, Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS
from safe_control_gym.envs.gym_control.pendulum_noise import build_noise_model
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel

# Default u_sat matches the source system (ML4KP-derived control saturation).
U_SAT_DEFAULT = 0.6371781908344007


def _wrap_to_pi(theta):
    '''Wrap an angle to [-pi, pi], matching the source env exactly.'''
    return (theta + math.pi) % (2 * math.pi) - math.pi


class InvertedPendulum(BenchmarkEnv):
    '''Single-link inverted pendulum with an explicit-Euler simulator.'''

    NAME = 'inverted_pendulum'

    AVAILABLE_CONSTRAINTS = deepcopy(GENERAL_CONSTRAINTS)

    DISTURBANCE_MODES = {'observation': {'dim': 2}, 'action': {'dim': 1}, 'dynamics': {'dim': 2}}

    INERTIAL_PROP_RAND_INFO = {
        'pendulum_mass': {'distrib': 'uniform', 'low': -0.01, 'high': 0.01},
        'pendulum_length': {'distrib': 'uniform', 'low': -0.01, 'high': 0.01},
    }

    # Nominal init is upright at rest; randomization covers the full state space
    # (theta over the circle, theta_dot over its bound).
    INIT_STATE_RAND_INFO = {
        'init_theta': {'distrib': 'uniform', 'low': -math.pi, 'high': math.pi},
        'init_theta_dot': {'distrib': 'uniform', 'low': -2 * math.pi, 'high': 2 * math.pi},
    }

    TASK_INFO = {
        'stabilization_goal': [0, 0],
        'stabilization_goal_tolerance': 0.075,
    }

    def __init__(self,
                 init_state=None,
                 inertial_prop=None,
                 # physical parameters (defaults match the source system)
                 gravity=9.81,
                 pendulum_mass=0.15,
                 pendulum_length=0.5,
                 damping=0.1,
                 u_sat=U_SAT_DEFAULT,
                 theta_dot_max=2 * math.pi,
                 goal_threshold=0.075,
                 noise=None,
                 # custom args
                 obs_goal_horizon=0,
                 rew_state_weight=1.0,
                 rew_act_weight=0.0001,
                 rew_exponential=True,
                 info_mse_metric_state_weight=None,
                 **kwargs
                 ):
        '''Initialize an inverted pendulum environment.

        Args:
            init_state (ndarray/dict, optional): initial ``[theta, theta_dot]``.
            inertial_prop (dict, optional): ground-truth ``pendulum_mass`` / ``pendulum_length``.
            gravity, pendulum_mass, pendulum_length, damping (float): physical constants.
            u_sat (float): control saturation; action clipped to ``[-u_sat, u_sat]``.
            theta_dot_max (float): angular-velocity bound; ``theta_dot`` clipped to it.
            goal_threshold (float): L2 goal tolerance around upright at rest.
            noise: source-system noise config -- ``None`` (deterministic), a preset
                name (e.g. ``'truncated_gaussian_act_med'``), a ``{'type': ...}``
                dict, or a ``NoiseModel`` instance (see ``pendulum_noise``).
            obs_goal_horizon (int): future goal states appended to obs (RL).
            rew_state_weight, rew_act_weight (list/float): rl_reward quadratic weights.
            rew_exponential (bool): exponentiate the negative quadratic reward.
            info_mse_metric_state_weight (list, optional): weights for the info-dict mse.
        '''
        self.GRAVITY_ACC = gravity
        self.PENDULUM_MASS = pendulum_mass
        self.PENDULUM_LENGTH = pendulum_length
        self.DAMPING = damping
        self.u_sat = float(u_sat)
        self.theta_dot_max = float(theta_dot_max)
        self.goal_threshold = float(goal_threshold)
        # Optional source-system noise model (None -> deterministic no-op).
        self.noise_model = build_noise_model(noise)
        self.obs_goal_horizon = obs_goal_horizon
        self.rew_state_weight = np.array(rew_state_weight, ndmin=1, dtype=float)
        self.rew_act_weight = np.array(rew_act_weight, ndmin=1, dtype=float)
        self.Q = get_cost_weight_matrix(self.rew_state_weight, 2)
        self.R = get_cost_weight_matrix(self.rew_act_weight, 1)
        self.rew_exponential = rew_exponential

        if info_mse_metric_state_weight is None:
            self.info_mse_metric_state_weight = np.array([1, 1], ndmin=1, dtype=float)
        else:
            if len(info_mse_metric_state_weight) == 2:
                self.info_mse_metric_state_weight = np.array(info_mse_metric_state_weight, ndmin=1, dtype=float)
            else:
                raise ValueError('[ERROR] in InvertedPendulum.__init__(), wrong info_mse_metric_state_weight argument size.')

        # BenchmarkEnv constructor after custom args (setup can depend on them).
        super().__init__(init_state=init_state, inertial_prop=inertial_prop, **kwargs)

        # Store ground-truth inertial properties.
        if inertial_prop is None:
            pass
        elif isinstance(inertial_prop, dict):
            self.PENDULUM_MASS = inertial_prop.get('pendulum_mass', self.PENDULUM_MASS)
            self.PENDULUM_LENGTH = inertial_prop.get('pendulum_length', self.PENDULUM_LENGTH)
        else:
            raise ValueError('[ERROR] in InvertedPendulum.__init__(), inertial_prop incorrect format.')

        # Parse the (optional) initial state.
        if init_state is None:
            self.INIT_THETA, self.INIT_THETA_DOT = 0.0, 0.0
        elif isinstance(init_state, (np.ndarray, list, tuple)):
            self.INIT_THETA, self.INIT_THETA_DOT = np.array(init_state, dtype=float)
        elif isinstance(init_state, dict):
            self.INIT_THETA = init_state.get('init_theta', 0.0)
            self.INIT_THETA_DOT = init_state.get('init_theta_dot', 0.0)
        else:
            raise ValueError('[ERROR] in InvertedPendulum.__init__(), init_state incorrect format.')

        # Reference for the stabilization task: upright at rest.
        self.U_GOAL = np.zeros(1)
        self.X_GOAL = np.array(self.TASK_INFO['stabilization_goal'], dtype=float)

        self._setup_symbolic()

    @property
    def inertia(self):
        return self.PENDULUM_MASS * self.PENDULUM_LENGTH ** 2

    def step(self, action):
        '''Advance one control step (returns the old-Gym 4-tuple).'''
        u = super().before_step(action)
        u = float(np.asarray(u).flat[0])

        g, l, b, inertia = self.GRAVITY_ACC, self.PENDULUM_LENGTH, self.DAMPING, self.inertia
        dt = self.PYB_TIMESTEP
        theta, thetadot = float(self.state[0]), float(self.state[1])
        self.goal_reached = False
        for _ in range(self.PYB_STEPS_PER_CTRL):
            # Explicit Euler: both updates use the pre-step (old) state.
            thetaddot = (g / l) * math.sin(theta) + u / inertia - (b / inertia) * thetadot
            theta = theta + dt * thetadot
            thetadot = thetadot + dt * thetaddot
            # Process (dynamics) noise, applied before wrap/clip so the state
            # invariants still hold after perturbation (matches the source env).
            noisy = self.noise_model.add_dynamics_noise(
                self.np_random, np.array([theta, thetadot], dtype=np.float64), u)
            theta, thetadot = float(noisy[0]), float(noisy[1])
            theta = _wrap_to_pi(theta)
            thetadot = float(np.clip(thetadot, -self.theta_dot_max, self.theta_dot_max))
            self.state = np.array([theta, thetadot], dtype=np.float64)
            if np.linalg.norm(self.state - self.X_GOAL) < self.goal_threshold:
                self.goal_reached = True
                break

        obs = self._get_observation()
        rew = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        obs, rew, done, info = super().after_step(obs, rew, done, info)
        return obs, rew, done, info

    def reset(self, seed=None):
        '''(Re-)initialize the environment; returns ``(obs, info)``.'''
        super().before_reset(seed=seed)
        init_values = {'init_theta': self.INIT_THETA, 'init_theta_dot': self.INIT_THETA_DOT}
        if self.RANDOMIZED_INIT:
            init_values = self._randomize_values_by_info(init_values, self.INIT_STATE_RAND_INFO)
        theta = _wrap_to_pi(init_values['init_theta'])
        thetadot = float(np.clip(init_values['init_theta_dot'], -self.theta_dot_max, self.theta_dot_max))
        self.state = np.array([theta, thetadot], dtype=np.float64)
        self.goal_reached = False
        obs, info = self._get_observation(), self._get_reset_info()
        obs, info = super().after_reset(obs, info)
        return obs, info

    def render(self, mode='human'):
        '''Rendering is not supported for the analytic pendulum.'''
        return None

    def close(self):
        '''No external simulator to tear down.'''
        return None

    def _setup_symbolic(self, prior_prop={}, **kwargs):
        '''Create the CasADi symbolic dynamics/observation/cost model.'''
        g = self.GRAVITY_ACC
        l = prior_prop.get('pendulum_length', self.PENDULUM_LENGTH)
        m = prior_prop.get('pendulum_mass', self.PENDULUM_MASS)
        b = self.DAMPING
        inertia = m * l ** 2
        dt = self.CTRL_TIMESTEP
        # Input variables.
        theta = cs.MX.sym('theta')
        theta_dot = cs.MX.sym('theta_dot')
        X = cs.vertcat(theta, theta_dot)
        U = cs.MX.sym('U')
        nx, nu = 2, 1
        # Dynamics (theta = 0 upright).
        theta_ddot = (g / l) * cs.sin(theta) + U / inertia - (b / inertia) * theta_dot
        X_dot = cs.vertcat(theta_dot, theta_ddot)
        # Observation (full state).
        Y = cs.vertcat(theta, theta_dot)
        # Cost (quadratic).
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)
        cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}
        cost = {'cost_func': cost_func, 'vars': {'X': X, 'U': U, 'Xr': Xr, 'Ur': Ur, 'Q': Q, 'R': R}}
        params = {
            'pendulum_mass': m,
            'pendulum_length': l,
            'gravity': g,
            'damping': b,
            'X_EQ': np.zeros(self.state_dim),
            'U_EQ': np.atleast_2d(self.U_GOAL)[0, :],
        }
        self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt, params=params)

    def _set_action_space(self):
        '''Physical torque action space ``[-u_sat, u_sat]``.'''
        self.physical_action_bounds = (-np.atleast_1d(self.u_sat), np.atleast_1d(self.u_sat))
        self.action_threshold = 1 if self.NORMALIZED_RL_ACTION_SPACE else self.u_sat
        self.action_space = spaces.Box(low=-self.action_threshold, high=self.action_threshold, shape=(1,))
        self.ACTION_LABELS = ['U']
        self.ACTION_UNITS = ['N·m'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-']

    def _set_observation_space(self):
        '''State/observation space over ``[-pi, pi] x [-theta_dot_max, theta_dot_max]``.'''
        obs_bound = np.array([math.pi, self.theta_dot_max])
        self.state_space = spaces.Box(low=-obs_bound, high=obs_bound, dtype=np.float64)
        # Append goal state(s) for RL when requested.
        if self.COST == Cost.RL_REWARD and self.TASK == Task.STABILIZATION and self.obs_goal_horizon > 0:
            obs_bound = np.concatenate([obs_bound] * 2)
        self.observation_space = spaces.Box(low=-obs_bound, high=obs_bound, dtype=np.float64)
        self.STATE_LABELS = ['theta', 'theta_dot']
        self.STATE_UNITS = ['rad', 'rad/s']

    def _preprocess_control(self, action):
        '''Denormalize, apply action disturbances, and clip to ``[-u_sat, u_sat]``.'''
        action = self.denormalize_action(action)
        self.current_physical_action = action
        if 'action' in self.disturbances:
            action = self.disturbances['action'].apply(action, self)
        if self.adversary_disturbance == 'action' and self.adv_action is not None:
            action = action + self.adv_action
        # Source-system actuation noise, applied to the command before the u_sat clip.
        u = self.noise_model.add_act_noise(self.np_random, float(np.asarray(action).reshape(-1)[0]))
        action = np.array([u], dtype=np.float64)
        self.current_noisy_physical_action = action
        force = np.clip(action, self.physical_action_bounds[0], self.physical_action_bounds[1])
        self.current_clipped_action = force
        return force[0]

    def normalize_action(self, action):
        '''Physical -> normalized action (identity unless the RL space is normalized).'''
        if self.NORMALIZED_RL_ACTION_SPACE:
            action = action / self.u_sat
        return action

    def denormalize_action(self, action):
        '''Normalized -> physical action (identity unless the RL space is normalized).'''
        if self.NORMALIZED_RL_ACTION_SPACE:
            action = self.u_sat * action
        return action

    def _get_observation(self):
        '''Return the observation (physical state, plus optional goal horizon).'''
        obs = deepcopy(self.state)
        if 'observation' in self.disturbances:
            obs = self.disturbances['observation'].apply(obs, self)
        # Source-system observation noise (on returned obs only, not on reset,
        # so the true state used for goal checks stays clean).
        if not self.at_reset:
            obs = self.noise_model.add_obs_noise(self.np_random, obs)
        if self.at_reset:
            obs = self.extend_obs(obs, 1)
        else:
            obs = self.extend_obs(obs, self.ctrl_step_counter + 2)
        return obs

    def _get_reward(self):
        '''Compute the step reward/cost.'''
        if self.COST == Cost.RL_REWARD:
            state = deepcopy(self.state)
            act = np.asarray(self.current_noisy_physical_action)
            state_error = state - self.X_GOAL
            dist = np.sum(self.rew_state_weight * state_error * state_error)
            dist += np.sum(self.rew_act_weight * act * act)
            rew = -dist
            if self.rew_exponential:
                rew = np.exp(rew)
            return rew
        if self.COST == Cost.QUADRATIC:
            return float(-1 * self.symbolic.loss(x=self.state,
                                                 Xr=self.X_GOAL,
                                                 u=self.current_clipped_action,
                                                 Ur=self.U_GOAL,
                                                 Q=self.Q,
                                                 R=self.R)['l'])

    def _get_done(self):
        '''Episode ends only on reaching the upright goal (no out-of-bounds).'''
        self.goal_reached = bool(np.linalg.norm(self.state - self.X_GOAL) < self.goal_threshold)
        return self.goal_reached

    def _get_info(self):
        '''Info dict: goal flag and (weighted) mse to the goal.'''
        info = {}
        info['goal_reached'] = self.goal_reached
        state_error = (deepcopy(self.state) - self.X_GOAL) * self.info_mse_metric_state_weight
        info['mse'] = np.sum(state_error ** 2)
        return info

    def _get_reset_info(self):
        '''Reset info: symbolic model, physical params, and references.'''
        info = {}
        info['symbolic_model'] = self.symbolic
        info['physical_parameters'] = {
            'pendulum_mass': self.PENDULUM_MASS,
            'pendulum_length': self.PENDULUM_LENGTH,
            'gravity': self.GRAVITY_ACC,
            'damping': self.DAMPING,
        }
        info['x_reference'] = self.X_GOAL
        info['u_reference'] = self.U_GOAL
        if self.constraints is not None:
            info['symbolic_constraints'] = self.constraints.get_all_symbolic_models()
            info['constraint_values'] = self.constraints.get_values(self, only_state=True)
        return info
