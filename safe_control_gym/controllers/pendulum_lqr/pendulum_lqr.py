'''Bounds-normalized LQR for the inverted pendulum.

A faithful port of the source system's ``LQRController``: it linearizes the
pendulum analytically at the upright equilibrium, **normalizes A/B by the
state/control bounds** (``Tx = diag(pi, theta_dot_max)``, ``Tu = u_sat``) before
solving the continuous-time ARE, and applies the resulting normalized-coordinate
gain directly to the physical state error. This exact computation is what the
region of attraction the RL policies were trained against depends on, so it must
not be replaced by safe-control-gym's generic symbolic LQR.
'''

import math

import numpy as np
from scipy.linalg import solve_continuous_are

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.lqr.lqr_utils import get_cost_weight_matrix


class PendulumLQR(BaseController):
    '''Static-gain LQR stabilizing the pendulum at upright.'''

    def __init__(self, env_func, q_lqr: list = None, r_lqr: list = None, **kwargs):
        '''Creates the task env and computes the LQR gain.

        Args:
            env_func (Callable): Function to instantiate the inverted pendulum env.
            q_lqr (list): Diagonal of the (normalized) state cost. Default identity.
            r_lqr (list): Diagonal of the (normalized) input cost. Default [1].
        '''
        super().__init__(env_func, **kwargs)
        self.env = env_func()

        g = self.env.GRAVITY_ACC
        l = self.env.PENDULUM_LENGTH
        b = self.env.DAMPING
        inertia = self.env.inertia
        self.u_sat = self.env.u_sat
        self.goal = np.array(self.env.X_GOAL, dtype=np.float64)

        Q = get_cost_weight_matrix([1, 1] if q_lqr is None else q_lqr, 2)
        R = get_cost_weight_matrix([1] if r_lqr is None else r_lqr, 1)

        # Analytic linearization at upright (theta = 0 unstable equilibrium).
        A = np.array([[0.0, 1.0], [g / l, -(b / inertia)]])
        B = np.array([[0.0], [1.0 / inertia]])
        # Normalize by the state/control bounds (matches the source controller).
        Tx = np.diag([math.pi, self.env.theta_dot_max])
        Tu = self.u_sat
        An = np.linalg.inv(Tx) @ A @ Tx
        Bn = np.linalg.inv(Tx) @ B * Tu

        S = solve_continuous_are(An, Bn, Q, R)
        self.gain = (np.linalg.inv(R) @ Bn.T @ S).ravel()
        # Backwards-compatible alias used by the source system and tests.
        self.K = self.gain

    def reset(self):
        '''Prepares for evaluation.'''
        self.env.reset()

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

    def select_action(self, obs, info=None):
        '''Return the saturated LQR torque for the current observation.'''
        error = np.asarray(obs, dtype=np.float64)[:2] - self.goal
        u = float(-self.K @ error)
        return np.array([np.clip(u, -self.u_sat, self.u_sat)], dtype=np.float64)
