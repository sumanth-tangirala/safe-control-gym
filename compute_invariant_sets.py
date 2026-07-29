#!/usr/bin/env python3
'''Compute and validate the invariant terminal ellipsoids for dataset recollection.

For each system, the true closed-loop step map f (env + controller, one control
step) is linearized at the closed-loop attractor s0 by central finite
differences; P solves the discrete Lyapunov equation A_d' P A_d - P = -Q; the
ellipsoid E = {(s - s0)' P (s - s0) <= c} is then validated empirically under
the full nonlinear dynamics: from states sampled on the boundary of E, the
Lyapunov value V must never exceed c (0.5% numerical tolerance) and every
rollout must converge back to s0.

Artifacts are written to invariant_sets/<system>.npz with fields
P, center, c, state_order, Q_diag and validation metadata. Generators load
these to label trajectories by terminal-state membership.

See plans/invariant-terminal-sets-recollection.md.

Usage:
    python compute_invariant_sets.py --systems pendulum cartpole quad2d quad3d
'''

import argparse
import json
import math
import os

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'invariant_sets')
VAL_TOL = 1.005  # allowed sup V/c (numerical tolerance)
CONV_TOL = 0.02  # required terminal distance to the attractor after validation rollout
FD_EPS = 1e-4


class System:
    '''A system exposes step-from-state rollouts in env-order coordinates.'''

    name = None
    state_order = None
    Q_diag = None
    c = None
    # distance from the attractor to each state-box face (np.inf = unbounded)
    box_dist = None
    val_samples = 60
    val_steps = 800

    def attractor(self):
        raise NotImplementedError

    def rollout(self, s_env, steps):
        '''Roll `steps` control steps from env-order state; return (steps+1, dim) states.'''
        raise NotImplementedError

    def close(self):
        pass


class Pendulum(System):
    name = 'pendulum'
    state_order = ['theta', 'theta_dot']
    Q_diag = [0.03, 1.0]
    c = 2.99
    box_dist = np.array([np.pi, 2 * np.pi])
    val_samples = 400
    val_steps = 1500

    DT = 0.01
    G_OVER_L = 9.81 / 0.5
    I = 0.15 * 0.5 ** 2
    B_OVER_I = 0.1 / I
    U_SAT = 0.6371781908344007
    K = np.array([7.390506186111498, 2.606118514678467])
    THETA_DOT_MAX = 2 * math.pi

    # The pendulum simulator is replicated exactly (explicit Euler, wrap, clip,
    # saturated LQR) so validation can run vectorized; fidelity to
    # InvertedPendulum + PendulumLQR is bit-for-bit for noise=None.
    def attractor(self):
        return np.zeros(2)

    def rollout(self, s_env, steps):
        th, td = float(s_env[0]), float(s_env[1])
        out = np.empty((steps + 1, 2))
        out[0] = (th, td)
        for t in range(steps):
            u = np.clip(-(self.K[0] * th + self.K[1] * td), -self.U_SAT, self.U_SAT)
            tdd = self.G_OVER_L * math.sin(th) + u / self.I - self.B_OVER_I * td
            th, td = th + self.DT * td, td + self.DT * tdd
            th = (th + math.pi) % (2 * math.pi) - math.pi
            td = min(max(td, -self.THETA_DOT_MAX), self.THETA_DOT_MAX)
            out[t + 1] = (th, td)
        return out


class Cartpole(System):
    name = 'cartpole'
    state_order = ['x', 'x_dot', 'theta', 'theta_dot']
    Q_diag = [1 / 6.0 ** 2, 1 / 5.0 ** 2, 1.0, 1 / 5.0 ** 2]
    c = 3.0
    box_dist = np.array([6.0, 5.0, np.inf, 5.0])
    val_samples = 60
    val_steps = 700

    def __init__(self):
        from functools import partial

        from safe_control_gym.utils.registration import make
        env_func = partial(make, 'cartpole', task='stabilization', ctrl_freq=100,
                           pyb_freq=5000, episode_len_sec=1000,
                           done_on_out_of_bound=True, cost='quadratic', gui=False,
                           randomized_init=False, obs_wrap_angle=True,
                           x_dot_limit=float('inf'), theta_dot_limit=float('inf'),
                           action_scale=2000.0)
        self.ctrl = make('lqr', env_func, q_lqr=[1, 1, 1, 1], r_lqr=[0.1],
                         discrete_dynamics=True)
        self.env = env_func()

    def attractor(self):
        return np.zeros(4)

    def rollout(self, s_env, steps):
        import pybullet as p
        env = self.env
        obs, info = env.reset()
        x, x_dot, theta, theta_dot = s_env
        p.resetJointState(env.CARTPOLE_ID, 0, targetValue=x, targetVelocity=x_dot,
                          physicsClientId=env.PYB_CLIENT)
        p.resetJointState(env.CARTPOLE_ID, 1, targetValue=theta, targetVelocity=theta_dot,
                          physicsClientId=env.PYB_CLIENT)
        env.state = np.array([x, x_dot, theta, theta_dot])
        obs = env._get_observation()
        env.out_of_bounds = False
        info = None
        out = np.empty((steps + 1, 4))
        out[0] = env.state
        for t in range(steps):
            action = self.ctrl.select_action(obs, info)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # noqa: F841 (unused, matches pre-migration behaviour)
            out[t + 1] = obs[:4]
        return out

    def close(self):
        self.env.close()


class Quad3D(System):
    name = 'quad3d'
    state_order = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                   'phi', 'theta', 'psi', 'p', 'q', 'r']
    Q_diag = [1.0] * 12
    c = 0.1
    box_dist = np.array([1.8, 3.0, 1.8, 3.0, 0.9, 3.0,
                         np.inf, np.inf, np.inf, 24.0, 24.0, 24.0])
    val_samples = 60
    val_steps = 500

    def __init__(self):
        from functools import partial

        from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
        from safe_control_gym.utils.registration import make
        task_info = {'stabilization_goal': [0, 0, 1],
                     'stabilization_goal_tolerance': 0.0}
        env_func = partial(make, 'quadrotor', quad_type=QuadType.THREE_D,
                           task='stabilization', task_info=task_info, ctrl_freq=100,
                           pyb_freq=5000, episode_len_sec=1000,
                           done_on_out_of_bound=True, cost='quadratic', gui=False,
                           randomized_init=False)
        self.ctrl = make('lqr', env_func, q_lqr=[1] * 12, r_lqr=[0.1] * 4,
                         discrete_dynamics=True)
        self.env = env_func()

    def attractor(self):
        return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

    def rollout(self, s_env, steps):
        import pybullet as p
        env = self.env
        obs, info = env.reset()
        x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, pb, qb, rb = s_env
        p.resetBasePositionAndOrientation(env.DRONE_ID, [x, y, z],
                                          p.getQuaternionFromEuler([phi, theta, psi]),
                                          physicsClientId=env.PYB_CLIENT)
        p.resetBaseVelocity(env.DRONE_ID, [x_dot, y_dot, z_dot], [pb, qb, rb],
                            physicsClientId=env.PYB_CLIENT)
        env._update_and_store_kinematic_information()
        obs = env._get_observation()
        env.out_of_bounds = False
        info = None
        out = np.empty((steps + 1, 12))
        out[0] = obs[:12]
        for t in range(steps):
            action = self.ctrl.select_action(obs, info)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # noqa: F841 (unused, matches pre-migration behaviour)
            out[t + 1] = obs[:12]
        return out

    def close(self):
        self.env.close()


class Quad2D(System):
    name = 'quad2d'
    state_order = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']
    Q_diag = [1.0] * 6
    c = 1.0
    val_samples = 60
    val_steps = 800

    def __init__(self):
        import tempfile
        from functools import partial

        import generate_quadrotor_2d_trajectories_rl as g
        from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
        from safe_control_gym.utils.registration import make
        self._g = g
        algo = 'safe_explorer_ppo'
        model_path = g.get_default_model_path(
            algo, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'examples/rl/models'))
        env_kwargs = {
            'quad_type': QuadType.TWO_D, 'task': 'stabilization', 'ctrl_freq': 100,
            'pyb_freq': 5000, 'episode_len_sec': 1000, 'done_on_out_of_bound': True,
            'cost': 'quadratic', 'normalized_rl_action_space': True, 'gui': False,
            'randomized_init': False,
            'task_info': {'stabilization_goal': [0, 1],
                          'stabilization_goal_tolerance': 0.0},
            'constraints': g.SAFE_EXPLORER_CONSTRAINTS, 'done_on_violation': False,
        }
        env_func = partial(make, 'quadrotor', **env_kwargs)
        self.ctrl = make(algo, env_func, **g.ALGO_CONFIGS[algo].copy(),
                         output_dir=tempfile.mkdtemp())
        self.ctrl.load(model_path)
        self.ctrl.obs_normalizer.set_read_only()
        self.env = env_func()
        self._s_star = None

    def attractor(self):
        if self._s_star is None:
            states = self.rollout(np.array([0.0, 0, 1.0, 0, 0, 0]), 3000)
            self._s_star = states[-1]
            drift = np.linalg.norm(states[-1] - states[-100])
            assert drift < 1e-6, f'quad2d attractor not settled (drift {drift})'
            x, x_dot, z, z_dot, _, theta_dot = self._s_star
            self.box_dist = np.array([1.0 - abs(x), 1.0 - abs(x_dot),
                                      min(z - 0.1, 1.5 - z), 1.0 - abs(z_dot),
                                      np.inf, 8.0 - abs(theta_dot)])
        return self._s_star

    def rollout(self, s_env, steps):
        import pybullet as p
        env = self.env
        obs, info = env.reset()
        x, x_dot, z, z_dot, theta, theta_dot = s_env
        p.resetBasePositionAndOrientation(env.DRONE_ID, [x, 0, z],
                                          p.getQuaternionFromEuler([0, theta, 0]),
                                          physicsClientId=env.PYB_CLIENT)
        p.resetBaseVelocity(env.DRONE_ID, [x_dot, 0, z_dot], [0, theta_dot, 0],
                            physicsClientId=env.PYB_CLIENT)
        env._update_and_store_kinematic_information()
        obs = env._get_observation()
        env.out_of_bounds = False
        info['constraint_values'] = env.constraints.get_values(env, only_state=True)
        out = np.empty((steps + 1, 6))
        out[0] = obs[:6]
        for t in range(steps):
            action = self.ctrl.select_action(self.ctrl.obs_normalizer(obs), info)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # noqa: F841 (unused, matches pre-migration behaviour)
            out[t + 1] = obs[:6]
        return out

    def close(self):
        self.env.close()
        self.ctrl.close()


SYSTEMS = {'pendulum': Pendulum, 'cartpole': Cartpole,
           'quad2d': Quad2D, 'quad3d': Quad3D}


def fd_linearize(system, s0):
    dim = len(s0)
    A = np.empty((dim, dim))
    for i in range(dim):
        d = np.zeros(dim)
        d[i] = FD_EPS
        fp = system.rollout(s0 + d, 1)[1] - s0
        fm = system.rollout(s0 - d, 1)[1] - s0
        A[:, i] = (fp - fm) / (2 * FD_EPS)
    return A


def validate(system, P, s0, c, seed=0):
    '''Sample the ellipsoid boundary; check V never exceeds c and all converge.'''
    dim = len(s0)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.linalg.inv(P))
    worst_ratio = 0.0
    worst_final = 0.0
    for _ in range(system.val_samples):
        u = rng.normal(size=dim)
        u /= np.linalg.norm(u)
        s = s0 + (L @ u) * math.sqrt(c)
        states = system.rollout(s, system.val_steps)
        dev = states - s0
        V = np.einsum('ti,ij,tj->t', dev, P, dev)
        worst_ratio = max(worst_ratio, V.max() / c)
        worst_final = max(worst_final, float(np.linalg.norm(dev[-1])))
    return worst_ratio, worst_final


def compute_system(name, skip_validation=False):
    print(f'=== {name} ===', flush=True)
    system = SYSTEMS[name]()
    try:
        s0 = system.attractor()
        A_d = fd_linearize(system, s0)
        rho = float(np.abs(np.linalg.eigvals(A_d)).max())
        print(f'attractor: {np.round(s0, 5)}', flush=True)
        print(f'A_d spectral radius: {rho:.6f}', flush=True)
        assert rho < 1, 'closed loop is not locally contracting'

        P = solve_discrete_lyapunov(A_d.T, np.diag(system.Q_diag))
        Pinv = np.linalg.inv(P)
        extents = np.sqrt(system.c * np.diag(Pinv))
        # the ellipsoid must fit inside the state box
        finite = np.isfinite(system.box_dist)
        assert (extents[finite] < system.box_dist[finite]).all(), \
            f'ellipsoid exceeds state box: extents {extents} vs {system.box_dist}'
        print(f'c={system.c}  extents ({system.state_order}): {np.round(extents, 4)}', flush=True)

        c = system.c
        if skip_validation:
            worst_ratio = worst_final = float('nan')
        else:
            # If the nominal level is not invariant under denser sampling,
            # reduce it until validation passes (the sublevel sets shrink
            # toward the linear regime, where invariance is guaranteed).
            for attempt in range(5):
                worst_ratio, worst_final = validate(system, P, s0, c)
                print(f'validation at c={c:.4g}: sup V/c = {worst_ratio:.4f} '
                      f'(tol {VAL_TOL}), worst final dist = {worst_final:.2e} '
                      f'(tol {CONV_TOL})', flush=True)
                if worst_ratio <= VAL_TOL and worst_final <= CONV_TOL:
                    break
                c /= 2
            else:
                raise AssertionError('no invariant level found after reductions')
            if c != system.c:
                extents = np.sqrt(c * np.diag(Pinv))
                print(f'level reduced to c={c:.4g}; extents now {np.round(extents, 4)}',
                      flush=True)

        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        path = os.path.join(ARTIFACT_DIR, f'{name}.npz')
        np.savez(path, P=P, center=s0, c=c,
                 state_order=np.array(system.state_order),
                 Q_diag=np.array(system.Q_diag), A_d=A_d,
                 validation=np.array([worst_ratio, worst_final]),
                 extents=extents)
        print(f'saved {path}', flush=True)
        return {'name': name, 'c': c, 'spectral_radius': rho,
                'sup_V_over_c': worst_ratio, 'worst_final_dist': worst_final,
                'extents': extents.tolist()}
    finally:
        system.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--systems', nargs='+', default=list(SYSTEMS),
                        choices=list(SYSTEMS))
    parser.add_argument('--skip_validation', action='store_true',
                        help='Compute P without the (slow) empirical invariance check')
    args = parser.parse_args()
    results = [compute_system(name, args.skip_validation) for name in args.systems]
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
