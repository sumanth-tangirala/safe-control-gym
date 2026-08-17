#!/usr/bin/env python3
'''Verification for the signal-dependent pendulum noise family.

Two checks, corresponding to steps 1 and 2 of
docs/superpowers/specs/2026-08-15-pendulum-signal-dependent-noise-design.md:

  ``sigma``    Draw ``w`` at a fixed ``u`` and confirm the empirical standard
               deviation is ``alpha + beta*|u|``. Cheap, and it catches a
               std/variance mix-up directly -- the two readings differ by 11x at
               ``u = 0`` and put the family on opposite sides of the tau sweep.

  ``gate``     Roll the eval grid through the new code path with
               ``alpha = beta = 0`` and compare the labels against the shipped
               ``tau_0.00`` set. Zero noise is a real gate rather than a
               formality: ``normal(0, 0)`` returns 0 but still advances a
               stream, so a disturbance sharing the env's generator would pass
               the sigma check and fail this.

Usage:
    python pend_sig_validate.py sigma
    python pend_sig_validate.py gate [n_cells]      # default: whole grid
'''
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from generate_inverted_pendulum_trajectories import (EVAL_SPLIT_ID, GRID_RESOLUTION, THETA_DOT_MAX, U_SAT,
                                                     make_controller, make_env_func, rollout_seed,
                                                     run_trajectory, sample_initial_states)
from safe_control_gym.envs.disturbances import DISTURBANCE_TYPES
from safe_control_gym.utils.registration import make

ALPHA, BETA = 0.008, 0.04
# From the shipped eval_description.json, NOT the collector defaults: the
# published pendulum tree runs at horizon 800 and pyb_freq 300.
HORIZON, CTRL_FREQ, PYB_FREQ, SEED = 800, 100, 300, 42
SHIPPED = os.environ.get(
    'PEND_TAU0',
    '/common/users/shared/pracsys/genMoPlan/data_trajectories/stochastic/pendulum/'
    'noisy_torque/lqr/tau_0.00/eval_success_prob.npz')


class _FakeEnv:
    '''Enough env for Disturbance.seed(); the class touches nothing else.'''

    def __init__(self, seed):
        self.np_random = np.random.default_rng(seed)


def env_config(signal_noise):
    return {'ctrl_freq': CTRL_FREQ, 'pyb_freq': PYB_FREQ,
            'episode_len_sec': math.ceil(HORIZON / CTRL_FREQ) + 1,
            'max_steps': HORIZON, 'noise': None, 'invariant': False,
            'torque_noise': None, 'signal_noise': signal_noise}


def check_sigma(n=200_000):
    '''The class in isolation, then the class as the env wires it.'''
    ok = True
    cls = DISTURBANCE_TYPES['signal_dependent']
    env = _FakeEnv(0)
    d = cls(env, dim=1, alpha=ALPHA, beta=BETA)
    d.seed(env)
    print(f'{"u":>8} {"predicted":>10} {"empirical":>10} {"rel err":>9}')
    for u in (0.0, 0.1, 0.3, U_SAT, -U_SAT):
        w = np.array([d.apply(np.array([u]), env)[0] - u for _ in range(n)])
        pred, emp = ALPHA + BETA * abs(u), w.std()
        rel = abs(emp - pred) / pred
        ok &= rel < 0.02 and abs(w.mean()) < 4 * pred / math.sqrt(n)
        print(f'{u:8.4f} {pred:10.5f} {emp:10.5f} {rel:9.2%}')

    penv = make('inverted_pendulum', ctrl_freq=CTRL_FREQ, pyb_freq=PYB_FREQ,
                episode_len_sec=11, cost='quadratic', gui=False, randomized_init=False,
                goal_threshold=0.0,
                disturbances={'action': [{'disturbance_func': 'signal_dependent',
                                          'alpha': ALPHA, 'beta': BETA}]})
    print(f'\n{"u":>8} {"predicted":>10} {"applied std":>12} {"rel err":>9} {"clipped":>8}')
    for u in (0.0, 0.1, 0.3, U_SAT):
        penv.reset(seed=7)
        applied = np.array([float(penv._preprocess_control(np.array([u]))[0])
                            for _ in range(20_000)])
        clipped = float(np.mean(np.abs(applied) >= U_SAT - 1e-12))
        pred, emp = ALPHA + BETA * abs(u), applied.std()
        rel = abs(emp - pred) / pred
        # Interior cells only: at u_sat half the draws are clipped by
        # construction (u + w is clipped, w is not), so a low std there is the
        # design rather than a defect. The clip itself is checked separately.
        if clipped < 0.001:
            ok &= rel < 0.05
        ok &= bool(np.all(np.abs(applied) <= U_SAT + 1e-12))
        print(f'{u:8.4f} {pred:10.5f} {emp:12.5f} {rel:9.2%} {clipped:8.1%}')

    zero = cls(_FakeEnv(1), dim=1, alpha=0.0, beta=0.0)
    zero.seed(_FakeEnv(1))
    worst = max(abs(zero.apply(np.array([u]), env)[0] - u)
                for u in np.linspace(-U_SAT, U_SAT, 5000))
    ok &= worst == 0.0
    print(f'\nalpha=beta=0: max |w| = {worst:.3e}')
    return ok


def _gate_worker(chunk):
    cfg, cells = chunk
    env_func = make_env_func(cfg)
    ctrl = make_controller('lqr', env_func)
    env = env_func()
    out = []
    for idx, state in cells:
        _, success, _ = run_trajectory(
            env, ctrl, state, HORIZON, seed=rollout_seed(SEED, EVAL_SPLIT_ID, idx, 0),
            box_rule=True)
        out.append((idx, int(bool(success))))
    return out


def check_gate(n=None):
    grid = sample_initial_states(0, False, SEED, THETA_DOT_MAX, GRID_RESOLUTION)
    shipped = np.load(SHIPPED)
    assert len(shipped['p_success']) == len(grid), 'grid size does not match the shipped set'
    assert np.allclose(shipped['starts'], grid), 'grid cell ORDER does not match'

    idxs = np.arange(len(grid))
    if n is not None and n < len(grid):
        # Stratified by the shipped label, so a subset gate cannot pass by
        # sampling only the wide all-failure region.
        rng = np.random.default_rng(0)
        lab = shipped['p_success'] > 0
        idxs = np.concatenate([rng.choice(idxs[lab], n // 2, replace=False),
                               rng.choice(idxs[~lab], n - n // 2, replace=False)])
    cells = [(int(i), grid[i]) for i in idxs]
    chunks = [(env_config((0.0, 0.0)), cells[i:i + 64]) for i in range(0, len(cells), 64)]
    got = {}
    with Pool(processes=len(os.sched_getaffinity(0))) as pool:
        for res in tqdm(pool.imap_unordered(_gate_worker, chunks), total=len(chunks)):
            got.update(dict(res))

    ours = np.array([got[int(i)] for i in idxs])
    theirs = (shipped['p_success'][idxs] > 0).astype(int)
    agree = float((ours == theirs).mean())
    print(f'\ncells {len(idxs)}  agreement {agree:.6f}  '
          f'ours p={ours.mean():.4f}  shipped p={theirs.mean():.4f}')
    bad = idxs[ours != theirs]
    if len(bad):
        print(f'{len(bad)} mismatches, first 10 cells: {grid[bad[:10]].tolist()}')
    return agree == 1.0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sigma'
    if mode == 'sigma':
        ok = check_sigma()
    elif mode == 'gate':
        ok = check_gate(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        raise SystemExit(f'unknown mode {mode!r} (expected sigma or gate)')
    print(f'\n{mode}: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
