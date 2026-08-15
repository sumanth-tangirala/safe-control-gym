#!/usr/bin/env python3
'''Beta sweep for the signal-dependent pendulum family (alpha fixed at 0.008).

Levels are not chosen a priori. They are coupled to the success rule and to the
horizon and do not transfer across either -- measured on quad3d, retention was
0.618 under a 0.1 box and 0.015 under a 0.05 ball at the same force -- so this
measures p_success on a subsample of the eval grid at each candidate beta and
the published levels are read off the curve.

The subsample is UNIFORM, not stratified: the number produced is an estimate of
the grid mean p_success, and a label-stratified sample would not be one. The
same cells and the same rollout seeds are used at every beta, so levels are
paired under common random numbers and differences between them are not sampling
noise.

Results are written to an npz after every level. A sweep that only prints has
already been lost once on this project, to a job that exited COMPLETED with no
stdout file.

Usage:  python pend_sig_sweep.py [n_cells] [K] [beta ...]
Env:    SIG_SWEEP_OUT   output npz (default: ./sig_sweep.npz)
        SIG_EXTERNAL    1 -> sat(u) + w instead of sat(u + w). Different mechanism,
                        so its levels are swept separately and do not transfer.
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

ALPHA = 0.008
# The published pendulum tree's parameters, so the sweep is comparable to it.
HORIZON, CTRL_FREQ, PYB_FREQ, SEED = 800, 100, 300, 42
OUT = os.environ.get('SIG_SWEEP_OUT', 'sig_sweep.npz')
EXTERNAL = os.environ.get('SIG_EXTERNAL', '') not in ('', '0', 'false', 'False')
DET = ('/common/users/shared/pracsys/genMoPlan/data_trajectories/stochastic/pendulum/'
       'noisy_torque/lqr/tau_0.00/eval_success_prob.npz')
# Spaced by powers of two from the spec's beta = 0.04, which is predicted to be
# nearly deterministic. The top of the range reaches sigma at saturation of
# 0.41, i.e. tau ~ 0.71 in std terms -- past the strongest published tau level
# of the old sweep, so the curve should be bracketed rather than truncated.
DEFAULT_BETAS = [0.04, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]


def _worker(chunk):
    beta, cells, K = chunk
    cfg = {'ctrl_freq': CTRL_FREQ, 'pyb_freq': PYB_FREQ,
           'episode_len_sec': math.ceil(HORIZON / CTRL_FREQ) + 1,
           'max_steps': HORIZON, 'noise': None, 'invariant': False,
           'torque_noise': None, 'signal_noise': (ALPHA, beta),
           'external_noise': EXTERNAL}
    env_func = make_env_func(cfg)
    ctrl = make_controller('lqr', env_func)
    env = env_func()
    out = []
    for idx, state in cells:
        hits = 0
        for k in range(K):
            _, success, _ = run_trajectory(
                env, ctrl, state, HORIZON,
                seed=rollout_seed(SEED, EVAL_SPLIT_ID, idx, k), box_rule=True)
            hits += int(bool(success))
        out.append((idx, hits))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    betas = [float(b) for b in sys.argv[3:]] or DEFAULT_BETAS

    grid = sample_initial_states(0, False, SEED, THETA_DOT_MAX, GRID_RESOLUTION)
    idxs = np.sort(np.random.default_rng(20260815).choice(len(grid), n, replace=False))
    cells = [(int(i), grid[i]) for i in idxs]

    workers = len(os.sched_getaffinity(0))
    rows = []
    print(f'{n} cells x K={K} x {len(betas)} levels on {workers} workers', flush=True)
    # Gains -- cells the deterministic controller fails that noise rescues -- are
    # the reason the external family exists, so the sweep reports them directly
    # rather than leaving them to be derived later.
    det = np.load(DET)['p_success'][idxs] > 0
    print(f'placement: {"sat(u) + w  EXTERNAL" if EXTERNAL else "sat(u + w)  internal"}',
          flush=True)
    print(f'{"beta":>7} {"sig(0)":>8} {"sig(sat)":>9} {"tau_equiv":>10} '
          f'{"p":>7} {"interior":>9} {"gain":>7} {"gain_p":>8}', flush=True)
    for beta in betas:
        chunks = [(beta, cells[i:i + 8], K) for i in range(0, len(cells), 8)]
        got = {}
        with Pool(processes=workers) as pool:
            for res in tqdm(pool.imap_unordered(_worker, chunks), total=len(chunks),
                            desc=f'beta={beta}', leave=False, mininterval=10.0):
                got.update(dict(res))
        p = np.array([got[int(i)] for i in idxs]) / K
        rows.append((beta, p))
        sig_sat = ALPHA + beta * U_SAT
        # Half-width of the uniform noise with the same std at saturation, so a
        # level can be read against the tau family it is meant to be compared to.
        gain = (~det) & (p > 0)
        print(f'{beta:7.3f} {ALPHA:8.4f} {sig_sat:9.4f} {sig_sat * math.sqrt(3):10.4f} '
              f'{p.mean():7.4f} {((p > 0) & (p < 1)).mean():9.4f} '
              f'{int(gain.sum()):7d} {(p[~det]).mean():8.4f}', flush=True)
        np.savez(OUT, betas=np.array([r[0] for r in rows]),
                 p=np.array([r[1] for r in rows]), cells=idxs, K=K, alpha=ALPHA,
                 horizon=HORIZON, ctrl_freq=CTRL_FREQ, pyb_freq=PYB_FREQ,
                 external=EXTERNAL, det=det)
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
