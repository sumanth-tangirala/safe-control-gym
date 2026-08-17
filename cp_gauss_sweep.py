#!/usr/bin/env python3
'''Cartpole gaussian_signal sweep: sigma = alpha + beta*|u| on the cart force.

One shard of one config. Sharded because a cartpole rollout is ~3.8 core-seconds
-- 1000 control steps at pyb_freq 5000, so 50,000 simulator steps -- which makes
the full 116,242-cell grid at K = 20 cost ~2,450 core-hours per config. This
sweeps a stratified subsample instead and leaves the full grid to collection.

The level ladder is ONE knob. Fixing the share of noise variance that comes from
the signal term at 50% fixes the ratio alpha : beta = 3.80 : 1, derived from the
measured command distribution (E|u| = 1.581, E[u^2] = 26.436 under the noiseless
LQR). So a level is just

    beta = k,  alpha = 3.80 * k

and the delivered noise scales linearly in k. k = 0.635 / 0.873 / 1.429 deliver
the same standard deviation as the published uniform levels low / med / high
(sigma 8 / 11 / 18, i.e. 4.62 / 6.35 / 10.39 N), so the two families can be
compared at matched strength.

Delivered noise is a SCALE MIXTURE -- each draw has its own sigma -- so its
standard deviation is sqrt(E[sigma^2]), not E[sigma]. The two differ by 4x at
alpha = 0, and reading the wrong one badly understates a level.

Usage:
    python cp_gauss_sweep.py --config 3 --shard 0 --nshards 8 --out shard.npz
    python cp_gauss_sweep.py --merge --out-dir <dir>
'''
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_collect import build, eval_starts, roll, rollout_seed  # noqa: E402

RATIO = 3.80
N_CELLS = int(os.environ.get('N_CELLS', 10_000))
K = int(os.environ.get('K', 20))
STRAT_SEED = 20260817

# (label, alpha, beta, sigma). alpha=None selects the uniform family.
CONFIGS = [
    ('gate_a0_b0', 0.0, 0.0, None),   # must reproduce the sigma_0 labels
    ('k0.318', 1.208, 0.318, None),
    ('k0.635', 2.413, 0.635, None),   # matches uniform low   in delivered std
    ('k0.873', 3.317, 0.873, None),   # matches uniform med
    ('k1.429', 5.428, 1.429, None),   # matches uniform high
    ('k2.000', 7.600, 2.000, None),
    ('pure_signal', 0.0, 0.873, None),   # no floor: noise vanishes at the goal
    ('uniform_low', None, None, 8.0),
    ('uniform_med', None, None, 11.0),
    ('uniform_high', None, None, 18.0),
]

DET_PUB = ('/common/users/shared/pracsys/genMoPlan/data_trajectories/stochastic/'
           'cartpole/noisy_torque/archive/sigma_0/eval_success_prob.npz')


def cell_index(det):
    '''Stratified by the deterministic label, so a subsample sees both sides.

    The cartpole grid is 82% deterministic failures; a uniform draw would spend
    most of its budget confirming that the far exterior stays dead.
    '''
    rng = np.random.default_rng(STRAT_SEED)
    pos, neg = np.flatnonzero(det), np.flatnonzero(~det)
    n = min(N_CELLS // 2, len(pos))
    idx = np.concatenate([rng.choice(pos, n, replace=False),
                          rng.choice(neg, N_CELLS - n, replace=False)])
    idx.sort()
    return idx


def _job(args):
    label, alpha, beta, sigma, cells, k = args
    env, ctrl = (build(sigma) if alpha is None else build(0.0, alpha=alpha, beta=beta))
    return [(i, sum(bool(roll(env, ctrl, s, rollout_seed(1, i, t))[0]) for t in range(k)))
            for i, s in cells]


def run_shard(cfg_i, shard, nshards, out):
    label, alpha, beta, sigma = CONFIGS[cfg_i]
    S, _ = eval_starts()
    det = np.load(DET_PUB)['p_success'] > 0
    idx = cell_index(det)
    edges = np.linspace(0, len(idx), nshards + 1).astype(int)
    mine = idx[edges[shard]:edges[shard + 1]]
    cells = [(int(i), S[i]) for i in mine]
    trials = 1 if label.startswith('gate') else K
    nproc = len(os.sched_getaffinity(0))
    chunks = [(label, alpha, beta, sigma, cells[i:i + 4], trials)
              for i in range(0, len(cells), 4)]
    print(f'{label} shard {shard}/{nshards}: {len(cells)} cells x K={trials} '
          f'on {nproc} procs', flush=True)
    got = {}
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(_job, chunks):
            got.update(dict(r))
    order = np.array([int(i) for i, _ in cells])
    np.savez(out, label=label, alpha=-1.0 if alpha is None else alpha,
             beta=-1.0 if beta is None else beta,
             sigma=-1.0 if sigma is None else sigma,
             cells=order, hits=np.array([got[int(i)] for i in order]), trials=trials)
    print(f'wrote {out}', flush=True)


def merge(out_dir):
    det = np.load(DET_PUB)['p_success'] > 0
    print(f'{"config":>14} {"alpha":>7} {"beta":>7} {"sigma":>6} {"K":>4} '
          f'{"mean p":>8} {"interior":>9} {"gained":>7} {"lost":>6}')
    for label, *_ in CONFIGS:
        files = sorted(glob.glob(os.path.join(out_dir, f'{label}_s*.npz')))
        if not files:
            continue
        cells = np.concatenate([np.load(f)['cells'] for f in files])
        hits = np.concatenate([np.load(f)['hits'] for f in files])
        z = np.load(files[0])
        p = hits / int(z['trials'])
        d = det[cells]
        a, b, s = float(z['alpha']), float(z['beta']), float(z['sigma'])
        print(f'{label:>14} {a if a >= 0 else float("nan"):7.3f} '
              f'{b if b >= 0 else float("nan"):7.3f} '
              f'{s if s >= 0 else float("nan"):6.1f} {int(z["trials"]):4d} '
              f'{p.mean():8.4f} {((p > 0) & (p < 1)).mean():9.4f} '
              f'{int(((~d) & (p > 0)).sum()):7d} {int((d & (p < 1)).sum()):6d}'
              f'   ({len(files)} shards, {len(cells)} cells)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=int)
    ap.add_argument('--shard', type=int)
    ap.add_argument('--nshards', type=int)
    ap.add_argument('--out')
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--out-dir')
    a = ap.parse_args()
    if a.merge:
        merge(a.out_dir)
        return
    if a.out and os.path.exists(a.out):
        print(f'{a.out} exists, skipping', flush=True)
        return
    run_shard(a.config, a.shard, a.nshards, a.out)


if __name__ == '__main__':
    main()
