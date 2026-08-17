#!/usr/bin/env python3
'''Cartpole gaussian_signal sweep targeting the pendulum's uncertainty band.

WHAT THIS MATCHES, AND WHY NOT DELIVERED VARIANCE
-------------------------------------------------
The pendulum gaussian_signal set is characterised by how much of the eval grid
is *uncertain* -- cells with 0 < p < 1 at K = 100, the blurred band around the
separatrix. Its three levels sit at

    low 11.2%    med 64.6%    high 82.4%

(`stochastic/pendulum/gaussian_signal/lqr/README.md`, column `interior`;
reproduced exactly from each level's `eval_success_prob.npz`). That fraction is
the thing being matched here, because it is what the noise level *looks* like
and it is comparable across two systems whose success rates are not.

Delivered standard deviation is NOT matched, and matching it was tried and
abandoned: it pins the cartpole levels to the second moment of its command
distribution, which is dominated by a rare heavy tail (p50 |u| = 0.53 N against
E[u^2] = 553), so a variance-matched level is nearly constant noise and loses
the one property that makes this family interesting -- going quiet at the goal.

TWO KNOBS, AND WHICH ONE SETS THE CHARACTER
-------------------------------------------
    w ~ Normal(0, alpha + beta*|u|)

`alpha` is the floor that survives at the goal, where a stabilising controller
commands almost nothing; `beta` is effort-proportional and acts only in the
transient. Their RATIO sets the character, their SCALE sets the strength.

At the cartpole median command (|u| = 0.53 N) the beta term contributes
`0.53*beta` against `alpha`. For beta to dominate near the goal the way it does
on the pendulum -- where |u| is pinned at u_sat and beta carries ~2/3 of sigma
-- the ratio wants to be alpha ~ 0.27*beta, not the alpha = 3.80*beta that
variance-matching produces. RATIOS below sweeps that choice rather than
assuming it.

SAMPLING
--------
A UNIFORM subsample of the grid, not the stratified one `cp_gauss_sweep.py`
uses. Interior fraction is a property of the whole grid, and a 50/50
stratification by deterministic label over-represents the boundary -- exactly
where interior cells live -- so its interior number is not comparable to the
pendulum's full-grid value.

Usage:
    python cp_interior_sweep.py --alpha A --beta B --shard S --nshards N --out f.npz
    python cp_interior_sweep.py --merge --out-dir <dir>
'''
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_collect import build, eval_starts, roll, rollout_seed  # noqa: E402

N_CELLS = int(os.environ.get('N_CELLS', 2000))
K = int(os.environ.get('K', 100))
SAMPLE_SEED = 20260817

# The pendulum levels this is trying to land on.
TARGETS = {'low': 0.112, 'med': 0.646, 'high': 0.824}


def cell_index(n_grid):
    '''Uniform without replacement -- an unbiased estimate of a whole-grid rate.'''
    rng = np.random.default_rng(SAMPLE_SEED)
    idx = rng.choice(n_grid, min(N_CELLS, n_grid), replace=False)
    idx.sort()
    return idx


def _job(args):
    alpha, beta, cells, k = args
    env, ctrl = build(0.0, alpha=alpha, beta=beta)
    out = [(i, sum(bool(roll(env, ctrl, s, rollout_seed(1, i, t))[0]) for t in range(k)))
           for i, s in cells]
    env.close()
    return out


def run_shard(alpha, beta, shard, nshards, out):
    S, det = eval_starts()
    idx = cell_index(len(S))
    edges = np.linspace(0, len(idx), nshards + 1).astype(int)
    mine = idx[edges[shard]:edges[shard + 1]]
    cells = [(int(i), S[i]) for i in mine]
    trials = 1 if (alpha == 0 and beta == 0) else K
    nproc = min(int(os.environ.get('NPROC', 0)) or 10 ** 6,
                len(os.sched_getaffinity(0)))
    chunks = [(alpha, beta, cells[i:i + 2], trials) for i in range(0, len(cells), 2)]
    print(f'alpha={alpha} beta={beta} shard {shard}/{nshards}: '
          f'{len(cells)} cells x K={trials} on {nproc} procs', flush=True)
    got = {}
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(_job, chunks):
            got.update(dict(r))
    order = np.array([int(i) for i, _ in cells])
    tmp = out + f'.tmp{os.getpid()}'
    np.savez(tmp, alpha=alpha, beta=beta, cells=order,
             hits=np.array([got[int(i)] for i in order]), trials=trials)
    os.replace(tmp + '.npz', out)
    print(f'wrote {out}', flush=True)


def merge(out_dir):
    S, det = eval_starts()
    det = det.astype(bool)
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, '*_s*.npz'))):
        z = np.load(f)
        rows.append((round(float(z['alpha']), 6), round(float(z['beta']), 6),
                     int(z['trials']), z['cells'], z['hits']))
    by_cfg = {}
    for a, b, t, c, h in rows:
        by_cfg.setdefault((a, b, t), []).append((c, h))
    print(f'{"alpha":>8} {"beta":>8} {"a/b":>6} {"K":>4} {"cells":>6} '
          f'{"mean p":>8} {"interior":>9} {"rescued":>8} {"broken":>7}')
    out = []
    for (a, b, t), parts in sorted(by_cfg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        cells = np.concatenate([c for c, _ in parts])
        hits = np.concatenate([h for _, h in parts])
        p = hits / t
        d = det[cells]
        interior = float(((p > 0) & (p < 1)).mean())
        ratio = a / b if b else float('nan')
        print(f'{a:8.3f} {b:8.3f} {ratio:6.2f} {t:4d} {len(cells):6d} '
              f'{p.mean():8.4f} {interior:9.4f} '
              f'{int(((~d) & (p > 0)).sum()):8d} {int((d & (p < 1)).sum()):7d}')
        out.append((a, b, interior))
    print()
    print('pendulum targets:  low 0.112   med 0.646   high 0.824')
    for name, tgt in TARGETS.items():
        near = sorted(out, key=lambda r: abs(r[2] - tgt))[:2]
        s = '   '.join(f'(a={a:.3f}, b={b:.3f} -> {i:.3f})' for a, b, i in near)
        print(f'  {name:5} target {tgt:.3f}: nearest {s}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float)
    ap.add_argument('--beta', type=float)
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
    run_shard(a.alpha, a.beta, a.shard, a.nshards, a.out)


if __name__ == '__main__':
    main()
