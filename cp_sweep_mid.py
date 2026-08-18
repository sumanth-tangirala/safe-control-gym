"""Re-read every swept config against the gradient, not the cell count.

`interior` (0 < p < 1) counts uncertain cells and is blind to how uncertain they
are. Measured on the collected cartpole low, 84.9% of its interior cells sit
below 0.1 or above 0.9 with a median of 0.98 -- a hard edge with a frayed pixel
on it, not a gradient. The pendulum's low, at the same interior fraction, has a
median interior p of 0.42.

So the quantity to match is `mid`: the fraction of ALL cells with p in
[0.2, 0.8]. Pendulum targets are low 0.0376, med 0.1334, high 0.6342.

Reads the same shards the interior sweeps already wrote; costs nothing to run.
"""
import glob
import os
import sys

import numpy as np

TARGETS = {'pendulum low': 0.0376, 'pendulum med': 0.1334, 'pendulum high': 0.6342}


def rows(out_dir):
    by = {}
    for f in glob.glob(os.path.join(out_dir, '*_s*.npz')):
        z = np.load(f)
        key = (round(float(z['alpha']), 6), round(float(z['beta']), 6), int(z['trials']))
        by.setdefault(key, []).append((z['cells'], z['hits']))
    out = []
    for (a, b, t), parts in by.items():
        hits = np.concatenate([h for _, h in parts])
        p = hits / t
        if t < 100:                      # the K=1 gate says nothing about a gradient
            continue
        inter = p[(p > 0) & (p < 1)]
        out.append(dict(alpha=a, beta=b, n=len(p), mean=p.mean(),
                        interior=float(((p > 0) & (p < 1)).mean()),
                        mid=float(((p >= 0.2) & (p <= 0.8)).mean()),
                        med_int=float(np.median(inter)) if len(inter) else float('nan')))
    return out


def main():
    allr = []
    for d in sys.argv[1:]:
        allr += rows(d)
    # one row per (alpha, beta); keep the one with the most cells
    best = {}
    for r in allr:
        k = (r['alpha'], r['beta'])
        if k not in best or r['n'] > best[k]['n']:
            best[k] = r
    allr = sorted(best.values(), key=lambda r: r['mid'])
    print(f"{'alpha':>8} {'beta':>7} {'a/b':>6} {'cells':>6} {'mean p':>8} "
          f"{'interior':>9} {'mid':>8} {'median int':>11}")
    for r in allr:
        ratio = r['alpha'] / r['beta'] if r['beta'] else float('inf')
        print(f"{r['alpha']:8.3f} {r['beta']:7.3f} {ratio:6.2f} {r['n']:6d} "
              f"{r['mean']:8.4f} {r['interior']:9.4f} {r['mid']:8.4f} {r['med_int']:11.3f}")
    print()
    for name, tgt in TARGETS.items():
        near = sorted(allr, key=lambda r: abs(r['mid'] - tgt))[:3]
        s = '  '.join(f"(a={r['alpha']:.3f},b={r['beta']:.3f} -> {r['mid']:.4f})" for r in near)
        print(f'  {name:14} mid {tgt:.4f}: {s}')


if __name__ == '__main__':
    main()
