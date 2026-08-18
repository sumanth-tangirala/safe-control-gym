"""Merge cartpole eval shards, measure how gradual the p field is, and plot it.

Interior fraction says how MANY cells are uncertain. It says nothing about how
gradual the transition is: a band of cells sitting at p = 0.02 and p = 0.98 is
technically interior and still looks like a hard edge. What separates "hazy"
from "sharp edge with noisy pixels" is where the interior cells' probabilities
actually sit, so that distribution is reported here alongside the picture.

The cartpole grid is 4-D (x, theta, x_dot, theta_dot), so the panel is the
(theta, theta_dot) plane at the central x and x_dot -- the slice directly
comparable to the pendulum's, which is that plane and nothing else.

Usage:  python cp_plot_eval.py <root-with-low-med-high> <out-dir>
"""
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_collect import eval_starts  # noqa: E402

SURFACE = '#fcfcfb'
PLANE = '#f9f9f7'
INK = '#0b0b0b'
AXIS = '#c3c2b7'
BLUE_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
             '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
             '#0d366b']
CMAP = LinearSegmentedColormap.from_list('p_success', [SURFACE] + BLUE_RAMP, N=512)
LEVELS = ['low', 'med', 'high']


def merge_level(d):
    files = sorted(glob.glob(os.path.join(d, 'eval_s*.npz')))
    if not files:
        return None
    lo, trials, alpha, beta = [], None, None, None
    for f in files:
        z = np.load(f)
        lo.append((int(z['lo']), z['hits']))
        trials = int(z['trials'])
        alpha, beta = float(z['alpha']), float(z['beta'])
    lo.sort(key=lambda t: t[0])
    return dict(p=np.concatenate([h for _, h in lo]) / trials,
                trials=trials, alpha=alpha, beta=beta, nshards=len(files))


def sharpness(p):
    '''Where the interior cells sit. Mass near 0/1 is a hard edge with noise on
    it; mass spread through the middle is a gradient.'''
    inter = p[(p > 0) & (p < 1)]
    if not len(inter):
        return {}
    return dict(
        interior=float(((p > 0) & (p < 1)).mean()),
        mid=float(((p >= 0.2) & (p <= 0.8)).mean()),
        of_interior_mid=float(((inter >= 0.2) & (inter <= 0.8)).mean()),
        of_interior_extreme=float(((inter < 0.1) | (inter > 0.9)).mean()),
        median_interior=float(np.median(inter)))


def panel(ax, img, extent, title):
    ax.set_facecolor(SURFACE)
    ax.imshow(img, origin='lower', extent=extent, aspect='auto', cmap=CMAP,
              vmin=0.0, vmax=1.0, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(AXIS)
        s.set_linewidth(0.6)
    ax.set_title(title, color=INK, fontsize=15, pad=10, fontweight='medium')


def main():
    root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    S, _ = eval_starts()                      # file order [x, theta, x_dot, thd]
    x, th, xd, thd = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    # central slice: the x / x_dot grid values closest to zero
    x0 = np.unique(x)[np.argmin(np.abs(np.unique(x)))]
    xd0 = np.unique(xd)[np.argmin(np.abs(np.unique(xd)))]
    sel = (x == x0) & (xd == xd0)
    uth, uthd = np.unique(th[sel]), np.unique(thd[sel])
    print(f'slice x={x0:.3f} x_dot={xd0:.3f} -> {sel.sum()} cells '
          f'({len(uth)} theta x {len(uthd)} theta_dot)')

    got = []
    for lv in LEVELS:
        m = merge_level(os.path.join(root, lv))
        if m is None:
            print(f'{lv}: no shards yet')
            continue
        p = m['p']
        if len(p) != len(S):
            print(f'{lv}: PARTIAL {len(p)}/{len(S)} cells '
                  f'({m["nshards"]}/30 shards) -- stats over what exists')
        st = sharpness(p)
        print(f'{lv:5} alpha={m["alpha"]:.3f} beta={m["beta"]:.3f} K={m["trials"]} '
              f'cells={len(p)}')
        print(f'      mean p {p.mean():.4f}   interior {st["interior"]:.4f}   '
              f'cells in p=[0.2,0.8] {st["mid"]:.4f}')
        print(f'      of the interior cells: {100 * st["of_interior_mid"]:.1f}% lie '
              f'in [0.2,0.8], {100 * st["of_interior_extreme"]:.1f}% are <0.1 or >0.9, '
              f'median {st["median_interior"]:.3f}')
        if len(p) == len(S):
            img = np.full((len(uthd), len(uth)), np.nan)
            ti = {v: i for i, v in enumerate(uth)}
            di = {v: i for i, v in enumerate(uthd)}
            for j in np.flatnonzero(sel):
                img[di[thd[j]], ti[th[j]]] = p[j]
            got.append((lv, m, img))

    if not got:
        return
    ext = [uth[0], uth[-1], uthd[0], uthd[-1]]
    fig = plt.figure(figsize=(3.5 * len(got) + 0.4, 4.05), facecolor=PLANE)
    gs = fig.add_gridspec(1, len(got), left=0.016, right=0.984, top=0.845,
                          bottom=0.022, wspace=0.055)
    for c, (lv, m, img) in enumerate(got):
        panel(fig.add_subplot(gs[0, c]), img, ext, lv)
    f = os.path.join(out_dir, 'cp_eval_p_success_grid.png')
    fig.savefig(f, dpi=200, facecolor=PLANE)
    print('wrote', f)


if __name__ == '__main__':
    main()
