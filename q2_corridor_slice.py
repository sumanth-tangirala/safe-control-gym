'''p_success heatmap slice: hover starts on a fine (x, z) grid.

Velocities and attitude are zero, so every start is a hover state and the two
plotted axes are the only thing that varies. The slice makes the corridor's
asymmetry directly visible: the push is +x toward a goal at x = 0, so the -x
side of the deterministic ROA should survive (or grow) while the +x side
erodes.

Usage:
  python q2_corridor_slice.py --model sine+ambient --f_max 0.08 --ambient 0.06 \
      --trials 20 --procs 64 --out slice_f0.08_a0.06.npz
  python q2_corridor_slice.py --model sine --f_max 0 --trials 1 \
      --out slice_baseline.npz          # deterministic reference
'''
import argparse
from multiprocessing import Pool

import numpy as np

from q2_corridor_common import build, roll, rollout_seed

NX, NZ = 61, 43
X_LO, X_HI = -1.0, 1.0
Z_LO, Z_HI = 0.1, 1.5
SLICE_SPLIT_ID = 2          # distinct from train (0) and eval (1)

ARGS = None
STATES = None


def grid_states(theta=0.0, xd=0.0, zd=0.0, td=0.0):
    xs = np.linspace(X_LO, X_HI, NX)
    zs = np.linspace(Z_LO, Z_HI, NZ)
    states = []
    for z in zs:
        for x in xs:
            # FILE order [x, z, theta, x_dot, z_dot, theta_dot]
            states.append([x, z, theta, xd, zd, td])
    return np.asarray(states), xs, zs


def _init(a, states):
    global ARGS, STATES
    ARGS, STATES = a, states


def _range(rng_pair):
    lo, hi = rng_pair
    env, ctrl = build(ARGS.f_max, model=ARGS.model, ambient=ARGS.ambient)
    p = np.zeros(hi - lo)
    try:
        for i in range(lo, hi):
            ok_count = 0
            for k in range(ARGS.trials):
                ok, _, _ = roll(env, ctrl, STATES[i],
                                rollout_seed(ARGS.base_seed, SLICE_SPLIT_ID, i, k))
                ok_count += int(ok)
            p[i - lo] = ok_count / ARGS.trials
    finally:
        env.close()
    return lo, hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='sine+ambient')
    ap.add_argument('--f_max', type=float, required=True)
    ap.add_argument('--ambient', type=float, default=None)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--procs', type=int, default=64)
    ap.add_argument('--base_seed', type=int, default=20260817)
    ap.add_argument('--theta', type=float, default=0.0)
    ap.add_argument('--xd', type=float, default=0.0)
    ap.add_argument('--zd', type=float, default=0.0)
    ap.add_argument('--td', type=float, default=0.0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    states, xs, zs = grid_states(args.theta, args.xd, args.zd, args.td)
    n = len(states)
    edges = np.linspace(0, n, args.procs + 1).astype(int)
    ranges = [(int(edges[k]), int(edges[k + 1])) for k in range(args.procs)]
    p = np.zeros(n)
    with Pool(args.procs, initializer=_init, initargs=(args, states)) as pool:
        for lo, hi, pv in pool.imap_unordered(_range, ranges):
            p[lo:hi] = pv

    np.savez(args.out, p=p.reshape(NZ, NX), xs=xs, zs=zs,
             model=args.model, f_max=args.f_max,
             theta=args.theta, xd=args.xd, zd=args.zd, td=args.td,
             ambient=-1.0 if args.ambient is None else args.ambient,
             trials=args.trials)
    print(f'{args.out}: mean p {p.mean():.4f}, '
          f'fuzzy {np.mean((p > 0) & (p < 1)):.4f}', flush=True)


if __name__ == '__main__':
    main()
