'''p_success heatmap slice for the quad3d twin curtain.

The quad2d slice plots (x, z) at zero attitude and velocity, because for quad2d
position decides the outcome. That does not transfer. Measured over 300,000 rows
of the shipped quad3d deterministic set, success rate by decile runs 0.13 to
0.27 across x, y and z, while attitude runs 0.08 to 0.67 across qw and body rate
runs 0.12 to 0.49 across r. Position barely moves it. A quad3d (x, z) panel
comes out close to uniform and shows nothing about how the curtain moves the
boundary.

So a slice here puts x on one axis, because that is where the curtains stand,
and a coordinate that actually crosses the deterministic boundary on the other.
Two are supported:

  --axis qw   sweeps attitude from level (qw=1) to inverted (qw=0), rolling
              about the body x-axis. Steepest gradient of any coordinate, and
              the only one reaching past 0.5 success, so it is the only slice
              that can show rescued and broken states in comparable numbers.
  --axis r    sweeps yaw rate. Success peaks near r=0 and falls off both sides,
              so the panel holds two boundaries rather than one.

The eval set cannot be sliced by filtering: it is 1,000,000 randomly drawn 13-D
points, not a lattice, so no two share the other eleven coordinates. The grid
here is generated, as q2_corridor_slice.py's is.

Usage:
  # gate: does the slice contain a deterministic boundary at all?
  python q3_corridor_slice.py --axis qw --model sine --f_max 0 --trials 1 \\
      --out slice_q3_qw_det.npz
  # then vary the disturbance
  python q3_corridor_slice.py --axis qw --model sine+ambient --f_max 0.20 \\
      --ambient 0.008 --trials 10 --out slice_q3_qw_f0.20_a0.008.npz
'''
import argparse
import math
from multiprocessing import Pool

import numpy as np

from q3_corridor_common import build, roll, rollout_seed

NX, NA = 61, 43
X_LO, X_HI = -1.8, 1.8
SLICE_SPLIT_ID = 2          # distinct from train (0) and eval (1)

# Sweep ranges per second axis, taken from the shipped set's own spans.
AXES = {
    'qw': (0.0, 1.0),
    'r': (-24.0, 24.0),
    'z_dot': (-3.0, 3.0),
}

ARGS = None
STATES = None
CELL_LO = 0        # global index of STATES[0]; the seed must not depend on sharding


def grid_states(axis, y=0.0, z=1.5):
    '''-> (states13, xs, avals). Row order is [x, y, z, qw, qx, qy, qz,
    x_dot, y_dot, z_dot, p, q, r], matching the shipped eval_states.txt.

    Everything not on an axis is held benign: centred in y, mid-height in z,
    level, still. So the two plotted axes are the only thing that varies.
    '''
    if axis not in AXES:
        raise ValueError(f'unknown axis {axis!r}; choose from {sorted(AXES)}')
    lo, hi = AXES[axis]
    xs = np.linspace(X_LO, X_HI, NX)
    avals = np.linspace(lo, hi, NA)

    states = []
    for a in avals:
        for x in xs:
            row = [x, y, z, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if axis == 'qw':
                # Roll about body x by 2*acos(qw). Put the remaining magnitude
                # in qx so the quaternion stays unit; an unnormalised one would
                # be silently wrong rather than an error.
                row[3] = float(a)
                row[4] = float(math.sqrt(max(0.0, 1.0 - a * a)))
            elif axis == 'r':
                row[12] = float(a)
            elif axis == 'z_dot':
                row[9] = float(a)
            states.append(row)
    return np.asarray(states, dtype=np.float64), xs, avals


def _init(a, states, cell_lo):
    global ARGS, STATES, CELL_LO
    ARGS, STATES, CELL_LO = a, states, cell_lo


def _range(rng_pair):
    lo, hi = rng_pair
    env, ctrl = build(ARGS.f_max, model=ARGS.model, ambient=ARGS.ambient)
    p = np.zeros(hi - lo)
    try:
        for i in range(lo, hi):
            ok_count = 0
            for k in range(ARGS.trials):
                # Seed on the GLOBAL cell index, so splitting the grid across
                # more tasks cannot change what any cell draws.
                ok, _, _ = roll(env, ctrl, STATES[i],
                                rollout_seed(ARGS.base_seed, SLICE_SPLIT_ID,
                                             CELL_LO + i, k))
                ok_count += int(ok)
            p[i - lo] = ok_count / ARGS.trials
    finally:
        env.close()
    return lo, hi, p


def _save(args, p, xs, avals):
    np.savez(args.out, p=p.reshape(NA, NX), xs=xs, avals=avals, axis=args.axis,
             model=args.model, f_max=args.f_max, y=args.y, z=args.z,
             ambient=-1.0 if args.ambient is None else args.ambient,
             trials=args.trials)
    fuzzy = float(np.mean((p > 0) & (p < 1)))
    print(f'{args.out}: mean p {p.mean():.4f}, fuzzy {fuzzy:.4f}, '
          f'span {p.min():.2f}-{p.max():.2f}', flush=True)
    # A slice with no boundary in it teaches nothing about how the curtain moves
    # that boundary, so say so rather than leave it to the eye.
    if args.f_max == 0 and args.trials == 1:
        verdict = 'HAS a boundary' if 0.02 < p.mean() < 0.98 else 'FLAT, discard this slice'
        print(f'  deterministic gate: {verdict} (mean {p.mean():.3f})', flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--axis', choices=sorted(AXES), required=True)
    ap.add_argument('--model', default='sine+ambient')
    ap.add_argument('--f_max', type=float, required=True)
    ap.add_argument('--ambient', type=float, default=None)
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--procs', type=int, default=64)
    ap.add_argument('--base_seed', type=int, default=20260817)
    ap.add_argument('--y', type=float, default=0.0)
    ap.add_argument('--z', type=float, default=1.5)
    ap.add_argument('--shard', type=int, default=0,
                    help='which cell-range of the grid this task computes')
    ap.add_argument('--nshards', type=int, default=1,
                    help='split the grid across this many tasks; merge with --merge')
    ap.add_argument('--merge', action='store_true',
                    help='combine shard partials named <out>.part<k> into <out>')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    if not 0 <= args.shard < args.nshards:
        ap.error(f'--shard {args.shard} out of range for --nshards {args.nshards}')

    states, xs, avals = grid_states(args.axis, args.y, args.z)
    n_all = len(states)

    if args.merge:
        # Partials cover disjoint cell ranges by construction, so a plain sum
        # reassembles the grid; assert full coverage rather than trust it.
        p = np.zeros(n_all)
        seen = np.zeros(n_all, dtype=bool)
        for k in range(args.nshards):
            d = np.load(f'{args.out}.part{k}.npz')
            lo, hi = int(d['lo']), int(d['hi'])
            p[lo:hi] = d['p']
            seen[lo:hi] = True
        if not seen.all():
            raise SystemExit(f'[ERROR] q3_corridor_slice.py: merge is missing '
                             f'{int((~seen).sum())} of {n_all} cells')
        _save(args, p, xs, avals)
        return

    # This task's slice of the grid.
    cut = np.linspace(0, n_all, args.nshards + 1).astype(int)
    c_lo, c_hi = int(cut[args.shard]), int(cut[args.shard + 1])
    states = states[c_lo:c_hi]
    n = len(states)
    edges = np.linspace(0, n, args.procs + 1).astype(int)
    ranges = [(int(edges[k]), int(edges[k + 1])) for k in range(args.procs)]
    p = np.zeros(n)
    with Pool(args.procs, initializer=_init, initargs=(args, states, c_lo)) as pool:
        for lo, hi, pv in pool.imap_unordered(_range, ranges):
            p[lo:hi] = pv

    if args.nshards > 1:
        np.savez(f'{args.out}.part{args.shard}.npz', p=p, lo=c_lo, hi=c_hi)
        print(f'{args.out}.part{args.shard}.npz: cells {c_lo}:{c_hi}, '
              f'mean p {p.mean():.4f}', flush=True)
        return
    _save(args, p, xs, avals)


if __name__ == '__main__':
    main()
