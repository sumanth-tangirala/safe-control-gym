'''What fraction of trajectories actually enter the corridor.

Run at f_max = 0: the question is whether the UNDISTURBED trajectory passes
through the band, which is what decides whether the disturbance can reach that
start state at all. Noise can pull a marginal trajectory in, which is what the
margin in q2_corridor_collect.py is for.

Rows are sampled at random from the full grid -- a prefix of the ordered grid
is one corner and would measure that corner, not the set (the same trap the
level-0 gate hit). Workers follow q2_collect.py's Pool pattern: each builds
its own env and rolls a contiguous slice of the picked rows.

Usage: python q2_corridor_entry.py --n 20000 --procs 24 --out entry_rate.npz
'''
import argparse
import os
from multiprocessing import Pool

import numpy as np

from q2_corridor_common import BAND, DET, build, roll, rollout_seed

ARGS = None
S_PICK = None
IDX_PICK = None


def _init(a, s_pick, idx_pick):
    global ARGS, S_PICK, IDX_PICK
    ARGS, S_PICK, IDX_PICK = a, s_pick, idx_pick


def _range(rng_pair):
    lo, hi = rng_pair
    env, ctrl = build(0.0)
    entered = np.zeros(hi - lo, dtype=np.uint8)
    try:
        for i in range(lo, hi):
            _, _, _, hit = roll(env, ctrl, S_PICK[i],
                                rollout_seed(ARGS.base_seed, 1, int(IDX_PICK[i]), 0),
                                track_band=True)
            entered[i - lo] = int(hit)
    finally:
        env.close()
    return lo, hi, entered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20_000)
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--base_seed', type=int, default=20260817)
    ap.add_argument('--out', default='entry_rate.npz')
    args = ap.parse_args()

    rows = np.loadtxt(os.path.join(DET, 'roa_labels.txt'), delimiter=',')
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(rows), args.n, replace=False))
    picked = rows[idx, 0:6]

    edges = np.linspace(0, args.n, args.procs + 1).astype(int)
    ranges = [(int(edges[k]), int(edges[k + 1])) for k in range(args.procs)]
    entered = np.zeros(args.n, dtype=np.uint8)
    with Pool(args.procs, initializer=_init,
              initargs=(args, picked, idx)) as pool:
        for lo, hi, e in pool.imap_unordered(_range, ranges):
            entered[lo:hi] = e

    z = picked[:, 1]
    np.savez(args.out, entered=entered, start_z=z, row_index=idx,
             band=np.asarray(BAND))
    for lo, hi, label in [(0.0, BAND[0], 'below band'),
                          (BAND[0], BAND[1], 'inside band'),
                          (BAND[1], 2.0, 'above band')]:
        sel = (z >= lo) & (z < hi)
        if sel.sum():
            print(f'{label:<12} n={sel.sum():>6}  entered={entered[sel].mean():.4f}')
    print(f'overall entered = {entered.mean():.4f}')


if __name__ == '__main__':
    main()
