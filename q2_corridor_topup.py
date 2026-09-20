'''Repair a shard collected with the unsound reachability shortcut.

Background. Until 2026-08-19 q2_corridor_collect.py's shard_eval applied the
reachability shortcut for every model: a start whose undisturbed trajectory
never came within MARGIN of the corridor band was rolled once, not `trials`
times, on the grounds that an altitude-gated disturbance cannot reach it. That
holds for the corridor gust. It does not hold once an ambient term is present,
because build() adds that as white_noise with no altitude gate. Ten shortcut
states from the sharp run were re-flown 20 times each and 7 varied, having been
recorded as certain successes.

Rather than discard the affected runs, this writes the MISSING flights as a
second k-window. For each start the earlier shard settled in one flight, it
rolls trials 1..trials-1 with the same seeds a full run would have drawn, and
records hits/trials_used for those starts only. Every other start gets 0 and 0,
so it keeps the count it already has.

The reducer already sums hits and trials_used across a shard's windows, so
summing the original k0-N file with this k1-N file gives, per start:

  shortcut start:      1 + 19 = 20 trials   (its trial 0, then trials 1..19)
  fully-rolled start: 20 +  0 = 20 trials   (untouched)

which is exactly what one uninterrupted no-shortcut run at `trials` would have
produced, because rollout_seed is a pure function of (split, state index, trial)
and does not depend on run order.

Usage:
  python q2_corridor_topup.py --in  SHARDS/eval_L0.08_A0.06_k0-20_s7.npz \\
                              --out SHARDS/eval_L0.08_A0.06_k1-20_s7.npz
'''
import argparse
import os
import sys

import numpy as np

from q2_corridor_collect import EVAL_SPLIT_ID
from q2_corridor_common import build, grid_states, roll, rollout_seed


def topup(in_path, out_path, base_seed):
    d = np.load(in_path)
    used, starts = d['trials_used'], d['starts']
    lo, hi = int(d['lo']), int(d['hi'])
    level, model = float(d['f_max']), str(d['model'])
    ambient = float(d['ambient'])
    trials = int(d['trials'])
    if int(d['trial_lo']) != 0:
        raise ValueError(f'{in_path}: trial_lo is {int(d["trial_lo"])}, expected 0; '
                         'the top-up complements the first window only')
    if ambient <= 0:
        raise ValueError(f'{in_path}: ambient is {ambient}, so the shortcut was sound '
                         'here and there is nothing to repair')

    todo = np.flatnonzero(used == 1)
    hits = np.zeros(len(starts), dtype=np.int32)
    add = np.zeros(len(starts), dtype=np.int32)
    if len(todo) == 0:
        # Nothing was shortcut. Still write the window so the reducer sees a
        # uniform set of files across shards rather than a ragged one.
        _save(out_path, d, starts, hits, add, level, model, ambient, trials, lo, hi)
        return 0, 0

    # grid_states is the authority on which start each index means; assert the
    # shard on disk agrees before spending an hour rolling against it.
    grid, _ = grid_states(lo, hi)
    if not np.allclose(grid, starts):
        raise ValueError(f'{in_path}: starts do not match grid_states({lo}, {hi})')

    env, ctrl = build(level, model=model, ambient=ambient)
    try:
        for i in todo:
            i = int(i)
            for k in range(1, trials):
                ok, _, _ = roll(env, ctrl, starts[i],
                                rollout_seed(base_seed, EVAL_SPLIT_ID, lo + i, k))
                hits[i] += int(ok)
                add[i] += 1
    finally:
        env.close()
    _save(out_path, d, starts, hits, add, level, model, ambient, trials, lo, hi)
    return len(todo), int(add.sum())


def _save(out_path, d, starts, hits, add, level, model, ambient, trials, lo, hi):
    np.savez(out_path, starts=starts, hits=hits, trials_used=add,
             det_labels=d['det_labels'], lo=lo, hi=hi, f_max=level,
             model=model, trial_lo=1, trials=trials, ambient=ambient)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='in_path', required=True, help='an existing k0-N shard')
    ap.add_argument('--out', required=True, help='the k1-N window to write')
    ap.add_argument('--base_seed', type=int, default=20260817)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if os.path.exists(args.out):
        print(f'{args.out} exists, skipping', flush=True)
        return
    if not os.path.exists(args.in_path):
        print(f'[ERROR] q2_corridor_topup.py: {args.in_path} does not exist', file=sys.stderr)
        sys.exit(1)
    n, flights = topup(args.in_path, args.out, args.base_seed)
    print(f'{args.out}: repaired {n} shortcut starts, {flights} flights', flush=True)


if __name__ == '__main__':
    main()
