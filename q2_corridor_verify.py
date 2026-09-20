'''Check that a collected config is complete and carries no shortcut leftovers.

Run this before reducing a config for publication. It answers three questions
the reducer does not:

  Is every shard index 0..n-1 present?
  Does every start reach exactly `trials` trials once its k-windows are summed?
  Do the windows agree with each other on starts and det_labels?

The trials check matters because of the 2026-08-19 shortcut bug: shard_eval used
to settle far-from-corridor starts in one flight, which is unsound whenever an
ambient term is present (see the FALSIFIED block in the corridor design spec).
Repaired shards carry a k0-20 file plus a k1-20 top-up that sums to 20. Shards
collected after the fix carry a single k0-20 file already at 20. Both layouts
are valid, so this checks the SUM rather than the file count.

Usage:
  python q2_corridor_verify.py --config sharp
  python q2_corridor_verify.py --level 0.05 --ambient 0.09 --nshards 560
'''
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

from q2_corridor_reduce import CONFIGS, find_shard_files


def verify(shard_dir, level, ambient, nshards, trials):
    by_idx = find_shard_files(shard_dir, 'eval', level, ambient)
    missing = [i for i in range(nshards) if i not in by_idx]

    bad_total = defaultdict(int)   # shard -> starts not at `trials`
    bad_starts, bad_det = [], []
    layout = defaultdict(int)      # window count -> how many shards
    n_starts = 0
    for idx in sorted(by_idx):
        windows = by_idx[idx]
        layout[len(windows)] += 1
        total = ref = det = None
        for path, _, _ in windows:
            d = np.load(path)
            u = d['trials_used'].astype(np.int64)
            total = u if total is None else total + u
            if ref is None:
                ref, det = d['starts'], d['det_labels']
            else:
                if not np.array_equal(d['starts'], ref):
                    bad_starts.append(idx)
                if not np.array_equal(d['det_labels'], det):
                    bad_det.append(idx)
        off = int((total != trials).sum())
        if off:
            bad_total[idx] = off
        n_starts += len(total)
    return dict(missing=missing, bad_total=dict(bad_total), bad_starts=sorted(set(bad_starts)),
                bad_det=sorted(set(bad_det)), layout=dict(layout), n_starts=n_starts,
                n_shards=len(by_idx))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', choices=sorted(CONFIGS))
    ap.add_argument('--level', type=float)
    ap.add_argument('--ambient', type=float)
    ap.add_argument('--nshards', type=int)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--shard_dir',
                    default=os.environ.get('Q2CORR_SHARDS',
                                           '/common/users/dm1487/q2_corridor_shards'))
    args = ap.parse_args(argv)
    if args.config:
        cfg = CONFIGS[args.config]
        level = cfg['level'] if args.level is None else args.level
        ambient = cfg['ambient'] if args.ambient is None else args.ambient
        nshards = cfg['n_eval'] if args.nshards is None else args.nshards
    else:
        if args.level is None or args.nshards is None:
            ap.error('--config, or both --level and --nshards, are required')
        level, ambient, nshards = args.level, args.ambient, args.nshards
    trials = 1 if level == 0 and not ambient else args.trials

    r = verify(args.shard_dir, level, ambient, nshards, trials)
    name = args.config or f'L{level} A{ambient}'
    print(f'{name}: {r["n_shards"]}/{nshards} shards, {r["n_starts"]} starts, expecting '
          f'{trials} trials each')
    print(f'  window layout (windows per shard -> shards): {r["layout"]}')

    fail = False
    if r['missing']:
        head = r['missing'][:12]
        more = '' if len(r['missing']) <= 12 else f' ... and {len(r["missing"]) - 12} more'
        print(f'  FAIL missing {len(r["missing"])} shard indices: {head}{more}')
        fail = True
    if r['bad_total']:
        worst = sorted(r['bad_total'].items(), key=lambda kv: -kv[1])[:5]
        print(f'  FAIL {len(r["bad_total"])} shards have starts not at {trials} trials; '
              f'worst (shard, count): {worst}')
        fail = True
    if r['bad_starts']:
        print(f'  FAIL windows disagree on starts for shards: {r["bad_starts"][:12]}')
        fail = True
    if r['bad_det']:
        print(f'  FAIL windows disagree on det_labels for shards: {r["bad_det"][:12]}')
        fail = True
    if not fail:
        print('  PASS complete, every start at the full trial count, windows consistent')
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
