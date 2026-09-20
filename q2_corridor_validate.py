'''Level-0 gate: at f_max = 0 the corridor collector must reproduce the shipped
deterministic quad2d labels. Anything less means the plant transcription is
wrong and every noisy level built on it would be wrong the same way.

Usage: python q2_corridor_validate.py [--n 300] [--min_agreement 0.98]
'''
import argparse
import os

import numpy as np

from q2_corridor_common import DET, build, roll, rollout_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--min_agreement', type=float, default=0.98)
    ap.add_argument('--base_seed', type=int, default=20260817)
    args = ap.parse_args()

    # Balanced sample, as in q2_validate.py: the grid is ordered, so a raw
    # prefix is all one corner where everything fails and passes trivially at
    # 100% without ever exercising the success branch.
    rows = np.loadtxt(os.path.join(DET, 'roa_labels.txt'), delimiter=',')
    lab = rows[:, 6].astype(int)
    rng = np.random.default_rng(0)
    pick = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(lab == 1), args.n // 2, replace=False),
        rng.choice(np.flatnonzero(lab == 0), args.n // 2, replace=False)]))
    starts, det_labels = rows[pick, 0:6], lab[pick]

    env, ctrl = build(0.0)
    try:
        got = np.zeros(len(starts), dtype=int)
        for i, s in enumerate(starts):
            ok, _, _ = roll(env, ctrl, s, rollout_seed(args.base_seed, 1, int(pick[i]), 0))
            got[i] = int(ok)
    finally:
        env.close()

    agree = int((got == det_labels).sum())
    frac = agree / len(starts)
    on1 = float((got[det_labels == 1] == 1).mean())
    on0 = float((got[det_labels == 0] == 0).mean())
    print(f'agreement {agree}/{len(starts)} = {frac:.4f}  '
          f'(success rows {on1:.3f}, failure rows {on0:.3f})')
    if frac < args.min_agreement:
        raise SystemExit(f'FAIL: below {args.min_agreement}')
    print('PASS')


if __name__ == '__main__':
    main()
