'''quad2d noise sweep: planar world-frame force, measured under the SHIPPED rule.

Success is the env's own goal_reached at radius 0.2 with entry-cut, matching
deterministic/quadrotor2D_rl. Levels are NOT scaled from quad3d: the two
systems differ in both directions (quad2d's goal ball is 4x looser but its
bounds are far tighter), so the usable range has to be measured here.

Common random numbers: seed is a function of (state index, trial) and excludes
the level, so every level sees the same noise stream per state.
'''
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from q2_common import DET, build, roll, rollout_seed  # noqa: E402

N_STATES = int(os.environ.get('N_STATES', 4000))
TRIALS = int(os.environ.get('TRIALS', 5))
NPROC = int(os.environ.get('NPROC', 32))
BASE = 20260813
WEIGHT_N = 0.027 * 9.81          # 0.2648 N, the scale levels are quoted against

LEVELS = [0.0, 0.005, 0.010, 0.020, 0.040, 0.070, 0.100, 0.150, 0.220, 0.300]

_rows = np.loadtxt(DET + '/roa_labels.txt', delimiter=',')
_rng = np.random.default_rng(0)
PICK = np.sort(_rng.choice(len(_rows), N_STATES, replace=False))
STATES = _rows[PICK, 0:6]
DET_LABELS = _rows[PICK, 6].astype(int)
del _rows


def work(task):
    level, lo, hi = task
    trials = 1 if level == 0 else TRIALS
    env, ctrl = build(level)
    hits = np.zeros(hi - lo, dtype=np.int32)
    steps = np.zeros(hi - lo, dtype=np.int32)
    for j, i in enumerate(range(lo, hi)):
        for k in range(trials):
            ok, st, _ = roll(env, ctrl, STATES[i], seed=rollout_seed(BASE, i, k))
            hits[j] += ok
            steps[j] = max(steps[j], st)
    env.close()
    return level, lo, hits, steps, trials


def main():
    edges = np.linspace(0, N_STATES, NPROC + 1).astype(int)
    tasks = [(lv, int(edges[c]), int(edges[c + 1]))
             for lv in LEVELS for c in range(NPROC) if edges[c + 1] > edges[c]]
    acc = {}
    with Pool(NPROC) as pool:
        for level, lo, hits, steps, trials in pool.imap_unordered(work, tasks):
            a = acc.setdefault(level, [np.zeros(N_STATES, np.int32),
                                       np.zeros(N_STATES, np.int32), trials])
            a[0][lo:lo + len(hits)] = hits
            a[1][lo:lo + len(steps)] = steps

    base = DET_LABELS == 1
    print(f'\n{N_STATES} states sampled from the {len(DET_LABELS)}-state grid, '
          f'{TRIALS} trials per level')
    print(f'shipped success rate on this sample: {base.mean():.4f}\n')
    print(f'{"f (N)":>7} {"%weight":>8} {"p(success)":>11} {"retained":>9} '
          f'{"gained":>8} {"interior":>9} {"maxsteps":>9}')
    for lv in LEVELS:
        hits, steps, trials = acc[lv]
        p = hits / trials
        ret = p[base].mean() if base.any() else float('nan')
        gain = p[~base].mean() if (~base).any() else float('nan')
        interior = ((p > 0) & (p < 1)).mean()
        print(f'{lv:7.3f} {100 * lv / WEIGHT_N:7.1f}% {p.mean():11.4f} '
              f'{ret:9.3f} {gain:8.4f} {interior:9.3f} {steps.max():9d}')


if __name__ == '__main__':
    main()
