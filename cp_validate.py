'''Cartpole sigma=0 gate. Balanced sample -- the grid is ordered, so a prefix
is one corner where everything fails and would pass trivially.'''
import os
import sys

import numpy as np

from cp_collect import EVAL_SPLIT_ID, build, eval_starts, roll, rollout_seed

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))

N = int(os.environ.get('VALIDATE_N', 300))
MIN = float(os.environ.get('VALIDATE_MIN', 0.97))
S, det = eval_starts()
rng = np.random.default_rng(0)
pick = np.sort(np.concatenate([
    rng.choice(np.flatnonzero(det == 1), N // 2, replace=False),
    rng.choice(np.flatnonzero(det == 0), N // 2, replace=False)]))
env, ctrl = build(0.0)
ok = np.array([roll(env, ctrl, S[i], rollout_seed(EVAL_SPLIT_ID, i, 0))[0]
               for i in pick], dtype=int)
env.close()
d = det[pick]
agree = (ok == d)
print(f'cartpole gate n={len(pick)}  agreement {agree.mean():.4f} (min {MIN})  '
      f'success rows {agree[d == 1].mean():.3f}  failure rows {agree[d == 0].mean():.3f}',
      flush=True)
sys.exit(0 if agree.mean() >= MIN else 1)
