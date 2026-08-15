'''quad2d level-0 gate. Balanced sample: the grid is ordered, so a prefix is
all one corner where everything fails and would pass trivially at 100%.'''
import os
import sys

import numpy as np

from q2_common import DET, build, roll

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))

N = int(os.environ.get('VALIDATE_N', 300))
MIN = float(os.environ.get('VALIDATE_MIN', 0.95))
rows = np.loadtxt(DET + '/roa_labels.txt', delimiter=',')
lab = rows[:, 6].astype(int)
rng = np.random.default_rng(0)
pick = np.sort(np.concatenate([
    rng.choice(np.flatnonzero(lab == 1), N // 2, replace=False),
    rng.choice(np.flatnonzero(lab == 0), N // 2, replace=False)]))
S, det = rows[pick, 0:6], lab[pick]
env, ctrl = build(0.0)
ok = np.array([roll(env, ctrl, S[i], seed=1)[0] for i in range(len(pick))], dtype=int)
env.close()
agree = (ok == det)
print(f'quad2d gate n={len(pick)}  agreement {agree.mean():.4f} (min {MIN})  '
      f'success rows {agree[det == 1].mean():.3f}  failure rows {agree[det == 0].mean():.3f}',
      flush=True)
sys.exit(0 if agree.mean() >= MIN else 1)
