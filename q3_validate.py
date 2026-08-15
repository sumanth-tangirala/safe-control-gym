'''Level-0 gate. Nothing else in the chain runs unless this passes.

Both splits must reproduce the shipped deterministic labels. The ceiling is
98-99%, not 100% -- the generating code is not in this repo in runnable form --
so the threshold is set at 96%, well above the measured ceiling's spread but far
below what a genuine config drift would produce. Every bug found during
development (wrong body/world frame, missing cost=quadratic, wrong bounds,
truncated horizon) landed at 50-94%, so this gate would have caught all of them.
'''
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from generate_quadrotor_3d_noisy import (DET, EVAL_SPLIT_ID, TRAIN_SPLIT_ID, build, eval_starts,  # noqa: E402
                                         inject_sampler, inject_stored, rollout_seed, run, sampler_starts)

N = int(os.environ.get('VALIDATE_N', 400))
THRESHOLD = float(os.environ.get('VALIDATE_MIN', 0.96))
BASE = int(os.environ.get('BASE_SEED', 20260813))


def check_eval():
    starts, det = eval_starts(0, N)
    env, ctrl = build(0.0)
    ok = np.zeros(N, dtype=int)
    for i in range(N):
        env.reset(seed=rollout_seed(BASE, EVAL_SPLIT_ID, i, 0))
        ctrl.reset()
        ok[i] = run(env, ctrl, inject_stored(env, starts[i]), keep_states=False)[0]
    env.close()
    return (ok == det).mean(), det.mean(), ok.mean()


def check_train():
    lab = {}
    with open(DET + '/trajectory_labels.txt') as fh:
        for line in fh:
            name, val = line.rsplit(',', 1)
            lab[int(name.split('_')[1].split('.')[0])] = int(val)
    shipped = np.array([lab[i] for i in range(N)])
    starts = np.asarray(sampler_starts()[:N])
    env, ctrl = build(0.0)
    ok = np.zeros(N, dtype=int)
    for i in range(N):
        env.reset(seed=rollout_seed(BASE, TRAIN_SPLIT_ID, i, 0))
        ctrl.reset()
        ok[i] = run(env, ctrl, inject_sampler(env, starts[i]), keep_states=False)[0]
    env.close()
    return (ok == shipped).mean(), shipped.mean(), ok.mean()


def main():
    fail = False
    for name, fn in [('eval', check_eval), ('train', check_train)]:
        agree, shipped_rate, our_rate = fn()
        status = 'PASS' if agree >= THRESHOLD else 'FAIL'
        if agree < THRESHOLD:
            fail = True
        print(f'[{status}] {name:5} n={N}  agreement {agree:.4f} '
              f'(min {THRESHOLD})  shipped rate {shipped_rate:.4f}  '
              f'ours {our_rate:.4f}', flush=True)
    if fail:
        print('GATE FAILED -- downstream jobs will be cancelled by afterok',
              flush=True)
        sys.exit(1)
    print('gate passed', flush=True)


if __name__ == '__main__':
    main()
