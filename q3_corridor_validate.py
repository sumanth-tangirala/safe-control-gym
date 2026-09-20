'''Level-0 gate: at f_max = 0 the corridor collector must reproduce the shipped
deterministic quad3d labels. Anything less means the plant transcription is
wrong and every noisy level built on it would be wrong the same way.

quad3d's own level-0 reproduces the shipped labels at ~0.97, not 1.0 -- chaos
amplification over ~500-step trajectories plus boundary ties, measured in
q3_validate.py and not closeable from this repo. 0.98 would false-alarm on
that residual, so the default threshold here is 0.95, below the historical
0.97. With HORIZON=2000 against the shipped set's effectively-unbounded
horizon (the corridor collector never truncates a success early), agreement
on SUCCESS rows should improve on the historical number; a drop in
FAILURE-row agreement instead of an improvement in success-row agreement
would signal a transcription bug, not a coincidence.

Usage: python q3_corridor_validate.py [--n 400] [--min_agreement 0.95]
'''
import argparse
import os

import numpy as np

from q3_corridor_common import DET, build, roll, rollout_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--min_agreement', type=float, default=0.95)
    ap.add_argument('--base_seed', type=int, default=20260817)
    args = ap.parse_args()

    # Balanced sample, as in q2_corridor_validate.py: eval_states.txt is
    # ordered by the sampler that produced it, so a raw prefix risks one
    # corner of the state space where everything fails (or succeeds) and the
    # gate would pass trivially without exercising both branches. Column
    # layout matches generate_quadrotor_3d_noisy's own eval loading: 0:13 is
    # the 13-D grouped/quaternion start state, 26 is the deterministic label.
    rows = np.loadtxt(os.path.join(DET, 'eval_states.txt'), delimiter=',')
    lab = rows[:, 26].astype(int)
    rng = np.random.default_rng(0)
    pick = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(lab == 1), args.n // 2, replace=False),
        rng.choice(np.flatnonzero(lab == 0), args.n // 2, replace=False)]))
    starts, det_labels = rows[pick, 0:13], lab[pick]

    # model='sine' at f_max=0 installs no disturbance (its fixed ambient is
    # 0.0) -- the same route q2_corridor_validate.py takes, and it avoids
    # 'sine+ambient' rejecting a bare build(0.0) for lack of an ambient value.
    env, ctrl = build(0.0, model='sine')
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
