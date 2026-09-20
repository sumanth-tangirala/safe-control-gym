'''Choose the twin-curtain F_max ladder, and measure what the design memo
says to measure.

Reports, per level: mean p_success, fraction_interior (the share of states
with 0 < p < 1, i.e. the transition shell), the horizon-hit count, and
p_success split by the sign of the start's y -- the rescue/hazard split
(design memo §B2 item 4): the curtains force +y, so a start with y_0 < 0 is
pushed TOWARD the goal (rescue) and y_0 > 0 is pushed AWAY (hazard). The
aggregate p_success averages the two in equal measure by the start grid's own
y-symmetry, so the split is where the biased force's effect is legible.

y column: eval_states.txt rows (and this script's picks) are the shipped
13-D GROUPED/quaternion layout -- [x, y, z, qw, qx, qy, qz, x_dot, y_dot,
z_dot, p, q, r] (generate_quadrotor_3d_noisy.to_row13). y is column index 1
in that 13-D row -- NOT the env's interleaved 12-D state order (where y is
index 2). Do not confuse the two.

States are sampled at random from the shipped eval grid (a raw prefix is one
corner) and each level runs the same picked states, seeded by original row
index so levels stay paired. Workers follow q2_corridor_sweep.py's Pool
pattern.

Usage: python q3_corridor_sweep.py --n 400 --trials 30 --procs 24 --out sweep.npz
'''
import argparse
import os
from multiprocessing import Pool

import numpy as np

from q3_corridor_common import DET, HORIZON, NOISE_MODELS, build, roll, rollout_seed

# Lower rungs are the "nothing should happen" floor (design memo §F stage 3):
# the shipped noisy_dynamics family's per-axis std tops out at 0.042 N, so
# 0.02 N is comfortably below anything that family registers. The upper
# rungs probe toward the goal-calm ceiling -- 0.372 N for the TWIN curtain
# (half the single-curtain memo's 0.743 N, because both curtains' skirts
# reach the goal and their means add; see q3_corridor_common's module
# docstring). 0.50 is ABOVE that ceiling on purpose: a deliberate
# over-the-line probe to see what happens once the goal itself is no longer
# calm, not an oversight.
LEVELS = [0.0, 0.02, 0.05, 0.08, 0.13, 0.20, 0.30, 0.50]

ARGS = None
S_PICK = None
IDX_PICK = None
LEVEL = None
MODEL = None
AMBIENT = None


def _init(a, s_pick, idx_pick, level, model, ambient):
    global ARGS, S_PICK, IDX_PICK, LEVEL, MODEL, AMBIENT
    ARGS, S_PICK, IDX_PICK, LEVEL, MODEL, AMBIENT = a, s_pick, idx_pick, level, model, ambient


def _range(rng_pair):
    lo, hi = rng_pair
    trials = 1 if (LEVEL == 0 and (AMBIENT or 0) == 0) else ARGS.trials
    env, ctrl = build(LEVEL, model=MODEL, ambient=AMBIENT)
    p = np.zeros(hi - lo)
    hits = 0
    try:
        for i in range(lo, hi):
            ok_count = 0
            for k in range(trials):
                ok, steps, _ = roll(env, ctrl, S_PICK[i],
                                    rollout_seed(ARGS.base_seed, 1,
                                                 int(IDX_PICK[i]), k))
                ok_count += int(ok)
                hits += int(steps >= HORIZON)
            p[i - lo] = ok_count / trials
    finally:
        env.close()
    return lo, hi, p, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--base_seed', type=int, default=20260817)
    ap.add_argument('--out', default='sweep.npz')
    ap.add_argument('--levels', type=float, nargs='+', default=LEVELS)
    ap.add_argument('--ambient', type=float, default=None)
    ap.add_argument('--model', choices=sorted(NOISE_MODELS), default='sine+ambient',
                    help='noise model (see q3_corridor_common.NOISE_MODELS)')
    args = ap.parse_args()

    model = args.model
    # A fixed-ambient model (its NOISE_MODELS entry's 'ambient' is a number,
    # not None) rejects an explicit ambient= override -- see
    # resolve_noise_model(). --ambient has no default value of its own, so a
    # mismatch here is always the caller's, and gets a hard CLI error rather
    # than a silently-dropped flag or build()'s later ValueError.
    entry_ambient = NOISE_MODELS[model]['ambient']
    if entry_ambient is not None and args.ambient is not None:
        ap.error(f'--ambient={args.ambient} is not valid with model {model!r} '
                 f'(fixed ambient {entry_ambient}); use --model sine+ambient to '
                 f'set an ambient std.')
    if entry_ambient is None and args.ambient is None:
        ap.error(f'model {model!r} requires --ambient (a std in newtons); '
                 f'it has no fixed value.')
    ambient = args.ambient if entry_ambient is None else None

    rows = np.loadtxt(os.path.join(DET, 'eval_states.txt'), delimiter=',')
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(rows), args.n, replace=False))
    picked = rows[idx, 0:13]
    y0 = picked[:, 1]   # 13-D grouped-row column 1 -- see the module docstring

    edges = np.linspace(0, args.n, args.procs + 1).astype(int)
    ranges = [(int(edges[k]), int(edges[k + 1])) for k in range(args.procs)]

    p_all, interior_all, horizon_all, p_states_all = [], [], [], []
    for level in args.levels:
        p = np.zeros(args.n)
        hits = 0
        with Pool(args.procs, initializer=_init,
                  initargs=(args, picked, idx, level, model, ambient)) as pool:
            for lo, hi, pv, h in pool.imap_unordered(_range, ranges):
                p[lo:hi] = pv
                hits += h
        interior = float(np.mean((p > 0) & (p < 1)))
        p_states_all.append(p.copy())
        p_all.append(float(p.mean()))
        interior_all.append(interior)
        horizon_all.append(hits)
        p_rescue = float(p[y0 < 0].mean()) if np.any(y0 < 0) else float('nan')
        p_hazard = float(p[y0 >= 0].mean()) if np.any(y0 >= 0) else float('nan')
        print(f'F_max={level:.3f}  p_success={p.mean():.4f}  '
              f'fraction_interior={interior:.4f}  hit_horizon={hits}  '
              f'p_success[y0<0 rescue]={p_rescue:.4f}  '
              f'p_success[y0>=0 hazard]={p_hazard:.4f}', flush=True)

    np.savez(args.out, levels=np.asarray(args.levels),
             p_success=np.asarray(p_all),
             fraction_interior=np.asarray(interior_all),
             hit_horizon=np.asarray(horizon_all),
             p_states=np.asarray(p_states_all),
             row_index=idx, y0=y0)


if __name__ == '__main__':
    main()
