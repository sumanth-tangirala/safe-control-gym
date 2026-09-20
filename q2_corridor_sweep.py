'''Choose the corridor level ladder, and measure what the spec says to measure.

Reports three things per level: mean p_success, fraction_interior (the share of
states with 0 < p < 1, i.e. the transition shell), and the horizon-hit count.
fraction_interior is the load-bearing one. Per-step draws average along a
crossing -- estimated at a 4-9% spread on the delivered impulse -- so if the
shell is too thin the family is not usable and the spec's frozen-field fallback
should be revisited rather than the levels pushed higher.

States are sampled at random from the full grid (a prefix is one corner) and
each level runs the same picked states, seeded by original row index so levels
stay paired. Workers follow q2_collect.py's Pool pattern.

Usage: python q2_corridor_sweep.py --n 400 --trials 30 --procs 24 --out sweep.npz
'''
import argparse
import os
from multiprocessing import Pool

import numpy as np

from q2_corridor_common import DET, HORIZON, NOISE_MODELS, build, roll, rollout_seed

# The 0.002-0.020 bracket came from an open-loop impulse estimate and measured
# far too weak: retention 0.97 at 0.016, fraction_interior 0.0025. The entry
# data shows 90.75% of successes cross the band, so there is no structural
# ceiling -- the ladder extends upward until retention actually falls.
LEVELS = [0.0, 0.002, 0.004, 0.006, 0.009, 0.012, 0.016, 0.020]
EXT_LEVELS = [0.03, 0.05, 0.08, 0.13, 0.20, 0.30]

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
    ap.add_argument('--model', choices=sorted(NOISE_MODELS), default=None,
                    help='noise model (see q2_corridor_common.NOISE_MODELS); '
                         "defaults to 'sine+ambient' when neither --model nor "
                         'the deprecated --draw is given')
    ap.add_argument('--draw', choices=['uniform', 'sine'], default=None,
                    help='deprecated alias for --model; ignored if --model is given')
    args = ap.parse_args()

    model = args.model if args.model is not None else (
        args.draw if args.draw is not None else 'sine+ambient')
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

    rows = np.loadtxt(os.path.join(DET, 'roa_labels.txt'), delimiter=',')
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(rows), args.n, replace=False))
    picked = rows[idx, 0:6]

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
        print(f'f_max={level:.3f}  p_success={p.mean():.4f}  '
              f'fraction_interior={interior:.4f}  hit_horizon={hits}', flush=True)

    np.savez(args.out, levels=np.asarray(args.levels),
             p_success=np.asarray(p_all),
             fraction_interior=np.asarray(interior_all),
             hit_horizon=np.asarray(horizon_all),
             p_states=np.asarray(p_states_all),
             row_index=idx)


if __name__ == '__main__':
    main()
