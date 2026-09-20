'''quad2d stochastic collection under the altitude corridor.

Eval implements the spec's mitigation, but ONLY for models with no ambient
term: a state whose undisturbed trajectory never comes within MARGIN of the
corridor cannot be reached by an altitude-gated disturbance, so it is rolled
once instead of `trials` times and its p_success is 0 or 1. The margin covers
trajectories that noise could pull into the band. `trials_used` is stored per
state so the reducer can report exactly how many were shortcut rather than
leaving it implicit.

The shortcut does NOT apply when ambient > 0. The ambient term is white noise
on the same mask with no altitude gate, so it reaches a start at any height and
the "cannot be reached" premise is simply false. This was measured, not
reasoned: 10 shortcut states from the sharp run (f_max 0.08, ambient 0.06) were
re-flown 20 times each, and 7 of them varied, having been recorded as certain
successes. No wider MARGIN fixes this, because there is no altitude at which
the ambient term switches off.

Usage:
  python q2_corridor_collect.py --split train --level 0.009 \\
      --shard 0 --nshards 40 --out train_0.npz
  python q2_corridor_collect.py --split eval --level 0.009 --trials 50 \\
      --shard 0 --nshards 56 --out eval_0.npz
'''
import argparse
import os

import numpy as np

from q2_corridor_common import BAND, NOISE_MODELS, build, grid_states, roll, rollout_seed

N_STATES = 489_789
N_TRAIN = 500_000
TRAIN_SPLIT_ID, EVAL_SPLIT_ID = 0, 1
MARGIN = 0.10          # metres of altitude slack around the band


def sample_starts(n, seed):
    '''Random starts within the shipped sampling bounds, off the eval lattice.'''
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(-1.0, 1.0, n),      # x
        rng.uniform(0.1, 1.5, n),       # z
        rng.uniform(-np.pi, np.pi, n),  # theta
        rng.uniform(-1.0, 1.0, n),      # x_dot
        rng.uniform(-1.0, 1.0, n),      # z_dot
        rng.uniform(-8.0, 8.0, n),      # theta_dot
    ])


def shard_train(args, lo, hi):
    starts = sample_starts(N_TRAIN, args.base_seed)[lo:hi]
    env, ctrl = build(args.level, model=args.model, ambient=args.ambient)
    states, offsets, labels, seeds = [], [0], [], []
    try:
        for i in range(len(starts)):
            seed = rollout_seed(args.base_seed, TRAIN_SPLIT_ID, lo + i, 0)
            ok, _, traj = roll(env, ctrl, starts[i], seed, keep=True)
            states.append(np.asarray(traj, dtype=np.float32))
            offsets.append(offsets[-1] + len(traj))
            labels.append(ok)
            seeds.append(seed)
    finally:
        env.close()
    np.savez(args.out,
             states=np.concatenate(states),
             offsets=np.asarray(offsets, np.int64),
             starts=starts.astype(np.float64),
             labels=np.asarray(labels, np.uint8),
             seeds=np.asarray(seeds, np.int64),
             lo=lo, hi=hi, f_max=args.level, model=args.model,
             ambient=-1.0 if args.ambient is None else args.ambient)
    return int(np.sum(labels)), len(labels)


def shard_eval(args, lo, hi):
    starts, det_labels = grid_states(lo, hi)
    env, ctrl = build(args.level, model=args.model, ambient=args.ambient)
    hits = np.zeros(len(starts), dtype=np.int32)
    used = np.zeros(len(starts), dtype=np.int32)
    ambient_on = args.ambient is not None and args.ambient > 0
    try:
        for i in range(len(starts)):
            # Trial 0 is always the reachability probe. In a top-up window
            # (trial_lo > 0) it replays identically by seeding and is NOT
            # counted -- its hits/used live in the first window's file.
            #
            # Only fly it when something reads it. The first window records
            # its hit, and the shortcut below needs `entered`. A top-up window
            # with an ambient term needs neither, so flying it is dead work:
            # one wasted rollout per state, ~1M across the quad2d K=20 to
            # K=50 campaign. Seeds are a pure function of (index, trial), so
            # skipping it changes no k >= 1 draw.
            entered = False
            if args.trial_lo == 0 or not ambient_on:
                ok, _, _, entered = roll(
                    env, ctrl, starts[i],
                    rollout_seed(args.base_seed, EVAL_SPLIT_ID, lo + i, 0),
                    track_band=True)
                if args.trial_lo == 0:
                    hits[i] = int(ok)
                    used[i] = 1
            # The shortcut is sound only when EVERY disturbance term is gated
            # on altitude. The corridor gust is, through sigma(z). The ambient
            # term is NOT -- build() adds it as white_noise with no altitude
            # condition, so it acts on a start at any height, 1200 steps per
            # rollout. Measured 2026-08-19 at f_max 0.08 / ambient 0.06: of 10
            # shortcut states re-flown 20 times, 7 varied. They were recorded
            # as p_success 1.0 and came back 13, 15, 17, 17, 17, 17 and 18 of
            # 20. So with an ambient term every state gets the full `trials`.
            if not ambient_on:
                reachable = entered or (
                    BAND[0] - MARGIN <= starts[i][1] <= BAND[1] + MARGIN)
                if args.level == 0 or not reachable:
                    continue
            for k in range(max(1, args.trial_lo), args.trials):
                ok, _, _ = roll(env, ctrl, starts[i],
                                rollout_seed(args.base_seed, EVAL_SPLIT_ID, lo + i, k))
                hits[i] += int(ok)
                used[i] += 1
    finally:
        env.close()
    np.savez(args.out, starts=starts, hits=hits, trials_used=used,
             det_labels=det_labels, lo=lo, hi=hi, f_max=args.level,
             model=args.model, trial_lo=args.trial_lo, trials=args.trials,
             ambient=-1.0 if args.ambient is None else args.ambient)
    return int(hits.sum()), int(used.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'eval'], required=True)
    ap.add_argument('--level', type=float, required=True)
    ap.add_argument('--model', choices=sorted(NOISE_MODELS), default='sine+ambient')
    ap.add_argument('--ambient', type=float, default=None)
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--trial_lo', type=int, default=0,
                    help='first trial index of this window; seeds are pure '
                         'functions of (index, trial), so a later top-up window '
                         'draws exactly what a single long run would have')
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--base_seed', type=int, default=20260817)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # Same guardrails as the sweep CLI: never silently drop or invent ambient.
    entry_ambient = NOISE_MODELS[args.model]['ambient']
    if entry_ambient is None and args.ambient is None:
        ap.error(f'model {args.model!r} requires --ambient')
    if entry_ambient is not None and args.ambient is not None:
        ap.error(f'model {args.model!r} has fixed ambient; drop --ambient '
                 f"or use 'sine+ambient'")

    # Idempotent: a completed shard is never redone, so resubmitting a partly
    # failed array costs only the missing work.
    if os.path.exists(args.out):
        print(f'{args.out} exists, skipping')
        return

    total = N_TRAIN if args.split == 'train' else N_STATES
    edges = np.linspace(0, total, args.nshards + 1).astype(int)
    lo, hi = int(edges[args.shard]), int(edges[args.shard + 1])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    fn = shard_train if args.split == 'train' else shard_eval
    got, n = fn(args, lo, hi)
    print(f'{args.split} f_max={args.level} shard {args.shard}/{args.nshards} '
          f'[{lo}:{hi}] -> {got}/{n}', flush=True)


if __name__ == '__main__':
    main()
