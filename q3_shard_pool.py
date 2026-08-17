'''Runs one SLURM array task's shard across that task's cores.

The collector is single-process per range; this splits the shard across the
allocated CPUs and merges the pieces into one npz. Sub-ranges are contiguous
slices of the shard, so a worker's output maps back to global indices without
bookkeeping, and rollout_seed stays a pure function of the global index -- the
core-count therefore does NOT change what any rollout draws.
'''
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from generate_quadrotor_3d_noisy import (EVAL_SPLIT_ID, N_TRAIN, TRAIN_SPLIT_ID, build,  # noqa: E402
                                         eval_starts, inject_sampler, inject_stored, rollout_seed, run,
                                         sampler_starts)

ARGS = None


def _train_range(rng):
    lo, hi = rng
    starts = np.asarray(sampler_starts()[lo:hi])
    env, ctrl = build(ARGS.level, ARGS.mechanism)
    states, lengths, labels, seeds = [], [], [], []
    for i in range(len(starts)):
        seed = rollout_seed(ARGS.base_seed, TRAIN_SPLIT_ID, lo + i, 0)
        env.reset(seed=seed)
        ctrl.reset()
        ok, traj = run(env, ctrl, inject_sampler(env, starts[i]), keep_states=True)
        states.append(np.asarray(traj, dtype=np.float32))
        lengths.append(len(traj))
        labels.append(ok)
        seeds.append(seed)
    env.close()
    return (lo, np.concatenate(states), np.asarray(lengths, np.int64),
            np.asarray(labels, np.uint8), np.asarray(seeds, np.int64),
            starts.astype(np.float64))


def _eval_range(rng):
    lo, hi = rng
    starts, det = eval_starts(lo, hi)
    trials = 1 if ARGS.level == 0 else ARGS.trials
    env, ctrl = build(ARGS.level, ARGS.mechanism)
    hits = np.zeros(len(starts), dtype=np.int32)
    for i in range(len(starts)):
        for k in range(trials):
            env.reset(seed=rollout_seed(ARGS.base_seed, EVAL_SPLIT_ID, lo + i, k))
            ctrl.reset()
            hits[i] += run(env, ctrl, inject_stored(env, starts[i]),
                           keep_states=False)[0]
    env.close()
    return lo, starts, hits, det, trials


def _init(a):
    global ARGS
    ARGS = a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'eval'], required=True)
    ap.add_argument('--level', type=float, required=True)
    ap.add_argument('--mechanism', default='dynamics', choices=['dynamics', 'action'])
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--base_seed', type=int, default=20260813)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f'{args.out} exists, skipping', flush=True)
        return

    total = N_TRAIN if args.split == 'train' else 1_000_000
    edges = np.linspace(0, total, args.nshards + 1).astype(int)
    lo, hi = int(edges[args.shard]), int(edges[args.shard + 1])
    nproc = len(os.sched_getaffinity(0))
    sub = np.linspace(lo, hi, nproc + 1).astype(int)
    ranges = [(int(sub[i]), int(sub[i + 1])) for i in range(nproc)
              if sub[i + 1] > sub[i]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f'{args.split} L={args.level} shard {args.shard}/{args.nshards} '
          f'[{lo}:{hi}] on {len(ranges)} procs', flush=True)

    # Build the starts cache once here rather than letting 40 workers race on it.
    if args.split == 'train':
        sampler_starts()

    fn = _train_range if args.split == 'train' else _eval_range
    with Pool(len(ranges), initializer=_init, initargs=(args,)) as pool:
        out = sorted(pool.map(fn, ranges), key=lambda r: r[0])

    tmp = args.out + f'.tmp{os.getpid()}'
    if args.split == 'train':
        states = np.concatenate([o[1] for o in out])
        lengths = np.concatenate([o[2] for o in out])
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        labels = np.concatenate([o[3] for o in out])
        np.savez(tmp, states=states, offsets=offsets,
                 starts=np.concatenate([o[5] for o in out]),
                 labels=labels, seeds=np.concatenate([o[4] for o in out]),
                 lo=lo, hi=hi, level=args.level, mechanism=args.mechanism)
        print(f'  -> {int(labels.sum())}/{len(labels)} successes, '
              f'{len(states)} states', flush=True)
    else:
        hits = np.concatenate([o[2] for o in out])
        trials = out[0][4]
        np.savez(tmp, starts=np.concatenate([o[1] for o in out]), hits=hits,
                 det_labels=np.concatenate([o[3] for o in out]), trials=trials,
                 lo=lo, hi=hi, level=args.level, mechanism=args.mechanism)
        print(f'  -> p_success mean {hits.sum() / (len(hits) * trials):.4f} '
              f'over {trials} trials', flush=True)
    # Atomic publish: a killed task never leaves a half-written shard that the
    # skip-if-exists check would then treat as complete.
    os.replace(tmp + '.npz' if not tmp.endswith('.npz') else tmp, args.out)


if __name__ == '__main__':
    main()
