'''quad2d stochastic collection under a planar external force disturbance.

Simpler than the quad3d collector in one respect: the shipped quad2d set has
489,789 trajectories and 489,789 roa_labels/eval_states rows, one per
trajectory, all from the same stratified grid. EVAL uses exactly those states; TRAIN
uses random starts within the same bounds (off-lattice coverage for learning,
while eval stays 1:1 with the shipped set for comparison). (quad3d differs: its
eval_states holds 1M intermediate states, distinct from the 800k starts.)

Also simpler on injection: for TWO_D the env stores ang_v[1] directly in the
world frame with no body conversion, so there is none of the quad3d
sampler-vs-stored rate asymmetry. One injection path serves both splits.

Everything else is transcribed from deterministic/quadrotor2D_rl:
    controller  safe_explorer_ppo, shipped model, obs_normalizer frozen
    success     ||state - [0,1,0,0,0,0]|| < 0.2, entry-cut
    horizon     1200 steps, INHERITED not chosen -- the shipped set already
                used it as a real limit (longest trajectory 709, timeouts 0)
    bounds      x +/-1.0, z [0.1,1.5], velocities +/-1.0, theta_dot +/-8.0,
                theta infinite

The horizon matters more here than for quad3d. Measured in the sweep, rollouts
start hitting the 1200 cap from f=0.020 upward, against a deterministic max of
674 -- so p_success is a bounded-time quantity and part of the decline is
trajectories crossing the deadline rather than failing.
'''
import argparse
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from q2_common import DET, HORIZON, build, roll, rollout_seed  # noqa: E402

N_STATES = 489_789
BASE_SEED = 20260813
ARGS = None
_CACHE = os.environ.get('Q2_STATES_CACHE', '')

# Train uses RANDOM starts; eval keeps the exact deterministic grid states. The
# two splits play different roles: eval is the comparison artefact and must
# align 1:1 with the shipped set, train is for learning and benefits from
# off-lattice coverage. Same convention as cartpole and quad3d.
SAMPLE = dict(x=1.0, z_lo=0.1, z_hi=1.5, theta=math.pi,
              x_dot=1.0, z_dot=1.0, theta_dot=8.0)
N_TRAIN = int(os.environ.get('N_TRAIN', 489_789))


def train_starts():
    """Random within bounds, rejecting states at or beyond a termination
    threshold -- the same filter the deterministic grid generator applied.
    Returns FILE order [x, z, theta, x_dot, z_dot, theta_dot]."""
    rng = np.random.default_rng(BASE_SEED)
    out = np.empty((N_TRAIN, 6))
    n = 0
    while n < N_TRAIN:
        m = (N_TRAIN - n) * 2
        c = np.column_stack([
            rng.uniform(-SAMPLE['x'], SAMPLE['x'], m),
            rng.uniform(SAMPLE['z_lo'], SAMPLE['z_hi'], m),
            rng.uniform(-SAMPLE['theta'], SAMPLE['theta'], m),
            rng.uniform(-SAMPLE['x_dot'], SAMPLE['x_dot'], m),
            rng.uniform(-SAMPLE['z_dot'], SAMPLE['z_dot'], m),
            rng.uniform(-SAMPLE['theta_dot'], SAMPLE['theta_dot'], m)])
        ok = ((np.abs(c[:, 0]) < SAMPLE['x'])
              & (c[:, 1] > SAMPLE['z_lo']) & (c[:, 1] < SAMPLE['z_hi'])
              & (np.abs(c[:, 3]) < SAMPLE['x_dot'])
              & (np.abs(c[:, 4]) < SAMPLE['z_dot'])
              & (np.abs(c[:, 5]) < SAMPLE['theta_dot']))
        c = c[ok][:N_TRAIN - n]
        out[n:n + len(c)] = c
        n += len(c)
    return out


def all_states():
    '''roa_labels.txt rows: 6 state columns + the shipped binary label.'''
    if _CACHE and os.path.exists(_CACHE):
        d = np.load(_CACHE, mmap_mode='r')
        return d['S'], d['lab']
    rows = np.loadtxt(DET + '/roa_labels.txt', delimiter=',')
    S, lab = rows[:, 0:6], rows[:, 6].astype(np.int8)
    if _CACHE:
        tmp = _CACHE + f'.tmp{os.getpid()}.npz'
        np.savez(tmp, S=S, lab=lab)
        os.replace(tmp, _CACHE)
    return S, lab


def _init(a):
    global ARGS
    ARGS = a


def _range(rng):
    lo, hi = rng
    if ARGS.split == 'train':
        S = train_starts()[lo:hi]
        lab = np.full(len(S), -1, np.int8)     # random starts: no shipped label
    else:
        S, lab = all_states()
        S = np.asarray(S[lo:hi])
        lab = np.asarray(lab[lo:hi])
    trials = 1 if ARGS.level == 0 else ARGS.trials
    env, ctrl = build(ARGS.level)
    if ARGS.split == 'train':
        states, lengths, labels, seeds = [], [], [], []
        for i in range(len(S)):
            seed = rollout_seed(BASE_SEED, lo + i, 0)
            ok, _, traj = roll(env, ctrl, S[i], seed=seed, keep=True)
            states.append(np.asarray(traj, dtype=np.float32))
            lengths.append(len(traj))
            labels.append(ok)
            seeds.append(seed)
        env.close()
        return (lo, np.concatenate(states), np.asarray(lengths, np.int64),
                np.asarray(labels, np.uint8), np.asarray(seeds, np.int64),
                S.astype(np.float64), lab)
    hits = np.zeros(len(S), dtype=np.int32)
    for i in range(len(S)):
        for k in range(trials):
            ok, _, _ = roll(env, ctrl, S[i], seed=rollout_seed(BASE_SEED, lo + i, k))
            hits[i] += ok
    env.close()
    return lo, S, hits, lab, trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'eval'], required=True)
    ap.add_argument('--level', type=float, required=True)
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f'{args.out} exists, skipping', flush=True)
        return

    total = N_TRAIN if args.split == 'train' else N_STATES
    edges = np.linspace(0, total, args.nshards + 1).astype(int)
    lo, hi = int(edges[args.shard]), int(edges[args.shard + 1])
    nproc = min(int(os.environ.get('NPROC', 0)) or 10 ** 6,
                len(os.sched_getaffinity(0)))
    sub = np.linspace(lo, hi, nproc + 1).astype(int)
    ranges = [(int(sub[i]), int(sub[i + 1])) for i in range(nproc)
              if sub[i + 1] > sub[i]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.split == 'eval':
        all_states()  # build the cache once, not in 30 racing workers
    print(f'{args.split} L={args.level} shard {args.shard}/{args.nshards} '
          f'[{lo}:{hi}] on {len(ranges)} procs, horizon {HORIZON}', flush=True)

    with Pool(len(ranges), initializer=_init, initargs=(args,)) as pool:
        out = sorted(pool.map(_range, ranges), key=lambda r: r[0])

    tmp = args.out + f'.tmp{os.getpid()}.npz'
    if args.split == 'train':
        states = np.concatenate([o[1] for o in out])
        lengths = np.concatenate([o[2] for o in out])
        labels = np.concatenate([o[3] for o in out])
        np.savez(tmp, states=states,
                 offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
                 starts=np.concatenate([o[5] for o in out]), labels=labels,
                 seeds=np.concatenate([o[4] for o in out]),
                 det_labels=np.concatenate([o[6] for o in out]),
                 lo=lo, hi=hi, level=args.level)
        print(f'  -> {int(labels.sum())}/{len(labels)} successes, '
              f'{len(states)} states', flush=True)
    else:
        hits = np.concatenate([o[2] for o in out])
        trials = out[0][4]
        np.savez(tmp, starts=np.concatenate([o[1] for o in out]), hits=hits,
                 det_labels=np.concatenate([o[3] for o in out]), trials=trials,
                 lo=lo, hi=hi, level=args.level)
        print(f'  -> p_success mean {hits.sum() / (len(hits) * trials):.4f} '
              f'over {trials} trials', flush=True)
    os.replace(tmp, args.out)


if __name__ == '__main__':
    main()
