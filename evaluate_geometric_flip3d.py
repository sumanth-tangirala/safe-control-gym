#!/usr/bin/env python3
'''Measure the hand-coded controller 1 (`quad_composition.geometric_flip3d`).

Two questions, one set of rollouts:

  1. HOW OFTEN DOES CONTROLLER 1 REACH G1, BY INITIAL TILT?  Initial states are
     drawn from the env's OWN closed state space (`sampling_bounds_from_env`),
     with the attitude's tilt confined to one bucket at a time so the answer is
     resolved against the thing that actually varies.

  2. WHAT DOES THE COMPOSITION DO?  Every rollout goes through
     `rollout3d.rollout_composite`, which owns the latch, so the four outcomes

         F1       controller 1 never reached G1
         S1->S2   handed off, and controller 2 (LQR) reached the goal ball
         S1->F2   handed off, and controller 2 did not

     are read straight off the same runs (S1 = S1->S2 + S1->F2).

`--attitude_only` answers a THIRD, diagnostic question -- how good is the
attitude law with the translational sampling attrition removed -- by pinning
position to the goal, and velocity and body rates to zero.  It is not the
headline number; it isolates the controller from the state distribution.

Sampling is done in the parent and handed to the workers, so the results
depend on `--seed` and on nothing else -- not on `--workers`, not on the order
in which rollouts finish.
'''

import argparse
import json
import multiprocessing as mp

import numpy as np

from quad_composition.flip_env3d import (G_NOM_3D, capped_tilt_quat, sample_uniform_state,
                                         sampling_bounds_from_env)
from quad_composition.g1 import G1Region
from quad_composition.geometric_flip3d import GeometricFlipController3D
from quad_composition.rollout3d import (MAX_STEPS, QUAT_SLICE, make_env, make_env_and_ctrl2,
                                        rollout_composite)

BUCKETS_DEG = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]

# Worker-local env/controllers: built once per process in `_init_worker`,
# because constructing a PyBullet env and an LQR gain per rollout would cost
# more than the rollouts themselves.
_WORKER = {}


def bucket_states(rng, bounds, lo_deg, hi_deg, count, margin=1.0, attitude_only=False):
    '''`count` dataset-order initial states whose tilt lies in [lo, hi) degrees.

    Everything except the attitude comes from `sample_uniform_state`, i.e. the
    same closed-state-space draw training and calibration use; only the
    quaternion is replaced, by a tilt-band sample.  `margin` shrinks the
    non-attitude box about its centre (1.0 = the full closed box); the
    feasibility probe used 0.97, and reporting both makes the comparison
    honest rather than flattering.

    `attitude_only` instead pins position to the stabilization goal and zeroes
    velocity and body rates -- the isolated attitude diagnostic.
    '''
    low, high = bounds
    mid = 0.5 * (low + high)
    states = []
    for _ in range(count):
        state = sample_uniform_state(rng, bounds)
        if attitude_only:
            state[:3] = [0.0, 0.0, 1.0]
            state[7:] = 0.0
        elif margin != 1.0:
            state = mid + margin * (state - mid)
        state[QUAT_SLICE] = capped_tilt_quat(rng, np.radians(hi_deg),
                                             min_tilt=np.radians(lo_deg))
        states.append(state)
    return states


def _init_worker(gains, g1_dict):
    # `randomized_init` is False and every initial state is handed in from the
    # parent, so the worker's env needs no seed of its own.
    env, ctrl2 = make_env_and_ctrl2()
    _WORKER['env'] = env
    _WORKER['ctrl2'] = ctrl2
    _WORKER['ctrl1'] = GeometricFlipController3D(env, **gains)
    _WORKER['g1'] = G1Region.from_dict(g1_dict)


def _run_one(job):
    '''One composite rollout.  Returns the bucket index and the outcome flags
    `rollout_composite` reports -- nothing is recomputed here, so this script
    cannot disagree with the module that owns the latch.
    '''
    bucket_index, init, max_steps = job
    res = rollout_composite(_WORKER['env'], _WORKER['ctrl1'], _WORKER['ctrl2'],
                            _WORKER['g1'], np.asarray(init, dtype=float),
                            max_steps=max_steps)
    return (bucket_index, bool(res.flip_success), bool(res.ctrl2_success),
            int(res.handoff_index), len(res.trajectory))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num_per_bucket', type=int, default=60)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--margin', type=float, default=1.0,
                        help='shrink the non-attitude sampling box about its centre '
                             '(1.0 = the full closed state space; the probe used 0.97)')
    parser.add_argument('--attitude_only', action='store_true',
                        help='pin position to the goal and zero velocity/rates')
    parser.add_argument('--k_p', type=float, default=None)
    parser.add_argument('--k_d', type=float, default=None)
    parser.add_argument('--thrust_up_frac', type=float, default=None)
    parser.add_argument('--no_saturate_inverted', action='store_true',
                        help="use the feasibility probe's exact (un-normalised) tilt error")
    parser.add_argument('--collective', default=None, choices=['bangbang', 'compensated'])
    parser.add_argument('--json_out', default=None)
    args = parser.parse_args()

    gains = {k: v for k, v in (('k_p', args.k_p), ('k_d', args.k_d),
                               ('thrust_up_frac', args.thrust_up_frac),
                               ('collective', args.collective)) if v is not None}
    if args.no_saturate_inverted:
        gains['saturate_inverted'] = False

    # G1 = G_nom (10 deg, 4 rad/s), the target controller 1 is built against.
    g1 = G_NOM_3D

    # Sample in the parent, then close its env before forking workers: the
    # draw must not depend on --workers, and a live PyBullet connection must
    # not be inherited.
    rng = np.random.default_rng(args.seed)
    env = make_env(seed=args.seed)
    bounds = sampling_bounds_from_env(env)
    jobs = []
    for index, (lo, hi) in enumerate(BUCKETS_DEG):
        for state in bucket_states(rng, bounds, lo, hi, args.num_per_bucket,
                                   margin=args.margin, attitude_only=args.attitude_only):
            jobs.append((index, state.tolist(), args.max_steps))
    env.close()

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=args.workers, initializer=_init_worker,
                  initargs=(gains, g1.to_dict())) as pool:
        results = pool.map(_run_one, jobs, chunksize=1)

    n = np.zeros(len(BUCKETS_DEG), dtype=int)
    s1 = np.zeros(len(BUCKETS_DEG), dtype=int)
    s1s2 = np.zeros(len(BUCKETS_DEG), dtype=int)
    handoff_steps = [[] for _ in BUCKETS_DEG]
    for index, flip_success, ctrl2_success, handoff_index, _ in results:
        n[index] += 1
        s1[index] += int(flip_success)
        s1s2[index] += int(flip_success and ctrl2_success)
        if flip_success:
            handoff_steps[index].append(handoff_index)

    print(f'gains={gains or "defaults"}  G1={g1.to_dict()}  margin={args.margin}'
          f'  attitude_only={args.attitude_only}  N/bucket={args.num_per_bucket}')
    print(f'{"tilt bucket":>14} {"N":>4} {"S1 (reach G1)":>15} {"S1->S2":>14} {"S1->F2":>14}'
          f' {"median handoff":>15}')
    for index, (lo, hi) in enumerate(BUCKETS_DEG):
        med = np.median(handoff_steps[index]) if handoff_steps[index] else float('nan')
        print(f'{lo:5d}-{hi:<3d}deg {n[index]:>4d} '
              f'{s1[index]:>5d} ({s1[index] / n[index] * 100:5.1f}%) '
              f'{s1s2[index]:>5d} ({s1s2[index] / n[index] * 100:5.1f}%) '
              f'{s1[index] - s1s2[index]:>5d} ({(s1[index] - s1s2[index]) / n[index] * 100:5.1f}%) '
              f'{med:>15.0f}')
    tot_n, tot_s1, tot_s2 = int(n.sum()), int(s1.sum()), int(s1s2.sum())
    print(f'{"ALL":>14} {tot_n:>4d} '
          f'{tot_s1:>5d} ({tot_s1 / tot_n * 100:5.1f}%) '
          f'{tot_s2:>5d} ({tot_s2 / tot_n * 100:5.1f}%) '
          f'{tot_s1 - tot_s2:>5d} ({(tot_s1 - tot_s2) / tot_n * 100:5.1f}%)')
    print(f'F1 (never reached G1): {tot_n - tot_s1}/{tot_n} '
          f'({(tot_n - tot_s1) / tot_n * 100:.1f}%)')

    if args.json_out:
        with open(args.json_out, 'w') as fh:
            json.dump({
                'config': {'gains': gains, 'g1': g1.to_dict(), 'seed': args.seed,
                           'num_per_bucket': args.num_per_bucket, 'margin': args.margin,
                           'attitude_only': args.attitude_only, 'max_steps': args.max_steps},
                'buckets': [{'lo_deg': lo, 'hi_deg': hi, 'n': int(n[i]),
                             's1': int(s1[i]), 's1_s2': int(s1s2[i]),
                             's1_f2': int(s1[i] - s1s2[i]), 'f1': int(n[i] - s1[i])}
                            for i, (lo, hi) in enumerate(BUCKETS_DEG)],
                'total': {'n': tot_n, 's1': tot_s1, 's1_s2': tot_s2,
                          's1_f2': tot_s1 - tot_s2, 'f1': tot_n - tot_s1},
            }, fh, indent=2)
        print(f'-> {args.json_out}')


if __name__ == '__main__':
    main()
