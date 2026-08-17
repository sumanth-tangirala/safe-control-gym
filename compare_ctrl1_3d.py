#!/usr/bin/env python3
'''PAIRED head-to-head of candidate controller 1s for the quadrotor-3D composition.

Every candidate is rolled out from THE SAME initial states -- sampled once, in
the parent, before any controller is built -- so the difference between two
rows of the output table is the controller and nothing else.  Two independent
runs of `evaluate_geometric_flip3d.py` with the same `--seed` would ALSO draw
the same states today, but only by construction of that script's RNG use; here
the states are drawn once and handed to every candidate explicitly, so the
pairing cannot be broken by a future change to sampling order, worker count, or
the number of candidates.

CANDIDATES are named on the command line as `--controllers`, a comma list of

    geometric              the hand-coded quad_composition.geometric_flip3d law
    sac:<path/to/model.pt> a trained SAC checkpoint (rollout3d.load_ctrl1)
    sac:                   a randomly-initialised SAC policy (pipeline check)

SAMPLING is stratified by INITIAL TILT into `evaluate_geometric_flip3d`'s six
30-degree buckets, `--num_per_bucket` states each, reusing that module's
`bucket_states` verbatim so this script's numbers are directly comparable with
the ones already on record for the geometric controller.  Note the consequence:
the OVERALL rates are over a tilt-uniform distribution, not over uniform SO(3)
(which concentrates near 90 deg) -- the same convention the existing geometric
measurement used.

OUTCOMES come out of `rollout3d.rollout_composite`, which owns the handoff
latch; nothing is recomputed here, so this script cannot disagree with it.

    F1       controller 1 never reached G1
    S1       controller 1 reached G1              (= S1->S2 + S1->F2)
    S1->S2   handed off, controller 2 (LQR) then reached the goal ball
    S1->F2   handed off, controller 2 then did not

G1 is `flip_env3d.G_NOM_3D` (10.03 deg, 4.0 rad/s).  The calibrated-from-exits
alternative is deliberately NOT offered: it fires the handoff while the drone
is still badly tilted and measured WORSE end-to-end (S1->S2 18.5% -> 8.0%).
'''

import argparse
import json
import multiprocessing as mp
import shutil
import tempfile

import numpy as np

from evaluate_geometric_flip3d import BUCKETS_DEG, bucket_states
from quad_composition.flip_env3d import G_NOM_3D, sampling_bounds_from_env
from quad_composition.g1 import G1Region
from quad_composition.geometric_flip3d import GeometricFlipController3D
from quad_composition.rollout3d import MAX_STEPS, load_ctrl1, make_env, make_env_and_ctrl2, rollout_composite

# Worker-local env/controllers, built once per process: a PyBullet env plus an
# LQR gain solve costs far more than the rollouts themselves.
_WORKER = {}


def parse_controller(spec):
    '''`'geometric'` or `'sac[:path]'` -> ('geometric', None) / ('sac', path or None).'''
    if spec == 'geometric':
        return ('geometric', None)
    if spec == 'sac' or spec.startswith('sac:'):
        path = spec[4:] if spec.startswith('sac:') else ''
        return ('sac', path or None)
    raise ValueError(f"unknown controller spec {spec!r}; expected 'geometric' or 'sac:<path>'")


def _init_worker(kind, path, g1_dict, tmp_root):
    '''Build this process's env, controller 2 (LQR) and controller 1 once.

    Each worker gets its own subdirectory of `tmp_root` for the controllers'
    output; `evaluate` owns `tmp_root` and removes the whole tree once the pool
    has been joined.  Never a `TemporaryDirectory` context manager -- that
    would unlink the directory while the controllers still hold it -- and never
    the default temp root, which here is an NFS mount that intermittently
    hangs.
    '''
    tmp = tempfile.mkdtemp(dir=tmp_root)
    env, ctrl2 = make_env_and_ctrl2(tmp)
    _WORKER['tmp'] = tmp
    _WORKER['env'] = env
    _WORKER['ctrl2'] = ctrl2
    _WORKER['g1'] = G1Region.from_dict(g1_dict)
    if kind == 'geometric':
        _WORKER['ctrl1'] = GeometricFlipController3D(env)
    else:
        _WORKER['ctrl1'] = load_ctrl1(path, env, tmp) if path else None
        if _WORKER['ctrl1'] is None:
            raise ValueError('sac candidate needs a checkpoint path in this script')


def _run_one(job):
    '''One composite rollout.  `job` is (job_index, bucket_index, init_state,
    max_steps); the job index is returned so results can be re-paired with the
    exact initial state that produced them regardless of completion order.
    '''
    job_index, bucket_index, init, max_steps = job
    res = rollout_composite(_WORKER['env'], _WORKER['ctrl1'], _WORKER['ctrl2'],
                            _WORKER['g1'], np.asarray(init, dtype=float), max_steps=max_steps)
    return (job_index, bucket_index, bool(res.flip_success), bool(res.ctrl2_success),
            int(res.handoff_index), len(res.trajectory))


def sample_jobs(seed, num_per_bucket, max_steps, margin=1.0):
    '''The shared initial states: `num_per_bucket` per tilt bucket, drawn once.

    The parent's env is closed before the pool is created -- a live PyBullet
    connection must not be inherited across a fork, and the draw must not
    depend on the worker count.
    '''
    rng = np.random.default_rng(seed)
    env = make_env(seed=seed)
    bounds = sampling_bounds_from_env(env)
    jobs = []
    for bucket_index, (lo, hi) in enumerate(BUCKETS_DEG):
        for state in bucket_states(rng, bounds, lo, hi, num_per_bucket, margin=margin):
            jobs.append((len(jobs), bucket_index, state.tolist(), max_steps))
    env.close()
    return jobs


def evaluate(spec, jobs, workers, g1):
    '''Roll every job through one candidate.  Returns a dict of per-bucket and
    overall counts plus the per-job outcome vectors (kept so callers can do a
    paired analysis, which aggregate rates alone cannot support).
    '''
    kind, path = parse_controller(spec)
    tmp_root = tempfile.mkdtemp(dir='/tmp', prefix='quad3d_cmp_')
    try:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=workers, initializer=_init_worker,
                      initargs=(kind, path, g1.to_dict(), tmp_root)) as pool:
            results = pool.map(_run_one, jobs, chunksize=1)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    results.sort()

    n_buckets = len(BUCKETS_DEG)
    n = np.zeros(n_buckets, dtype=int)
    s1 = np.zeros(n_buckets, dtype=int)
    s1s2 = np.zeros(n_buckets, dtype=int)
    handoff_steps = [[] for _ in BUCKETS_DEG]
    flip_flags, ctrl2_flags, traj_lens, handoffs = [], [], [], []
    for _, bucket_index, flip_success, ctrl2_success, handoff_index, traj_len in results:
        n[bucket_index] += 1
        s1[bucket_index] += int(flip_success)
        s1s2[bucket_index] += int(flip_success and ctrl2_success)
        if flip_success:
            handoff_steps[bucket_index].append(handoff_index)
        flip_flags.append(bool(flip_success))
        ctrl2_flags.append(bool(flip_success and ctrl2_success))
        traj_lens.append(int(traj_len))
        handoffs.append(int(handoff_index))

    return {
        'controller': spec,
        'buckets': [{'lo_deg': lo, 'hi_deg': hi, 'n': int(n[i]), 's1': int(s1[i]),
                     's1_s2': int(s1s2[i]), 's1_f2': int(s1[i] - s1s2[i]),
                     'f1': int(n[i] - s1[i]),
                     'median_handoff': (float(np.median(handoff_steps[i]))
                                        if handoff_steps[i] else None)}
                    for i, (lo, hi) in enumerate(BUCKETS_DEG)],
        'total': {'n': int(n.sum()), 's1': int(s1.sum()), 's1_s2': int(s1s2.sum()),
                  's1_f2': int(s1.sum() - s1s2.sum()), 'f1': int(n.sum() - s1.sum())},
        'per_job': {'s1': flip_flags, 's1_s2': ctrl2_flags, 'trajectory_len': traj_lens,
                    'handoff_index': handoffs},
    }


def format_table(report):
    '''The per-bucket table for one candidate, as printed lines.'''
    lines = [f'--- {report["controller"]} ---',
             f'{"tilt bucket":>14} {"N":>4} {"S1 (reach G1)":>15} {"S1->S2":>14} '
             f'{"S1->F2":>14} {"median handoff":>15}']
    for b in report['buckets']:
        n = max(b['n'], 1)
        med = '-' if b['median_handoff'] is None else f'{b["median_handoff"]:.0f}'
        lines.append(f'{b["lo_deg"]:5d}-{b["hi_deg"]:<3d}deg {b["n"]:>4d} '
                     f'{b["s1"]:>5d} ({b["s1"] / n * 100:5.1f}%) '
                     f'{b["s1_s2"]:>5d} ({b["s1_s2"] / n * 100:5.1f}%) '
                     f'{b["s1_f2"]:>5d} ({b["s1_f2"] / n * 100:5.1f}%) '
                     f'{med:>15}')
    t = report['total']
    n = max(t['n'], 1)
    lines.append(f'{"ALL":>14} {t["n"]:>4d} '
                 f'{t["s1"]:>5d} ({t["s1"] / n * 100:5.1f}%) '
                 f'{t["s1_s2"]:>5d} ({t["s1_s2"] / n * 100:5.1f}%) '
                 f'{t["s1_f2"]:>5d} ({t["s1_f2"] / n * 100:5.1f}%)')
    lines.append(f'F1 (never reached G1): {t["f1"]}/{t["n"]} ({t["f1"] / n * 100:.1f}%)')
    return lines


def rank(reports):
    '''Candidates best-first on S1->S2 rate, S1 rate breaking ties -- the
    decision rule this comparison exists to settle (end-to-end success is what
    matters; reaching G1 is only the means).
    '''
    return sorted(reports, key=lambda r: (r['total']['s1_s2'], r['total']['s1']), reverse=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--controllers', required=True,
                        help="comma list of 'geometric' and/or 'sac:<checkpoint>'")
    parser.add_argument('--num_per_bucket', type=int, default=30)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--json_out', default=None)
    args = parser.parse_args(argv)

    specs = [s.strip() for s in args.controllers.split(',') if s.strip()]
    for spec in specs:
        parse_controller(spec)

    g1 = G_NOM_3D
    jobs = sample_jobs(args.seed, args.num_per_bucket, args.max_steps, margin=args.margin)
    print(f'{len(jobs)} shared initial states (seed={args.seed}, '
          f'{args.num_per_bucket}/bucket)  G1={g1.to_dict()}')

    reports = []
    for spec in specs:
        report = evaluate(spec, jobs, args.workers, g1)
        reports.append(report)
        print('\n'.join(format_table(report)))
        print()

    ranked = rank(reports)
    print('RANKING (S1->S2 rate, S1 rate breaks ties):')
    for i, r in enumerate(ranked):
        t = r['total']
        print(f'  {i + 1}. {r["controller"]:<60} '
              f'S1->S2 {t["s1_s2"] / max(t["n"], 1) * 100:5.1f}%  '
              f'S1 {t["s1"] / max(t["n"], 1) * 100:5.1f}%')

    if args.json_out:
        with open(args.json_out, 'w') as fh:
            json.dump({'config': {'seed': args.seed, 'num_per_bucket': args.num_per_bucket,
                                  'max_steps': args.max_steps, 'margin': args.margin,
                                  'g1': g1.to_dict(), 'controllers': specs},
                       'init_states': [j[2] for j in jobs],
                       'reports': reports,
                       'winner': ranked[0]['controller'] if ranked else None}, fh, indent=2)
        print(f'-> {args.json_out}')
    return reports


if __name__ == '__main__':
    main()
