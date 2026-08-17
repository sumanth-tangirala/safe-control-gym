#!/usr/bin/env python3
'''Measure the primary result in 3D: is G1 subsumed by RoA2?

non_subsumption = 1 - P(ctrl2_success = 1 | flip_success = 1), measured over
REAL handoffs. The claim under test is that it is bounded away from both 0
(G1 subsumed, composition trivially sound) and 1 (G1 disjoint, composition
useless). This is the 3D port of analyze_quad2d_composition.py.

WHY A DIFFERENT SHAPE THAN THE 2D SCRIPT: the 2D analyzer reads two
pre-generated dataset directories (composite, baseline) written by
generate_quadrotor_2d_composition.py. No 3D analogue of that generation
pipeline exists yet, and none is needed for this measurement: this script
samples its own initial states and rolls them straight through
`quad_composition.rollout3d.rollout_composite`, in-process, in parallel. The
non_subsumption() function itself is unchanged from 2D -- it is a pure
function of two boolean arrays and does not care where they came from.

WILSON, NOT BOOTSTRAP -- carried over unchanged from the 2D script's own
finding: a bootstrap over the real handoff count (~245,000 in 2D) allocates
an (n_boot, n_handoffs) index array that is ~20 GB and OOMs. non_subsumption
is a plain Bernoulli proportion over the handoff rows; Wilson's closed form is
exact enough and O(1) in memory whatever the sample size. It also stays
inside [0, 1] at the two regimes this experiment is designed to distinguish,
p near 0 and p near 1, where the textbook Wald interval misbehaves.

REAL HANDOFFS VS ALL HANDOFFS (2D's Finding I6, unresolved there, resolved
here): a row with handoff_index == 0 means the INITIAL state was already
inside G1, so controller 1 never acted. That row is not a real handoff --
it is a statement about the initial-state distribution, not about the
composition. Both figures are reported below, never one silently substituted
for the other.

CHECKPOINT ATTRIBUTION: models/quad3d_ctrl1_selected.pt may be rewritten
concurrently by a re-selection pass running elsewhere. To make the reported
number attributable to one specific checkpoint even if the source file
changes mid-run, main() takes a single stat+copy snapshot of the source file
into a frozen /tmp path BEFORE building any worker, and every worker loads
that frozen copy, never the live path. The source file's mtime/size at the
moment of the snapshot are recorded in the output JSON.
'''

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import tempfile
import time

import numpy as np

from quad_composition.flip_env3d import G_NOM_3D, sample_uniform_state, sampling_bounds_from_env
from quad_composition.g1 import G1Region
from quad_composition.rollout3d import (GOAL_STATE, MAX_STEPS, QUAT_SLICE, load_ctrl1, make_env,
                                        make_env_and_ctrl2, omega_norm, rollout_composite,
                                        tilt_from_quat_wxyz)

# Two-sided 95% normal quantile, for the Wilson interval below.
Z_95 = 1.959963984540054

# 30-degree tilt buckets spanning the full [0, 180] tilt range, matching the
# convention already used by evaluate_geometric_flip3d.py / compare_ctrl1_3d.py.
BUCKETS_DEG = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]

# Mechanism-check variables read off the handoff state. tilt/omega are the two
# quantities G1 DOES constrain (every handoff state satisfies tilt < tilt_c,
# |omega| < w_c by construction of the handoff condition); horiz_dist,
# alt_err and speed are quantities G1 does NOT constrain at all.
MECHANISM_VARS = ['handoff_tilt_deg', 'handoff_omega', 'handoff_horiz_dist',
                  'handoff_alt_err', 'handoff_speed']


def non_subsumption(flip_success, ctrl2_success, z=Z_95):
    '''(point estimate, lo, hi) of 1 - P(ctrl2 succeeds | handoff fired).

    Unchanged from analyze_quad2d_composition.py: a Wilson score interval on
    the Bernoulli proportion of handoff rows where ctrl2 succeeded. See this
    module's docstring for why Wilson over a bootstrap.
    '''
    flip = np.asarray(flip_success).astype(bool)
    ctrl2 = np.asarray(ctrl2_success).astype(bool)
    handed = ctrl2[flip]
    n = int(handed.size)
    if n == 0:
        raise ValueError('no handoffs: cannot measure non-subsumption')

    point = 1.0 - float(handed.mean())
    denom = 1.0 + z * z / n
    center = (point + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(point * (1.0 - point) / n + z * z / (4 * n * n))
    return point, max(0.0, center - half), min(1.0, center + half)


def safe_non_subsumption(flip_mask, ctrl2_success, z=Z_95):
    '''non_subsumption, but returns a dict with n_handoffs=0 and point=None
    instead of raising, for callers (bucket tables) that must keep going over
    empty subsets rather than aborting the whole report.
    '''
    n = int(np.asarray(flip_mask).astype(bool).sum())
    if n == 0:
        return {'point': None, 'ci95': None, 'n_handoffs': 0}
    point, lo, hi = non_subsumption(flip_mask, ctrl2_success, z=z)
    return {'point': point, 'ci95': [lo, hi], 'n_handoffs': n}


def assign_tilt_bucket(tilt_deg, buckets=BUCKETS_DEG):
    '''Bucket index (0..len(buckets)-1) for each tilt in degrees, half-open
    [lo, hi) except the last bucket, which is closed at 180 so a tilt of
    exactly 180 degrees still lands somewhere. -1 if out of range (should not
    happen for a valid tilt in [0, 180]).
    '''
    tilt_deg = np.asarray(tilt_deg, dtype=float)
    idx = np.full(tilt_deg.shape, -1, dtype=int)
    last = len(buckets) - 1
    for i, (lo, hi) in enumerate(buckets):
        mask = (tilt_deg >= lo) & (tilt_deg <= hi if i == last else tilt_deg < hi)
        idx[mask] = i
    return idx


def quintile_success(values, success, n_bins=5):
    '''Split rows into n_bins equal-COUNT groups by rank of `values` (not by
    value threshold -- with a skewed distribution, equal-width bins can be
    nearly empty at one end), and report the success rate in each.

    Returns a list of n_bins dicts (lowest values first), each with n,
    value_lo, value_hi, success_rate. Ties in `values` do not matter: the
    split is by sorted position.
    '''
    values = np.asarray(values, dtype=float)
    success = np.asarray(success).astype(bool)
    if values.shape != success.shape:
        raise ValueError('values and success must be the same shape')
    order = np.argsort(values, kind='stable')
    groups = np.array_split(order, n_bins)
    out = []
    for g in groups:
        if len(g) == 0:
            out.append({'n': 0, 'value_lo': None, 'value_hi': None, 'success_rate': None})
            continue
        v = values[g]
        out.append({'n': int(len(g)), 'value_lo': float(v.min()), 'value_hi': float(v.max()),
                    'success_rate': float(success[g].mean())})
    return out


def quintile_spread(quintiles):
    '''max(success_rate) - min(success_rate) over non-empty quintiles -- the
    separation summary used to rank mechanism variables against each other.
    '''
    rates = [q['success_rate'] for q in quintiles if q['success_rate'] is not None]
    return (max(rates) - min(rates)) if rates else None


# ---------------------------------------------------------------------------
# Checkpoint freezing
# ---------------------------------------------------------------------------

def snapshot_checkpoint(src_path, tmp_root, retries=3):
    '''Copy `src_path` into `tmp_root` and return
    (frozen_path, {'mtime': ..., 'size': ..., 'path': src_path}).

    A re-selection pass may be rewriting src_path concurrently (operational
    constraint for this task). Every worker loads the FROZEN copy, never the
    live path, so all workers see identical bytes regardless of what happens
    to src_path after this function returns. The stat is taken immediately
    before and after the copy; if they disagree the copy is retried (the copy
    may have read a half-written file), up to `retries` times, and the last
    disagreement is recorded as a warning rather than silently trusted.
    '''
    frozen_path = os.path.join(tmp_root, 'ctrl1_frozen.pt')
    warning = None
    for attempt in range(retries):
        before = os.stat(src_path)
        shutil.copy2(src_path, frozen_path)
        after = os.stat(src_path)
        if before.st_mtime == after.st_mtime and before.st_size == after.st_size:
            return frozen_path, {'path': os.path.abspath(src_path),
                                 'mtime': before.st_mtime, 'size': before.st_size,
                                 'warning': None}
        warning = (f'source checkpoint changed during snapshot attempt {attempt + 1} '
                  f'(mtime {before.st_mtime} -> {after.st_mtime}, size {before.st_size} '
                  f'-> {after.st_size}); retrying')
    # Out of retries: report the last observed post-copy stat, flagged.
    after = os.stat(src_path)
    return frozen_path, {'path': os.path.abspath(src_path), 'mtime': after.st_mtime,
                         'size': after.st_size, 'warning': warning}


# ---------------------------------------------------------------------------
# Sampling and parallel rollout
# ---------------------------------------------------------------------------

def sample_jobs(seed, num_states):
    '''`num_states` initial states, uniform over the closed 3D state space
    with FULL SO(3) attitude coverage (`sample_uniform_state` with
    max_init_tilt=None -- Shoemake's uniform-rotation method, not a tilt-
    banded sample). Drawn once in the parent so the result depends only on
    --seed, not on --workers or completion order.
    '''
    rng = np.random.default_rng(seed)
    env = make_env(seed=seed)
    bounds = sampling_bounds_from_env(env)
    jobs = [(i, sample_uniform_state(rng, bounds, max_init_tilt=None).tolist())
           for i in range(num_states)]
    env.close()
    return jobs


_WORKER = {}


def _init_worker(frozen_ckpt_path, g1_dict, tmp_root, max_steps):
    tmp = tempfile.mkdtemp(dir=tmp_root)
    env, ctrl2 = make_env_and_ctrl2(tmp)
    ctrl1 = load_ctrl1(frozen_ckpt_path, env, tmp)
    _WORKER['env'] = env
    _WORKER['ctrl1'] = ctrl1
    _WORKER['ctrl2'] = ctrl2
    _WORKER['g1'] = G1Region.from_dict(g1_dict)
    _WORKER['max_steps'] = max_steps


def _run_one(job):
    '''One composite rollout. Returns a flat dict of everything the report
    needs: outcome labels, the initial tilt (for the by-bucket table), and --
    only when a handoff fired -- the handoff-state mechanism variables.
    '''
    job_index, init_state = job
    init = np.asarray(init_state, dtype=float)
    res = rollout_composite(_WORKER['env'], _WORKER['ctrl1'], _WORKER['ctrl2'], _WORKER['g1'],
                            init, max_steps=_WORKER['max_steps'])
    out = {
        'job_index': job_index,
        'init_tilt_deg': math.degrees(tilt_from_quat_wxyz(init[QUAT_SLICE])),
        'flip_success': bool(res.flip_success),
        'handoff_index': int(res.handoff_index),
        'ctrl2_success': bool(res.ctrl2_success),
    }
    if res.flip_success:
        hs = np.asarray(res.trajectory[res.handoff_index], dtype=float)
        out['handoff_tilt_deg'] = math.degrees(tilt_from_quat_wxyz(hs[QUAT_SLICE]))
        out['handoff_omega'] = omega_norm(hs)
        out['handoff_horiz_dist'] = float(np.hypot(hs[0] - GOAL_STATE[0], hs[1] - GOAL_STATE[1]))
        out['handoff_alt_err'] = float(abs(hs[2] - GOAL_STATE[2]))
        out['handoff_speed'] = float(np.linalg.norm(hs[7:10]))
    return out


def run_rollouts(jobs, frozen_ckpt_path, g1, workers, max_steps):
    tmp_root = tempfile.mkdtemp(dir='/tmp', prefix='quad3d_analyze_')
    try:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=workers, initializer=_init_worker,
                      initargs=(frozen_ckpt_path, g1.to_dict(), tmp_root, max_steps)) as pool:
            results = pool.map(_run_one, jobs, chunksize=1)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    results.sort(key=lambda r: r['job_index'])
    return results


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(results, g1, ckpt_info, config):
    n = len(results)
    flip = np.array([r['flip_success'] for r in results], dtype=bool)
    ctrl2 = np.array([r['ctrl2_success'] for r in results], dtype=bool)
    handoff_index = np.array([r['handoff_index'] for r in results], dtype=int)
    init_tilt_deg = np.array([r['init_tilt_deg'] for r in results], dtype=float)
    real = handoff_index > 0
    at_zero = handoff_index == 0

    point, lo, hi = non_subsumption(flip, ctrl2)
    all_result = {'point': point, 'ci95': [lo, hi], 'n_handoffs': int(flip.sum())}
    real_result = safe_non_subsumption(real, ctrl2)

    # --- by initial-tilt bucket, both scopes ---------------------------------
    bucket_idx = assign_tilt_bucket(init_tilt_deg)
    bucket_table = []
    for i, (blo, bhi) in enumerate(BUCKETS_DEG):
        in_bucket = bucket_idx == i
        bucket_table.append({
            'lo_deg': blo, 'hi_deg': bhi,
            'n_states': int(in_bucket.sum()),
            'all_handoffs': safe_non_subsumption(flip & in_bucket, ctrl2),
            'real_handoffs': safe_non_subsumption(real & in_bucket, ctrl2),
        })

    # --- mechanism check, over REAL handoffs only ---------------------------
    # Real handoffs (not index-0) isolate cases where controller 1 actually
    # acted; index-0 rows are the raw initial state and would mix "what
    # controller 1 delivers" with "what got sampled" for the tilt/omega
    # variables specifically, muddying the comparison the mechanism check is
    # for.
    real_rows = [r for r in results if r['handoff_index'] > 0]
    mechanism = {}
    for var in MECHANISM_VARS:
        values = np.array([r[var] for r in real_rows], dtype=float)
        succ = np.array([r['ctrl2_success'] for r in real_rows], dtype=bool)
        quintiles = quintile_success(values, succ) if len(values) else []
        mechanism[var] = {'quintiles': quintiles, 'spread': quintile_spread(quintiles)}

    handoff_rate = float(flip.mean()) if n else None
    return {
        'config': config,
        'controller_1_checkpoint': ckpt_info,
        'g1': g1.to_dict(),
        'n_initial_states': n,
        'handoff_rate': handoff_rate,
        'n_handoffs_at_index_zero': int(at_zero.sum()),
        'non_subsumption': dict(all_result, scope='all handoffs, including handoff_index == 0'),
        'non_subsumption_real_handoffs': dict(
            real_result, scope='handoff_index > 0 only (controller 1 actually acted)'),
        'non_subsumption_by_tilt_bucket': bucket_table,
        'mechanism_check': dict(
            mechanism,
            note='quintiles computed over REAL handoffs only (handoff_index > 0); '
                 'handoff_tilt_deg/handoff_omega are the variables G1 constrains, '
                 'handoff_horiz_dist/handoff_alt_err/handoff_speed are not',
            n_real_handoffs=len(real_rows)),
        'ci_method': 'wilson_score_95',
    }


def format_summary(report):
    lines = []
    ns = report['non_subsumption']
    nsr = report['non_subsumption_real_handoffs']
    lines.append(f'controller 1 checkpoint: {report["controller_1_checkpoint"]["path"]}')
    lines.append(f'  mtime={report["controller_1_checkpoint"]["mtime"]}  '
                 f'size={report["controller_1_checkpoint"]["size"]} bytes')
    if report['controller_1_checkpoint'].get('warning'):
        lines.append(f'  WARNING: {report["controller_1_checkpoint"]["warning"]}')
    lines.append(f'G1 = {report["g1"]}')
    lines.append(f'{report["n_initial_states"]} initial states, '
                 f'handoff rate {report["handoff_rate"] * 100:.1f}%, '
                 f'{report["n_handoffs_at_index_zero"]} at handoff_index==0')
    lines.append('')
    lines.append(f'non-subsumption (all handoffs)  : {ns["point"]:.4f}  '
                 f'95% CI [{ns["ci95"][0]:.4f}, {ns["ci95"][1]:.4f}]  '
                 f'over {ns["n_handoffs"]} handoffs')
    if nsr['point'] is None:
        lines.append('non-subsumption (real handoffs) : n/a -- no handoff had handoff_index > 0')
    else:
        lines.append(f'non-subsumption (real handoffs) : {nsr["point"]:.4f}  '
                     f'95% CI [{nsr["ci95"][0]:.4f}, {nsr["ci95"][1]:.4f}]  '
                     f'over {nsr["n_handoffs"]} handoffs')
    lines.append('')
    lines.append('by initial tilt bucket (real handoffs):')
    lines.append(f'{"bucket":>12} {"n states":>9} {"n handoffs":>11} '
                 f'{"non_subsumption":>16} {"95% CI":>20}')
    for b in report['non_subsumption_by_tilt_bucket']:
        r = b['real_handoffs']
        label = f'{b["lo_deg"]}-{b["hi_deg"]}deg'
        if r['point'] is None:
            lines.append(f'{label:>12} {b["n_states"]:>9d} {r["n_handoffs"]:>11d} '
                         f'{"n/a":>16} {"":>20}')
        else:
            ci = f'[{r["ci95"][0]:.3f}, {r["ci95"][1]:.3f}]'
            lines.append(f'{label:>12} {b["n_states"]:>9d} {r["n_handoffs"]:>11d} '
                         f'{r["point"]:>16.4f} {ci:>20}')
    lines.append('')
    lines.append(f'mechanism check (n={report["mechanism_check"]["n_real_handoffs"]} real '
                 f'handoffs), success-rate spread across quintiles:')
    ranked = sorted(
        ((v, d['spread']) for v, d in report['mechanism_check'].items()
         if v in MECHANISM_VARS and d['spread'] is not None),
        key=lambda t: t[1], reverse=True)
    for var, spread in ranked:
        lines.append(f'  {var:<20} spread = {spread:.3f}')
    lines.append('')
    headline = ns['point'] if nsr['point'] is None else nsr['point']
    if headline < 0.02:
        lines.append('WARNING: G1 is effectively subsumed by RoA2 -- the primary claim fails.')
    elif headline > 0.98:
        lines.append('WARNING: G1 barely intersects RoA2 -- handoffs almost never succeed.')
    else:
        lines.append(f'Non-subsumption ({headline:.3f}) is bounded away from both 0 and 1: '
                     'G1 is neither subsumed by nor disjoint from RoA2.')
    return '\n'.join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num_states', type=int, default=2000,
                        help='initial states sampled uniformly over the closed state space '
                             'with full SO(3) attitude coverage (must be >= 1500)')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    parser.add_argument('--ctrl1_path', default='models/quad3d_ctrl1_selected.pt')
    parser.add_argument('--output', default='results/quad3d_composition.json')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.num_states < 1500:
        raise ValueError(f'--num_states {args.num_states} < 1500: the measurement requires at '
                         'least 1500 initial states')

    g1 = G_NOM_3D
    t0 = time.time()
    jobs = sample_jobs(args.seed, args.num_states)
    print(f'{len(jobs)} initial states sampled (seed={args.seed}), full SO(3) attitude '
          f'coverage. G1={g1.to_dict()}')

    tmp_root = tempfile.mkdtemp(dir='/tmp', prefix='quad3d_analyze_ckpt_')
    try:
        frozen_ckpt_path, ckpt_info = snapshot_checkpoint(args.ctrl1_path, tmp_root)
        print(f'controller 1 checkpoint frozen: {ckpt_info["path"]} '
              f'mtime={ckpt_info["mtime"]} size={ckpt_info["size"]}'
              + (f'  WARNING: {ckpt_info["warning"]}' if ckpt_info['warning'] else ''))

        results = run_rollouts(jobs, frozen_ckpt_path, g1, args.workers, args.max_steps)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    elapsed = time.time() - t0
    print(f'{len(results)} rollouts completed in {elapsed:.1f}s')

    config = {'num_states': args.num_states, 'seed': args.seed, 'workers': args.workers,
             'max_steps': args.max_steps, 'ctrl1_path': args.ctrl1_path,
             'ctrl2': 'lqr (quad_composition.rollout3d.make_env_and_ctrl2)',
             'elapsed_sec': elapsed}
    report = build_report(results, g1, ckpt_info, config)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump(report, fh, indent=2)

    print()
    print(format_summary(report))
    print(f'\n-> {args.output}')
    return report


if __name__ == '__main__':
    main()
