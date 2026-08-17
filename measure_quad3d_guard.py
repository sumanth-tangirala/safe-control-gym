#!/usr/bin/env python3
'''Four-arm measurement of the supervisory guard, on a FRESH set of paired
initial states never seen by `fit_quad3d_guard.py` (which only touched
`quadrotor3D_lqr_regenerated/eval_states.txt`).

SAMPLING: `flip_env3d.sample_uniform_state` over `sampling_bounds_from_env`
-- the SAME mechanism `analyze_quad3d_composition.py::sample_jobs` already
uses to draw fresh evaluation states for this composition experiment (that
script produced the primary non-subsumption result). Positions, velocities
and rates are uniform within `rollout3d.TERMINATION` (the closed state
space); attitude is uniform on SO(3) via Shoemake's method -- NOT
independently-uniform Euler angles, which CLAUDE.md and flip_env3d.py's own
docstring flag as non-uniform on SO(3) and biased toward the gimbal poles.

CAVEAT, checked and worth stating plainly: this is NOT the same empirical
distribution as `quadrotor3D_lqr_regenerated/eval_states.txt`. That file's
40000 rows trace back to a 40000-row `eval_states.txt` inside the archived,
800000-trajectory `quadrotor3D_lqr` dataset (8.19% baseline success over the
full 800k) -- some curated subsample of it, not literally "fresh uniform
draws with p/q/r in +-24 rad/s" as its own generation_parameters claim: a
direct reproduction of that literal recipe (independently-uniform Euler
angles, p/q/r each uniform in +-24 rad/s -- tried and discarded here) gives
mean |omega| ~23 rad/s, matching the +-24 uniform-cube theoretical value
exactly, whereas the archived eval_states.txt's own p/q/r columns have mean
|omega| ~14.2 rad/s with a visibly narrower spread. Whatever selected that
40000-row file, it was not a plain fresh uniform draw, and that selection
mechanism could not be recovered from the code in this repo. The SO(3)-
uniform sample used here is therefore a HARDER, more adversarial distribution
than the guard was fit on (higher mean tilt and |omega|) -- an honest, if
imperfect, choice: it is this codebase's own established "fresh, paired,
closed-state-space" sampling convention, not an invented one, but it means
the guard is being evaluated somewhat out-of-distribution relative to its
fit. See the guard3d report for how this affects interpretation.

ONLY TWO ARMS ARE ACTUALLY ROLLED OUT PER STATE: baseline (ctrl1=None) and
the unguarded composite (ctrl1=real). The other two arms this script reports
-- guarded composition and the oracle guard -- are DERIVED, not re-simulated:

    guarded_success(x) = baseline_success(x) if guard(x)      else composite_success(x)
    oracle_success(x)  = baseline_success(x) if baseline(x)   else composite_success(x)
                        = baseline_success(x) OR composite_success(x)

This is exact, not an approximation: `rollout3d.rollout_composite`'s guard
hook (guard=True) collapses to precisely the ctrl1=None baseline path -- same
env, same initial state, same controller 2, same deterministic PyBullet
physics -- so simulating it separately would reproduce the already-simulated
baseline outcome bit for bit. `--verify_n` real end-to-end rollouts through
`rollout_composite(..., guard=guard3d.lqr_success_guard)` confirm this
equivalence holds on the real (non-fake) simulation stack, not just in the
unit tests' scripted-env fakes.

Oracle guard = TRUE baseline label, not a prediction: whenever baseline
succeeds it is chosen (trivially correct), so oracle_success is an upper
bound on ANY guard restricted to the same two choices (run LQR alone / run
the existing composition) -- it can never do better than knowing the truth.
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

from quad_composition.flip_env3d import sample_uniform_state, sampling_bounds_from_env
from quad_composition.guard3d import FITTED_GUARD
from quad_composition.rollout3d import MAX_STEPS, load_ctrl1, make_env, make_env_and_ctrl2, rollout_composite

CTRL1_MODEL = 'models/quad3d_ctrl1_selected.pt'
Z_95 = 1.959963984540054


def sample_paired_states(n, seed):
    '''n dataset-order 13-dim states, uniform over the closed state space with
    full SO(3) attitude coverage -- see module docstring for the sampling
    method and its caveat relative to the archived eval_states.txt.
    '''
    rng = np.random.default_rng(seed)
    env = make_env(seed=seed)
    bounds = sampling_bounds_from_env(env)
    states = [sample_uniform_state(rng, bounds, max_init_tilt=None).tolist() for _ in range(n)]
    env.close()
    return states


# ---------------------------------------------------------------------------
# Parallel rollout: one job per state, TWO rollout_composite calls per job
# (baseline, unguarded composite) against fresh envs so neither run can leak
# state into the other. A `verify` subset ALSO runs a real guarded rollout.
# ---------------------------------------------------------------------------

_WORKER = {}


def _init_worker(ctrl1_path, tmp_root, max_steps):
    from quad_composition.flip_env3d import G_NOM_3D
    tmp = tempfile.mkdtemp(dir=tmp_root)
    env, ctrl2 = make_env_and_ctrl2(tmp)
    ctrl1 = load_ctrl1(ctrl1_path, env, tmp)
    _WORKER.update(env=env, ctrl1=ctrl1, ctrl2=ctrl2, g1=G_NOM_3D, max_steps=max_steps)


def _run_one(job):
    idx, state, verify = job
    w = _WORKER
    base = rollout_composite(w['env'], None, w['ctrl2'], w['g1'], state, max_steps=w['max_steps'])
    comp = rollout_composite(w['env'], w['ctrl1'], w['ctrl2'], w['g1'], state,
                             max_steps=w['max_steps'])
    out = {'idx': idx, 'baseline_success': bool(base.ctrl2_success),
          'composite_success': bool(comp.ctrl2_success),
          'composite_flip_success': bool(comp.flip_success)}
    if verify:
        from quad_composition.guard3d import lqr_success_guard
        guarded = rollout_composite(w['env'], w['ctrl1'], w['ctrl2'], w['g1'], state,
                                    max_steps=w['max_steps'], guard=lqr_success_guard)
        out['verify_guarded_success'] = bool(guarded.ctrl2_success)
        out['verify_guard_pred'] = bool(lqr_success_guard(state))
    return out


def run_rollouts(states, ctrl1_path, workers, max_steps, verify_n, verify_seed):
    rng = np.random.default_rng(verify_seed)
    verify_idx = set(rng.choice(len(states), size=min(verify_n, len(states)),
                                replace=False).tolist()) if verify_n else set()
    jobs = [(i, s, i in verify_idx) for i, s in enumerate(states)]

    tmp_root = tempfile.mkdtemp(dir='/tmp', prefix='quad3d_guard_measure_')
    try:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=workers, initializer=_init_worker,
                      initargs=(ctrl1_path, tmp_root, max_steps)) as pool:
            results = pool.map(_run_one, jobs, chunksize=1)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    results.sort(key=lambda r: r['idx'])
    return results


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def wilson_ci(successes, n, z=Z_95):
    if n == 0:
        return None
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n))
    return p_hat, max(0.0, center - half), min(1.0, center + half)


def paired_counts(a, b):
    '''(a wins b loses, b wins a loses, both, neither) over two boolean arrays.'''
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    return {'a_wins_b_loses': int((a & ~b).sum()), 'b_wins_a_loses': int((~a & b).sum()),
           'both': int((a & b).sum()), 'neither': int((~a & ~b).sum())}


def build_report(states, results, guard_preds, config):
    n = len(results)
    baseline = np.array([r['baseline_success'] for r in results], dtype=bool)
    composite = np.array([r['composite_success'] for r in results], dtype=bool)
    guard_pred = np.array(guard_preds, dtype=bool)

    guarded = np.where(guard_pred, baseline, composite)
    oracle = baseline | composite   # equivalently: baseline if baseline else composite

    arms = {'baseline': baseline, 'unguarded_composite': composite,
           'guarded_composite': guarded, 'oracle_guard': oracle}
    rates = {}
    for name, arr in arms.items():
        succ = int(arr.sum())
        p_hat, lo, hi = wilson_ci(succ, n)
        rates[name] = {'n': n, 'successes': succ, 'rate': p_hat, 'ci95': [lo, hi]}

    # The 1051-style set: baseline wins, unguarded composite loses.
    recoverable = baseline & ~composite
    recovered_by_guard = int((recoverable & guard_pred).sum())
    n_recoverable = int(recoverable.sum())

    # The cost set: unguarded composite wins, baseline loses. Guard costs a
    # win here iff it (wrongly) predicts baseline success.
    costly = ~baseline & composite
    regressed_by_guard = int((costly & guard_pred).sum())
    n_costly = int(costly.sum())

    report = {
        'config': config,
        'n_states': n,
        'rates': rates,
        'paired_vs_baseline': {
            'guarded_composite': paired_counts(guarded, baseline),
            'unguarded_composite': paired_counts(composite, baseline),
            'oracle_guard': paired_counts(oracle, baseline),
        },
        'paired_vs_unguarded_composite': {
            'guarded_composite': paired_counts(guarded, composite),
            'oracle_guard': paired_counts(oracle, composite),
        },
        'recoverable_1051_style': {
            'n': n_recoverable, 'recovered_by_guard': recovered_by_guard,
            'recovered_by_oracle': n_recoverable,   # oracle recovers ALL of these, by construction
            'recovery_rate_guard': (recovered_by_guard / n_recoverable) if n_recoverable else None,
        },
        'costly_regressions': {
            'n': n_costly, 'regressed_by_guard': regressed_by_guard,
            'regressed_by_oracle': 0,   # oracle never regresses these, by construction
            'regression_rate_guard': (regressed_by_guard / n_costly) if n_costly else None,
        },
        'oracle_gap': {
            'oracle_rate_minus_guarded_rate': rates['oracle_guard']['rate'] - rates['guarded_composite']['rate'],
            'fraction_of_achievable_gain_captured': (
                (rates['guarded_composite']['rate'] - rates['unguarded_composite']['rate'])
                / (rates['oracle_guard']['rate'] - rates['unguarded_composite']['rate'])
                if rates['oracle_guard']['rate'] > rates['unguarded_composite']['rate'] else None),
        },
    }

    verify_rows = [r for r in results if 'verify_guarded_success' in r]
    if verify_rows:
        mismatches = [r['idx'] for r in verify_rows
                     if r['verify_guarded_success'] != (
                         r['baseline_success'] if r['verify_guard_pred'] else r['composite_success'])]
        report['verification'] = {
            'n_verified': len(verify_rows), 'n_mismatches': len(mismatches),
            'mismatched_indices': mismatches,
        }
    return report


def format_summary(report):
    lines = []
    lines.append(f'{report["n_states"]} fresh initial states (seed={report["config"]["seed"]})')
    lines.append('')
    lines.append(f'{"arm":<22} {"success rate":>14} {"95% CI":>22}')
    for name, r in report['rates'].items():
        ci = f'[{r["ci95"][0]:.4f}, {r["ci95"][1]:.4f}]'
        lines.append(f'{name:<22} {r["rate"] * 100:>13.2f}% {ci:>22}')
    lines.append('')
    lines.append('paired vs. baseline:')
    for name, c in report['paired_vs_baseline'].items():
        lines.append(f'  {name:<22} wins={c["a_wins_b_loses"]:>5} losses={c["b_wins_a_loses"]:>5} '
                     f'both={c["both"]:>5} neither={c["neither"]:>5}')
    lines.append('')
    lines.append('paired vs. unguarded composite:')
    for name, c in report['paired_vs_unguarded_composite'].items():
        lines.append(f'  {name:<22} wins={c["a_wins_b_loses"]:>5} losses={c["b_wins_a_loses"]:>5} '
                     f'both={c["both"]:>5} neither={c["neither"]:>5}')
    lines.append('')
    rec = report['recoverable_1051_style']
    lines.append(f'1051-style recoverable states (baseline wins, unguarded composite loses): '
                 f'n={rec["n"]}, guard recovers {rec["recovered_by_guard"]} '
                 f'({(rec["recovery_rate_guard"] or 0) * 100:.1f}%), oracle recovers all {rec["n"]}')
    cost = report['costly_regressions']
    lines.append(f'costly states (unguarded composite wins, baseline loses): n={cost["n"]}, '
                 f'guard wrongly regresses {cost["regressed_by_guard"]} '
                 f'({(cost["regression_rate_guard"] or 0) * 100:.1f}%), oracle regresses 0')
    lines.append('')
    gap = report['oracle_gap']
    lines.append(f'oracle gap (oracle rate - guarded rate): '
                 f'{gap["oracle_rate_minus_guarded_rate"] * 100:+.2f} points')
    frac = gap['fraction_of_achievable_gain_captured']
    lines.append(f'fraction of achievable gain captured: '
                 f'{"n/a (oracle <= unguarded composite)" if frac is None else f"{frac * 100:.1f}%"}')
    if 'verification' in report:
        v = report['verification']
        lines.append('')
        lines.append(f'end-to-end verification: {v["n_verified"]} real guarded rollouts vs. the '
                     f'derived formula -- {v["n_mismatches"]} mismatches')
    return '\n'.join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n_states', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=314159)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    parser.add_argument('--ctrl1_path', default=CTRL1_MODEL)
    parser.add_argument('--verify_n', type=int, default=150,
                        help='real end-to-end rollouts through the guard= hook, checked against '
                             'the derived arm formula')
    parser.add_argument('--verify_seed', type=int, default=271828)
    parser.add_argument('--output', default='results/quad3d_guard_measurement.json')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    t0 = time.time()

    states = sample_paired_states(args.n_states, args.seed)
    print(f'{len(states)} fresh initial states sampled (seed={args.seed})')

    guard_preds = [FITTED_GUARD.predict(s) for s in states]
    print(f'guard predicts LQR-alone success for {sum(guard_preds)}/{len(states)} states '
         f'({100 * sum(guard_preds) / len(states):.2f}%)')

    results = run_rollouts(states, args.ctrl1_path, args.workers, args.max_steps,
                           args.verify_n, args.verify_seed)
    elapsed = time.time() - t0
    print(f'{len(results)} states rolled out (2 rollouts each + {args.verify_n} verification '
         f'rollouts) in {elapsed:.1f}s')

    config = {'n_states': args.n_states, 'seed': args.seed, 'workers': args.workers,
             'max_steps': args.max_steps, 'ctrl1_path': args.ctrl1_path,
             'verify_n': args.verify_n, 'verify_seed': args.verify_seed, 'elapsed_sec': elapsed}
    report = build_report(states, results, guard_preds, config)

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
