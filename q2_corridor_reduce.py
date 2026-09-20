'''Aggregate the quad2d altitude-corridor shards into one dataset directory
per config, in a STAGING directory only -- publication to the shared root is
a separate, human-gated step.

Mirrors q2_reduce.py, which is the precedent for the schema: same per-level
directory contents, same key names, so the corridor family is readable by
whatever already reads the shipped quad2d set. It differs from q2_reduce.py
in three ways, all forced by how q2_corridor_collect.py's shards differ from
the planar-force collector's:

  Shard files live flat in one directory and their names carry the config,
  not a level string alone: `{split}_L<level>_A<ambient|none>_k<lo>-<hi>_
  s<shard>.npz` (see q2_campaign.sbatch, the launcher that names them). The
  family is three configs -- baseline (level 0, model 'sine', no ambient),
  sharp (L0.08 A0.06), smooth (L0.05 A0.09) -- named via --config, or given
  directly with --level/--ambient/--model for one-off reproduction.

  Eval shards carry trials_used PER STATE, not a single scalar trials -- the
  reachability shortcut in shard_eval rolls a state once instead of `trials`
  times when its undisturbed trajectory never comes within MARGIN of the
  corridor. A shard's trial budget can also be split across multiple
  k-windows (k0-20, then a top-up k20-50): trial 0 is the reachability
  probe and is only counted by the window whose trial_lo is 0, so summing
  hits and trials_used across a shard's windows recovers the same total a
  single uninterrupted run would have produced, including the shortcut's
  trials_used == 1. This reducer merges every window found for a shard
  before concatenating across shards.

  Train shards carry no det_labels (shard_train never rolls the
  deterministic grid), so train.npz and train_statistics drop the
  deterministic-agreement fields q2_reduce.py reports.

The description also gains `generation_parameters.noise_model` (the resolved
disturbance stack, from q2_corridor_common.resolve_noise_model) and
`generation_parameters.corridor` (the band geometry), and eval_statistics
gains trials_shortcut / trials_full / num_windows / k_total so the shortcut's
effect on the estimate, and how much of the trial budget has landed, are
visible next to the numbers they produced.

Refuses to reduce a config whose shards are incomplete (missing a shard index
0..nshards-1) unless run with --allow_partial, which is for previews only.

Usage:
  python q2_corridor_reduce.py --config sharp --out_dir /tmp/reduce_preview/sharp
  python q2_corridor_reduce.py --level 0.08 --ambient 0.06 --model sine+ambient \\
      --out_dir /tmp/x --allow_partial
'''
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from q2_common import HORIZON, STATE_BOUNDS, TOL  # noqa: E402
from q2_corridor_common import (BAND, CENTRE, PROFILE, SINE_PERIOD, WIDTH, resolve_noise_model,  # noqa: E402
                                sigma)

CAL_N = 10_000
SPLIT_SEED = 20260813
WEIGHT_N = 0.027 * 9.81
SKIP_MARGIN_M = 0.10  # q2_corridor_collect.py's MARGIN; kept as a literal here
# since the collector does not export the name.

# The collected family: level/ambient/model per config name, plus the
# expected shard count that makes a reduction "complete" (q2_campaign.sbatch
# submissions for this family -- train 100 shards throughout; eval 112 for
# the baseline's smaller deterministic-agreement pass, 560 for the noisy
# configs' full corridor sweep).
CONFIGS = {
    'baseline': dict(level=0.0, ambient=None, model='sine', n_train=100, n_eval=112),
    'sharp': dict(level=0.08, ambient=0.06, model='sine+ambient', n_train=100, n_eval=560),
    'smooth': dict(level=0.05, ambient=0.09, model='sine+ambient', n_train=100, n_eval=560),
}

SHARD_RE = re.compile(
    r'^(?P<split>train|eval)_L(?P<level>[0-9.]+)_A(?P<amb>none|[0-9.]+)_'
    r'k(?P<klo>\d+)-(?P<khi>\d+)_s(?P<shard>\d+)\.npz$')


class MissingShardsError(Exception):
    '''Raised when a config's shards do not cover every index 0..n-1 and
    --allow_partial was not given.'''

    def __init__(self, split, missing):
        self.split, self.missing = split, missing
        super().__init__(f'{split}: missing shard indices {missing}')


def _shard_match(name, split, level, ambient, tol=1e-9):
    m = SHARD_RE.match(name)
    if not m or m['split'] != split:
        return None
    if abs(float(m['level']) - level) > tol:
        return None
    if ambient is None:
        if m['amb'] != 'none':
            return None
    elif m['amb'] == 'none' or abs(float(m['amb']) - ambient) > tol:
        return None
    return dict(klo=int(m['klo']), khi=int(m['khi']), shard=int(m['shard']))


def find_shard_files(shard_dir, split, level, ambient):
    '''-> {shard_idx: [(path, klo, khi), ...]}, windows sorted by klo.

    Matches on the parsed (level, ambient) values rather than a formatted
    string, so it is indifferent to how the submitting sbatch call spelled
    the CLI floats (e.g. '0.08' vs '0.080').
    '''
    by_idx = defaultdict(list)
    pattern = os.path.join(shard_dir, f'{split}_L*_A*_k*-*_s*.npz')
    for path in sorted(glob.glob(pattern)):
        info = _shard_match(os.path.basename(path), split, level, ambient)
        if info is None:
            continue
        by_idx[info['shard']].append((path, info['klo'], info['khi']))
    for idx in by_idx:
        by_idx[idx].sort(key=lambda t: t[1])
    return dict(by_idx)


def missing_shards(by_idx, n_expected):
    '''Shard indices in 0..n_expected-1 with no file at all.'''
    return [i for i in range(n_expected) if i not in by_idx]


def _check_shard_params(d, level, ambient, model, path):
    if abs(float(d['f_max']) - level) > 1e-9:
        raise ValueError(f"{path}: f_max {float(d['f_max'])} != expected {level}")
    if str(d['model']) != model:
        raise ValueError(f"{path}: model {str(d['model'])!r} != expected {model!r}")
    expected_amb = -1.0 if ambient is None else ambient
    if abs(float(d['ambient']) - expected_amb) > 1e-9:
        raise ValueError(f"{path}: ambient {float(d['ambient'])} != expected {expected_amb}")


def reduce_train(by_idx, level, ambient, model, out_dir):
    '''by_idx: shard_idx -> [(path, klo, khi), ...] from find_shard_files.

    Train shards have exactly one window (k0-1); asserts that rather than
    silently merging, since a top-up scheme has never applied to train.
    '''
    if not by_idx:
        return None
    states, offsets, starts, labels, seeds = [], [0], [], [], []
    for idx in sorted(by_idx):
        windows = by_idx[idx]
        if len(windows) != 1:
            raise ValueError(f'train shard {idx} has {len(windows)} window files, expected 1: '
                             f'{[p for p, _, _ in windows]}')
        path, _, _ = windows[0]
        d = np.load(path)
        _check_shard_params(d, level, ambient, model, path)
        states.append(d['states'])
        offsets.extend((d['offsets'][1:] + offsets[-1]).tolist())
        starts.append(d['starts'])
        labels.append(d['labels'])
        seeds.append(d['seeds'])
    states = np.concatenate(states)
    labels = np.concatenate(labels)
    offsets = np.asarray(offsets, dtype=np.int64)
    assert offsets[-1] == len(states), 'offsets do not span the state array'
    assert len(offsets) - 1 == len(labels), 'offset/label count mismatch'
    np.savez(os.path.join(out_dir, 'train.npz'),
             states=states, offsets=offsets, starts=np.concatenate(starts),
             labels=labels, seeds=np.concatenate(seeds))
    os.makedirs(os.path.join(out_dir, 'train_test_splits'), exist_ok=True)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(labels))
    with open(os.path.join(out_dir, 'train_test_splits', 'shuffled_indices_0.txt'), 'w') as fh:
        fh.write('\n'.join(f'sequence_{i}.txt' for i in order) + '\n')
    with open(os.path.join(out_dir, 'train_test_splits', 'shuffled_labels_0.txt'), 'w') as fh:
        fh.write('\n'.join(str(int(labels[i])) for i in order) + '\n')
    lengths = np.diff(offsets)
    return dict(num_trajectories=int(len(labels)),
                success_count=int(labels.sum()),
                success_rate=float(labels.mean()),
                mean_length=float(lengths.mean()), max_length=int(lengths.max()),
                hit_horizon=int((lengths - 1 >= HORIZON).sum()),
                total_states=int(len(states)), shards=len(by_idx), model=model,
                bounds=achieved_bounds(states))


# Units per 6-D state column, for the achieved_bounds block. File order is
# [x, z, theta, x_dot, z_dot, theta_dot], grouped, not the env's interleaved order.
_UNITS = dict(x='m', z='m', theta='rad', x_dot='m/s', z_dot='m/s', theta_dot='rad/s')
_COLS = ['x', 'z', 'theta', 'x_dot', 'z_dot', 'theta_dot']


def achieved_bounds(states):
    """Per-column min/max over every state in train.npz, including the final
    out-of-bounds state of a terminated rollout.

    Sumanth added this block by hand to the published descriptions on
    2026-08-20, following the deterministic dataset's convention. Computing it
    here instead means a republish regenerates it rather than dropping it.
    """
    lo, hi = states.min(axis=0), states.max(axis=0)
    out = {'description': ('Actual min/max values achieved across all trajectories in '
                           "this level's train.npz (includes final out-of-bounds "
                           'states). Computed by the reducer; same convention as the '
                           'deterministic dataset_description.json.')}
    for i, name in enumerate(_COLS):
        out[name] = {'min': round(float(lo[i]), 6), 'max': round(float(hi[i]), 6),
                     'unit': _UNITS[name]}
    out['n_states_measured'] = int(len(states))
    return out


def reduce_eval(by_idx, level, ambient, model, out_dir):
    '''by_idx: shard_idx -> [(path, klo, khi), ...] from find_shard_files.

    Merges every k-window of a shard by SUMMING hits and trials_used per
    state before concatenating across shards -- the windows partition a
    state's trials (see module docstring), so the sum is the same total a
    single uninterrupted run at `trials=k_total` would have produced.
    '''
    if not by_idx:
        return None
    starts_chunks, hits_chunks, used_chunks, det_chunks = [], [], [], []
    window_tags = set()
    for idx in sorted(by_idx):
        merged_hits = merged_used = starts_ref = det_ref = None
        for path, klo, khi in by_idx[idx]:
            window_tags.add((klo, khi))
            d = np.load(path)
            _check_shard_params(d, level, ambient, model, path)
            s = d['starts']
            h = d['hits'].astype(np.int64)
            u = d['trials_used'].astype(np.int64)
            det = d['det_labels']
            if starts_ref is None:
                starts_ref, det_ref = s, det
                merged_hits, merged_used = np.zeros_like(h), np.zeros_like(u)
            elif not np.array_equal(s, starts_ref):
                raise ValueError(f'starts mismatch across windows for shard {idx}: {path}')
            merged_hits += h
            merged_used += u
        starts_chunks.append(starts_ref)
        hits_chunks.append(merged_hits)
        used_chunks.append(merged_used)
        det_chunks.append(det_ref)
    starts = np.concatenate(starts_chunks)
    hits = np.concatenate(hits_chunks)
    used = np.concatenate(used_chunks)
    det = np.concatenate(det_chunks)
    p = np.divide(hits, used, out=np.zeros(len(hits), dtype=np.float64), where=used > 0)

    body = '\n'.join(','.join(f'{v:.6f}' for v in r) + f',{q:.4f}'
                     for r, q in zip(starts, p))
    for name in ('roa_labels.txt', 'eval_states.txt'):
        with open(os.path.join(out_dir, name), 'w') as fh:
            fh.write(body + '\n')
    lines = body.split('\n')
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(lines))
    cal_n = min(CAL_N, len(lines) // 10)
    for name, idx in (('cal_set.txt', perm[:cal_n]), ('test_set.txt', perm[cal_n:])):
        with open(os.path.join(out_dir, name), 'w') as fh:
            fh.write('\n'.join(lines[i] for i in idx) + '\n')

    np.savez(os.path.join(out_dir, 'eval_success_prob.npz'),
             starts=starts, successes=hits, trials=used, p_success=p, det_labels=det)
    return dict(num_states=int(len(p)), mean_trials=float(used.mean()),
                mean_p_success=float(p.mean()),
                fraction_interior=float(((p > 0) & (p < 1)).mean()),
                agreement_with_deterministic=float(((hits > 0).astype(int) == det).mean()),
                deterministic_rate=float((det == 1).mean()), shards=len(by_idx), model=model,
                trials_shortcut=int((used == 1).sum()), trials_full=int((used > 1).sum()),
                num_windows=len(window_tags),
                k_total=max((khi for _, khi in window_tags), default=0))


# The draw law, per noise model. These fields USED to be hardcoded to the
# per-step uniform draw regardless of --model, which shipped a
# dataset_description.json contradicting its own generation_parameters.
# noise_model: the collected family is 'sine+ambient', whose corridor term is
# a per-rollout sinusoid, while NOISE_MODELS marks 'uniform' as falsified and
# keeps it only for reproduction. A consumer reading `mechanism` would have
# been told the wrong law.
_SINE_TERM = 'sigma(z) * (0.5 + 0.5*A*sin(2*pi/period*t + phi))'
_UNIFORM_TERM = 'U(0, sigma(z))'
_AMBIENT_TERM = 'N(0, ambient_std)'

# model -> (corridor term, carries the ambient term)
_DRAWS = {
    'sine': (_SINE_TERM, False),
    'sine+ambient': (_SINE_TERM, True),
    'uniform': (_UNIFORM_TERM, False),
    'ambient': (None, True),
}


def draw_law(model):
    '''-> the mechanism-block fields that depend on which model produced the data.

    Keys mirror q2_reduce.py's `mechanism` schema (distribution / low / high /
    hold / applied_as) so the corridor family stays readable by whatever reads
    the planar-force set, plus `formula` for the corridor block.
    '''
    if model not in _DRAWS:
        raise ValueError(f'unknown noise model {model!r}; expected one of {sorted(_DRAWS)}')
    corridor, has_ambient = _DRAWS[model]
    terms = [t for t in (corridor, _AMBIENT_TERM if has_ambient else None) if t]

    if corridor is _SINE_TERM:
        dist = ('per-rollout coherent sinusoid -- A ~ U(0,1) and phi ~ U(-pi,pi) drawn '
                'once in reset(), NOT a per-step draw')
        hold = ('coherent per-rollout: A and phi are fixed for the episode, so the '
                'corridor force is a deterministic function of (z, t)')
        low, high = 0.0, 'sigma(z) (altitude-gated envelope)'
    elif corridor is _UNIFORM_TERM:
        dist = 'uniform'
        hold = 'zero-order, redrawn each control step (100 Hz)'
        low, high = 0.0, 'sigma(z) (altitude-gated)'
    else:
        dist = 'no corridor term -- ambient wobble only'
        hold = 'n/a: no corridor term'
        low, high = None, None

    if has_ambient:
        dist += '; plus zero-mean N(0, ambient_std) drawn i.i.d. every control step'
        if corridor is not None:
            hold += '. The ambient term IS redrawn every step, everywhere, not just in the band'

    # The corridor term is one-sided (+x); the ambient term is zero-mean and
    # therefore two-sided. Both ride the same mask [1, 0] -- see
    # q2_corridor_common.build().
    applied_as = ('[Fx, 0, 0] at the COM link -- NO torque; corridor term is one-sided (+x), '
                  'ambient term is zero-mean and two-sided' if has_ambient and corridor is not None
                  else '[Fx, 0, 0] at the COM link -- zero-mean, two-sided, NO torque' if has_ambient
                  else '[Fx, 0, 0] at the COM link -- one-sided, +x only, NO torque')

    return dict(distribution=dist, low=low, high=high, hold=hold, applied_as=applied_as,
                formula='F_x = ' + ' + '.join(terms) if terms else 'F_x = 0')


def describe(level, ambient, model, tr, ev):
    stack = resolve_noise_model(model, level, ambient)
    law = draw_law(model)
    desc = {
        'dataset_name': ('2D Quadrotor RL (safe_explorer_ppo) under an '
                         f'altitude-gated corridor disturbance, model={model} '
                         f'f_max={level} ambient={ambient}'),
        'mechanism': {
            'kind': 'dynamics', 'dim': 1, 'frame': 'world',
            'applied_as': law['applied_as'],
            'distribution': law['distribution'], 'low': law['low'], 'high': law['high'],
            'hold': law['hold'],
            'formula': law['formula'],
            'matched': False,
            'reference_scale': {'body_weight_N': WEIGHT_N,
                                'level_as_fraction_of_weight': float(level) / WEIGHT_N
                                if level else 0.0},
            'note': ('A corridor of disturbed air below the goal, gated by '
                     'altitude rather than uniform over the state space -- see '
                     "the 'corridor' block below for the geometry."),
        },
        'controller': {'type': 'safe_explorer_ppo',
                       'model': 'safe_explorer_ppo_model_quadrotor_2D_stab.pt',
                       'obs_normalizer': 'frozen (set_read_only) and applied every step',
                       'note': "info['constraint_values'] is seeded before the first step; "
                               'the policy reads it internally'},
        'success_criteria': {
            'type': 'radius', 'threshold': TOL,
            'goal_state': [0, 1, 0, 0, 0, 0],
            'entry_cut': True,
            'note': ('Stops at first entry, so the label is a function of the '
                     'terminal state.'),
        },
        'horizon': {
            'steps': HORIZON, 'seconds': HORIZON / 100.0,
            'inherited': True,
            'note': 'INHERITED from the deterministic set and the planar-force family.',
        },
        'termination_thresholds': {
            'x': 1.0, 'z_min': 0.1, 'z_max': 1.5, 'theta': 'inf',
            'x_dot': 1.0, 'z_dot': 1.0, 'theta_dot': 8.0,
            'source': ('deterministic dataset_description.json; the quad2d RL '
                       'generator sets state_space explicitly'),
            'env_state_indices': {str(k): list(v) for k, v in STATE_BOUNDS.items()},
        },
        'plant': {'quad_type': 2, 'ctrl_freq': 100, 'pyb_freq': 5000,
                  'cost': 'quadratic', 'randomized_init': False,
                  'normalized_rl_action_space': True,
                  'constraints': 'SAFE_EXPLORER_CONSTRAINTS, done_on_violation=False'},
        'data_format': {
            'state_order': ['x', 'z', 'theta', 'x_dot', 'z_dot', 'theta_dot'],
            'angular_velocity_frame': ('world -- for TWO_D the env stores ang_v[1] '
                                       'directly with no body conversion'),
            'theta': 'wrapped to [-pi, pi] when stored',
            'train': 'train.npz -- states(float32,6) offsets starts labels seeds',
            'eval': 'roa_labels.txt / eval_states.txt -- 6 state cols + p_success',
            'precision': {'state': 6, 'p_success': 4},
        },
        'sampling': {'type': 'stratified_grid',
                     'note': ('eval takes the same 489,789 deterministic grid '
                              'states as the planar-force family; train uses '
                              'off-lattice random starts within the same bounds.')},
        'reproducibility': {
            'seed_fn': 'rollout_seed(base, split_id, index, trial), base 20260817',
            'note': ('The disturbance strength is excluded from the seed, so '
                     'levels are paired under common random numbers.'),
        },
        'train_statistics': {k: v for k, v in tr.items() if k != 'bounds'} if tr else tr,
        'eval_statistics': ev,
        'achieved_bounds': tr.get('bounds') if tr else None,
    }
    desc['generation_parameters'] = {
        'noise_model': stack,
        'corridor': {
            'mechanism': 'altitude_gated',
            'channel': 'dynamics',
            'one_sided': True,
            'direction': '+x',
            'profile': PROFILE,
            'formula': law['formula'],
            'sigma': 'sigma(z) = f_max * exp(-0.5*((z-centre)/width)**2)',
            'f_max': float(level),
            'centre': CENTRE,
            'width': WIDTH,
            'sigma_at_goal': float(sigma(1.0, level)),
            # 3dp, not 4: BAND is (0.1858175, 0.9141825), and the canonical
            # label used everywhere else (q2_corridor_common.py's own BAND
            # comment, the design spec) is the rounder [0.186, 0.914].
            # round(BAND, 4) would give [0.1858, 0.9142], off by a digit.
            'band_1pct': [round(BAND[0], 3), round(BAND[1], 3)],
            'draw_law': model,
            'period_s': SINE_PERIOD,
            'ambient_std': stack['ambient'],
            'redraw': 'per-rollout phase and amplitude; deterministic within a rollout',
            'skip_margin_m': SKIP_MARGIN_M,
        },
    }
    return desc


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', choices=sorted(CONFIGS),
                    help='one of the collected family; sets level/ambient/model/nshards')
    ap.add_argument('--level', type=float, help='overrides --config, or stands alone')
    ap.add_argument('--ambient', type=float, help='overrides --config; omit for none')
    ap.add_argument('--model', help='overrides --config')
    ap.add_argument('--n_train', type=int, help='overrides --config; expected train shard count')
    ap.add_argument('--n_eval', type=int, help='overrides --config; expected eval shard count')
    ap.add_argument('--shard_dir',
                    default=os.environ.get('Q2CORR_SHARDS', '/common/users/dm1487/q2_corridor_shards'))
    ap.add_argument('--out_dir', required=True,
                    help='STAGING directory. Never a shared/published path -- '
                         'publication is a separate, human-gated step.')
    ap.add_argument('--allow_partial', action='store_true',
                    help='reduce even if some shard indices are missing (previews only)')
    args = ap.parse_args(argv)
    if args.config is None and args.level is None:
        ap.error('--config or --level is required')
    return args


def resolve_config(args):
    if args.config is not None:
        cfg = dict(CONFIGS[args.config])
    else:
        cfg = dict(level=None, ambient=None, model='sine+ambient', n_train=100, n_eval=560)
    if args.level is not None:
        cfg['level'] = args.level
    if args.ambient is not None:
        cfg['ambient'] = args.ambient
    if args.model is not None:
        cfg['model'] = args.model
    if args.n_train is not None:
        cfg['n_train'] = args.n_train
    if args.n_eval is not None:
        cfg['n_eval'] = args.n_eval
    if cfg['level'] is None:
        raise SystemExit('[ERROR] q2_corridor_reduce.py: --level required when --config '
                         'is not given')
    return cfg


def main(argv=None):
    args = parse_args(argv)
    cfg = resolve_config(args)
    level, ambient, model = cfg['level'], cfg['ambient'], cfg['model']

    train_by_idx = find_shard_files(args.shard_dir, 'train', level, ambient)
    eval_by_idx = find_shard_files(args.shard_dir, 'eval', level, ambient)
    missing_train = missing_shards(train_by_idx, cfg['n_train'])
    missing_eval = missing_shards(eval_by_idx, cfg['n_eval'])
    if (missing_train or missing_eval) and not args.allow_partial:
        if missing_train:
            print(f'[ERROR] q2_corridor_reduce.py: missing train shards '
                  f"({len(missing_train)}/{cfg['n_train']}) for level={level} "
                  f'ambient={ambient} model={model!r}: {missing_train}', file=sys.stderr)
        if missing_eval:
            print(f'[ERROR] q2_corridor_reduce.py: missing eval shards '
                  f"({len(missing_eval)}/{cfg['n_eval']}) for level={level} "
                  f'ambient={ambient} model={model!r}: {missing_eval}', file=sys.stderr)
        print('[ERROR] q2_corridor_reduce.py: refusing to reduce a partial config; '
              'pass --allow_partial for a preview.', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    tr = reduce_train(train_by_idx, level, ambient, model, args.out_dir)
    ev = reduce_eval(eval_by_idx, level, ambient, model, args.out_dir)
    desc = describe(level, ambient, model, tr, ev)
    for name, payload in (('dataset_description.json', desc),
                          ('train_description.json', {**desc, 'split': 'train'}),
                          ('eval_description.json', {**desc, 'split': 'eval'})):
        with open(os.path.join(args.out_dir, name), 'w') as fh:
            json.dump(payload, fh, indent=2)
    print(f'{args.out_dir}: train={tr} eval={ev}', flush=True)


if __name__ == '__main__':
    main()
