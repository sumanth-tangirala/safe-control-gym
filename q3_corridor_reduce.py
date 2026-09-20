'''Aggregate the quad3d twin-curtain shards into one dataset directory per
config, in a STAGING directory only. Publication to the shared root is a
separate, human-gated step.

Mirrors q2_corridor_reduce.py's output contract exactly, same file set and key
names, so anything that reads the quad2d corridor family reads this too. Four
things differ, all forced by quad3d itself:

  States are 13-D quaternion rows [x, y, z, qw, qx, qy, qz, x_dot, y_dot,
  z_dot, p, q, r], not quad2d's 6-D. The text files carry 13 state columns plus
  p_success.

  Shard names are flat and carry the config, not a level string alone:
  `{split}_f<level>_a<ambient>[_k<trials>]_s<shard>.npz` for eval and
  `train_f<level>_a<ambient>_s<shard>.npz` for train, as q3eval.sbatch and
  q3train.sbatch write them.

  The corridor is a TWIN curtain gated on x, so `generation_parameters.corridor`
  carries both centres and the two bands, not a single altitude band.

  Train shards carry no det_labels, so train_statistics drops the
  deterministic-agreement fields. Eval shards do carry them, from the shipped
  deterministic set, which is what makes rescued and broken computable without
  a separate zero-force baseline.

The k-window merge is kept even though nothing has used it yet: the collector
writes trial_lo and trials per shard, so a K=20 to K=50 top-up would land as a
second window and must sum rather than replace. Same argument as quad2d, where
that mechanism is what let a spoiled campaign be repaired instead of recollected.

describe() derives the noise law from the model rather than hardcoding one.
The quad2d reducer hardcoded the falsified per-step uniform draw into every
emitted description regardless of --model, so the file contradicted its own
generation_parameters.noise_model. Not repeating that here.

Usage:
  python q3_corridor_reduce.py --config f_0.25 --out_dir /tmp/stage/f_0.25
  python q3_corridor_reduce.py --level 0.30 --ambient 0.008 --out_dir /tmp/x \\
      --allow_partial
'''
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

from q3_corridor_common import BAND, PROFILE, SINE_PERIOD, WIDTH, X_C, resolve_noise_model, sigma

CAL_N = 10_000
SPLIT_SEED = 20260813
WEIGHT_N = 0.027 * 9.81
N_EVAL, N_TRAIN_TOTAL = 1_000_000, 800_000

# The collected family. Names are force values, matching the sibling
# stochastic/quadrotor3D/noisy_dynamics/lqr/f_0.032 rather than quad2d's word
# names: ambient is fixed at 0.008 across both configs and measured nearly
# inert (fuzziness moved 0.08 points across an 8x sweep), so force alone
# identifies a config here.
CONFIGS = {
    'f_0.25': dict(level=0.25, ambient=0.008, model='sine+ambient', n_eval=3360, n_train=3360),
    'f_0.30': dict(level=0.30, ambient=0.008, model='sine+ambient', n_eval=3360, n_train=3360),
}

EVAL_RE = re.compile(r'^eval_f(?P<level>[0-9.]+)_a(?P<amb>[0-9.]+)_'
                     r'k(?P<trials>\d+)_s(?P<shard>\d+)\.npz$')
TRAIN_RE = re.compile(r'^train_f(?P<level>[0-9.]+)_a(?P<amb>[0-9.]+)_s(?P<shard>\d+)\.npz$')


class MissingShardsError(Exception):
    '''Raised when a split's shards do not cover every index 0..n-1 and
    --allow_partial was not given.'''

    def __init__(self, split, missing):
        self.split, self.missing = split, missing
        super().__init__(f'{split}: missing shard indices {missing}')


def find_shard_files(shard_dir, split, level, ambient, tol=1e-9):
    '''-> {shard_idx: [(path, trials), ...]}, windows sorted by trial count.

    Matches on the parsed float values rather than a formatted string, so it is
    indifferent to how the submitting sbatch spelled the CLI numbers.
    '''
    rx = EVAL_RE if split == 'eval' else TRAIN_RE
    by_idx = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(shard_dir, f'{split}_f*_a*_s*.npz'))):
        m = rx.match(os.path.basename(path))
        if m is None:
            continue
        if abs(float(m['level']) - level) > tol or abs(float(m['amb']) - ambient) > tol:
            continue
        by_idx[int(m['shard'])].append((path, int(m.groupdict().get('trials', 1))))
    for idx in by_idx:
        by_idx[idx].sort(key=lambda t: t[1])
    return dict(by_idx)


def missing_shards(by_idx, n_expected):
    return [i for i in range(n_expected) if i not in by_idx]


# Column counts, asserted per shard so a layout regression fails at reduce
# time rather than silently reaching the downstream model. Trajectory states
# and eval starts are the 13-D quaternion row; train starts stay the 12-D
# Euler row the sampler emits.
NCOL_ROW13, NCOL_SAMPLER = 13, 12


def _cols(a, want, path, what):
    if a.ndim != 2 or a.shape[1] != want:
        raise ValueError(f'{path}: {what} has shape {a.shape}, expected (n, {want})')


def _check(d, level, ambient, model, path):
    if abs(float(d['f_max']) - level) > 1e-9:
        raise ValueError(f"{path}: f_max {float(d['f_max'])} != expected {level}")
    if str(d['model']) != model:
        raise ValueError(f"{path}: model {str(d['model'])!r} != expected {model!r}")
    if abs(float(d['ambient']) - ambient) > 1e-9:
        raise ValueError(f"{path}: ambient {float(d['ambient'])} != expected {ambient}")


def reduce_eval(by_idx, level, ambient, model, out_dir):
    '''Sums hits and trials_used across a shard's k-windows before
    concatenating across shards. The windows partition a start's trials, so the
    sum is what one uninterrupted run at k_total would have produced.'''
    if not by_idx:
        return None
    starts_c, hits_c, used_c, det_c = [], [], [], []
    window_tags = set()
    for idx in sorted(by_idx):
        merged_h = merged_u = starts_ref = det_ref = None
        for path, trials in by_idx[idx]:
            window_tags.add(trials)
            d = np.load(path)
            _check(d, level, ambient, model, path)
            s, h = d['starts'], d['hits'].astype(np.int64)
            _cols(s, NCOL_ROW13, path, 'eval starts')
            u, det = d['trials_used'].astype(np.int64), d['det_labels']
            if starts_ref is None:
                starts_ref, det_ref = s, det
                merged_h, merged_u = np.zeros_like(h), np.zeros_like(u)
            elif not np.array_equal(s, starts_ref):
                raise ValueError(f'starts mismatch across windows for shard {idx}: {path}')
            merged_h += h
            merged_u += u
        starts_c.append(starts_ref)
        hits_c.append(merged_h)
        used_c.append(merged_u)
        det_c.append(det_ref)
    starts = np.concatenate(starts_c)
    hits, used = np.concatenate(hits_c), np.concatenate(used_c)
    det = np.concatenate(det_c).astype(int)
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

    ok = det == 1
    return dict(num_states=int(len(p)), mean_trials=float(used.mean()),
                mean_p_success=float(p.mean()),
                fraction_interior=float(((p > 0) & (p < 1)).mean()),
                agreement_with_deterministic=float(((hits > 0).astype(int) == det).mean()),
                deterministic_rate=float(ok.mean()),
                # The two directions of label change, which the quad2d family
                # reports only by hand. A rescued start failed deterministically
                # and sometimes reaches the goal under disturbance; a broken one
                # is the reverse.
                rescued=int(((~ok) & (p > 0)).sum()), broken=int((ok & (p < 1)).sum()),
                shards=len(by_idx), model=model,
                trials_shortcut=int((used == 1).sum()), trials_full=int((used > 1).sum()),
                num_windows=len(window_tags), k_total=max(window_tags, default=0))


# Units per 13-D state column, for the achieved_bounds block.
_UNITS = dict(x='m', y='m', z='m', qw='unitless', qx='unitless', qy='unitless',
              qz='unitless', x_dot='m/s', y_dot='m/s', z_dot='m/s',
              p='rad/s', q='rad/s', r='rad/s')
_COLS = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'x_dot', 'y_dot', 'z_dot', 'p', 'q', 'r']


def achieved_bounds(states):
    '''Per-column min/max over every state in train.npz, including the final
    out-of-bounds state of a terminated rollout.

    Sumanth added this block by hand to the published descriptions on
    2026-08-20, following the deterministic dataset's convention. Computing it
    here instead means a republish regenerates it rather than dropping it.
    '''
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


def reduce_train(by_idx, level, ambient, model, out_dir):
    '''Train shards have exactly one window; asserts that rather than silently
    merging, since a top-up scheme has never applied to train.'''
    if not by_idx:
        return None
    states, offsets, starts, labels, seeds = [], [0], [], [], []
    for idx in sorted(by_idx):
        windows = by_idx[idx]
        if len(windows) != 1:
            raise ValueError(f'train shard {idx} has {len(windows)} window files, expected 1: '
                             f'{[p for p, _ in windows]}')
        path, _ = windows[0]
        d = np.load(path)
        _check(d, level, ambient, model, path)
        _cols(d['states'], NCOL_ROW13, path, 'train states')
        _cols(d['starts'], NCOL_SAMPLER, path, 'train starts')
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
    return dict(num_trajectories=int(len(labels)), success_count=int(labels.sum()),
                success_rate=float(labels.mean()),
                mean_length=float(lengths.mean()), max_length=int(lengths.max()),
                total_states=int(len(states)), shards=len(by_idx), model=model,
                bounds=achieved_bounds(states))


# The draw law per model, derived rather than hardcoded. See the module
# docstring for why.
_SINE = 'sigma(x) * (0.5 + 0.5*A*sin(2*pi/period*t + phi))'
_AMBIENT = 'N(0, ambient_std)'
_DRAWS = {'sine': (_SINE, False), 'sine+ambient': (_SINE, True),
          'uniform': ('U(0, sigma(x))', False), 'ambient': (None, True)}


def draw_law(model):
    if model not in _DRAWS:
        raise ValueError(f'unknown noise model {model!r}; expected one of {sorted(_DRAWS)}')
    corridor, has_amb = _DRAWS[model]
    terms = [t for t in (corridor, _AMBIENT if has_amb else None) if t]
    if corridor is _SINE:
        dist = ('per-rollout coherent sinusoid -- A ~ U(0,1) and phi ~ U(-pi,pi) drawn '
                'once in reset(), NOT a per-step draw')
        hold = ('coherent per-rollout: A and phi are fixed for the episode, so each '
                'curtain force is a deterministic function of (x, t)')
    elif corridor is None:
        dist, hold = 'no curtain term -- ambient wobble only', 'n/a: no curtain term'
    else:
        dist, hold = 'uniform', 'zero-order, redrawn each control step (100 Hz)'
    if has_amb:
        dist += '; plus zero-mean N(0, ambient_std) drawn i.i.d. every control step'
        if corridor is not None:
            hold += '. The ambient term IS redrawn every step, everywhere, not just in a band'
    # Measured, not assumed. mask=[0,1,0] on a THREE_D quad selects index 1 of
    # the disturb_force 3-vector, and quadrotor.py passes that vector through
    # unchanged, so the force lands on y. The gate is on x. Probing the env
    # with the drone parked at the curtain peak gives [0, 0.068, 0].
    applied = ('[0, Fy, 0] at the COM link -- NO torque; the curtain is gated on x but '
               'pushes along y, so the drone is shoved sideways as it crosses. Curtain '
               'term is one-sided (+y); ambient term is zero-mean and two-sided'
               if has_amb and corridor is not None else
               '[0, Fy, 0] at the COM link -- gated on x, pushes along +y only, NO torque')
    return dict(distribution=dist, hold=hold, applied_as=applied,
                formula='F_y = ' + ' + '.join(terms) if terms else 'F_y = 0')


def describe(level, ambient, model, tr, ev):
    stack = resolve_noise_model(model, level, ambient)
    law = draw_law(model)
    return {
        'dataset_name': ('3D Quadrotor LQR under a twin-curtain corridor disturbance, '
                         f'model={model} f_max={level} ambient={ambient}'),
        'mechanism': {
            'kind': 'dynamics', 'dim': 3, 'active_component': 'y', 'frame': 'world',
            'applied_as': law['applied_as'], 'distribution': law['distribution'],
            'hold': law['hold'], 'formula': law['formula'], 'matched': False,
            'reference_scale': {'body_weight_N': WEIGHT_N,
                                'level_as_fraction_of_weight': float(level) / WEIGHT_N
                                if level else 0.0},
            'note': ('Two vertical sheets of crosswind standing at x = +/-0.9 m. The '
                     'gate is the x coordinate, the push is along y, so a drone '
                     'crossing a sheet is shoved perpendicular to its crossing '
                     'direction. Each curtain draws independently, so the MEAN force '
                     'at x is sigma(x)/2 while sigma is the sum of both peaks.'),
        },
        'controller': {'type': 'lqr',
                       'note': 'the shipped quadrotor3D_lqr controller, unchanged'},
        'success_criteria': {'type': 'goal_reached', 'entry_cut': True,
                             'note': ('Stops at first entry, so the label is a function '
                                      'of the terminal state.')},
        # The two splits do NOT share a deadline, so reporting one number
        # would be wrong for one of them. Eval goes through
        # q3_corridor_common.roll(), which runs to that module's HORIZON=2000
        # (the memo-D deadline). Train goes through
        # generate_quadrotor_3d_noisy.run(), hardcoded to the older
        # HORIZON=1000 that the noisy_dynamics family was labelled at.
        # Measured 2026-08-20: the train deadline truncated 1 of 800,000
        # trajectories at f_max 0.25 and 0 at 0.30, so the mismatch is
        # recorded rather than recollected.
        'horizon': {
            'eval': {'steps': 2000, 'seconds': 20.0, 'ctrl_freq_hz': 100},
            'train': {'steps': 1000, 'seconds': 10.0, 'ctrl_freq_hz': 100},
            'note': ('Split-dependent deadline. Eval labels are bounded-time reach '
                     'probabilities at 20 s; train labels at 10 s. Under this '
                     'disturbance the controller mostly still reaches the goal and '
                     'just takes longer, so neither is an asymptotic reach '
                     'probability.'),
        },
        'data_format': {
            # TWO layouts, not one. Trajectory states and eval starts are the
            # 13-D quaternion row; train STARTS are the 12-D Euler row the
            # sampler emits, kept verbatim so index i matches the shipped
            # deterministic set row for row.
            'state_order': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz',
                            'x_dot', 'y_dot', 'z_dot', 'p', 'q', 'r'],
            'train_start_order': ['x', 'y', 'z', 'phi', 'theta', 'psi',
                                  'x_dot', 'y_dot', 'z_dot', 'p', 'q', 'r'],
            'angular_rate_frame': ('body -- p,q,r in a 13-D row are body rates; '
                                   'the injector converts them to world'),
            'train': ('train.npz -- states(float32, N x 13) offsets(int64, T+1) '
                      'starts(float64, T x 12) labels(uint8, T) seeds(int64, T). '
                      'Trajectory t is states[offsets[t]:offsets[t+1]].'),
            'eval': ('roa_labels.txt / eval_states.txt -- 13 state cols + '
                     'p_success. eval_success_prob.npz carries starts successes '
                     'trials p_success det_labels.'),
            'precision': {'state': 6, 'p_success': 4},
        },
        'sampling': {
            'type': 'index_aligned_with_deterministic',
            'note': ('eval takes the shipped quadrotor3D_lqr eval_states.txt rows '
                     f'({N_EVAL} of them, 1,000,000-row grid); train regenerates the '
                     f'{N_TRAIN_TOTAL} sampler starts with seed 42, reproducing the '
                     'shipped starts to 5e-7. Index i is the same trajectory in both.'),
        },
        'reproducibility': {
            'seed_fn': 'rollout_seed(base, split_id, index, trial), base 20260817',
            'note': ('The disturbance strength is excluded from the seed, so levels are '
                     'paired under common random numbers.'),
        },
        'generation_parameters': {
            'noise_model': stack,
            'corridor': {
                'mechanism': 'x_gated_twin_crosswind_curtain', 'channel': 'dynamics',
                'one_sided': True, 'gated_on': 'x', 'direction': '+y', 'twin': True,
                'func_name_note': ('the registered function is called '
                                   "'altitude_gated_sine', inherited from the quad2d "
                                   'altitude corridor. Here it gates on x '
                                   '(state_index 0) and pushes on y. The name is '
                                   'legacy; gated_on and direction are authoritative.'),
                'profile': PROFILE, 'formula': law['formula'],
                'sigma': ('sigma(x) = f_max * [exp(-0.5*((x-x_c)/width)**2) '
                          '+ exp(-0.5*((x+x_c)/width)**2)]'),
                'f_max': float(level), 'centres': [X_C, -X_C], 'width': WIDTH,
                'band_1pct': [[round(b[0], 3), round(b[1], 3)] for b in BAND],
                'sigma_at_origin': float(sigma(0.0, level)),
                'draw_law': model, 'period_s': SINE_PERIOD,
                'ambient_std': stack['ambient'],
                'redraw': 'per-rollout phase and amplitude, independent per curtain',
                'independent_curtains': True,
            },
        },
        'train_statistics': {k: v for k, v in tr.items() if k != 'bounds'} if tr else tr,
        'eval_statistics': ev,
        'achieved_bounds': tr.get('bounds') if tr else None,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', choices=sorted(CONFIGS),
                    help='one of the collected family; sets level/ambient/model/nshards')
    ap.add_argument('--level', type=float, help='overrides --config, or stands alone')
    ap.add_argument('--ambient', type=float, help='overrides --config')
    ap.add_argument('--model', help='overrides --config')
    ap.add_argument('--n_eval', type=int, help='expected eval shard count')
    ap.add_argument('--n_train', type=int, help='expected train shard count')
    ap.add_argument('--eval_dir', default=os.environ.get('Q3EVAL_SHARDS',
                                                         '/scratch/dm1487/q3eval'))
    ap.add_argument('--train_dir', default=os.environ.get('Q3TRAIN_SHARDS',
                                                          '/scratch/dm1487/q3train'))
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
    cfg = dict(CONFIGS[args.config]) if args.config else dict(
        level=None, ambient=0.008, model='sine+ambient', n_eval=3360, n_train=3360)
    for k in ('level', 'ambient', 'model', 'n_eval', 'n_train'):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if cfg['level'] is None:
        raise SystemExit('[ERROR] q3_corridor_reduce.py: --level required when --config '
                         'is not given')
    return cfg


def main(argv=None):
    args = parse_args(argv)
    cfg = resolve_config(args)
    level, ambient, model = cfg['level'], cfg['ambient'], cfg['model']

    ev_idx = find_shard_files(args.eval_dir, 'eval', level, ambient)
    tr_idx = find_shard_files(args.train_dir, 'train', level, ambient)
    miss_e = missing_shards(ev_idx, cfg['n_eval'])
    miss_t = missing_shards(tr_idx, cfg['n_train'])
    if (miss_e or miss_t) and not args.allow_partial:
        for name, miss, tot in (('eval', miss_e, cfg['n_eval']),
                                ('train', miss_t, cfg['n_train'])):
            if miss:
                print(f'[ERROR] q3_corridor_reduce.py: missing {name} shards '
                      f'({len(miss)}/{tot}) for level={level} ambient={ambient}: '
                      f'{miss[:12]}{"..." if len(miss) > 12 else ""}', file=sys.stderr)
        print('[ERROR] q3_corridor_reduce.py: refusing to reduce a partial config; '
              'pass --allow_partial for a preview.', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    tr = reduce_train(tr_idx, level, ambient, model, args.out_dir)
    ev = reduce_eval(ev_idx, level, ambient, model, args.out_dir)
    desc = describe(level, ambient, model, tr, ev)
    for name, payload in (('dataset_description.json', desc),
                          ('train_description.json', {**desc, 'split': 'train'}),
                          ('eval_description.json', {**desc, 'split': 'eval'})):
        with open(os.path.join(args.out_dir, name), 'w') as fh:
            json.dump(payload, fh, indent=2)
    print(f'{args.out_dir}: train={tr} eval={ev}', flush=True)


if __name__ == '__main__':
    main()
