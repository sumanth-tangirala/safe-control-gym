'''Aggregate the cartpole gaussian_signal shards into one directory per level.

cp_reduce.py cannot be reused as-is: it reads the uniform family's layout
(`<root>/train/S<sigma>_s*.npz`), names outputs `s_<sigma>`, and pulls
`det_labels` out of the train shards, which cp_collect does not write. This
reads the layout the gaussian collection actually produced --
`<root>/<level>/{train,eval}_s*.npz` -- and emits the file set the published
sibling `stochastic/cartpole/noisy_torque/lqr/<level>/` has, so the two compare
directory-for-directory.

Levels here are matched to the pendulum gaussian_signal set on MID, the
fraction of eval cells with p in [0.2, 0.8], which is what tracks the width and
depth of the blur around the separatrix. Not on `interior` (0 < p < 1): that
counts uncertain cells while ignoring how uncertain they are, and a level
matched that way came out with 84.9% of its interior cells below 0.1 or above
0.9, median 0.98 -- a hard edge with one frayed pixel.
'''
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from cp_collect import BOUNDS, CUT, FORCE, HORIZON, N_EVAL, N_TRAIN, TOL  # noqa: E402

ROOT = os.environ.get('CP_GAUSS_OUT', '/scratch/%s/cartpole_gaussian_signal'
                      % os.environ.get('USER', 'nobody'))
DEST = os.environ.get('CP_DATASET', os.path.expanduser('~/cp_gauss_dataset'))
CAL_N = 10_000
SPLIT_SEED = 20260813

# What each level was aimed at, from stochastic/pendulum/gaussian_signal/lqr.
PENDULUM_MID = {'low': 0.0376, 'med': 0.1334, 'high': 0.6342}
PENDULUM_INTERIOR = {'low': 0.1120, 'med': 0.6459, 'high': 0.8239}


def reduce_train(level, out_dir):
    files = sorted(glob.glob(os.path.join(ROOT, level, 'train_s*.npz')))
    if not files:
        return None
    states, offsets, starts, labels, seeds = [], [0], [], [], []
    alpha = beta = None
    for f in files:
        d = np.load(f)
        states.append(d['states'])
        offsets.extend((d['offsets'][1:] + offsets[-1]).tolist())
        starts.append(d['starts'])
        labels.append(d['labels'])
        seeds.append(d['seeds'])
        alpha, beta = float(d['alpha']), float(d['beta'])
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
                success_rate=float(labels.mean()), mean_length=float(lengths.mean()),
                max_length=int(lengths.max()),
                hit_horizon=int((lengths - 1 >= HORIZON).sum()),
                total_states=int(len(states)), shards=len(files),
                alpha=alpha, beta=beta)


def reduce_eval(level, out_dir):
    files = sorted(glob.glob(os.path.join(ROOT, level, 'eval_s*.npz')))
    if not files:
        return None
    keyed = []
    alpha = beta = trials = None
    for f in files:
        d = np.load(f)
        keyed.append((int(d['lo']), d['starts'], d['hits'], d['det_labels']))
        trials = int(d['trials'])
        alpha, beta = float(d['alpha']), float(d['beta'])
    keyed.sort(key=lambda t: t[0])            # shard order is not lexical
    starts = np.concatenate([k[1] for k in keyed])
    hits = np.concatenate([k[2] for k in keyed])
    det = np.concatenate([k[3] for k in keyed])
    assert len(starts) == N_EVAL, f'{len(starts)} eval rows, expected {N_EVAL}'
    p = hits / trials

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
             starts=starts, successes=hits.astype(np.int32),
             trials=np.full(len(hits), trials, dtype=np.int32),
             p_success=p, det_labels=det)

    inter = p[(p > 0) & (p < 1)]
    return dict(num_states=int(len(p)), trials=trials,
                mean_p_success=float(p.mean()),
                fraction_mid=float(((p >= 0.2) & (p <= 0.8)).mean()),
                fraction_interior=float(((p > 0) & (p < 1)).mean()),
                median_interior_p=float(np.median(inter)) if len(inter) else None,
                interior_in_mid=float(((inter >= 0.2) & (inter <= 0.8)).mean()),
                interior_at_extremes=float(((inter < 0.1) | (inter > 0.9)).mean()),
                rescued=int(((det == 0) & (p > 0)).sum()),
                broken=int(((det == 1) & (p < 1)).sum()),
                deterministic_rate=float((det == 1).mean()),
                shards=len(files), alpha=alpha, beta=beta)


def describe(level, tr, ev):
    a, b = ev['alpha'], ev['beta']
    return {
        'dataset_name': f'CartPole LQR under signal-dependent action noise ({level})',
        'level_name': level,
        'level_name_note': ('low/med/high are matched to the pendulum '
                            'gaussian_signal set on the uncertainty band, not on '
                            'noise magnitude. The parameters are alpha/beta below.'),
        'mechanism': {
            'kind': 'action', 'dim': 1,
            'applied_as': ('gaussian on the commanded cart force, PRE-saturation, '
                           'with a scale that grows with the command'),
            'distribution': 'normal', 'alpha': a, 'beta': b,
            'sigma_formula': 'sigma = alpha + beta*|u|, a STANDARD DEVIATION not a variance',
            'hold': 'zero-order, redrawn each control step (100 Hz)',
            'matched': True,
            'alpha_meaning': ('the floor surviving as the command goes to zero -- the '
                              'only term acting at the goal, so it is what blurs the '
                              'success boundary'),
            'beta_meaning': ('effort-proportional, acting only in the transient, so it '
                             'is what randomises the outcome rather than the edge'),
            'ratio_note': ('alpha = 0.27*beta throughout. A larger ratio reaches a given '
                           'band at less delivered noise, but it does so with a constant '
                           'floor that turns p=1 cells into p=0.98: it widens the band '
                           'without deepening it. Measured, ratio 3.80 at matched band '
                           'left 84.9% of interior cells below 0.1 or above 0.9.'),
            'delivered_std_note': ('Delivered noise is a scale mixture -- every draw has '
                                   'its own sigma -- so its standard deviation is '
                                   'sqrt(E[sigma^2]), NOT E[sigma].'),
            'control_bound_N': FORCE,
            'saturation_caveat': ('Noise is added before the clip to +/-2000 N, but the '
                                  'LQR never saturates: median demand 0.53 N, p99 137.6 '
                                  'over 25,845 measured steps. clip(u+w) and clip(u)+w '
                                  'are the same function here, so the pendulum family\'s '
                                  'pre/post-saturation distinction has nothing to bite '
                                  'on. It is also why this family cannot rescue at scale '
                                  '-- noise adds no authority the controller lacks.'),
        },
        'controller': {'type': 'LQR', 'q_lqr': [1, 1, 1, 1], 'r_lqr': [0.1],
                       'discrete_dynamics': True, 'goal_state': [0, 0, 0, 0]},
        'success_criteria': {
            'type': 'radius', 'threshold': TOL, 'entry_cut': True,
            'computed_over': 'the full 4-D state, env-native goal_reached',
            'IMPORTANT': (
                'The deterministic cartpole_pybullet description claims per-channel '
                'tolerances (x < 0.01, others < 0.05) held for 10 consecutive steps. '
                'That was NEVER IMPLEMENTED. Every shipped success ends with '
                '||state|| in [0.0497, 0.0500] and not one satisfies |x| < 0.01 -- '
                'the signature of first entry into an L2 ball of radius 0.05 with no '
                'dwell. This dataset uses the real rule.'),
            'verification': ('alpha = beta = 0 reproduces the deterministic labels '
                             'exactly: 58/58 on a contiguous shard, on both iLab and '
                             'Amarel, with mean p 0.1724 against the deterministic '
                             '0.1724.'),
        },
        'level_matching': {
            'matched_on': 'fraction of eval cells with p_success in [0.2, 0.8]',
            'why': ('It is what the width AND depth of the separatrix blur reduce to, '
                    'and it is comparable between two systems whose success rates are '
                    'not. Delivered variance was tried first and rejected: it is not '
                    'even monotone in the band, since 22.3 N of noise at one ratio '
                    'gives a wider band than 35.4 N at another.'),
            'target_pendulum_mid': PENDULUM_MID[level],
            'achieved_mid': ev['fraction_mid'],
            'pendulum_reference': 'stochastic/pendulum/gaussian_signal/lqr',
            'high_caveat': (
                'The pendulum high level sits at mid 0.6342 and is NOT reachable on '
                'cartpole; the measured ceiling is ~0.17 across every ratio and both '
                'noise families, and the published uniform cartpole levels sit at '
                'interior 0.187 regardless of sigma. The pendulum saturates 70-98% of '
                'steps so its post-clip noise adds authority and rescues nearly every '
                'failing cell (mean p RISES 0.386 -> 0.546); cartpole never saturates, '
                'so noise only breaks cells and the far exterior stays a hard zero. '
                'This level is the cartpole ceiling, not a match.'),
            'resolution_caveat': (
                'The eval grid is 4-D and coarse per axis -- 14 theta values at step '
                '0.462 and 19 theta_dot at 0.500, against the pendulum grid\'s 158 and '
                '315 at step 0.040. The blur is only 2-4 cells wide, so a rendered '
                'slice looks blocky next to the pendulum\'s however much noise is '
                'applied. That is sampling, not sharpness; the cell statistics are '
                'unaffected.'),
        },
        'horizon': {'steps': HORIZON, 'seconds': HORIZON / 100.0},
        'termination_thresholds': dict(CUT, theta='inf (periodic)',
                                       source='deterministic dataset_description.json'),
        'plant': {'gravity': 9.8, 'cart_mass': 1.0, 'pole_mass': 0.1,
                  'pole_length': 0.5, 'ctrl_freq': 100, 'pyb_freq': 5000,
                  'cost': 'quadratic',
                  'cost_note': ("cost='quadratic' is REQUIRED: without it goal_reached "
                                'never reaches info, so the env terminates in the right '
                                'place while every label reads 0.'),
                  'damping': 0, 'joint_motor': 'disabled'},
        'sampling': {
            'train': {'kind': 'random', 'n': N_TRAIN, 'bounds': BOUNDS,
                      'rejection': 'states at or beyond a termination threshold dropped'},
            'eval': {'kind': 'the exact deterministic eval states', 'n': N_EVAL,
                     'source': 'deterministic/cartpole_pybullet/eval_states.txt cols 0:4',
                     'note': 'row i indexes the same physical state in both datasets'},
        },
        'data_format': {
            'state_order': ['x', 'theta', 'x_dot', 'theta_dot'],
            'env_state_order': ['x', 'x_dot', 'theta', 'theta_dot'],
            'permutation': [0, 2, 1, 3],
            'train': 'train.npz -- states(float32,4) offsets starts labels seeds',
            'eval': 'roa_labels.txt / eval_states.txt -- 4 state cols + p_success',
            'precision': {'state': 6, 'p_success': 4},
        },
        'reproducibility': {
            'seed_fn': 'rollout_seed(split_id, index, trial), base 20260815',
            'note': ('alpha/beta are excluded from the seed, so levels are paired '
                     'under common random numbers'),
            'code': 'cp_collect.py, reduced by cp_reduce_gauss.py',
        },
        'collection_note': {
            'halk_nodes': ('Amarel main-redhat halk* nodes exit COMPLETED with exit '
                           'code 0 while writing nothing. Excluded in the sbatch files.'),
            'env_defect_fixed': (
                'cartpole._get_reward branched on Cost.SHAPED_DMC and Cost.SHAPED, '
                'neither of which existed in the enum, so every rollout raised '
                'AttributeError before this collection could run at all. Fixed by '
                'adding SHAPED_DMC and removing the SHAPED branches.'),
            'nproc_defect_fixed': (
                'The sbatch scripts sized their pool with $(nproc), which honours '
                'OMP_NUM_THREADS -- pinned to 1 for BLAS a few lines earlier -- so each '
                'whole-node task ran a single worker. Now SLURM_CPUS_ON_NODE.'),
        },
        'deterministic_reference': 'deterministic/cartpole_pybullet',
        'train_statistics': tr,
        'eval_statistics': ev,
    }


def main():
    levels = sys.argv[1:] or ['low', 'med', 'high']
    for lv in levels:
        out_dir = os.path.join(DEST, lv)
        os.makedirs(out_dir, exist_ok=True)
        ev = reduce_eval(lv, out_dir)
        tr = reduce_train(lv, out_dir)
        if ev is None or tr is None:
            print(f'{lv}: MISSING shards (train={tr is not None} eval={ev is not None})')
            continue
        assert abs(tr['alpha'] - ev['alpha']) < 1e-9 and abs(tr['beta'] - ev['beta']) < 1e-9, \
            f'{lv}: train and eval shards disagree on the noise parameters'
        desc = describe(lv, tr, ev)
        for name, payload in (('dataset_description.json', desc),
                              ('train_description.json', {**desc, 'split': 'train'}),
                              ('eval_description.json', {**desc, 'split': 'eval'})):
            with open(os.path.join(out_dir, name), 'w') as fh:
                json.dump(payload, fh, indent=2)
        print(f"{lv}: alpha={ev['alpha']} beta={ev['beta']}  "
              f"mid={ev['fraction_mid']:.4f} (pendulum {PENDULUM_MID[lv]})  "
              f"interior={ev['fraction_interior']:.4f}  mean_p={ev['mean_p_success']:.4f}  "
              f"train={tr['num_trajectories']} trajs / {tr['total_states']} states",
              flush=True)


if __name__ == '__main__':
    main()
