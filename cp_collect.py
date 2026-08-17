'''Cartpole stochastic collection, matching deterministic/cartpole_pybullet.

WHAT THE SHIPPED STOCHASTIC SET GOT WRONG, AND WHAT THIS FIXES
--------------------------------------------------------------
  control bound   100 N          -> 2000 N   (the description's control_bound)
  success         uniform 0.1 box-> env-native L2 ball, radius 0.05, entry-cut
  termination     x_dot/th_dot 20-> 5.0
  eval states     131,859 random -> the exact 116,242 deterministic states
  baseline        none           -> sigma = 0 included
  state order     env order,          -> file order written directly
                  reordered post-hoc

THE SUCCESS RULE IS NOT WHAT THE DESCRIPTION SAYS
-------------------------------------------------
generation_parameters.termination_conditions.success claims per-channel
tolerances (x < 0.01, others < 0.05) held for 10 consecutive steps. That was
never implemented. Every shipped success ends with ||state|| in [0.0497, 0.0500]
and NOT ONE satisfies |x| < 0.01 -- the signature of first-entry into an L2 ball
of radius 0.05, with no dwell.

Verified by reproducing the deterministic trajectories: labels 300/300, final
states median 4.97e-07 (the 6-decimal storage floor).

Note that labels CANNOT validate a success rule here. The first gate scored
300/300 against the wrong rule, because converging and diverging trajectories
are separated by a wide gap and many rules partition them identically. Only the
final states discriminate.

cost='quadratic' is required or goal_reached never reaches info: the env still
terminates in the right place, so trajectories look perfect while every label
reads 0.

STATE ORDER
-----------
env  [x, x_dot, theta, theta_dot]
file [x, theta, x_dot, theta_dot]      -> permutation [0, 2, 1, 3]
Applied here, at write time, rather than as a post-hoc canonicalization pass.
'''
import argparse
import math
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import pybullet as pb

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from safe_control_gym.utils.registration import make  # noqa: E402

DET = os.environ.get(
    'CP_DET_DIR',
    '/common/users/shared/pracsys/genMoPlan/data_trajectories/'
    'deterministic/cartpole_pybullet')

INF = float('inf')
FORCE = 2000.0
HORIZON = 1000
TOL = 0.05
N_EVAL = 116_242
N_TRAIN = int(os.environ.get('N_TRAIN', 116_242))
BASE_SEED = 20260815
TRAIN_SPLIT_ID, EVAL_SPLIT_ID = 0, 1
ENV2FILE = [0, 2, 1, 3]

# description: initial_state_bounds, and the rejection filter that turned the
# raw 25x21x13x21 grid into 116,242 states.
BOUNDS = dict(x=6.0, x_dot=5.0, theta=math.pi, theta_dot=5.0)
CUT = dict(x=6.0, x_dot=5.0, theta_dot=5.0)      # theta unbounded (periodic)

ARGS = None


def rollout_seed(split_id, index, trial):
    return int((BASE_SEED + split_id * 1_000_003 + index * 7919
                + trial * 104_729) % (2 ** 31 - 1))


def _scratch():
    base = os.environ.get('SLURM_TMPDIR') or '/tmp'
    d = os.path.join(base, f'cpc-{os.getpid()}')
    os.makedirs(d, exist_ok=True)
    return d


def build(sigma, alpha=None, beta=None):
    '''Env + LQR at one noise level.

    ``sigma`` selects the original uniform family, ``U(-sigma, sigma)`` on the
    commanded cart force. ``alpha``/``beta`` select the gaussian_signal family,
    ``w ~ Normal(0, alpha + beta*|u|)``, and are mutually exclusive with it.

    Both are applied PRE-saturation, and on this system that is not a choice
    with consequences: the LQR's demand is a median 0.27 N against a 2000 N
    bound and never saturates in 16,494 measured steps, so clip(u + w) and
    clip(u) + w are the same function here. The pendulum's placement
    distinction has nothing to bite on.
    '''
    if (alpha is not None or beta is not None) and sigma > 0:
        raise ValueError('uniform sigma and gaussian alpha/beta are different '
                         'mechanisms; pass one')
    kw = dict(task='stabilization', ctrl_freq=100, pyb_freq=5000, gui=False,
              output_dir=_scratch(), randomized_init=False,
              randomized_inertial_prop=False, action_scale=FORCE,
              episode_len_sec=math.ceil(HORIZON / 100) + 1,
              terminate_on_goal=True, cost='quadratic',
              task_info={'stabilization_goal': [0],
                         'stabilization_goal_tolerance': TOL},
              x_dot_limit=INF, theta_dot_limit=INF, obs_wrap_angle=True)
    if alpha is not None or beta is not None:
        alpha, beta = float(alpha or 0.0), float(beta or 0.0)
        if alpha or beta:
            kw['disturbances'] = {'action': [{'disturbance_func': 'signal_dependent',
                                              'alpha': alpha, 'beta': beta}]}
    elif sigma > 0:
        kw['disturbances'] = {'action': [{'disturbance_func': 'uniform',
                                          'low': -sigma, 'high': sigma}]}
    ef = partial(make, 'cartpole', **kw)
    env = ef()
    env.x_threshold = CUT['x']
    env.x_dot_threshold = CUT['x_dot']
    env.theta_threshold_radians = INF
    env.theta_dot_threshold = CUT['theta_dot']
    return env, make('lqr', ef, q_lqr=[1, 1, 1, 1], r_lqr=[0.1],
                     discrete_dynamics=True)


def roll(env, ctrl, s_file, seed, keep=False):
    '''s_file is FILE order [x, theta, x_dot, theta_dot].'''
    x, theta, x_dot, theta_dot = s_file
    env.reset(seed=int(seed))
    # The start MUST go through the joints: the state lives in PyBullet and
    # every step reads it back, so assigning env.state alone is discarded.
    pb.resetJointState(env.CARTPOLE_ID, 0, targetValue=x, targetVelocity=x_dot,
                       physicsClientId=env.PYB_CLIENT)
    pb.resetJointState(env.CARTPOLE_ID, 1, targetValue=theta,
                       targetVelocity=theta_dot, physicsClientId=env.PYB_CLIENT)
    env.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
    obs = env._get_observation()
    info = {'current_step': 0}
    traj = [env.state[ENV2FILE].copy()] if keep else None
    for _ in range(HORIZON):
        obs, _, term, trunc, info = env.step(ctrl.select_action(obs, info))
        if keep:
            traj.append(env.state[ENV2FILE].copy())
        if info.get('goal_reached', False):
            return True, traj
        if term or trunc:
            return False, traj
    return False, traj


def train_starts():
    '''Random within the sampling bounds, rejecting states at or beyond a
    termination threshold -- the same filter the deterministic grid used.'''
    rng = np.random.default_rng(BASE_SEED)
    out = np.empty((N_TRAIN, 4))
    n = 0
    while n < N_TRAIN:
        m = (N_TRAIN - n) * 2
        c = np.column_stack([
            rng.uniform(-BOUNDS['x'], BOUNDS['x'], m),
            rng.uniform(-BOUNDS['theta'], BOUNDS['theta'], m),
            rng.uniform(-BOUNDS['x_dot'], BOUNDS['x_dot'], m),
            rng.uniform(-BOUNDS['theta_dot'], BOUNDS['theta_dot'], m)])
        ok = ((np.abs(c[:, 0]) < CUT['x']) & (np.abs(c[:, 2]) < CUT['x_dot'])
              & (np.abs(c[:, 3]) < CUT['theta_dot']))
        c = c[ok][:N_TRAIN - n]
        out[n:n + len(c)] = c
        n += len(c)
    return out


def eval_starts():
    '''The exact deterministic eval states, columns 0:4 (file order).'''
    r = np.loadtxt(DET + '/eval_states.txt', delimiter=',')
    return r[:, 0:4], r[:, 8].astype(np.int8)


def _init(a):
    global ARGS
    ARGS = a


def _range(rng_):
    lo, hi = rng_
    if ARGS.split == 'train':
        S = train_starts()[lo:hi]
        det = np.full(len(S), -1, np.int8)
    else:
        S, det = eval_starts()
        S, det = S[lo:hi], det[lo:hi]
    gaussian = getattr(ARGS, 'alpha', None) is not None or getattr(ARGS, 'beta', None) is not None
    noiseless = (ARGS.level == 0) if not gaussian else not (ARGS.alpha or ARGS.beta)
    trials = 1 if noiseless else ARGS.trials
    env, ctrl = (build(0.0, alpha=ARGS.alpha, beta=ARGS.beta) if gaussian
                 else build(ARGS.level))
    if ARGS.split == 'train':
        states, lengths, labels, seeds = [], [], [], []
        for i in range(len(S)):
            sd = rollout_seed(TRAIN_SPLIT_ID, lo + i, 0)
            ok, traj = roll(env, ctrl, S[i], sd, keep=True)
            states.append(np.asarray(traj, dtype=np.float32))
            lengths.append(len(traj))
            labels.append(ok)
            seeds.append(sd)
        env.close()
        return (lo, np.concatenate(states), np.asarray(lengths, np.int64),
                np.asarray(labels, np.uint8), np.asarray(seeds, np.int64), S, det)
    hits = np.zeros(len(S), np.int32)
    for i in range(len(S)):
        for k in range(trials):
            hits[i] += roll(env, ctrl, S[i], rollout_seed(EVAL_SPLIT_ID, lo + i, k))[0]
    env.close()
    return lo, S, hits, det, trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'eval'], required=True)
    ap.add_argument('--level', type=float, default=0.0,
                    help='uniform family: half-width sigma on the commanded cart force')
    ap.add_argument('--alpha', type=float, default=None,
                    help='gaussian_signal family: the sigma floor, in N. Needs --beta.')
    ap.add_argument('--beta', type=float, default=None,
                    help='gaussian_signal family: effort-proportional term, so '
                         'sigma = alpha + beta*|u|. Needs --alpha. On cartpole |u| is '
                         'heavily skewed (p50 0.27 N, p99 28.9), so beta near 1 leaves '
                         'the median untouched and multiplies the tail.')
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if (args.alpha is None) != (args.beta is None):
        ap.error('--alpha and --beta must be given together')
    if args.alpha is not None and args.level:
        ap.error('--level (uniform) and --alpha/--beta (gaussian) are different '
                 'mechanisms; pass one')

    if os.path.exists(args.out):
        print(f'{args.out} exists, skipping', flush=True)
        return

    total = N_TRAIN if args.split == 'train' else N_EVAL
    edges = np.linspace(0, total, args.nshards + 1).astype(int)
    lo, hi = int(edges[args.shard]), int(edges[args.shard + 1])
    nproc = min(int(os.environ.get('NPROC', 0)) or 10 ** 6,
                len(os.sched_getaffinity(0)))
    sub = np.linspace(lo, hi, nproc + 1).astype(int)
    ranges = [(int(sub[i]), int(sub[i + 1])) for i in range(nproc)
              if sub[i + 1] > sub[i]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tag = (f'alpha={args.alpha} beta={args.beta}' if args.alpha is not None
           else f'sigma={args.level}')
    print(f'{args.split} {tag} shard {args.shard}/{args.nshards} '
          f'[{lo}:{hi}] on {len(ranges)} procs', flush=True)

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
