'''Quad3d stochastic collection under an unmatched external force disturbance.

Produces, per noise level, a train split (trajectories, binary label) and an
eval split (p_success over K trials at fixed start states).

WHAT THIS MATCHES, AND WHERE IT CANNOT
--------------------------------------
Every plant setting is taken from the shipped deterministic dataset -- its
dataset_description.json where that is explicit, and the dataset-era collector
otherwise. At level 0 this reproduces the shipped labels to 98-99%, not 100%.
The residual is not a config error: the code that generated the shipped data is
not in this repo in runnable form (its env_func omits task_info, which the 3D
branch indexes, so it raises IndexError on construction). The disagreements are
chaos amplification over ~500-step trajectories plus boundary ties, measured as
1 tie at 0.0499 against the 0.05 threshold and 6 divergences out of 400.

THE TWO INJECTION PATHS -- easiest thing here to get backwards
--------------------------------------------------------------
Angular velocity enters PyBullet in the WORLD frame, but the env stores it as
`Rbo @ ang_v`, i.e. a BODY rate. That makes the two splits asymmetric:

  train starts come from the sampler. The original collector passed those
  body-labelled values straight to resetBaseVelocity as world, so the shipped
  data was generated with that conflation baked in. Reproducing it means
  repeating it. Converting instead drops label agreement 393/400 -> 367/400.

  eval starts are read back from eval_states.txt. Those were written by the env
  as true body rates, so replaying one needs world = R @ body. Skipping the
  conversion drops 148/150 -> 138/150 and inflates final-state error 29x.

Both are correct. They differ because one is a sampler output and the other is
an env output.

HORIZON IS LOAD-BEARING
-----------------------
Labels are BOUNDED-TIME reach probabilities at H=1000 steps (10 s). Under this
disturbance the controller mostly still reaches the goal -- it just takes far
longer. Given the deterministic run's own 100,000-step allowance, success at
f=0.072 is ~0.24, statistically identical to f=0's ~0.25; at H=1000 it reads
0.058. Roughly 15% of f=0.072 rollouts would succeed with unlimited time. The
falling success rate is therefore mostly deadline, not failure, and the number
is not comparable to an asymptotic reach probability.
'''
import argparse
import os
from functools import partial

import numpy as np
import pybullet as pb

from safe_control_gym.utils.registration import make

# Amarel is a separate filesystem and cannot see the CS shared mount, so the
# two files the eval split reads (eval_states.txt, trajectory_labels.txt) are
# copied there and this is pointed at the copy.
DET = os.environ.get(
    'Q3_DET_DIR',
    '/common/users/shared/pracsys/genMoPlan/data_trajectories/'
    'deterministic/quadrotor3D_lqr')

HORIZON = 1000
N_TRAIN = 800_000          # all of them; index-aligned with the shipped set
SAMPLER_N = 800_000
STARTS_CACHE = os.environ.get('Q3_STARTS_CACHE', '')
TRAIN_SPLIT_ID, EVAL_SPLIT_ID = 0, 1

# From dataset_description.json: generation_parameters.initial_state_bounds
BOUNDS = dict(x=1.8, y=1.8, z_min=0.1, z_max=3.0, phi=np.pi, theta=np.pi,
              psi=np.pi, x_dot=3.0, y_dot=3.0, z_dot=3.0,
              p_body=24.0, q_body=24.0, r_body=24.0)
# Sampler rejection thresholds. Angles are unbounded: theta wraps, so there is
# no such thing as out-of-range attitude.
SAMPLER_TH = dict(BOUNDS, phi=np.inf, theta=np.inf, psi=np.inf)

# From generation_parameters.termination_thresholds, applied to state_space
# AFTER construction -- the env swallows these as kwargs. Indices are the env's
# interleaved 12-D order: x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r
STATE_BOUNDS = {0: (-1.8, 1.8), 1: (-3.0, 3.0), 2: (-1.8, 1.8), 3: (-3.0, 3.0),
                4: (0.1, 3.0), 5: (-3.0, 3.0),
                9: (-24.0, 24.0), 10: (-24.0, 24.0), 11: (-24.0, 24.0)}

TASK_INFO = {'stabilization_goal': [0, 0, 1], 'stabilization_goal_tolerance': 0.05}


def rollout_seed(base, split_id, index, trial):
    '''Pure function of the coordinates -- deliberately excludes the noise level.

    Two consequences. A resumed shard draws exactly what an uninterrupted run
    would have drawn. And every level sees the same noise stream per (start,
    trial), so levels are paired and level-to-level differences carry far less
    variance than the individual estimates.
    '''
    return int((base + split_id * 1_000_003 + index * 7919 + trial * 104_729)
               % (2 ** 31 - 1))


def build(level, mechanism='dynamics'):
    kw = dict(quad_type=3, task='stabilization', task_info=TASK_INFO,
              ctrl_freq=100, pyb_freq=5000, gui=False, randomized_init=False,
              episode_len_sec=1000, cost='quadratic', done_on_out_of_bound=True)
    if level > 0:
        kw['disturbances'] = {mechanism: [{'disturbance_func': 'uniform',
                                           'low': -level, 'high': level}]}
    env_func = partial(make, 'quadrotor', **kw)
    env = env_func()
    for i, (lo, hi) in STATE_BOUNDS.items():
        env.state_space.low[i], env.state_space.high[i] = lo, hi
    ctrl = make('lqr', env_func, q_lqr=[1] * 12, r_lqr=[0.1] * 4,
                discrete_dynamics=True)
    return env, ctrl


def to_row13(state12):
    '''env 12-D (interleaved, Euler) -> shipped 13-D (grouped, quaternion).

    Scalar-first and canonicalised qw >= 0, since q and -q are the same rotation
    and the shipped data picks the positive-scalar representative.
    '''
    x, xd, y, yd, z, zd, phi, th, psi, p_, q_, r_ = state12
    qx, qy, qz, qw = pb.getQuaternionFromEuler([phi, th, psi])
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    return np.array([x, y, z, qw, qx, qy, qz, xd, yd, zd, p_, q_, r_])


def inject_sampler(env, state12):
    '''Train path: repeat the original collector's body-as-world conflation.

    NOTE the ordering. generate_random_initial_states returns GROUPED order --
    [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r] -- which is NOT the
    env's interleaved state order [x, x_dot, y, y_dot, z, z_dot, ...]. The
    dataset-era collector unpacks it with interleaved variable NAMES, which is a
    mislabelling in that source, not a different behaviour. Reading it as
    interleaved feeds theta in as z and psi as z_dot: every rollout terminates
    on step 1 and the success rate goes to zero.
    '''
    x, y, z, phi, th, psi, xd, yd, zd, p_, q_, r_ = state12
    pb.resetBasePositionAndOrientation(
        env.DRONE_ID, [x, y, z], pb.getQuaternionFromEuler([phi, th, psi]),
        physicsClientId=env.PYB_CLIENT)
    pb.resetBaseVelocity(env.DRONE_ID, [xd, yd, zd], [p_, q_, r_],
                         physicsClientId=env.PYB_CLIENT)
    env._update_and_store_kinematic_information()
    return env._get_observation()


def inject_stored(env, row13):
    '''Eval path: stored rates are true body rates, so convert to world.'''
    qw, qx, qy, qz = row13[3:7]
    quat = [qx, qy, qz, qw]
    pb.resetBasePositionAndOrientation(env.DRONE_ID, list(row13[0:3]), quat,
                                       physicsClientId=env.PYB_CLIENT)
    rot = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
    pb.resetBaseVelocity(env.DRONE_ID, list(row13[7:10]),
                         list(rot @ np.asarray(row13[10:13])),
                         physicsClientId=env.PYB_CLIENT)
    env._update_and_store_kinematic_information()
    return env._get_observation()


def run(env, ctrl, obs, keep_states):
    '''One rollout. Stops at first goal entry (terminate_on_goal), on leaving
    the bounds, or at the horizon. The entry state IS included as the last
    stored state -- that is what keeps "terminal state in the goal set" and
    "label 1" the same statement, which the terminal-state consumer requires.
    '''
    info = {'current_step': 0}
    traj = [to_row13(env.state)] if keep_states else None
    success = False
    for _ in range(HORIZON):
        obs, _, terminated, truncated, info = env.step(ctrl.select_action(obs, info))
        if keep_states:
            traj.append(to_row13(env.state))
        if info.get('goal_reached', False):
            success = True
            break
        if terminated or truncated:
            break
    return success, traj


def sampler_starts():
    '''Draw the full 800k then slice, so our index i is the shipped index i.

    Regenerating rather than parsing sequence_*.txt removes the 6-decimal
    rounding and the quaternion round trip; verified to reproduce the shipped
    starts to 5e-7, against a 5e-7 storage floor.
    '''
    # The draw costs 17 s, which is wasted once per task across hundreds of
    # tasks, so cache it. mmap keeps each worker from faulting in all 77 MB.
    if STARTS_CACHE and os.path.exists(STARTS_CACHE):
        return np.load(STARTS_CACHE, mmap_mode='r')
    from generate_quadrotor_3d_trajectories import generate_random_initial_states
    allstates = generate_random_initial_states(BOUNDS, SAMPLER_N, SAMPLER_TH, seed=42)
    out = np.asarray(allstates[:N_TRAIN])
    if STARTS_CACHE:
        # np.save appends .npy, so name the temp accordingly or the rename
        # target will not exist.
        tmp = STARTS_CACHE + f'.tmp{os.getpid()}.npy'
        np.save(tmp, out)
        os.replace(tmp, STARTS_CACHE)   # atomic: concurrent tasks never see a partial file
    return out


def eval_starts(lo, hi):
    '''Rows [lo:hi) of the shipped eval_states.txt, first 13 columns.'''
    rows = np.loadtxt(DET + '/eval_states.txt', delimiter=',',
                      skiprows=lo, max_rows=hi - lo, ndmin=2)
    return rows[:, 0:13], rows[:, 26].astype(np.int8)


def shard_train(args, lo, hi):
    starts = np.asarray(sampler_starts()[lo:hi])
    env, ctrl = build(args.level, args.mechanism)
    states, offsets, labels, seeds = [], [0], [], []
    for i in range(len(starts)):
        seed = rollout_seed(args.base_seed, TRAIN_SPLIT_ID, lo + i, 0)
        env.reset(seed=seed)
        ctrl.reset()
        obs = inject_sampler(env, starts[i])
        ok, traj = run(env, ctrl, obs, keep_states=True)
        states.append(np.asarray(traj, dtype=np.float32))
        offsets.append(offsets[-1] + len(traj))
        labels.append(ok)
        seeds.append(seed)
    env.close()
    np.savez(args.out,
             states=np.concatenate(states), offsets=np.asarray(offsets, np.int64),
             starts=starts.astype(np.float64),
             labels=np.asarray(labels, np.uint8), seeds=np.asarray(seeds, np.int64),
             lo=lo, hi=hi, level=args.level, mechanism=args.mechanism)
    return int(np.sum(labels)), len(labels)


def shard_eval(args, lo, hi):
    starts, det_labels = eval_starts(lo, hi)
    trials = 1 if args.level == 0 else args.trials   # no noise -> all trials identical
    env, ctrl = build(args.level, args.mechanism)
    hits = np.zeros(len(starts), dtype=np.int32)
    for i in range(len(starts)):
        for k in range(trials):
            env.reset(seed=rollout_seed(args.base_seed, EVAL_SPLIT_ID, lo + i, k))
            ctrl.reset()
            obs = inject_stored(env, starts[i])
            ok, _ = run(env, ctrl, obs, keep_states=False)
            hits[i] += ok
    env.close()
    np.savez(args.out, starts=starts, hits=hits, trials=trials,
             det_labels=det_labels, lo=lo, hi=hi,
             level=args.level, mechanism=args.mechanism)
    return int(hits.sum()), len(starts) * trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'eval'], required=True)
    ap.add_argument('--level', type=float, required=True)
    ap.add_argument('--mechanism', default='dynamics',
                    choices=['dynamics', 'action'])
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--base_seed', type=int, default=20260813)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # Idempotent: a completed shard is never redone, so resubmitting a partly
    # failed array costs only the missing work.
    if os.path.exists(args.out):
        print(f'{args.out} exists, skipping')
        return

    total = N_TRAIN if args.split == 'train' else 1_000_000
    edges = np.linspace(0, total, args.nshards + 1).astype(int)
    lo, hi = int(edges[args.shard]), int(edges[args.shard + 1])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    fn = shard_train if args.split == 'train' else shard_eval
    got, n = fn(args, lo, hi)
    print(f'{args.split} level={args.level} shard {args.shard}/{args.nshards} '
          f'[{lo}:{hi}] -> {got}/{n} successes', flush=True)


if __name__ == '__main__':
    main()
