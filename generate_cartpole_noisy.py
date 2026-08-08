#!/usr/bin/env python3
'''Collect cartpole datasets under action noise, with a train/eval split.

`generate_cartpole_trajectories.py` is the older single-pass collector: no
split, no noise, no probability field. Rather than grow it, this reuses the
pendulum collector's machinery -- `rollout_seed`, the atomic writers, the
sharded eval and its merge -- and supplies only the cartpole plant. Those
helpers are imported, not copied, so the two systems cannot drift apart.

REGIME (measured 2026-08-06, see docs/superpowers/specs/):

  plant   ideal frictionless cartpole. Damping is zero and the joint motor is
          disabled (cartpole.py:368,370) BY DESIGN -- the LQR gain comes from
          the symbolic model of exactly that plant, so touching the physics
          would make the controller mismatched. Untouched here.
  force   100 N. p saturates at 100 (measured 0.176 at both 100 N and 2000 N
          under the shipped cut), so the shipped 2000 N buys nothing and is
          2x the URDF's own effort="1000".
  cut     x 6.0, x_dot 20, theta_dot 20, theta unbounded. The shipped cut puts
          x_dot at 5.0, which EQUALS the sampling bound: 60% of rollouts ended
          on it with a median of 4 control steps, before the controller could
          act, because one saturated step at 2000 N adds 18 m/s. Relaxing it to
          20 raises the deterministic rate from 0.180 to 0.295 -- those starts
          do stabilise, they were just cut mid-recovery.
  rule    every channel within 0.1, NO dwell, rollout stops at first entry.
          Same choice as the pendulum: a failure can then never end inside the
          box, so the label is a function of the terminal state.

The four thresholds are set as ATTRIBUTES, not constructor kwargs -- CartPole
swallows them in **kwargs and silently keeps its defaults. See
configs/system/README.md.
'''
import argparse
import json
import math
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pybullet as pb

from generate_inverted_pendulum_trajectories import (EVAL_SPLIT_ID, TRAIN_SPLIT_ID, atomic_savez,
                                                     atomic_write_text, get_available_cpus, merge_eval_shards,
                                                     rollout_seed, shard_path)
from safe_control_gym.utils.registration import make

INF = float('inf')
BOX_TOL = np.array([0.1, 0.1, 0.1, 0.1])      # x, x_dot, theta, theta_dot
FORCE = 100.0
CUT = {'x': 6.0, 'x_dot': 20.0, 'theta_dot': 20.0}
SAMPLE = {'x': 6.0, 'x_dot': 5.0, 'theta': math.pi, 'theta_dot': 5.0}
DATA_ROOT = '/common/users/shared/pracsys/genMoPlan/data_trajectories'
STATE_ORDER = ['x', 'x_dot', 'theta', 'theta_dot']


def default_output_dir(sigma):
    return os.path.join(DATA_ROOT, 'stochastic', 'cartpole', 'noisy_action', 'lqr',
                        f'sigma_{sigma:05.1f}')


def grid_states(resolution):
    '''The eval grid, filtered to starts that are not already outside the cut.

    Same construction as generate_cartpole_trajectories.py, but filtered on THIS
    regime's cut rather than the shipped one -- a start beyond the avoid set is
    not a question worth asking.
    '''
    axes = [np.arange(-SAMPLE[k], SAMPLE[k] + resolution, resolution) for k in
            ['x', 'x_dot', 'theta', 'theta_dot']]
    g = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 4)
    keep = ((np.abs(g[:, 0]) < CUT['x']) & (np.abs(g[:, 1]) < CUT['x_dot'])
            & (np.abs(g[:, 3]) < CUT['theta_dot']))
    return g[keep]


def random_states(n, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(-SAMPLE['x'], SAMPLE['x'], n),
                     rng.uniform(-SAMPLE['x_dot'], SAMPLE['x_dot'], n),
                     rng.uniform(-SAMPLE['theta'], SAMPLE['theta'], n),
                     rng.uniform(-SAMPLE['theta_dot'], SAMPLE['theta_dot'], n)], 1)


def _scratch_dir():
    '''Node-local directory for the env's per-reset URDF write.

    cartpole.py:357-365 writes a URDF, loads it into PyBullet and deletes it on
    EVERY reset, into `output_dir`, which defaults to os.getcwd(). On a cluster
    the cwd is shared NFS, so every rollout does three network filesystem
    operations. Measured: ~4 min/batch when this lands on local disk against
    60-180 min when it lands on NFS -- a 15-45x collapse. TMPDIR is set per-job
    by SLURM on the compute node.
    '''
    # /tmp first, NOT $TMPDIR: TMPDIR is set to shared storage on some hosts
    # here, which is exactly what we are avoiding. SLURM_TMPDIR, where set, is
    # node-local by definition. The file is a few KB and transient.
    base = os.environ.get('SLURM_TMPDIR') or '/tmp'
    d = os.path.join(base, f'scg-{os.getpid()}')
    os.makedirs(d, exist_ok=True)
    return d


def build(sigma, horizon):
    kw = dict(task='stabilization', ctrl_freq=100, pyb_freq=5000, gui=False,
              output_dir=_scratch_dir(),
              randomized_init=False, randomized_inertial_prop=False,
              action_scale=FORCE, episode_len_sec=math.ceil(horizon / 100) + 1,
              terminate_on_goal=False, x_dot_limit=INF, theta_dot_limit=INF,
              obs_wrap_angle=True)
    if sigma > 0:
        kw['disturbances'] = {'action': [{'disturbance_func': 'uniform',
                                          'low': -sigma, 'high': sigma}]}
    env_func = partial(make, 'cartpole', **kw)
    env = env_func()
    env.x_threshold = CUT['x']
    env.x_dot_threshold = CUT['x_dot']
    env.theta_threshold_radians = INF
    env.theta_dot_threshold = CUT['theta_dot']
    ctrl = make('lqr', env_func, q_lqr=[1, 1, 1, 1], r_lqr=[0.1], discrete_dynamics=True)
    return env, ctrl


def run_trajectory(env, ctrl, init_state, horizon, seed):
    '''Roll out, stopping at the first state inside the box.

    The start MUST be injected through the joints: the cartpole's state lives in
    PyBullet and every step reads it back, so assigning env.state alone is
    silently discarded and the rollout starts from the default upright state.
    '''
    env.reset(seed=seed)
    ctrl.reset()
    x, x_dot, theta, theta_dot = (float(v) for v in init_state)
    pb.resetJointState(env.CARTPOLE_ID, jointIndex=0, targetValue=x,
                       targetVelocity=x_dot, physicsClientId=env.PYB_CLIENT)
    pb.resetJointState(env.CARTPOLE_ID, jointIndex=1, targetValue=theta,
                       targetVelocity=theta_dot, physicsClientId=env.PYB_CLIENT)
    env.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
    obs = env._get_observation()
    info = {'current_step': 0}
    traj = [env.state.copy()]
    for _ in range(horizon):
        obs, _, term, trunc, info = env.step(ctrl.select_action(obs, info))
        traj.append(env.state.copy())
        if np.all(np.abs(env.state) < BOX_TOL):
            return np.array(traj), True
        if term or trunc:
            break
    return np.array(traj), False


def _eval_worker(task):
    chunk, sigma, horizon, base_seed, batch = task
    env, ctrl = build(sigma, horizon)
    idx, out = [], []
    try:
        for i, s in chunk:
            _, ok = run_trajectory(env, ctrl, s, horizon,
                                   rollout_seed(base_seed, EVAL_SPLIT_ID, i, batch))
            idx.append(i)
            out.append(int(ok))
    finally:
        env.close()
    return np.array(idx, np.int64), np.array(out, np.int64)


def _train_worker(task):
    chunk, sigma, horizon, base_seed = task
    env, ctrl = build(sigma, horizon)
    idx, states, lengths, labels, seeds = [], [], [], [], []
    try:
        for i, s in chunk:
            sd = rollout_seed(base_seed, TRAIN_SPLIT_ID, i)
            traj, ok = run_trajectory(env, ctrl, s, horizon, sd)
            idx.append(i)
            states.append(traj.astype(np.float32))
            lengths.append(len(traj))
            labels.append(int(ok))
            seeds.append(sd)
    finally:
        env.close()
    return idx, states, lengths, labels, seeds


def describe(split, sigma, horizon, seed, n, extra):
    d = {'dataset_name': f'CartPole LQR under action noise (sigma={sigma} N), {split} split',
         'split': split, 'controller': 'lqr', 'system': 'cartpole',
         'action_noise': sigma, 'action_noise_unit': 'N',
         'noise_mechanism': ('uniform on the commanded cart force, pre-saturation'
                             if sigma > 0 else 'none'),
         'control_bound': FORCE, 'fraction_of_control_bound': sigma / FORCE,
         'state_order': STATE_ORDER, 'ctrl_freq': 100, 'pyb_freq': 5000,
         'steps_per_control': 50, 'horizon_steps': horizon, 'seed': seed,
         'termination_cut': {**CUT, 'theta': 'unbounded',
                             'note': ('RELAXED from the shipped x_dot/theta_dot of 5.0, which '
                                      'equals the sampling bound and ended 60% of rollouts in a '
                                      'median of 4 control steps')},
         'success_rule': {'kind': 'per_channel_box_entry', 'tol': BOX_TOL.tolist(),
                          'hold_steps': 1,
                          'cut': 'rollout stops at, and stores, the first state inside the box'},
         'plant': {'damping': 0, 'joint_motor': 'disabled',
                   'note': 'ideal frictionless cartpole, untouched -- the LQR gain is '
                           'derived from the symbolic model of exactly this plant'},
         'num_items': n}
    d.update(extra)
    return d


def collect_train(sigma, out_dir, num_trajs, seed, horizon, workers):
    os.makedirs(out_dir, exist_ok=True)
    starts = random_states(num_trajs, seed)
    items = list(enumerate(starts))
    c = max(1, len(items) // (workers * 4))
    tasks = [(items[i:i + c], sigma, horizon, seed) for i in range(0, len(items), c)]
    allst = [None] * num_trajs
    lengths = np.zeros(num_trajs, np.int64)
    labels = np.zeros(num_trajs, np.uint8)
    seeds = np.zeros(num_trajs, np.int64)
    with Pool(workers) as pool:
        for idx, st, ln, lb, sd in pool.imap_unordered(_train_worker, tasks):
            for j, i in enumerate(idx):
                allst[i], lengths[i], labels[i], seeds[i] = st[j], ln[j], lb[j], sd[j]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    atomic_savez(os.path.join(out_dir, 'train.npz'),
                 states=np.concatenate(allst, 0), offsets=offsets, starts=starts,
                 labels=labels, seeds=seeds)
    desc = describe('train', sigma, horizon, seed, num_trajs,
                    {'success_rate': float(labels.mean()),
                     'mean_length': float(lengths.mean())})
    atomic_write_text(os.path.join(out_dir, 'train_description.json'),
                      json.dumps(desc, indent=2, default=str))
    return desc


def collect_eval(sigma, out_dir, seed, horizon, resolution, workers,
                 batch_offset, batch_count):
    os.makedirs(out_dir, exist_ok=True)
    grid = grid_states(resolution)
    items = list(enumerate(grid))
    c = max(1, len(items) // (workers * 4))
    chunks = [items[i:i + c] for i in range(0, len(items), c)]
    lo, hi = batch_offset, batch_offset + batch_count
    sp = shard_path(out_dir, lo, hi)
    succ = np.zeros(len(grid), np.int64)
    trials = np.zeros(len(grid), np.int64)
    n = lo
    with Pool(workers) as pool:
        while n < hi:
            tasks = [(ch, sigma, horizon, seed, n) for ch in chunks]
            batch = np.zeros(len(grid), np.int64)
            for idx, vals in pool.imap_unordered(_eval_worker, tasks):
                batch[idx] = vals
            succ += batch
            trials += 1
            n += 1
            atomic_savez(sp, successes=succ.astype(np.int32),
                         trials=trials.astype(np.int32),
                         batch_lo=np.int64(lo), batch_hi=np.int64(n))
            print(f'[eval sigma={sigma} shard {lo}-{hi}] batch {n}', flush=True)
    return {'cells': len(grid), 'batches': n - lo}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sigma', type=float, required=True, help='action noise half-width, N')
    ap.add_argument('--split', choices=['train', 'eval'], required=True)
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--num_trajs', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--horizon', type=int, default=1000)
    ap.add_argument('--resolution', type=float, default=0.5)
    ap.add_argument('--num_workers', type=int, default=None)
    ap.add_argument('--batch_offset', type=int, default=0)
    ap.add_argument('--batch_count', type=int, default=1)
    ap.add_argument('--merge_eval_shards', action='store_true')
    args = ap.parse_args()

    out = args.output_dir or default_output_dir(args.sigma)
    workers = args.num_workers or get_available_cpus()

    if args.merge_eval_shards:
        grid = grid_states(args.resolution)
        desc = describe('eval', args.sigma, args.horizon, args.seed, len(grid), {})
        stats = merge_eval_shards(out, grid, np.array([]), np.array([]), desc)
        print(f"merged {stats['shards']} shards, {stats['n_batches']} batches, "
              f"mean p {stats['success_rate']:.4f}, mean SE {stats['mean_se']:.5f}")
        return

    if args.split == 'train':
        d = collect_train(args.sigma, out, args.num_trajs, args.seed, args.horizon, workers)
        print(f"train sigma={args.sigma}: {d['num_items']} trajectories, "
              f"p={d['success_rate']:.4f}, mean length {d['mean_length']:.1f}")
    else:
        s = collect_eval(args.sigma, out, args.seed, args.horizon, args.resolution,
                         workers, args.batch_offset, args.batch_count)
        print(f"eval sigma={args.sigma}: {s['cells']} cells, {s['batches']} batches -> {out}")


if __name__ == '__main__':
    main()
