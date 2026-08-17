'''Quad3d noise sweep, SLURM array form: action vs dynamics, refined levels.

One array task per (level, chunk-of-starts). Each task writes its own npz;
`--reduce` aggregates them into the table.

Levels refine what the 1000-start pilot found:
  action   collapses between 22% and 57% of hover thrust (retention 0.972 ->
           0.296), so the grid is dense across that band.
  dynamics degrades gently and is still at 0.787 retention at 30% of weight,
           so the grid extends past body weight to find its 50% point.

Common random numbers: seed is a pure function of (start index, trial) and
excludes the level, so every level sees the same noise stream per start and
level-to-level differences are paired.
'''
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import pybullet as pb

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from generate_quadrotor_3d_trajectories import generate_random_initial_states  # noqa: E402
from safe_control_gym.utils.registration import make  # noqa: E402

N_STARTS = int(os.environ.get('N_STARTS', 5000))
TRIALS = int(os.environ.get('TRIALS', 20))
CHUNKS = int(os.environ.get('CHUNKS', 8))
OUT = os.environ.get('OUT_DIR', os.path.dirname(os.path.abspath(__file__)))
HOR = 1000
BOX_TOL = np.full(12, 0.1)

# hover thrust/motor 0.06615 N ; physical_action_bounds [0.02816, 0.14834]
ACTION_LEVELS = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.022]
# total weight 0.2646 N ; top level is 136% of weight
DYN_LEVELS = [0.0, 0.002, 0.004, 0.007, 0.010, 0.015, 0.022, 0.032, 0.048, 0.070]
LEVELS = ([('action', v) for v in ACTION_LEVELS]
          + [('dynamics', v) for v in DYN_LEVELS])

TI = {'stabilization_goal': [0, 0, 1], 'stabilization_goal_tolerance': 0.05}
BOUNDS = dict(x=1.8, y=1.8, z_min=0.1, z_max=3.0, phi=np.pi, theta=np.pi, psi=np.pi,
              x_dot=3.0, y_dot=3.0, z_dot=3.0, p_body=24.0, q_body=24.0, r_body=24.0)
TH = dict(BOUNDS, phi=np.inf, theta=np.inf, psi=np.inf)
BOX = {0: (-1.8, 1.8), 2: (-1.8, 1.8), 4: (0.1, 3.0), 1: (-3, 3), 3: (-3, 3),
       5: (-3, 3), 9: (-24, 24), 10: (-24, 24), 11: (-24, 24)}

S = np.array(generate_random_initial_states(BOUNDS, N_STARTS, TH, seed=42))


def rollout_seed(idx, trial):
    return int((idx * 7919 + trial * 104729) % (2 ** 31 - 1))


def build(mech, level):
    kw = dict(quad_type=3, task='stabilization', task_info=TI, ctrl_freq=100,
              pyb_freq=5000, gui=False, randomized_init=False, episode_len_sec=1000,
              cost='quadratic', done_on_out_of_bound=True)
    if level > 0:
        kw['disturbances'] = {mech: [{'disturbance_func': 'uniform',
                                      'low': -level, 'high': level}]}
    ef = partial(make, 'quadrotor', **kw)
    env = ef()
    for i, (lo, hi) in BOX.items():
        env.state_space.low[i], env.state_space.high[i] = lo, hi
    ctrl = make('lqr', ef, q_lqr=[1] * 12, r_lqr=[0.1] * 4, discrete_dynamics=True)
    return env, ctrl


def work(args):
    mech, level, lo, hi = args
    env, ctrl = build(mech, level)
    hits = np.zeros(hi - lo, dtype=np.int32)
    for j, idx in enumerate(range(lo, hi)):
        s = S[idx]
        for trial in range(TRIALS):
            env.reset(seed=rollout_seed(idx, trial))
            ctrl.reset()
            pb.resetBasePositionAndOrientation(
                env.DRONE_ID, list(s[0:3]),
                pb.getQuaternionFromEuler(list(s[3:6])), physicsClientId=env.PYB_CLIENT)
            pb.resetBaseVelocity(env.DRONE_ID, list(s[6:9]), list(s[9:12]),
                                 physicsClientId=env.PYB_CLIENT)
            env._update_and_store_kinematic_information()
            obs = env._get_observation()
            info = {'current_step': 0}
            for _ in range(HOR):
                obs, _, term, trunc, info = env.step(ctrl.select_action(obs, info))
                if info.get('goal_reached', False):
                    hits[j] += 1
                    break
                if term or trunc:
                    break
    env.close()
    return lo, hits


def run_task(task_id, nproc):
    level_idx, chunk = divmod(task_id, CHUNKS)
    mech, level = LEVELS[level_idx]
    edges = np.linspace(0, N_STARTS, CHUNKS + 1).astype(int)
    lo, hi = int(edges[chunk]), int(edges[chunk + 1])
    # split the chunk again across the task's own cores
    sub = np.linspace(lo, hi, nproc + 1).astype(int)
    jobs = [(mech, level, int(sub[i]), int(sub[i + 1]))
            for i in range(nproc) if sub[i + 1] > sub[i]]
    print(f'task {task_id}: {mech} {level} starts[{lo}:{hi}] on {len(jobs)} procs',
          flush=True)
    hits = np.zeros(hi - lo, dtype=np.int32)
    with Pool(len(jobs)) as pool:
        for l0, h in pool.imap_unordered(work, jobs):
            hits[l0 - lo:l0 - lo + len(h)] = h
    np.savez(os.path.join(OUT, f'sweep_{task_id:04d}.npz'),
             mech=mech, level=level, lo=lo, hi=hi, hits=hits, trials=TRIALS)
    print(f'task {task_id}: done, {hits.sum()} hits', flush=True)


def reduce_all():
    import glob
    counts = {}
    files = sorted(glob.glob(os.path.join(OUT, 'sweep_*.npz')))
    for f in files:
        d = np.load(f, allow_pickle=True)
        key = (str(d['mech']), float(d['level']))
        c = counts.setdefault(key, np.full(N_STARTS, -1, dtype=np.int32))
        c[int(d['lo']):int(d['hi'])] = d['hits']
    print(f'{len(files)} files, {len(counts)} levels')
    missing = {k: int((v < 0).sum()) for k, v in counts.items() if (v < 0).any()}
    if missing:
        print('INCOMPLETE levels (starts with no data):', missing)
    base = counts[('action', 0.0)] > 0
    n = base.size
    print(f'\ndeterministic successes: {base.sum()}/{n} ({100 * base.mean():.2f}%)')
    print(f'trials per start: {TRIALS}\n')
    print(f'{"mech":9} {"level":>7} {"%ref":>7} {"p(success)":>11} {"retained":>9} '
          f'{"+/-":>6} {"gained":>8}')
    for mech, lvls, ref in [('action', ACTION_LEVELS, 0.06615),
                            ('dynamics', DYN_LEVELS, 0.2646)]:
        for lv in lvls:
            c = counts.get((mech, lv))
            if c is None:
                continue
            ok = c >= 0
            p = c[ok].sum() / (ok.sum() * TRIALS)
            nb = base & ok
            nf = (~base) & ok
            ret = c[nb].sum() / max(nb.sum() * TRIALS, 1)
            gain = c[nf].sum() / max(nf.sum() * TRIALS, 1)
            se = np.sqrt(max(ret * (1 - ret), 1e-12) / max(nb.sum() * TRIALS, 1))
            print(f'{mech:9} {lv:7.3f} {100 * lv / ref:6.1f}% {p:11.4f} {ret:9.3f} '
                  f'{se:6.3f} {gain:8.4f}')


if __name__ == '__main__':
    if '--reduce' in sys.argv:
        reduce_all()
    else:
        # `or`, not a .get default -- the default expression is evaluated
        # eagerly, so sys.argv[1] raises IndexError under sbatch where no
        # positional argument is passed.
        tid = int(os.environ.get('SLURM_ARRAY_TASK_ID') or sys.argv[1])
        nc = len(os.sched_getaffinity(0))
        run_task(tid, nc)
