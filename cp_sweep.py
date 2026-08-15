'''Cartpole noise sweep under the CORRECTED config.

The shipped stochastic cartpole was collected against a 100 N control bound; the
deterministic set uses 2000 N. Its levels {15,20,30,40} are therefore ~1-2% of
the real actuator range and would barely register, so the range has to be found
from scratch rather than rescaled.

Success is the env's native L2 ball at 0.05 with entry-cut, verified against
deterministic/cartpole_pybullet: labels 300/300, final states median 4.97e-07
(the 6-decimal storage floor). NOT the per-channel tolerance + 10-step dwell the
description claims -- that text was never implemented, and labels alone cannot
tell the difference.

Common random numbers: seed excludes the level, so levels are paired.
'''
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
N_STATES = int(os.environ.get('N_STATES', 4000))
TRIALS = int(os.environ.get('TRIALS', 5))
NPROC = int(os.environ.get('NPROC', 32))
BASE = 20260815

# Wide: 0 to 80% of the 2000 N bound. No prior to anchor on.
# Refined: the coarse pass showed retention 0.708 at 10 N and 0.000 by 50 N.
# The 2000 N bound is nearly irrelevant -- the LQR commands small forces near
# the goal, so noise swamps the command long before saturation matters.
LEVELS = [0.0, 2.0, 4.0, 6.0, 8.0, 11.0, 14.0, 18.0, 23.0, 30.0]

_rows = np.loadtxt(DET + '/eval_states.txt', delimiter=',')
_rng = np.random.default_rng(0)
PICK = np.sort(_rng.choice(len(_rows), N_STATES, replace=False))
STATES = _rows[PICK, 0:4]          # file order [x, theta, x_dot, theta_dot]
DET_LABELS = _rows[PICK, 8].astype(int)
del _rows


def rollout_seed(idx, trial):
    return int((BASE + idx * 7919 + trial * 104_729) % (2 ** 31 - 1))


def _scratch():
    base = os.environ.get('SLURM_TMPDIR') or '/tmp'
    d = os.path.join(base, f'cps-{os.getpid()}')
    os.makedirs(d, exist_ok=True)
    return d


def build(sigma):
    kw = dict(task='stabilization', ctrl_freq=100, pyb_freq=5000, gui=False,
              output_dir=_scratch(), randomized_init=False,
              randomized_inertial_prop=False, action_scale=FORCE,
              episode_len_sec=math.ceil(HORIZON / 100) + 1,
              terminate_on_goal=True, cost='quadratic',
              task_info={'stabilization_goal': [0],
                         'stabilization_goal_tolerance': 0.05},
              x_dot_limit=INF, theta_dot_limit=INF, obs_wrap_angle=True)
    if sigma > 0:
        kw['disturbances'] = {'action': [{'disturbance_func': 'uniform',
                                          'low': -sigma, 'high': sigma}]}
    ef = partial(make, 'cartpole', **kw)
    env = ef()
    env.x_threshold = 6.0
    env.x_dot_threshold = 5.0
    env.theta_threshold_radians = INF
    env.theta_dot_threshold = 5.0
    return env, make('lqr', ef, q_lqr=[1, 1, 1, 1], r_lqr=[0.1],
                     discrete_dynamics=True)


def roll(env, ctrl, s_file, seed):
    x, theta, x_dot, theta_dot = s_file
    env.reset(seed=int(seed))
    # State lives in PyBullet; assigning env.state alone is silently discarded.
    pb.resetJointState(env.CARTPOLE_ID, 0, targetValue=x, targetVelocity=x_dot,
                       physicsClientId=env.PYB_CLIENT)
    pb.resetJointState(env.CARTPOLE_ID, 1, targetValue=theta,
                       targetVelocity=theta_dot, physicsClientId=env.PYB_CLIENT)
    env.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
    obs = env._get_observation()
    info = {'current_step': 0}
    for step in range(1, HORIZON + 1):
        obs, _, term, trunc, info = env.step(ctrl.select_action(obs, info))
        if info.get('goal_reached', False):
            return True, step
        if term or trunc:
            return False, step
    return False, HORIZON


def work(task):
    level, lo, hi = task
    trials = 1 if level == 0 else TRIALS
    env, ctrl = build(level)
    hits = np.zeros(hi - lo, dtype=np.int32)
    mx = np.zeros(hi - lo, dtype=np.int32)
    for j, i in enumerate(range(lo, hi)):
        for k in range(trials):
            ok, st = roll(env, ctrl, STATES[i], rollout_seed(i, k))
            hits[j] += ok
            mx[j] = max(mx[j], st)
    env.close()
    return level, lo, hits, mx, trials


def main():
    edges = np.linspace(0, N_STATES, NPROC + 1).astype(int)
    tasks = [(lv, int(edges[c]), int(edges[c + 1]))
             for lv in LEVELS for c in range(NPROC) if edges[c + 1] > edges[c]]
    acc = {}
    with Pool(NPROC) as pool:
        for level, lo, hits, mx, trials in pool.imap_unordered(work, tasks):
            a = acc.setdefault(level, [np.zeros(N_STATES, np.int32),
                                       np.zeros(N_STATES, np.int32), trials])
            a[0][lo:lo + len(hits)] = hits
            a[1][lo:lo + len(mx)] = mx

    # Persist BEFORE printing. A previous run exited COMPLETED after 57 min with
    # no stdout file written at all, losing everything; stdout is not a durable
    # result channel here.
    out = os.environ.get('CP_SWEEP_OUT', os.path.expanduser('~/scg-repo/cp_sweep_result.npz'))
    np.savez(out, levels=np.array(LEVELS),
             hits=np.stack([acc[lv][0] for lv in LEVELS]),
             maxstep=np.stack([acc[lv][1] for lv in LEVELS]),
             trials=np.array([acc[lv][2] for lv in LEVELS]),
             det_labels=DET_LABELS, states=STATES)
    print(f'saved {out}', flush=True)

    base = DET_LABELS == 1
    print(f'\n{N_STATES} states from the {116242}-state deterministic grid, '
          f'{TRIALS} trials, control bound {FORCE} N')
    print(f'shipped success rate on this sample: {base.mean():.4f}\n')
    print(f'{"sigma":>8} {"%bound":>8} {"p(success)":>11} {"retained":>9} '
          f'{"gained":>8} {"interior":>9} {"maxstep":>8}')
    for lv in LEVELS:
        hits, mx, trials = acc[lv]
        p = hits / trials
        ret = p[base].mean() if base.any() else float('nan')
        gain = p[~base].mean() if (~base).any() else float('nan')
        print(f'{lv:8.1f} {100 * lv / FORCE:7.1f}% {p.mean():11.4f} {ret:9.3f} '
              f'{gain:8.4f} {((p > 0) & (p < 1)).mean():9.3f} {mx.max():8d}')


if __name__ == '__main__':
    main()
