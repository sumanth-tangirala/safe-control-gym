#!/usr/bin/env python3
'''Calibrate G1 from controller 1's exit attitudes, then freeze it.

The 3D analogue of `calibrate_quad2d_g1.py` -- ported structure, ported
enforcement, dimension-agnostic `G1Region`/`fit_from_exits` reused verbatim.

RULING (task-5, ported to 3D): RoA2 -- controller 2 (LQR), and the shipped
`quadrotor3D_lqr` dataset (including its `roa_labels.txt`) -- is not loaded,
imported, or referenced anywhere in this script. G1's parameters must come
only from what controller 1 actually achieves; if RoA2 leaked in here, G1
would be fitted to the answer and the downstream composition result would be
a construction, not a discovery (spec D1). In particular this script builds
its env with `quad_composition.rollout3d.make_env`, not `make_env_and_ctrl2`
-- the latter also builds controller 2 (LQR), which would be a needless (if
inert, since LQR is analytic and has no checkpoint) dependency on RoA2's
controller inside the calibration path.

Run this BEFORE generating any 3D composition dataset.
'''

import argparse
import json
import os
import shutil
import tempfile

import numpy as np

from quad_composition.flip_env3d import potential, sample_uniform_state, sampling_bounds_from_env
from quad_composition.g1 import fit_from_exits
from quad_composition.geometric_flip3d import GeometricFlipController3D
from quad_composition.rollout3d import (QUAT_SLICE, ctrl1_observation, load_ctrl1, make_env,
                                        omega_norm, set_initial_state, state_from_env,
                                        tilt_from_quat_wxyz)


def collect_exit_attitudes(env, ctrl1, rng, num_rollouts, settle_steps):
    '''Roll out controller 1 from uniform initial states (full SO(3) attitude,
    spec D1 step 2) and record each rollout's best-attained attitude.

    "Best" means the state maximizing `flip_env3d.potential` (equivalently,
    minimizing tilt/TILT_SCALE + |omega|/RATE_SCALE) seen along the rollout --
    controller 1's closest approach to upright-and-still within `settle_steps`
    ticks, scored by the exact potential it was trained to climb, not a
    separately maintained copy of its constants. A rollout that terminates
    (`done`) contributes its best state up to that point; a rollout that takes
    zero steps (`settle_steps == 0`) contributes nothing.

    Returns (tilts, omegas) as numpy arrays -- G1's only inputs. Nothing about
    controller 2 or RoA2 is read here or by any function this one calls.

    Tilt comes from the ROTATION MATRIX (`rollout3d.tilt_from_quat_wxyz`),
    NEVER from Euler angles: PyBullet's `getEulerFromQuaternion` folds pitch
    into [-pi/2, pi/2], so a nearly-inverted drone would read as nearly
    upright (rollout3d.py's docstring, item 1; the 3D analogue of Finding C1).
    |omega| is the body-rate norm (`rollout3d.omega_norm`).

    Controller 1 is fed `ctrl1_observation(env, obs)` here, the same 18-dim
    euler-free observation it was trained on (spec D6) and that
    `rollout3d._act_ctrl1` feeds it at composition-rollout time.
    '''
    bounds = sampling_bounds_from_env(env)
    tilts, omegas = [], []
    for _ in range(num_rollouts):
        obs, info = set_initial_state(env, sample_uniform_state(rng, bounds))
        best = None
        for _ in range(settle_steps):
            action = ctrl1.select_action(ctrl1.obs_normalizer(ctrl1_observation(env, obs)), info)
            obs, _, done, info = env.step(action)
            state = state_from_env(env, obs)
            phi = potential(state)
            if best is None or phi > best[0]:
                best = (phi, tilt_from_quat_wxyz(state[QUAT_SLICE]), omega_norm(state))
            if done:
                break
        if best is not None:
            tilts.append(best[1])
            omegas.append(best[2])
    return np.array(tilts), np.array(omegas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', choices=['sac', 'geometric'], default='sac',
                        help='Controller 1 to calibrate against: a SAC checkpoint (default, '
                             'needs --flip_model) or the hand-coded geometric controller '
                             '(quad_composition.geometric_flip3d), which needs no checkpoint.')
    parser.add_argument('--flip_model', default=None,
                        help='SAC checkpoint for controller 1. Required when --controller sac; '
                             'unused (and unnecessary) when --controller geometric.')
    parser.add_argument('--output', default='models/quad3d_flip/g1.json')
    parser.add_argument('--num_rollouts', type=int, default=5000)
    parser.add_argument('--quantile', type=float, default=0.9)
    parser.add_argument('--settle_steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    if args.controller == 'sac' and not args.flip_model:
        parser.error('--flip_model is required when --controller sac')

    rng = np.random.default_rng(args.seed)

    # NFS temp dirs intermittently hang; use /tmp explicitly and best-effort
    # cleanup rather than TemporaryDirectory's strict (and occasionally
    # failing) teardown.
    tmp = tempfile.mkdtemp(dir='/tmp', prefix='calibrate_quad3d_g1_')
    env = ctrl1 = None
    try:
        env = make_env(seed=args.seed)
        ctrl1 = (GeometricFlipController3D(env) if args.controller == 'geometric'
                else load_ctrl1(args.flip_model, env, tmp))

        tilts, omegas = collect_exit_attitudes(
            env, ctrl1, rng, args.num_rollouts, args.settle_steps)

        g1 = fit_from_exits(tilts, omegas, quantile=args.quantile)

        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as fh:
            json.dump({
                'g1': g1.to_dict(),
                'calibration': {
                    'num_rollouts': args.num_rollouts,
                    'settle_steps': args.settle_steps,
                    'quantile': args.quantile,
                    'seed': args.seed,
                    'controller': args.controller,
                    'flip_model': args.flip_model,
                    'exit_tilt_quantiles': {
                        str(q): float(np.quantile(tilts, q))
                        for q in (0.5, 0.75, 0.9, 0.95)},
                    'exit_rate_quantiles': {
                        str(q): float(np.quantile(omegas, q))
                        for q in (0.5, 0.75, 0.9, 0.95)},
                },
                # True by construction: controller 2 is never loaded above.
                'roa2_consulted': False,
            }, fh, indent=2)
        print(f'G1 = {g1.to_dict()}  -> {args.output}')
    finally:
        if ctrl1 is not None:
            ctrl1.close()
        if env is not None:
            env.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
