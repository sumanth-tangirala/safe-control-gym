#!/usr/bin/env python3
'''Calibrate G1 from controller 1's exit attitudes, then freeze it.

RULING (task-5): RoA2 -- controller 2, its checkpoint, and the shipped
`quadrotor2D_rl` labels -- is not loaded, imported, or referenced anywhere in
this script.  G1's parameters must come only from what controller 1 actually
achieves; if RoA2 leaked in here, G1 would be fitted to the answer and the
downstream composition result would be a construction, not a discovery
(spec D1).  In particular this script builds its env with
`quad_composition.rollout2d.make_env`, not `make_env_and_ctrl2` -- the latter
loads controller 2 just to get an env, which would be a needless (if inert)
dependency on RoA2's controller inside the calibration path.

Run this BEFORE generating any composition dataset.
'''

import argparse
import json
import os
import shutil
import tempfile

import numpy as np

from quad_composition.flip_env2d import potential, sample_uniform_state
from quad_composition.g1 import fit_from_exits
from quad_composition.rollout2d import load_ctrl1, make_env, set_initial_state, state_from_env


def collect_exit_attitudes(env, ctrl1, rng, num_rollouts, settle_steps):
    '''Roll out controller 1 from uniform initial states and record each
    rollout's best-attained attitude (spec D1 step 2).

    "Best" means the state maximizing `flip_env2d.potential` (equivalently,
    minimizing |theta|/TILT_SCALE + |theta_dot|/RATE_SCALE) seen along the
    rollout -- controller 1's closest approach to upright-and-still within
    `settle_steps` ticks, scored by the exact potential it was trained to
    climb, not a separately maintained copy of its constants. A rollout that
    terminates (`done`) contributes its best state up to that point; a
    rollout that takes zero steps (`settle_steps == 0`) contributes nothing.

    Returns (tilts, omegas) as numpy arrays of |theta|, |theta_dot| -- G1's
    only inputs. Nothing about controller 2 or RoA2 is read here or by any
    function this one calls.

    Attitudes are TRUE attitude (`state_from_env`), not the env's
    gimbal-folded observation theta (Finding C1). Fitting G1 to folded exits
    would fit it to a distribution structurally capped at pi/2, and would
    score a fully inverted exit as a perfect one.
    '''
    tilts, omegas = [], []
    for _ in range(num_rollouts):
        obs, info = set_initial_state(env, sample_uniform_state(rng))
        best = None
        for _ in range(settle_steps):
            action = ctrl1.select_action(ctrl1.obs_normalizer(obs), info)
            obs, _, done, info = env.step(action)
            state = state_from_env(env, obs)
            phi = potential(state)
            if best is None or phi > best[0]:
                best = (phi, abs(state[2]), abs(state[5]))
            if done:
                break
        if best is not None:
            tilts.append(best[1])
            omegas.append(best[2])
    return np.array(tilts), np.array(omegas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flip_model', required=True)
    parser.add_argument('--output', default='models/quad2d_flip/g1.json')
    parser.add_argument('--num_rollouts', type=int, default=5000)
    parser.add_argument('--quantile', type=float, default=0.9)
    parser.add_argument('--settle_steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # NFS temp dirs intermittently hang; use /tmp explicitly and best-effort
    # cleanup rather than TemporaryDirectory's strict (and occasionally
    # failing) teardown.
    tmp = tempfile.mkdtemp(dir='/tmp', prefix='calibrate_quad2d_g1_')
    env = ctrl1 = None
    try:
        env = make_env(seed=args.seed)
        ctrl1 = load_ctrl1(args.flip_model, env, tmp)

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
