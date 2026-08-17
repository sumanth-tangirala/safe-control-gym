#!/usr/bin/env python3
'''Train controller 1 (the flip controller) for quadrotor-3D.

Attitude-only objective (spec D2), on exactly controller 2's action space
(spec D6).  Unlike the 2D branch, that action space is the PHYSICAL one:
`normalized_rl_action_space` is left at its False default in
`rollout3d.ENV_CONFIG`, because the shipped quadrotor3D_lqr dataset was
generated with the physical actuator (TWR 2.24) and controller 1 must have
exactly controller 2's authority.  Do not normalize it here.

WHY 3D AT ALL: quadrotor-2D cannot be flipped from inversion by any
controller (a hand-coded bang-bang flip scores 0.000 above 90 deg).  A
feasibility probe showed the 3D system CAN be: a geometric attitude
controller recovers from full inversion at 35-42% under these limits, and at
100% in an isolated attitude-only test.

Usage (smoke-sized run):
    python3 train_quadrotor_3d_flip.py --output_dir /tmp/flip3d_smoke \\
        --max_env_steps 2000 --seed 0

Usage (the real training run -- a multi-hour job, NOT launched by the tests):
    python3 train_quadrotor_3d_flip.py --output_dir models/quad3d_flip \\
        --max_env_steps 1000000 --seed 0
'''

import argparse
import os

import numpy as np

from quad_composition.flip_env3d import G_NOM_3D, FlipTrainingEnv3D
from quad_composition.rollout3d import ENV_CONFIG, SAC_CONFIG, apply_termination
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import set_seed

_MAX_INIT_TILT = None


def env_func(seed=None, **kwargs):
    '''Build one training instance: the 3D quadrotor with TERMINATION bounds
    applied, wrapped in FlipTrainingEnv3D (attitude-only reward, spec D2).

    Must accept a `seed` kwarg and swallow any others: SAC's training-mode
    runner calls this once per parallel rollout worker via `make_vec_envs`
    (`env_func(seed=base_seed + rank)`) and once more to build its eval env
    (`env_func(seed=seed * 111)`) -- a zero-arg env_func raises TypeError on
    either call.

    TERMINATION is applied HERE, to the freshly built env, BEFORE wrapping:
    FlipTrainingEnv3D does not apply these bounds itself, and it reads
    `env.state_space` in its constructor to derive its sampling box, so doing
    this in the wrong order would both break out-of-bounds termination and
    silently sample the env's much wider default box (CLAUDE.md's closed state
    space).

    `seed` also seeds FlipTrainingEnv3D's own initial-state sampler, so
    parallel rollout workers don't all draw the identical sequence of initial
    states.  Caveat: SAC hard-codes its eval env's seed to `seed * 111`, which
    is 0 for `--seed 0` and therefore collides with rollout worker rank 0.
    '''
    env = apply_termination(make('quadrotor', seed=seed, **ENV_CONFIG))
    return FlipTrainingEnv3D(env, G_NOM_3D, seed=seed if seed is not None else 0,
                             max_init_tilt=_MAX_INIT_TILT)


def main():
    global _MAX_INIT_TILT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_env_steps', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=0)
    # sac.yaml sets all three intervals to 0, which disables logging, periodic
    # checkpointing and eval entirely -- so a run killed before max_env_steps
    # leaves NO usable policy behind, and there is no way to observe
    # throughput while it runs.  Both matter for any long run.  Intervals must
    # divide evenly by rollout_batch_size (4), since total_steps advances in
    # steps of that size and SAC tests `total_steps % interval == 0`.
    parser.add_argument('--log_interval', type=int, default=5000,
                        help='Steps between progress log lines (0 disables).')
    parser.add_argument('--save_interval', type=int, default=20000,
                        help='Steps between periodic checkpoints (0 disables).')
    parser.add_argument('--eval_interval', type=int, default=25000,
                        help='Steps between evaluations (0 disables).')
    parser.add_argument('--eval_batch_size', type=int, default=5,
                        help='Episodes per evaluation.')
    parser.add_argument('--max_init_tilt_deg', type=float, default=None,
                        help='Cap the INITIAL tilt sampled during training, in degrees. '
                             'Uncapped, attitude is uniform on SO(3) (mean tilt 90 deg). '
                             'Training only -- evaluation and dataset generation are '
                             'unaffected.')
    args = parser.parse_args()

    for name in ('log_interval', 'save_interval', 'eval_interval'):
        value = getattr(args, name)
        if value and value % 4 != 0:
            parser.error(f'--{name} must be a multiple of 4 (rollout_batch_size); got {value}')

    _MAX_INIT_TILT = (np.radians(args.max_init_tilt_deg)
                      if args.max_init_tilt_deg is not None else None)

    os.makedirs(args.output_dir, exist_ok=True)

    # Repo convention (matches train_rl_controller.py's set_seed_from_config).
    # It does NOT control network initialisation: SAC.__init__ builds all
    # rollout_batch_size vec-env workers first and each reseeds the global
    # RNGs to seed + rank, so weights end up drawn at seed + 3.  --seed does
    # still vary runs, via that same path.
    set_seed(args.seed)

    # SAC_CONFIG is rollout3d's verified copy of
    # safe_control_gym/controllers/sac/sac.yaml; it carries training=False for
    # eval use, overridden to True here.
    config = dict(SAC_CONFIG, max_env_steps=args.max_env_steps, seed=args.seed, training=True,
                  log_interval=args.log_interval, save_interval=args.save_interval,
                  eval_interval=args.eval_interval, eval_batch_size=args.eval_batch_size,
                  eval_save_best=bool(args.eval_interval))

    # checkpoint_path must be given explicitly: SAC's own default
    # ('model_latest.pt', no directory component) makes learn()'s final save
    # call os.makedirs('') and raise FileNotFoundError.
    ctrl = make('sac', env_func, **config, output_dir=args.output_dir,
                checkpoint_path=os.path.join(args.output_dir, 'model_latest.pt'))
    ctrl.reset()
    ctrl.learn()
    ctrl.save(os.path.join(args.output_dir, 'flip_model.pt'))
    ctrl.close()


if __name__ == '__main__':
    main()
