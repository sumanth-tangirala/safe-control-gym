#!/usr/bin/env python3
'''Train controller 1 (the flip controller) for quadrotor-2D.

Attitude-only objective (spec D2), on exactly controller 2's action space
(spec D6: `normalized_rl_action_space=True`, `norm_act_scale` left at its
0.1 default via ENV_CONFIG -- see rollout2d.py). Consumes the training
environment built in Task 3 (`quad_composition.flip_env2d.FlipTrainingEnv`)
and the shared env/algo config from Task 2 (`quad_composition.rollout2d`).

Usage (smoke-sized run):
    python3 train_quadrotor_2d_flip.py --output_dir /tmp/flip_smoke \\
        --max_env_steps 2000 --seed 0

Usage (the real training run -- NOT launched by this script's tests, a
multi-hour job):
    python3 train_quadrotor_2d_flip.py --output_dir models/quad2d_flip \\
        --max_env_steps 1000000 --seed 0
'''

import argparse
import os

from quad_composition.flip_env2d import G_NOM, FlipTrainingEnv
from quad_composition.rollout2d import ENV_CONFIG, SAC_CONFIG, TERMINATION
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import set_seed


def env_func(seed=None, **kwargs):
    '''Build one training instance: the 2D quadrotor with TERMINATION bounds
    applied, wrapped in FlipTrainingEnv (attitude-only reward, spec D2).

    Must accept a `seed` kwarg and swallow any others (Ruling D-H): SAC's
    training-mode runner calls this once per parallel rollout worker via
    `make_vec_envs` (`env_func(seed=base_seed + rank)`) and once more to
    build its eval env (`env_func(seed=seed * 111)`) -- a zero-arg env_func
    raises TypeError on either call.

    TERMINATION is applied here, to the freshly built env, before wrapping
    in FlipTrainingEnv (Task 3 review carried into this task):
    FlipTrainingEnv does not apply these bounds itself, so skipping this
    step would silently leave the env on its own default state bounds --
    breaking both out-of-bounds termination and the closed state space
    (CLAUDE.md) -- exactly as `make_env_and_ctrl2` in rollout2d.py already
    does it for controller 2's env.

    `seed` also seeds FlipTrainingEnv's own initial-state sampler, so
    parallel rollout workers don't all draw the identical sequence of
    initial states. Caveat for anyone who later turns evaluation on:
    SAC hard-codes its eval env's seed to `seed * 111`, which is 0 for
    `--seed 0` and therefore collides with rollout worker rank 0, so the two
    would draw the same initial states. It is inert today because
    SAC_CONFIG sets `eval_interval: 0` and the eval env is never stepped.
    '''
    env = make('quadrotor', seed=seed, **ENV_CONFIG)
    for idx, (lo, hi) in TERMINATION.items():
        env.state_space.low[idx] = lo
        env.state_space.high[idx] = hi
    return FlipTrainingEnv(env, G_NOM, seed=seed if seed is not None else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_env_steps', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Matches train_rl_controller.py's set_seed_from_config, and seeds the
    # global random/numpy/torch state from process start.
    #
    # It does NOT control network initialisation, despite running before
    # make('sac', ...). SAC.__init__ calls make_vec_envs first, and
    # DummyVecEnv builds all rollout_batch_size workers eagerly; each one's
    # make_env_fn thunk reseeds all three global RNGs to seed + rank. So by
    # the time SACAgent.__init__ initialises weights, the torch RNG is at
    # seed + rollout_batch_size - 1, not args.seed -- verified empirically:
    # for --seed S the checkpoint's weights are exactly those produced by
    # torch.manual_seed(S + 3), and are NOT reproduced by manual_seed(S).
    # --seed does still vary runs, via that same seed + rank path.
    #
    # The call is kept anyway: it is the repo's convention, it covers
    # anything drawing on the global RNGs before make_vec_envs (and any
    # future path that does not route through the vec-env workers), and
    # dropping it is an untested behavioural change for no gain.
    set_seed(args.seed)

    # Ruling D-C: SAC_CONFIG comes from rollout2d (Task 2's verified copy of
    # safe_control_gym/controllers/sac/sac.yaml), NOT the brief's inline
    # dict -- that dict omits `activation`, `use_entropy_tuning`, and
    # `target_entropy`, which SACAgent.__init__ reads unconditionally and
    # would crash on construction without. rollout2d.SAC_CONFIG carries
    # training=False (it is built there for eval-only use); override to
    # True for this training run.
    config = dict(SAC_CONFIG, max_env_steps=args.max_env_steps, seed=args.seed, training=True)

    # checkpoint_path must be given explicitly: SAC's own default
    # ('model_latest.pt', no directory component) makes `learn()`'s final
    # checkpoint save call `os.makedirs(os.path.dirname('model_latest.pt'))`
    # == `os.makedirs('')`, which raises FileNotFoundError. Joining it under
    # output_dir avoids that and matches
    # safe_control_gym/experiments/train_rl_controller.py's own pattern.
    ctrl = make('sac', env_func, **config, output_dir=args.output_dir,
                checkpoint_path=os.path.join(args.output_dir, 'model_latest.pt'))
    ctrl.reset()
    ctrl.learn()
    ctrl.save(os.path.join(args.output_dir, 'flip_model.pt'))
    ctrl.close()


if __name__ == '__main__':
    main()
