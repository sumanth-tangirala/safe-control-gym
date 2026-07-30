'''Task-agnostic stable-baselines3 training.

The only module in the package permitted to import stable-baselines3; envs/ and
controllers/ stay SB3-free so inference and dataset collection never gain the
dependency.

Shaping is configuration, not code: --kv_overrides sb3_config.angle_obs=... and
sb3_config.action_repeat=... select the optional wrappers. No task is special
cased here.

    python -m safe_control_gym.experiments.train_sb3 \\
        --env_id cartpole_stabilization --algo sac --output_dir logs \\
        --overrides configs/sb3/cartpole_stabilization_sac.yaml

--env_id is an alias for --task. They mean the same registry lookup; --task is
defined once in configuration.py and shared by every entry point in the repo, so
it is not renamed, while --env_id is what a composite (system, task) id actually
is. Pass either.

Runs are written to <output_dir>/<algo>/<env_id>_<run>/, following RL Baselines3
Zoo, with the merged config, the CLI arguments and the verbatim command stored
beside the weights. <run> auto-increments, so re-running never clobbers.
config.yml is what makes a run rebuildable: eval_policy reconstructs the
environment and its wrappers from it rather than re-deriving them from flags.

--use_gpu (like every other entry point in this repo) selects SB3's device;
it defaults to CPU.

**Pass --use_gpu if a GPU is free.** Measured on an idle ilab2 (64 cores,
RTX A4500), SAC on the pendulum with `net_arch: [256, 256]`, 4000 steps, both
devices back to back on the same host with OMP/MKL threads pinned to 8:

    cpu     65.6 steps/s   ->  200k steps ~ 51 min
    cuda   111.0 steps/s   ->  200k steps ~ 30 min   (1.69x faster)

An earlier version of this docstring claimed GPU would be *slower* here, on the
general principle that small MLP policies favour CPU. That was asserted, not
measured, and it is wrong for this workload -- the first attempt to check it ran
on a box at load average 74 and measured contention rather than devices. Benchmark
on an idle host before trusting any claim of this kind, including this one.

The environment is never the bottleneck either way: the pendulum steps at
~12,500/s, so 200k steps is ~16s of simulation against tens of minutes of
gradient work.
'''
import os
import shlex
import sys

import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from safe_control_gym.envs.env_wrappers.shaping import ActionRepeat, AngleObservation
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import set_device_from_config, set_seed_from_config

ALGOS = {'sac': SAC, 'ppo': PPO}

# CLI arguments worth recording separately from the merged config: reading
# args.yml answers "what was typed" without diffing a fully-resolved config
# against the defaults it came from.
CLI_KEYS = ('algo', 'task', 'seed', 'use_gpu', 'output_dir', 'tag', 'restore')


def apply_env_id_alias(argv=None):
    '''Rewrite --env_id to --task, which is what ConfigFactory parses.

    Returns the argv list to parse. Passing both is an error rather than a
    silent precedence rule -- they name the same thing, so disagreeing values
    mean the caller believes something untrue.
    '''
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a == '--env_id' or a.startswith('--env_id=') for a in argv):
        return argv
    if any(a == '--task' or a.startswith('--task=') for a in argv):
        raise SystemExit('Pass --env_id or --task, not both: they are the same argument.')
    return [a.replace('--env_id', '--task', 1) if a.startswith('--env_id') else a
            for a in argv]


def build_env(config):
    '''Registered env plus whatever shaping the config asks for.'''
    env = make(config.task, **config.task_config)
    sb3_config = config.get('sb3_config', {})
    angle_obs = sb3_config.get('angle_obs', None)
    if angle_obs is not None:
        env = AngleObservation(env, angle_obs['angle_index'],
                               angle_obs['rate_index'], angle_obs['rate_max'])
    repeat = int(sb3_config.get('action_repeat', 1))
    if repeat > 1:
        env = ActionRepeat(env, repeat)
    return env


def claim_run_dir(root, algo, env_id):
    '''<root>/<algo>/<env_id>_<n>, the lowest n not already taken.

    Claimed with exist_ok=False rather than checked-then-created, so two runs
    launched at once take different directories instead of both believing they
    own the same one.

    The parent, by contrast, must tolerate already existing: utils.mkdirs is
    os.makedirs without exist_ok, so four systems launched together raced on
    creating <root>/<algo> and two of them died with FileExistsError before
    training a single step.
    '''
    parent = os.path.join(root, algo)
    os.makedirs(parent, exist_ok=True)
    run = 1
    while True:
        candidate = os.path.join(parent, f'{env_id}_{run}')
        try:
            os.makedirs(candidate)
            return candidate
        except FileExistsError:
            run += 1


def write_run_metadata(run_dir, config):
    '''config.yml, args.yml and command.txt beside the weights.'''
    with open(os.path.join(run_dir, 'config.yml'), 'w') as handle:
        yaml.safe_dump(dict(config), handle, default_flow_style=False, sort_keys=True)
    args = {key: config.get(key) for key in CLI_KEYS if config.get(key) is not None}
    with open(os.path.join(run_dir, 'args.yml'), 'w') as handle:
        yaml.safe_dump(args, handle, default_flow_style=False, sort_keys=True)
    with open(os.path.join(run_dir, 'command.txt'), 'w') as handle:
        handle.write(shlex.join([sys.executable, '-m',
                                 'safe_control_gym.experiments.train_sb3'] + sys.argv[1:]) + '\n')


def train():
    '''Train and checkpoint; returns (model, run_dir).'''
    sys.argv[1:] = apply_env_id_alias()
    config = ConfigFactory().merge()
    set_seed_from_config(config)
    set_device_from_config(config)

    sb3_config = config.get('sb3_config', {})
    algo = ALGOS[config.algo]
    total_timesteps = int(sb3_config.get('total_timesteps', 100000))
    save_freq = int(sb3_config.get('save_freq', 10000))
    eval_freq = int(sb3_config.get('eval_freq', max(total_timesteps // 10, 1)))
    n_eval_episodes = int(sb3_config.get('n_eval_episodes', 5))

    run_dir = claim_run_dir(config.output_dir, config.algo, config.task)
    write_run_metadata(run_dir, config)
    print(f'device={config.device} run_dir={run_dir}')

    env = build_env(config)
    # A second env, alive alongside the training env for the whole run. That was
    # unsafe for the quadrotors until base_aviary.py's changeDynamics call was
    # given its physicsClientId -- without it this env would silently corrupt the
    # training env's dynamics. See tests/test_envs/test_concurrent_pybullet_envs.py.
    eval_env = build_env(config)

    model = algo('MlpPolicy', env, seed=config.seed, verbose=1,
                 device=config.device,
                 policy_kwargs={'net_arch': list(sb3_config.get('net_arch', [256, 256]))})
    callbacks = [
        # Periodic checkpoints, not only best: the shipped strong/weak model pairs
        # are best-vs-intermediate checkpoints of one run, so dropping intermediates
        # would make that axis unreproducible.
        CheckpointCallback(save_freq=save_freq,
                           save_path=os.path.join(run_dir, 'checkpoints'),
                           name_prefix='step'),
        EvalCallback(eval_env, best_model_save_path=run_dir,
                     log_path=run_dir, eval_freq=eval_freq,
                     n_eval_episodes=n_eval_episodes, deterministic=True),
    ]
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        model.save(os.path.join(run_dir, 'model_final'))
    finally:
        env.close()
        eval_env.close()
    return model, run_dir


if __name__ == '__main__':
    train()
