'''Task-agnostic stable-baselines3 training.

The only module in the package permitted to import stable-baselines3; envs/ and
controllers/ stay SB3-free so inference and dataset collection never gain the
dependency.

Shaping is configuration, not code: --kv_overrides sb3_config.angle_obs=... and
sb3_config.action_repeat=... select the optional wrappers. No task is special
cased here.

    python -m safe_control_gym.experiments.train_sb3 \\
        --task inverted_pendulum --algo sac --output_dir results/pendulum_sac \\
        --kv_overrides sb3_config.total_timesteps=200000

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

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from safe_control_gym.envs.env_wrappers.shaping import ActionRepeat, AngleObservation
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_device_from_config, set_seed_from_config

ALGOS = {'sac': SAC, 'ppo': PPO}


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


def train():
    '''Train and checkpoint; returns the fitted model.'''
    config = ConfigFactory().merge()
    set_seed_from_config(config)
    set_device_from_config(config)
    mkdirs(config.output_dir)
    print(f'device={config.device}')

    sb3_config = config.get('sb3_config', {})
    algo = ALGOS[config.algo]
    total_timesteps = int(sb3_config.get('total_timesteps', 100000))
    save_freq = int(sb3_config.get('save_freq', 10000))

    env = build_env(config)
    model = algo('MlpPolicy', env, seed=config.seed, verbose=1,
                 device=config.device,
                 policy_kwargs={'net_arch': list(sb3_config.get('net_arch', [256, 256]))})
    # Periodic checkpoints, not only best: the shipped strong/weak model pairs
    # are best-vs-intermediate checkpoints of one run, so dropping intermediates
    # would make that axis unreproducible.
    callback = CheckpointCallback(save_freq=save_freq,
                                  save_path=os.path.join(config.output_dir, 'checkpoints'),
                                  name_prefix='step')
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(config.output_dir, 'model_final'))
    env.close()
    return model


if __name__ == '__main__':
    train()
