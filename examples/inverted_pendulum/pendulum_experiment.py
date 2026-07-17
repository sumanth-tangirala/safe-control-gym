'''Run an inverted-pendulum controller under an optional noise config.

Mirrors the other examples' ConfigFactory + BaseExperiment flow, so noise is
selected the safe-control-gym way -- with ``--overrides`` pointing at a file in
``config_overrides/noise/``.

Examples:
    # LQR under control-proportional dynamics noise
    python examples/inverted_pendulum/pendulum_experiment.py \
        --algo pendulum_lqr --task inverted_pendulum \
        --overrides examples/inverted_pendulum/config_overrides/noise/control_proportional_high.yaml

    # A trained RL swing-up policy under truncated actuation noise
    python examples/inverted_pendulum/pendulum_experiment.py \
        --algo pendulum_rl --task inverted_pendulum \
        --overrides examples/inverted_pendulum/config_overrides/noise/truncated_gaussian_act_med.yaml \
        --kv_overrides 'algo_config.model_path="v3_strong"'

``n_episodes`` can be set via ``--kv_overrides n_episodes=10``.
'''

from functools import partial

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run():
    '''Evaluate the chosen controller on the pendulum and print metrics.'''
    config = ConfigFactory().merge()

    env_func = partial(make, config.task, **config.task_config)
    ctrl = make(config.algo, env_func, **config.algo_config)
    env = env_func()

    n_episodes = int(config.get('n_episodes', 5))
    experiment = BaseExperiment(env=env, ctrl=ctrl)
    trajs_data, metrics = experiment.run_evaluation(training=False, n_episodes=n_episodes)

    env.close()
    ctrl.close()

    noise = config.task_config.get('noise', None)
    print(f'[{config.algo} on {config.task}] noise={noise} n_episodes={n_episodes}')
    print('FINAL METRICS - ' + ', '.join(f'{k}: {v}' for k, v in metrics.items()))
    return metrics


if __name__ == '__main__':
    run()
