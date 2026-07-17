'''Tests for the noise override YAMLs and the example experiment runner.'''

import glob
import os
import subprocess
import sys

import yaml

from safe_control_gym.envs.gym_control.pendulum_noise import NOISE_PRESETS, NoiseModel
from safe_control_gym.envs.gym_control.inverted_pendulum import InvertedPendulum

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOISE_DIR = os.path.join(REPO, 'examples/inverted_pendulum/config_overrides/noise')
EXPERIMENT = os.path.join(REPO, 'examples/inverted_pendulum/pendulum_experiment.py')


def test_all_noise_override_yamls_exist_and_are_valid():
    files = glob.glob(os.path.join(NOISE_DIR, '*.yaml'))
    assert len(files) == len(NOISE_PRESETS) == 25
    for f in files:
        name = os.path.basename(f)[:-len('.yaml')]
        spec = yaml.safe_load(open(f))
        noise = spec['task_config']['noise']
        if name == 'none':
            assert noise is None
        else:
            assert noise == name and name in NOISE_PRESETS


def test_env_built_from_override_has_the_noise_model():
    spec = yaml.safe_load(open(os.path.join(NOISE_DIR, 'truncated_gaussian_act_med.yaml')))
    env = InvertedPendulum(randomized_init=False, **spec['task_config'])
    assert type(env.noise_model).__name__ == 'TruncatedActuationGaussianNoiseModel'
    env.close()
    # 'none' override -> deterministic no-op model.
    spec_none = yaml.safe_load(open(os.path.join(NOISE_DIR, 'none.yaml')))
    env2 = InvertedPendulum(randomized_init=False, **spec_none['task_config'])
    assert isinstance(env2.noise_model, NoiseModel) and type(env2.noise_model) is NoiseModel
    env2.close()


def test_experiment_script_runs_lqr_with_noise_override():
    cmd = [sys.executable, EXPERIMENT,
           '--algo', 'pendulum_lqr', '--task', 'inverted_pendulum',
           '--overrides', os.path.join(NOISE_DIR, 'gaussian_act_high.yaml'),
           '--kv_overrides', 'n_episodes=2', 'task_config.episode_len_sec=2']
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, f'experiment failed:\n{out.stdout}\n{out.stderr}'
    assert 'FINAL METRICS' in out.stdout
