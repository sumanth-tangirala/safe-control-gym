'''The trainer must be genuinely task-agnostic, not pendulum-shaped.'''
import os
import subprocess
import sys

import pytest

REPO = os.path.join(os.path.dirname(__file__), '..', '..')

TASKS = ['inverted_pendulum', 'cartpole', 'quadrotor']


@pytest.mark.parametrize('task', TASKS)
def test_trains_briefly(task, tmp_path):
    out = tmp_path / task
    result = subprocess.run(
        [sys.executable, '-m', 'safe_control_gym.experiments.train_sb3',
         '--task', task, '--algo', 'sac', '--seed', '1',
         '--output_dir', str(out),
         '--kv_overrides', 'sb3_config.total_timesteps=256',
         'sb3_config.save_freq=128'],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    assert (out / 'model_final.zip').exists()
    assert list((out / 'checkpoints').glob('step_*.zip'))


def test_sb3_not_imported_by_library():
    '''envs/ and controllers/ must stay importable without SB3.'''
    probe = (
        'import sys, importlib\n'
        'sys.modules["stable_baselines3"] = None\n'
        'import safe_control_gym.envs, safe_control_gym.controllers\n'
        'print("ok")\n'
    )
    result = subprocess.run([sys.executable, '-c', probe],
                            cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'ok' in result.stdout
