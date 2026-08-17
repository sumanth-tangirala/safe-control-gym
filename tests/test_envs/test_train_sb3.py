'''The trainer must be genuinely task-agnostic, not pendulum-shaped.'''
import os
import subprocess
import sys

import pytest
import yaml

REPO = os.path.join(os.path.dirname(__file__), '..', '..')

# Base ids and composite ids both, because they are separate lookups: a
# composite id that failed to register would still leave the base ids training
# happily, and the trainer would look fine.
TASKS = ['inverted_pendulum', 'cartpole', 'quadrotor',
         'inverted_pendulum_stabilization', 'cartpole_stabilization',
         'quadrotor2d_stabilization', 'quadrotor3d_stabilization']

SHORT = ['--kv_overrides', 'sb3_config.total_timesteps=256',
         'sb3_config.save_freq=128', 'sb3_config.eval_freq=128',
         'sb3_config.n_eval_episodes=1']


def _train(out, env_id, flag='--task', extra=()):
    return subprocess.run(
        [sys.executable, '-m', 'safe_control_gym.experiments.train_sb3',
         flag, env_id, '--algo', 'sac', '--seed', '1',
         '--output_dir', str(out)] + SHORT + list(extra),
        cwd=REPO, capture_output=True, text=True)


def _only_run(out, env_id):
    runs = sorted((out / 'sac').glob(f'{env_id}_*'))
    assert len(runs) == 1, f'expected one run directory, found {runs}'
    return runs[0]


@pytest.mark.parametrize('task', TASKS)
def test_trains_briefly(task, tmp_path):
    result = _train(tmp_path, task)
    assert result.returncode == 0, result.stderr[-3000:]
    run = _only_run(tmp_path, task)
    assert (run / 'model_final.zip').exists()
    assert list((run / 'checkpoints').glob('step_*.zip'))


def test_run_directory_is_self_describing(tmp_path):
    '''A run must say what it is without its command being remembered.

    config.yml in particular is load-bearing: eval_policy rebuilds the
    environment and its wrappers from it, so a run missing it cannot be
    evaluated at all.
    '''
    assert _train(tmp_path, 'cartpole_stabilization').returncode == 0
    run = _only_run(tmp_path, 'cartpole_stabilization')

    for name in ('config.yml', 'args.yml', 'command.txt'):
        assert (run / name).exists(), f'{name} missing from {run}'

    config = yaml.safe_load((run / 'config.yml').read_text())
    assert config['task'] == 'cartpole_stabilization'
    assert config['algo'] == 'sac'
    assert config['task_config']['task'] == 'stabilization'

    args = yaml.safe_load((run / 'args.yml').read_text())
    assert args['seed'] == 1
    assert 'train_sb3' in (run / 'command.txt').read_text()


def test_runs_do_not_clobber(tmp_path):
    '''A second run of the same config takes _2, not the same directory.'''
    for _ in range(2):
        assert _train(tmp_path, 'cartpole_stabilization').returncode == 0
    runs = sorted(p.name for p in (tmp_path / 'sac').glob('cartpole_stabilization_*'))
    assert runs == ['cartpole_stabilization_1', 'cartpole_stabilization_2']


def test_concurrent_launches_all_get_a_directory(tmp_path):
    '''Launching several systems at once must not kill some of them.

    claim_run_dir creates <root>/<algo> before claiming the run directory under
    it. Built on utils.mkdirs -- os.makedirs with no exist_ok -- that raced: of
    four systems launched together, two died with FileExistsError on
    <root>/sac before training a step. The run directories themselves were
    already claimed atomically; the parent was not.
    '''
    from concurrent.futures import ThreadPoolExecutor

    ids = ['cartpole_stabilization', 'inverted_pendulum_stabilization',
           'quadrotor2d_stabilization', 'quadrotor3d_stabilization']
    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        results = list(pool.map(lambda i: (i, _train(tmp_path, i)), ids))

    for env_id, result in results:
        assert result.returncode == 0, f'{env_id} failed:\n{result.stderr[-2000:]}'
    assert len(list((tmp_path / 'sac').iterdir())) == len(ids)


def test_env_id_is_an_alias_for_task(tmp_path):
    result = _train(tmp_path, 'cartpole_stabilization', flag='--env_id')
    assert result.returncode == 0, result.stderr[-3000:]
    run = _only_run(tmp_path, 'cartpole_stabilization')
    assert yaml.safe_load((run / 'config.yml').read_text())['task'] == 'cartpole_stabilization'


def test_env_id_and_task_together_is_an_error(tmp_path):
    '''Two names for one argument must not silently pick a winner.'''
    result = _train(tmp_path, 'cartpole_stabilization', flag='--env_id',
                    extra=('--task', 'inverted_pendulum'))
    assert result.returncode != 0
    assert 'not both' in result.stderr


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
