'''Round-trip: train a real (short) SAC policy, export it, and check that the
native pendulum_rl actor reproduces SB3's own deterministic predict().

This is the test the exporter exists to satisfy: scripts/export_sb3_pendulum.py
is the bridge between safe_control_gym/experiments/train_sb3.py's .zip output
and the pendulum_rl controller's .pt input, and the only thing that makes it
trustworthy is that the two agree on actions, not just on file shape.
'''

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.join(os.path.dirname(__file__), '..', '..')

THETA_DOT_MAX = 2 * math.pi
ACTION_REPEAT = 4
FWD_TOL = 1e-6


def _train_short_sac(tmp_path):
    '''A genuinely short SAC run on inverted_pendulum, wrapped exactly as the
    pendulum_rl policies are: AngleObservation ([cos, sin, thdot/thdot_max])
    plus action_repeat 4.'''
    out = tmp_path / 'train'
    result = subprocess.run(
        [sys.executable, '-m', 'safe_control_gym.experiments.train_sb3',
         '--task', 'inverted_pendulum', '--algo', 'sac', '--seed', '3',
         '--output_dir', str(out),
         '--kv_overrides',
         'sb3_config.total_timesteps=1024', 'sb3_config.save_freq=1024',
         'sb3_config.angle_obs.angle_index=0', 'sb3_config.angle_obs.rate_index=1',
         f'sb3_config.angle_obs.rate_max={THETA_DOT_MAX}',
         f'sb3_config.action_repeat={ACTION_REPEAT}'],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    zip_path = out / 'model_final.zip'
    assert zip_path.exists()
    return zip_path


def test_export_round_trips_to_sb3_predict(tmp_path):
    zip_path = _train_short_sac(tmp_path)

    out_pt = tmp_path / 'export_test_policy.pt'
    result = subprocess.run(
        [sys.executable, os.path.join('scripts', 'export_sb3_pendulum.py'),
         str(zip_path), str(out_pt),
         '--action_repeat', str(ACTION_REPEAT), '--theta_dot_max', str(THETA_DOT_MAX)],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    assert out_pt.exists()

    # Provenance landed in a manifest.json next to the .pt.
    manifest_path = tmp_path / 'manifest.json'
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    entry = manifest['models'][-1]
    assert entry['variant'] == 'export_test_policy'
    assert entry['git_sha'] is None or len(entry['git_sha']) == 40
    assert entry['source_zip'] == str(zip_path.resolve()) or entry['source_zip'] == str(zip_path)
    assert entry['sb3_version']
    assert entry['torch_version']
    assert entry['action_repeat'] == ACTION_REPEAT
    assert entry['theta_dot_max'] == pytest.approx(THETA_DOT_MAX)
    assert entry['forward_max_err'] <= 1e-5

    from stable_baselines3 import SAC

    from safe_control_gym.controllers.pendulum_rl.pendulum_rl import PendulumRL
    from safe_control_gym.envs.gym_control.inverted_pendulum import InvertedPendulum

    model = SAC.load(str(zip_path), device='cpu')

    def env_func(**kwargs):
        cfg = dict(ctrl_freq=100, pyb_freq=100, randomized_init=False, cost='quadratic')
        cfg.update(kwargs)
        return InvertedPendulum(**cfg)

    # action_repeat=1: query the policy every call, matching model.predict 1:1.
    ctrl = PendulumRL(env_func, model_path=str(out_pt), action_repeat=1)

    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        theta = float(rng.uniform(-math.pi, math.pi))
        thetadot = float(rng.uniform(-THETA_DOT_MAX, THETA_DOT_MAX))
        obs3 = np.array([math.cos(theta), math.sin(theta), thetadot / THETA_DOT_MAX],
                        dtype=np.float32)
        sb3_action, _ = model.predict(obs3, deterministic=True)
        got = ctrl.select_action(np.array([theta, thetadot], dtype=np.float64))
        got_val = float(np.asarray(got).reshape(-1)[0])
        sb3_val = float(np.asarray(sb3_action).reshape(-1)[0])
        max_err = max(max_err, abs(got_val - sb3_val))
    ctrl.close()

    assert max_err <= FWD_TOL, f'round-trip mismatch {max_err:.2e} > {FWD_TOL:.0e}'


def test_export_refuses_shipped_model_name(tmp_path):
    '''Guard rail: exporting must not be pointable at a shipped v1..v4 .pt.'''
    shipped_dir = os.path.join(REPO, 'safe_control_gym/controllers/pendulum_rl/models')
    target = os.path.join(shipped_dir, 'v1_strong.pt')
    fake_zip = tmp_path / 'unused.zip'
    fake_zip.write_bytes(b'')
    result = subprocess.run(
        [sys.executable, os.path.join('scripts', 'export_sb3_pendulum.py'),
         str(fake_zip), target],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'refusing to overwrite shipped model' in result.stderr
