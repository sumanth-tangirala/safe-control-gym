'''Controller 1 training must run end-to-end and emit a loadable checkpoint.

Spec: docs/superpowers/specs/2026-08-16-quad-composition-g1-design.md (D2, D6)
'''
import os
import shutil
import subprocess
import sys
import tempfile

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# $TMPDIR on this machine points at an NFS mount that intermittently hangs, and
# these tests write a multi-megabyte checkpoint plus a logger directory. Pin the
# scratch space to local disk instead of inheriting it.
LOCAL_TMP = '/tmp'


@pytest.mark.slow
def test_training_emits_a_loadable_checkpoint():
    '''End-to-end subprocess run at a tiny step budget (2000, not the real
    1,000,000 -- production training is launched separately, out of scope
    for this smoke test) so this stays a smoke test rather than a
    multi-hour training job.

    2000 is not arbitrary: SAC_CONFIG's warm_up_steps is 1000, so a shorter
    budget would never leave the random-action warm-up and would exercise
    neither the policy forward pass nor a single gradient update -- exactly
    the two paths that break under this machine's numpy 1.x/2.x torch build
    (see sac_utils._tensor_to_numpy / _numpy_to_tensor).

    tempfile.TemporaryDirectory()'s strict cleanup raises OSError on this NFS
    mount whenever a controller's logger still holds an open file handle
    (see test_rollout2d.py); use plain mkdtemp + best-effort rmtree instead.
    '''
    tmp = tempfile.mkdtemp(prefix='train_flip_smoke_', dir=LOCAL_TMP)
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, 'train_quadrotor_2d_flip.py'),
             '--output_dir', tmp, '--max_env_steps', '2000', '--seed', '0'],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=1800)
        assert result.returncode == 0, result.stderr[-3000:]
        checkpoint = os.path.join(tmp, 'flip_model.pt')
        assert os.path.exists(checkpoint)

        state = torch.load(checkpoint, weights_only=False)
        assert set(('agent', 'obs_normalizer', 'reward_normalizer')) <= set(state)

        # The contract is "loadable by make('sac', ...).load()", and the one
        # caller that will do that is rollout2d.load_ctrl1 -- go through it, so
        # a future drift between the training config and rollout2d.SAC_CONFIG
        # (hidden_dim, activation, ...) fails here rather than in Task 6's
        # dataset generation.
        import train_quadrotor_2d_flip as script
        from quad_composition.rollout2d import load_ctrl1

        env = script.env_func()
        try:
            ctrl1 = load_ctrl1(checkpoint, env, tmp)
            assert ctrl1.obs_normalizer.read_only
        finally:
            env.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.slow
def test_env_func_applies_termination_bounds_before_wrapping():
    '''Carried forward from Task 3's review: FlipTrainingEnv does not apply
    the TERMINATION bounds itself -- the caller must. If
    train_quadrotor_2d_flip.env_func skipped this step, the env would
    silently fall back to its own default state bounds, and both
    out-of-bounds termination and the closed state space (CLAUDE.md) would
    be wrong.
    '''
    import train_quadrotor_2d_flip as script
    from quad_composition.rollout2d import TERMINATION

    env = script.env_func()
    try:
        for idx, (lo, hi) in TERMINATION.items():
            assert env.state_space.low[idx] == pytest.approx(lo)
            assert env.state_space.high[idx] == pytest.approx(hi)
    finally:
        env.close()


@pytest.mark.slow
def test_env_func_gives_controller_1_exactly_controller_2s_actuator_authority():
    '''Spec D6: controller 1 must have exactly controller 2's action space --
    normalized, at the inherited norm_act_scale=0.1. Giving controller 1 more
    authority than the controller it hands off to invalidates the experiment,
    and nothing else in the training path would notice.
    '''
    from functools import partial

    import numpy as np

    import train_quadrotor_2d_flip as script
    from quad_composition.rollout2d import ENV_CONFIG
    from safe_control_gym.utils.registration import make

    train_env = script.env_func()
    ctrl2_env = partial(make, 'quadrotor', **ENV_CONFIG)()
    try:
        assert train_env.action_space.shape == ctrl2_env.action_space.shape
        assert np.allclose(train_env.action_space.low, ctrl2_env.action_space.low)
        assert np.allclose(train_env.action_space.high, ctrl2_env.action_space.high)
        assert train_env.norm_act_scale == ctrl2_env.norm_act_scale == 0.1
        assert train_env.NORMALIZED_RL_ACTION_SPACE
    finally:
        train_env.close()
        ctrl2_env.close()


def test_env_func_accepts_a_seed_kwarg_and_extra_kwargs():
    '''Ruling D-H: SAC's training-mode runner calls `env_func(seed=...)` for
    every parallel rollout worker (via make_vec_envs) and again for its eval
    env (`env_func(seed=seed * 111)`). A zero-arg env_func -- as in the
    brief's Step 3 sample -- raises TypeError at that call. This does not
    boot PyBullet (signature check only), so it is not marked slow.
    '''
    import inspect

    import train_quadrotor_2d_flip as script

    sig = inspect.signature(script.env_func)
    sig.bind(seed=123)
    sig.bind(seed=123, some_other_kwarg='ignored')
