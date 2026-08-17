'''The 3D flip controller's attitude must come from the rotation matrix, its
observation must carry no Euler angles, and its reward must be attitude-only.

Spec: docs/superpowers/specs/2026-08-16-quad-composition-g1-design.md (D2, D6)

These are deliberately few and fast: they guard the four specific defects the
2D branch was bitten by, not the whole surface.
'''
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pybullet as p
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from quad_composition.flip_env3d import (FlipTrainingEnv3D, G_NOM_3D, potential,  # noqa: E402
                                         sample_uniform_state, sampling_bounds_from_env,
                                         shaped_reward)
from quad_composition.rollout3d import (canonical_quat_wxyz, ctrl1_observation,  # noqa: E402
                                        make_env, set_initial_state, tilt_from_quat,
                                        tilt_from_quat_wxyz)

# $TMPDIR on this machine points at an NFS mount that intermittently hangs.
LOCAL_TMP = '/tmp'


def _state(pos=(0.0, 0.0, 1.0), euler=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0),
           rates=(0.0, 0.0, 0.0)):
    '''A dataset-order 13-dim row [x, y, z, qw, qx, qy, qz, vx, vy, vz, p, q, r]
    built from an Euler attitude (convenient for writing a KNOWN attitude down;
    nothing under test ever reads Euler angles back out).
    '''
    quat = canonical_quat_wxyz(p.getQuaternionFromEuler(list(euler)))
    return np.array(list(pos) + quat + list(vel) + list(rates), dtype=float)


# ---------------------------------------------------------------------------
# 1. Tilt comes from the rotation matrix.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('euler, expected', [
    ((0.0, 0.0, 0.0), 0.0),                    # upright
    ((0.0, 0.0, 2.0), 0.0),                    # pure yaw is not tilt
    ((np.pi / 2, 0.0, 0.0), np.pi / 2),        # on its side (roll)
    ((0.0, np.pi / 2, 0.0), np.pi / 2),        # on its side (pitch)
    ((np.pi, 0.0, 0.0), np.pi),                # fully inverted (roll)
    ((0.0, np.pi, 0.0), np.pi),                # fully inverted (pitch)
])
def test_tilt_is_correct_at_known_attitudes(euler, expected):
    quat = p.getQuaternionFromEuler(list(euler))
    assert tilt_from_quat(quat) == pytest.approx(expected, abs=1e-6)
    # Same answer through the dataset-order (scalar-first) entry point.
    assert tilt_from_quat_wxyz(canonical_quat_wxyz(quat)) == pytest.approx(expected, abs=1e-6)


def test_euler_derived_tilt_would_call_an_inverted_drone_upright():
    '''The 2D killer bug, restated in 3D.

    `p.getEulerFromQuaternion` folds pitch into [-pi/2, pi/2], so a drone at
    pitch pi -- FULLY INVERTED -- reads back as `[pi, 0, pi]`, i.e. pitch
    EXACTLY ZERO.  Any attitude scalar built from that Euler triple's pitch
    scores the inverted drone as perfectly upright, which in 2D paid the
    G_nom bonus to an inverted drone and trained controller 1 to stay there.
    `tilt_from_quat` returns pi, as it must.
    '''
    quat = p.getQuaternionFromEuler([0.0, np.pi, 0.0])
    folded_pitch = p.getEulerFromQuaternion(quat)[1]
    assert abs(folded_pitch) == pytest.approx(0.0, abs=1e-6), 'the fold, measured'
    assert tilt_from_quat(quat) == pytest.approx(np.pi, abs=1e-6)

    inverted = _state(euler=(0.0, np.pi, 0.0))
    upright = _state(euler=(0.0, 0.0, 0.0))
    assert potential(inverted) < potential(upright)
    assert not G_NOM_3D.contains(tilt_from_quat_wxyz(inverted[3:7]), 0.0)
    assert G_NOM_3D.contains(tilt_from_quat_wxyz(upright[3:7]), 0.0)


# ---------------------------------------------------------------------------
# 2. Controller 1's observation is 18-dim and Euler-free.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ctrl1_observation_is_18_dim_and_separates_upright_from_inverted():
    '''Spec D6: the three Euler elements are replaced by the nine rotation
    matrix entries (12 - 3 + 9 = 18).

    The two states below differ ONLY by a pi pitch, and the env's NATIVE
    observation cannot tell them apart in its pitch element (the fold).
    Controller 1's observation must.
    '''
    env = make_env(seed=0)
    try:
        assert env.observation_space.shape == (12,)
        upright = _state(pos=(0.0, 0.0, 1.5), euler=(0.0, 0.0, 0.0))
        inverted = _state(pos=(0.0, 0.0, 1.5), euler=(0.0, np.pi, 0.0))

        obs_u, _ = set_initial_state(env, upright)
        ctrl1_u = ctrl1_observation(env, obs_u)
        native_pitch_u = float(obs_u[7])

        obs_i, _ = set_initial_state(env, inverted)
        ctrl1_i = ctrl1_observation(env, obs_i)
        native_pitch_i = float(obs_i[7])

        assert ctrl1_u.shape == (18,)
        assert ctrl1_i.shape == (18,)
        # The trap: the native Euler pitch is the same for both.
        assert native_pitch_i == pytest.approx(native_pitch_u, abs=1e-6)
        # R22 lives at index 6 + 8 == 14 of the 18-dim layout.
        assert ctrl1_u[14] == pytest.approx(1.0, abs=1e-5)
        assert ctrl1_i[14] == pytest.approx(-1.0, abs=1e-5)
        assert np.abs(ctrl1_u - ctrl1_i).max() > 1.0
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 3. The reward is attitude-only.
# ---------------------------------------------------------------------------

def test_reward_is_invariant_to_position_and_translational_velocity():
    '''Spec D2.  Two (state, next_state) pairs differing ONLY in x, y, z and
    the translational velocities -- attitude and body rates held fixed -- must
    score identically.  A leak here (e.g. routing `info['out_of_bounds']` into
    the reward, which `Quadrotor._get_done` computes from position and
    translational velocity) would pull G1 toward RoA2.
    '''
    att_a = dict(euler=(0.3, 1.9, -0.7), rates=(2.0, -1.0, 0.5))
    att_b = dict(euler=(0.1, 0.05, 2.0), rates=(0.2, 0.1, -0.1))

    state_a = _state(pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), **att_a)
    state_b = _state(pos=(-1.7, 1.2, 0.2), vel=(2.9, -2.5, 1.1), **att_a)
    next_a = _state(pos=(0.5, -0.4, 2.6), vel=(-1.0, 0.3, -2.8), **att_b)
    next_b = _state(pos=(1.4, 0.9, 0.15), vel=(0.7, 2.2, 0.4), **att_b)

    assert potential(state_a) == pytest.approx(potential(state_b))
    for in_g_nom in (False, True):
        assert shaped_reward(state_a, next_a, in_g_nom=in_g_nom) == pytest.approx(
            shaped_reward(state_b, next_b, in_g_nom=in_g_nom))


def test_potential_prefers_upright_and_still():
    upright = _state(euler=(0.0, 0.02, 0.0), rates=(0.1, 0.0, 0.0))
    tumbling = _state(euler=(0.0, np.pi, 0.0), rates=(10.0, -8.0, 3.0))
    assert potential(upright) > potential(tumbling)


# ---------------------------------------------------------------------------
# 4. Sampling bounds track the env; attitude covers SO(3).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sampling_bounds_track_the_envs_own_state_space():
    '''Closed state space (CLAUDE.md): initialized == achieved.  Hardcoding the
    sampling box would silently desynchronise it from termination the moment a
    caller overrode a bound.
    '''
    env = make_env(seed=0, termination_overrides={1: (-5.0, 5.0), 9: (-30.0, 30.0)})
    try:
        wrapped = FlipTrainingEnv3D(env, G_NOM_3D, seed=0)
        low, high = wrapped.sampling_bounds
        ss_low, ss_high = env.state_space.low, env.state_space.high
        # dataset idx -> env idx, for every Euclidean dimension.
        for d_idx, e_idx in {0: 0, 1: 2, 2: 4, 7: 1, 8: 3, 9: 5,
                             10: 9, 11: 10, 12: 11}.items():
            assert low[d_idx] == pytest.approx(ss_low[e_idx])
            assert high[d_idx] == pytest.approx(ss_high[e_idx])
        # The overrides specifically must have propagated.
        assert high[7] == pytest.approx(5.0)    # x_dot
        assert high[10] == pytest.approx(30.0)  # p

        rng = np.random.default_rng(0)
        samples = np.array([sample_uniform_state(rng, (low, high)) for _ in range(400)])
        for d_idx in (0, 1, 2, 7, 8, 9, 10, 11, 12):
            assert samples[:, d_idx].min() >= low[d_idx] - 1e-9
            assert samples[:, d_idx].max() <= high[d_idx] + 1e-9
        # Attitude covers the whole of SO(3), not a slab around upright.
        tilts = np.array([tilt_from_quat_wxyz(s[3:7]) for s in samples])
        assert tilts.max() > np.radians(170)
        assert tilts.min() < np.radians(10)
        # Uniform SO(3) has cos(tilt) uniform on [-1, 1], hence mean tilt pi/2.
        assert np.mean(tilts) == pytest.approx(np.pi / 2, abs=0.15)

        # The cap narrows tilt ONLY.
        capped = np.array([sample_uniform_state(rng, (low, high), np.radians(45))
                           for _ in range(400)])
        capped_tilts = np.array([tilt_from_quat_wxyz(s[3:7]) for s in capped])
        assert capped_tilts.max() <= np.radians(45) + 1e-9
        for d_idx in (0, 1, 2, 7, 8, 9, 10, 11, 12):
            spread = capped[:, d_idx].max() - capped[:, d_idx].min()
            assert spread > 0.85 * (high[d_idx] - low[d_idx]), \
                f'dataset dimension {d_idx} was narrowed along with tilt'
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 5. Training runs end to end and emits a loadable checkpoint.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_training_emits_a_loadable_checkpoint():
    '''End-to-end subprocess run at a tiny step budget (2000, not the real
    1,000,000).  2000 is not arbitrary: SAC_CONFIG's warm_up_steps is 1000, so
    a shorter budget would never leave the random-action warm-up and would
    exercise neither the policy forward pass nor a single gradient update.

    tempfile.TemporaryDirectory()'s strict cleanup raises OSError on this
    machine's NFS mount whenever a controller's logger still holds an open
    file handle; use plain mkdtemp on local disk + best-effort rmtree.
    '''
    import torch

    tmp = tempfile.mkdtemp(prefix='train_flip3d_smoke_', dir=LOCAL_TMP)
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, 'train_quadrotor_3d_flip.py'),
             '--output_dir', tmp, '--max_env_steps', '2000', '--seed', '0',
             '--eval_interval', '0'],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=3600)
        assert result.returncode == 0, result.stderr[-3000:]
        checkpoint = os.path.join(tmp, 'flip_model.pt')
        assert os.path.exists(checkpoint)

        state = torch.load(checkpoint, weights_only=False)
        assert {'agent', 'obs_normalizer', 'reward_normalizer'} <= set(state)

        # The contract is "loadable by make('sac', ...).load()", and the one
        # caller that will do that is rollout3d.load_ctrl1 -- go through it, so
        # a future drift between the training config and rollout3d.SAC_CONFIG
        # (hidden_dim, activation, the 18-dim observation space, ...) fails
        # here rather than during dataset generation.
        from quad_composition.rollout3d import load_ctrl1
        env = make_env(seed=0)
        try:
            ctrl1 = load_ctrl1(checkpoint, env, output_dir=tmp)
            obs, info = set_initial_state(env, _state(euler=(0.0, np.pi, 0.0)))
            action = ctrl1.select_action(
                ctrl1.obs_normalizer(ctrl1_observation(env, obs)), info)
            assert np.asarray(action).shape == (4,)
            assert np.all(np.isfinite(action))
            ctrl1.close()
        finally:
            env.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
