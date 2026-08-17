'''The rollout core must match the reference generation implementation, and
be statistically consistent with the shipped quadrotor2D_rl dataset.

Exact per-trajectory reproduction of the shipped dataset is NOT achievable
on this machine and is not what these tests assert -- see RULING D-I in
task-2-report.md ("Fix round 2") for the full investigation and the two
tests below for the actual equivalence claims this task supports.

Spec: docs/superpowers/specs/2026-08-16-quad-composition-g1-design.md (D3, D4, D5)
'''
import math
import os
import sys
import tempfile

import numpy as np
import pybullet as p
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SHIPPED = ('/common/users/shared/pracsys/genMoPlan/data_trajectories/'
           'deterministic/quadrotor2D_rl/eval_states.txt')
MODEL = os.path.join(REPO_ROOT, 'examples/rl/models/safe_explorer_ppo/'
                                'safe_explorer_ppo_model_quadrotor_2D_stab.pt')


def quat_for(theta):
    '''Body orientation for a pure pitch of `theta`, i.e. what a real env
    caches on `.quat`.

    Every fake env below carries one, because supervisory decisions read TRUE
    attitude off the rotation matrix (`rollout2d.true_theta`) rather than out
    of the gimbal-folded observation (Finding C1). A fake that only produced
    an obs would let the folded/true distinction go untested -- which is
    exactly how the bug survived: every attitude test used synthetic state
    vectors, where folding never happens.
    '''
    return p.getQuaternionFromEuler([0.0, float(theta), 0.0])


def fold_pitch(theta):
    '''The env's own gimbal fold, applied to a TRUE pitch.

    `p.getEulerFromQuaternion` returns the branch with pitch in [-pi/2, pi/2],
    so a true pitch t outside that range is reported as sign(t)*pi - t. This
    is the map that turns the rollout core's TRUE theta column back into what
    the reference generation script stores, and it exists only so the
    reference-equivalence test can compare the two conventions like for like.
    '''
    theta = math.atan2(math.sin(theta), math.cos(theta))
    if abs(theta) <= math.pi / 2:
        return theta
    return math.copysign(math.pi, theta) - theta


def fake_set_initial_state(env, init_state):
    '''Stand-in for rollout2d.set_initial_state against a fake env: places the
    drone at `init_state` (dataset order), including the quaternion.

    The returned obs carries the FOLDED pitch, computed by PyBullet itself,
    exactly as a real env's would -- so the step-0 latch decision is genuinely
    exercised against the fold rather than handed a pre-unfolded value.
    '''
    x, z, theta, x_dot, z_dot, theta_dot = init_state
    env.quat = quat_for(theta)
    folded = p.getEulerFromQuaternion(env.quat)[1]
    return np.array([x, x_dot, z, z_dot, folded, theta_dot], dtype=float), {}


def test_state_from_obs_reorders_env_obs_into_dataset_order():
    from quad_composition.rollout2d import state_from_obs

    # env order [x, x_dot, z, z_dot, theta, theta_dot]
    obs = np.array([0.1, 0.2, 1.3, 0.4, 0.5, 0.6])
    # dataset order [x, z, theta, x_dot, z_dot, theta_dot]
    assert state_from_obs(obs) == pytest.approx([0.1, 1.3, 0.5, 0.2, 0.4, 0.6])


def test_env_uses_controller_2s_restricted_action_space():
    '''Spec D6: TWR 1.10, alpha 53.1 -- not the physical actuator.'''
    from quad_composition.rollout2d import ENV_CONFIG
    assert ENV_CONFIG['normalized_rl_action_space'] is True
    assert 'norm_act_scale' not in ENV_CONFIG, 'must inherit the 0.1 default'


def test_sac_config_matches_sac_yaml_with_training_false():
    '''Ruling D-D: SAC_CONFIG is the full key set from sac.yaml, not the
    safe_explorer_ppo ALGO_CONFIG -- a later task loads a SAC checkpoint with
    it and must not silently get the wrong controller's hyperparameters.
    '''
    from quad_composition.rollout2d import SAC_CONFIG

    yaml_path = os.path.join(REPO_ROOT, 'safe_control_gym', 'controllers', 'sac', 'sac.yaml')
    with open(yaml_path) as f:
        expected = yaml.safe_load(f)
    expected['training'] = False

    assert SAC_CONFIG == expected


def test_a_step_that_enters_g1_and_finishes_does_not_step_a_dead_env(monkeypatch):
    '''Ruling D-E: a step that both enters G1 and sets done must be handled by
    evaluating `done` exactly once, after the latch update -- not skipped
    because latching took the `if not latched` branch instead of `elif done`.
    '''
    from quad_composition import rollout2d

    class FakeG1:
        def contains(self, tilt, omega):
            return bool(tilt < 0.05 and omega < 0.05)

    class FakeCtrl:
        def obs_normalizer(self, obs):
            return obs

        def select_action(self, obs, info):
            return np.zeros(2)

    calls = {'steps': 0}

    class FakeEnv:
        quat = quat_for(0.0)

        def step(self, action):
            calls['steps'] += 1
            # Lands inside G1 (theta=0, theta_dot=0) AND finishes, same tick.
            self.quat = quat_for(0.0)
            obs = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            info = {'goal_reached': True}
            return obs, 0.0, True, info

    monkeypatch.setattr(rollout2d, 'set_initial_state', fake_set_initial_state)

    # Outside G1 initially (theta=0.5, theta_dot=0.5).
    init_state = [0.0, 1.0, 0.5, 0.0, 0.0, 0.5]
    res = rollout2d.rollout_composite(FakeEnv(), FakeCtrl(), FakeCtrl(), FakeG1(),
                                       init_state, max_steps=5)

    assert calls['steps'] == 1, 'env must not be stepped again after it is done'
    assert res.handoff_index == 1
    assert res.ctrl2_success is True


def test_ctrl1_none_baseline_has_no_handoff_and_flip_success_false(monkeypatch):
    '''Ruling D-F: on the baseline path (ctrl1=None), handoff_index must be -1
    and flip_success must be False, not handoff_index=0 as the brief's sample
    implementation had it -- that would silently claim a handoff which never
    happened, contradicting the brief's own equivalence test below.
    '''
    from quad_composition import rollout2d

    class FakeCtrl:
        def obs_normalizer(self, obs):
            return obs

        def select_action(self, obs, info):
            return np.zeros(2)

    class FakeEnv:
        quat = quat_for(0.0)

        def __init__(self):
            self.n = 0

        def step(self, action):
            self.n += 1
            done = self.n >= 2
            obs = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            info = {'goal_reached': done}
            return obs, 0.0, done, info

    monkeypatch.setattr(rollout2d, 'set_initial_state', fake_set_initial_state)

    res = rollout2d.rollout_composite(FakeEnv(), None, FakeCtrl(), None,
                                       [0, 1, 0, 0, 0, 0], max_steps=5)

    assert res.handoff_index == -1
    assert res.flip_success is False
    assert res.ctrl2_success is True


def test_seed_row_theta_is_normalized(monkeypatch):
    '''The seed (row 0) trajectory entry must go through the same theta
    normalization as every later row (state_from_obs), matching
    generate_quadrotor_2d_trajectories_rl.py's run_trajectory (line ~494).
    This is dormant on the shipped dataset (its init states are already
    normalized) but a later task calibrating G1 will pass raw, unnormalized
    theta and would otherwise silently write a bad row 0.
    '''
    from quad_composition import rollout2d

    class FakeCtrl:
        def obs_normalizer(self, obs):
            return obs

        def select_action(self, obs, info):
            return np.zeros(2)

    class FakeEnv:
        quat = quat_for(0.0)

        def step(self, action):
            obs = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            return obs, 0.0, True, {'goal_reached': False}

    monkeypatch.setattr(rollout2d, 'set_initial_state', fake_set_initial_state)

    theta_unnormalized = 4.0  # outside [-pi, pi]
    init_state = [0.0, 1.0, theta_unnormalized, 0.0, 0.0, 0.0]
    res = rollout2d.rollout_composite(FakeEnv(), None, FakeCtrl(), None,
                                       init_state, max_steps=1)

    assert res.trajectory[0][2] == pytest.approx(rollout2d.normalize_angle(theta_unnormalized))
    assert abs(res.trajectory[0][2]) <= math.pi


# ---------------------------------------------------------------------------
# Finding C1: TRUE vs GIMBAL-FOLDED attitude.
#
# These are REAL-ENV tests on purpose. Every attitude test that existed before
# this fix used synthetic state vectors, where a theta is whatever you wrote
# down and the fold never happens -- which is precisely why a whole branch of
# attitude logic could be computed on the wrong quantity with a green suite.
# The fold only exists on the far side of PyBullet's quaternion.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_true_theta_recovers_the_attitude_the_observation_folds_away():
    '''`p.getEulerFromQuaternion` returns the branch with pitch in
    [-pi/2, pi/2], so the env's observed theta for a nearly-inverted drone is
    a small number. `true_theta` must recover the real one, for both signs,
    over the whole range up to |theta| = pi. Fails on the pre-fix code, where
    the only available theta was the folded one.
    '''
    from quad_composition.rollout2d import (make_env, set_initial_state, state_from_env, state_from_obs,
                                            true_theta)

    env = make_env(seed=0)
    try:
        for theta in (0.0, 1.0, -1.0, 1.5, 2.0, -2.0, 3.0, -3.0, math.pi - 1e-4):
            obs, _ = set_initial_state(env, [0.0, 1.0, theta, 0.0, 0.0, 0.0])
            assert true_theta(env) == pytest.approx(theta, abs=1e-3), \
                f'true attitude not recovered for theta={theta}'
            assert state_from_env(env, obs)[2] == pytest.approx(theta, abs=1e-3)

        # Not vacuous: the observation really is folded, by a lot.
        obs, _ = set_initial_state(env, [0.0, 1.0, 3.0, 0.0, 0.0, 0.0])
        assert state_from_obs(obs)[2] == pytest.approx(math.pi - 3.0, abs=1e-3)
        assert abs(state_from_obs(obs)[2]) < 0.2, 'an inverted drone reads as upright'
    finally:
        env.close()


@pytest.mark.slow
def test_a_fully_inverted_drone_is_not_in_g1():
    '''G1 is attitude-only, so it is exactly the decision the fold corrupts:
    an upside-down drone must not be handed to controller 2 as "upright".
    '''
    from quad_composition.g1 import G1Region
    from quad_composition.rollout2d import make_env, set_initial_state, state_from_env, state_from_obs

    g1 = G1Region(tilt_c=0.175, w_c=1.0)    # G_NOM's numbers: 10 deg, 1 rad/s
    env = make_env(seed=0)
    try:
        obs, _ = set_initial_state(env, [0.0, 1.0, math.pi - 0.05, 0.0, 0.0, 0.0])

        state = state_from_env(env, obs)
        assert not bool(g1.contains(abs(state[2]), abs(state[5]))), \
            'a drone 0.05 rad from fully inverted must not be inside G1'

        # And this is what the pre-fix code was actually asking, on the same
        # physical state -- kept as a live demonstration of the bug.
        folded = state_from_obs(obs)
        assert bool(g1.contains(abs(folded[2]), abs(folded[5]))), \
            'the folded observation says an inverted drone IS in G1'
    finally:
        env.close()


@pytest.mark.slow
def test_rollout_composite_does_not_hand_off_an_inverted_drone_on_the_real_env():
    '''End to end through the real env: starting fully inverted, the handoff
    must never fire. Before the fix the folded obs put the very first step
    inside G1, so controller 2 was handed an upside-down drone at step ~1.
    '''
    from quad_composition.g1 import G1Region
    from quad_composition.rollout2d import make_env, rollout_composite

    class FakeCtrl:
        def obs_normalizer(self, obs):
            return obs

        def select_action(self, obs, info):
            return np.zeros(2)

    g1 = G1Region(tilt_c=0.175, w_c=1.0)
    env = make_env(seed=0)
    try:
        res = rollout_composite(env, FakeCtrl(), FakeCtrl(), g1,
                                [0.0, 1.0, math.pi - 0.05, 0.0, 0.0, 0.0], max_steps=40)
        assert res.handoff_index == -1, \
            f'spurious handoff at step {res.handoff_index} on an inverted drone'
        assert res.flip_success is False
        # The stored theta column is TRUE attitude, so it must stay near pi --
        # a folded column would show |theta| <= pi/2 throughout.
        assert max(abs(row[2]) for row in res.trajectory) > math.pi / 2
    finally:
        env.close()


def test_the_same_true_attitude_is_classified_identically_at_step_0_and_step_1(monkeypatch):
    '''Finding C2: `rollout_composite` used to test the RAW (true) init theta
    at step 0 and the FOLDED obs theta at step >= 1, so one physical attitude
    got two different answers depending on when it appeared. theta = 3.0
    (folded: 0.1416) is the discriminating case for a G1 with tilt_c = 0.2:
    true says "outside", folded says "inside".

    The fake env below folds for real, via PyBullet, so it is not begging the
    question -- it reports exactly what a real env reports.
    '''
    from quad_composition import rollout2d
    from quad_composition.g1 import G1Region

    class FakeCtrl:
        def obs_normalizer(self, obs):
            return obs

        def select_action(self, obs, info):
            return np.zeros(2)

    class FoldingEnv:
        '''Lands at `next_theta` (TRUE) on its first step and finishes.'''

        def __init__(self, next_theta):
            self.next_theta = next_theta
            self.quat = quat_for(0.0)

        def step(self, action):
            self.quat = quat_for(self.next_theta)
            folded = p.getEulerFromQuaternion(self.quat)[1]
            return np.array([0.0, 0.0, 1.0, 0.0, folded, 0.0]), 0.0, True, {}

    g1 = G1Region(tilt_c=0.2, w_c=1.0)
    monkeypatch.setattr(rollout2d, 'set_initial_state', fake_set_initial_state)

    def handoff_for(step0_theta, step1_theta):
        env = FoldingEnv(step1_theta)
        res = rollout2d.rollout_composite(
            env, FakeCtrl(), FakeCtrl(), g1,
            [0.0, 1.0, step0_theta, 0.0, 0.0, 0.0], max_steps=3)
        return res.handoff_index

    # theta = 3.0 is outside G1 wherever it appears.
    assert handoff_for(3.0, 0.0) == 1, 'sanity: an upright step 1 must latch'
    assert handoff_for(3.0, 3.0) == -1, 'true theta 3.0 at step 1 must not latch'
    assert handoff_for(0.5, 3.0) == -1, 'true theta 3.0 at step 1 must not latch'
    # theta = 0.1 is inside G1 wherever it appears.
    assert handoff_for(0.1, 0.1) == 0, 'true theta 0.1 at step 0 must latch at 0'
    assert handoff_for(3.0, 0.1) == 1, 'true theta 0.1 at step 1 must latch at 1'


# ---------------------------------------------------------------------------
# Spec D6: controller 1's observation is UNFOLDED. Ruling on the OPEN ITEM
# flagged at the end of the C1/C2 fix wave (final-fix-report.md) -- folding
# aliases true pitch t with sign(t)*pi - t, so an upright drone and an
# inverted one present the identical raw observation while their dynamics
# are opposite (thrust up vs thrust down). All real-env: every attitude bug
# on this branch so far survived exactly because its tests used synthetic
# state vectors, where the fold never happens.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ctrl1_observation_is_7d_and_cos_sin_matches_true_attitude():
    '''`ctrl1_observation` must be 7-dim, with the folded theta element (env
    index 4) replaced by (cos(true_theta), sin(true_theta)) and the other
    five elements unchanged in their existing positions. Swept over both
    signs and past the fold boundary (|theta| > pi/2), through the real env.
    '''
    from quad_composition.rollout2d import ctrl1_observation, make_env, set_initial_state

    env = make_env(seed=0)
    try:
        for theta in (0.5, 2.0, 3.0, -3.0):
            obs, _ = set_initial_state(env, [0.0, 1.0, theta, 0.0, 0.0, 0.0])
            obs7 = ctrl1_observation(env, obs)

            assert obs7.shape == (7,)
            assert obs7[4] == pytest.approx(math.cos(theta), abs=1e-3), f'cos mismatch at theta={theta}'
            assert obs7[5] == pytest.approx(math.sin(theta), abs=1e-3), f'sin mismatch at theta={theta}'
            # The other five elements (x, x_dot, z, z_dot, theta_dot) are
            # unchanged and keep their positions.
            np.testing.assert_allclose(obs7[[0, 1, 2, 3]], np.asarray(obs)[[0, 1, 2, 3]])
            assert obs7[6] == pytest.approx(obs[5])
            # cos^2 + sin^2 == 1 to tight tolerance (obs7 is float32, whose
            # machine epsilon is ~1.19e-7, so 1e-6 is tight relative to the
            # dtype rather than an arbitrary loosening).
            assert obs7[4] ** 2 + obs7[5] ** 2 == pytest.approx(1.0, abs=1e-6), \
                f'cos^2+sin^2 != 1 at theta={theta}'
    finally:
        env.close()


@pytest.mark.slow
def test_ctrl1_observation_distinguishes_upright_from_inverted():
    '''The whole point of spec D6. On the RAW folded observation, an upright
    drone (true theta 0.05) and a nearly-inverted one (true theta pi - 0.05)
    are almost indistinguishable -- both observe theta ~ 0.05 (asserted below,
    non-vacuously). `ctrl1_observation` must tell them apart: this fails on
    the pre-fix code, which handed the policy the raw (folded) obs unchanged.
    '''
    from quad_composition.rollout2d import ctrl1_observation, make_env, set_initial_state, state_from_obs

    env = make_env(seed=0)
    try:
        # ctrl1_observation reads env.quat -- the env's CURRENT state -- not
        # something embedded in `obs`, so it must be called immediately after
        # the set_initial_state call that produced that obs, exactly as
        # rollout_composite/FlipTrainingEnv/collect_exit_attitudes all do.
        # Deferring it past the second set_initial_state call below would
        # read both "up" and "down" off the SAME (latest) env.quat -- the
        # very same class of step-0-vs-step-N mixup as Finding C2.
        obs_up, _ = set_initial_state(env, [0.0, 1.0, 0.05, 0.0, 0.0, 0.0])
        up7 = ctrl1_observation(env, obs_up)
        folded_up = state_from_obs(obs_up)[2]

        obs_down, _ = set_initial_state(env, [0.0, 1.0, math.pi - 0.05, 0.0, 0.0, 0.0])
        down7 = ctrl1_observation(env, obs_down)
        folded_down = state_from_obs(obs_down)[2]

        # Not vacuous: the RAW folded observations are nearly identical.
        assert folded_up == pytest.approx(folded_down, abs=1e-3)

        assert not np.allclose(up7, down7, atol=1e-2), \
            'controller 1 must see upright and inverted as different observations'
        # cos flips sign between upright and (nearly) inverted -- the
        # discriminating feature the folded observation could never carry.
        assert up7[4] > 0.9
        assert down7[4] < -0.9
    finally:
        env.close()


@pytest.mark.slow
def test_act_ctrl1_and_act_ctrl2_feed_different_observations_on_the_real_env():
    '''`_act_ctrl1` must feed controller 1 the 7-dim unfolded observation;
    `_act_ctrl2` must feed controller 2 the SAME raw obs the env actually
    produced, byte-for-byte -- exactly 6 dims, folded theta untouched. Real
    env so the folded theta is genuine PyBullet output, not a value nobody's
    Euler-angle solver ever computed.
    '''
    from quad_composition.rollout2d import _act_ctrl1, _act_ctrl2, make_env, set_initial_state

    class RecordingCtrl:
        def obs_normalizer(self, obs):
            return obs

        def select_action(self, obs, info):
            self.seen = np.array(obs, dtype=float)
            return np.zeros(2)

    env = make_env(seed=0)
    try:
        obs, info = set_initial_state(env, [0.1, 0.9, 3.0, 0.05, -0.05, 0.3])

        ctrl1, ctrl2 = RecordingCtrl(), RecordingCtrl()
        _act_ctrl1(env, ctrl1, obs, info)
        _act_ctrl2(ctrl2, obs, info)

        assert ctrl1.seen.shape == (7,), 'controller 1 must receive the 7-dim unfolded observation'
        assert ctrl2.seen.shape == (6,), 'controller 2 must receive exactly 6 dims'
        np.testing.assert_array_equal(ctrl2.seen, np.asarray(obs, dtype=float))
    finally:
        env.close()


@pytest.mark.slow
def test_load_ctrl1_smoke_builds_a_sac_controller_against_the_shared_env(monkeypatch):
    '''Construction smoke test only: controller 1 has no trained checkpoint
    yet (a later task trains and exercises it), so this stops short of an
    actual `.load()` against a real file.

    Spec D6: controller 1's network must be sized to the 7-dim
    `ctrl1_observation_space`, not the raw env's native 6-dim one, and
    `select_action` must actually run against a real 7-dim observation
    computed from the real env -- otherwise a checkpoint trained under
    `train_quadrotor_2d_flip.py` (which wraps the SAME 7-dim space via
    `FlipTrainingEnv`) would fail to load with a shape mismatch the first
    time anyone tried it.
    '''
    from quad_composition.rollout2d import ENV_CONFIG, ctrl1_observation, load_ctrl1, set_initial_state
    from safe_control_gym.controllers.sac.sac import SAC
    from safe_control_gym.utils.registration import make

    monkeypatch.setattr(SAC, 'load', lambda self, path: None)

    with tempfile.TemporaryDirectory() as tmp:
        env = make('quadrotor', **ENV_CONFIG)
        try:
            ctrl1 = load_ctrl1('unused/path.pt', env, tmp)
            assert isinstance(ctrl1, SAC)
            assert ctrl1.obs_normalizer.read_only is True
            assert ctrl1.env.observation_space.shape == (7,), \
                "controller 1's network must be sized to the 7-dim observation space"

            obs, info = set_initial_state(env, [0.0, 1.0, 2.0, 0.0, 0.0, 0.0])
            action = ctrl1.select_action(ctrl1.obs_normalizer(ctrl1_observation(env, obs)), info)
            assert action.shape == env.action_space.shape

            ctrl1.close()
        finally:
            env.close()


# The only window verified to contain both label classes in one contiguous
# span (Fix round 1). Used by both tests below.
MIXED_WINDOW_SKIPROWS = 4102
MIXED_WINDOW_ROWS = 20


@pytest.mark.slow
def test_rollout_core_matches_the_reference_implementation():
    '''RULING D-I(a): reference equivalence -- the real gate.

    Exact bit-level reproduction of the shipped quadrotor2D_rl dataset is not
    achievable on this machine (Fix round 2, task-2-report.md): even the
    UNTOUCHED, unmodified generate_quadrotor_2d_trajectories_rl.run_trajectory
    does not reproduce its own shipped eval_states.txt on this mixed-class
    window (measured directly: label agreement 19/20, final-state agreement
    12/20 at atol=1e-4; one row's discrete outcome flips). That is a chaotic
    divergence from whatever PyBullet/library/hardware state generated the
    dataset -- not something any implementation run today can fix.

    What this task can and must guarantee is that quad_composition.rollout2d
    exactly matches the reference generation script when both run in the
    same process, on the same machine, against the same checkpoint. That is
    the equivalence this test asserts, at atol=1e-9 (not 1e-4) on the final
    state, plus label equality: if this ever fails, the rollout core has
    drifted from the reference implementation, which is a real bug.

    ONE CONVENTION DIFFERENCE IS EXPECTED, AND IS CHECKED RATHER THAN WAIVED
    (Finding C1): the reference script stores the env's GIMBAL-FOLDED
    observation theta, while the rollout core now stores TRUE attitude. The
    theta column is therefore compared through `fold_pitch`, which maps our
    true value onto the reference's branch; every other column is still
    compared directly at atol=1e-9. That still pins our theta to the
    reference's physics up to a known, invertible map -- it is not a dropped
    column. What pins the REPRESENTATION (true, not folded) is the separate
    set of real-env attitude tests further down this file.
    '''
    if not os.path.exists(SHIPPED):
        pytest.skip('shipped dataset not mounted')
    import shutil
    from functools import partial

    import generate_quadrotor_2d_trajectories_rl as gen_script
    from quad_composition.rollout2d import make_env_and_ctrl2, rollout_composite
    from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
    from safe_control_gym.utils.registration import make

    rows = np.loadtxt(SHIPPED, delimiter=',', skiprows=MIXED_WINDOW_SKIPROWS,
                       max_rows=MIXED_WINDOW_ROWS)
    inits, labels = rows[:, :6], rows[:, 12].astype(int)
    assert (labels == 1).any(), 'window must include at least one success (label 1)'
    assert (labels == 0).any(), 'window must include at least one failure (label 0)'

    # Independently reconstructed from the ORIGINAL script's own definitions
    # (not quad_composition.rollout2d's ENV_CONFIG/ALGO_CONFIG), so this test
    # also catches config drift between the two, not just loop-logic drift.
    ref_env_kwargs = {
        'quad_type': QuadType.TWO_D, 'task': 'stabilization',
        'ctrl_freq': 100, 'pyb_freq': 5000, 'episode_len_sec': 1000,
        'done_on_out_of_bound': True, 'cost': 'quadratic',
        'normalized_rl_action_space': True, 'gui': False, 'randomized_init': False,
        'constraints': gen_script.SAFE_EXPLORER_CONSTRAINTS, 'done_on_violation': False,
        'task_info': {'stabilization_goal': [0, 1], 'stabilization_goal_tolerance': 0.2},
    }
    ref_termination = {0: (-1.0, 1.0), 1: (-1.0, 1.0), 2: (0.1, 1.5),
                       3: (-1.0, 1.0), 4: (-np.inf, np.inf), 5: (-8.0, 8.0)}

    # tempfile.TemporaryDirectory's strict cleanup raises OSError on this
    # NFS mount whenever a controller's logger still holds an open file
    # handle (Fix round 1); use plain mkdtemp + best-effort rmtree instead.
    out_ours = tempfile.mkdtemp(prefix='rollout2d_ref_ours_')
    out_ref = tempfile.mkdtemp(prefix='rollout2d_ref_orig_')
    env = ctrl2 = env_ref = ctrl_ref = None
    try:
        env, ctrl2 = make_env_and_ctrl2(MODEL, out_ours)

        env_func = partial(make, 'quadrotor', **ref_env_kwargs)
        ctrl_ref = make('safe_explorer_ppo', env_func,
                        **gen_script.ALGO_CONFIGS['safe_explorer_ppo'], output_dir=out_ref)
        ctrl_ref.load(MODEL)
        ctrl_ref.obs_normalizer.set_read_only()
        env_ref = env_func()
        for idx, (lo, hi) in ref_termination.items():
            env_ref.state_space.low[idx] = lo
            env_ref.state_space.high[idx] = hi

        for init in inits:
            res = rollout_composite(env, None, ctrl2, None, init)
            traj, success, _, _ = gen_script.run_trajectory(
                env_ref, ctrl_ref, init.tolist(), 'safe_explorer_ppo', max_steps=1200)

            assert res.ctrl2_success == bool(success), f'label mismatch from {init}'
            ours, ref = np.asarray(res.trajectory[-1]), np.asarray(traj[-1])
            non_theta = [0, 1, 3, 4, 5]
            assert np.allclose(ours[non_theta], ref[non_theta], atol=1e-9), \
                f'final state mismatch from {init}'
            assert fold_pitch(ours[2]) == pytest.approx(ref[2], abs=1e-9), \
                f'final theta mismatch from {init} (ours is TRUE attitude, ' \
                f"the reference's is the env's folded observation)"
    finally:
        for obj in (env, ctrl2, env_ref, ctrl_ref):
            if obj is not None:
                obj.close()
        shutil.rmtree(out_ours, ignore_errors=True)
        shutil.rmtree(out_ref, ignore_errors=True)


@pytest.mark.slow
def test_baseline_rollout_is_statistically_consistent_with_the_shipped_labels():
    '''RULING D-I(b): statistical consistency, not exact reproduction.

    Exact per-row reproduction of the shipped quadrotor2D_rl dataset is
    impossible on this machine -- even the UNTOUCHED
    generate_quadrotor_2d_trajectories_rl.run_trajectory only reproduces its
    own shipped labels on 19/20 rows and its own shipped final states on
    12/20 rows (atol=1e-4) on this exact mixed-class window (measured
    directly, Fix round 2, task-2-report.md). That is chaotic divergence
    from whatever environment generated the dataset, not a rollout-core
    defect: test_rollout_core_matches_the_reference_implementation pins the
    core to the reference implementation exactly (atol=1e-9).

    So this test does not assert per-row equality -- that would be asserting
    something false. It asserts a >=0.85 label-agreement rate against the
    shipped file (comfortably below the reference script's own 19/20 = 0.95
    self-consistency, so passing does not depend on chaotic luck matching
    exactly), plus both-classes presence. This still catches gross
    regressions (wrong checkpoint, wrong action-space scale, wrong goal
    tolerance) without asserting an untrue exact match.
    '''
    if not os.path.exists(SHIPPED):
        pytest.skip('shipped dataset not mounted')
    import shutil

    from quad_composition.rollout2d import make_env_and_ctrl2, rollout_composite

    rows = np.loadtxt(SHIPPED, delimiter=',', skiprows=MIXED_WINDOW_SKIPROWS,
                       max_rows=MIXED_WINDOW_ROWS)
    inits, labels = rows[:, :6], rows[:, 12].astype(int)
    assert (labels == 1).any(), 'window must include at least one success (label 1)'
    assert (labels == 0).any(), 'window must include at least one failure (label 0)'

    out = tempfile.mkdtemp(prefix='rollout2d_stat_')
    env = ctrl2 = None
    try:
        env, ctrl2 = make_env_and_ctrl2(MODEL, out)
        n_match = 0
        for init, label in zip(inits, labels):
            res = rollout_composite(env, None, ctrl2, None, init)
            n_match += int(res.ctrl2_success == bool(label))
            assert res.handoff_index == -1, 'no handoff without ctrl1'
            assert res.flip_success is False, 'not meaningful on the baseline path'
        rate = n_match / len(labels)
        assert rate >= 0.85, f'label agreement {rate:.2f} too low ({n_match}/{len(labels)})'
    finally:
        if env is not None:
            env.close()
        if ctrl2 is not None:
            ctrl2.close()
        shutil.rmtree(out, ignore_errors=True)
