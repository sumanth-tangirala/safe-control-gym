'''The rollout core must reproduce the shipped quadrotor2D_rl dataset.

Spec: docs/superpowers/specs/2026-08-16-quad-composition-g1-design.md (D3, D4, D5)
'''
import math
import os
import sys
import tempfile

import numpy as np
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SHIPPED = ('/common/users/shared/pracsys/genMoPlan/data_trajectories/'
           'deterministic/quadrotor2D_rl/eval_states.txt')
MODEL = os.path.join(REPO_ROOT, 'examples/rl/models/safe_explorer_ppo/'
                                'safe_explorer_ppo_model_quadrotor_2D_stab.pt')


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
        def step(self, action):
            calls['steps'] += 1
            # Lands inside G1 (theta=0, theta_dot=0) AND finishes, same tick.
            obs = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            info = {'goal_reached': True}
            return obs, 0.0, True, info

    monkeypatch.setattr(rollout2d, 'set_initial_state',
                         lambda env, init_state: (np.zeros(6), {}))

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
        def __init__(self):
            self.n = 0

        def step(self, action):
            self.n += 1
            done = self.n >= 2
            obs = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            info = {'goal_reached': done}
            return obs, 0.0, done, info

    monkeypatch.setattr(rollout2d, 'set_initial_state',
                         lambda env, init_state: (np.zeros(6), {}))

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
        def step(self, action):
            obs = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            return obs, 0.0, True, {'goal_reached': False}

    monkeypatch.setattr(rollout2d, 'set_initial_state',
                         lambda env, init_state: (np.zeros(6), {}))

    theta_unnormalized = 4.0  # outside [-pi, pi]
    init_state = [0.0, 1.0, theta_unnormalized, 0.0, 0.0, 0.0]
    res = rollout2d.rollout_composite(FakeEnv(), None, FakeCtrl(), None,
                                       init_state, max_steps=1)

    assert res.trajectory[0][2] == pytest.approx(rollout2d.normalize_angle(theta_unnormalized))
    assert abs(res.trajectory[0][2]) <= math.pi


@pytest.mark.slow
def test_load_ctrl1_smoke_builds_a_sac_controller_against_the_shared_env(monkeypatch):
    '''Construction smoke test only: controller 1 has no trained checkpoint
    yet (a later task trains and exercises it), so this stops short of an
    actual `.load()` against a real file.
    '''
    from quad_composition.rollout2d import ENV_CONFIG, load_ctrl1
    from safe_control_gym.controllers.sac.sac import SAC
    from safe_control_gym.utils.registration import make

    monkeypatch.setattr(SAC, 'load', lambda self, path: None)

    with tempfile.TemporaryDirectory() as tmp:
        env = make('quadrotor', **ENV_CONFIG)
        try:
            ctrl1 = load_ctrl1('unused/path.pt', env, tmp)
            assert isinstance(ctrl1, SAC)
            assert ctrl1.obs_normalizer.read_only is True
            ctrl1.close()
        finally:
            env.close()


@pytest.mark.slow
def test_baseline_rollout_reproduces_the_shipped_labels():
    '''ctrl1=None must reproduce quadrotor2D_rl on its own initial states.

    Covers both label classes. Rows 0..39 are early-grid failures (fast,
    out-of-bounds terminations) -- but every one of them is class 0, so on
    their own they only exercise the OOB path and prove nothing about
    `goal_reached`, the success path that produces ctrl2_success, this
    experiment's primary label. Rows 4102..4121 are a contiguous window
    that is known to contain both classes (labels
    [0 0 0 0 0 1 0 0 1 1 1 1 0 0 0 0 0 0 0 0]), so they exercise the success
    path too. The assertion below on `labels` guards against the sample
    silently degenerating back to one class if the shipped file changes.

    KNOWN FAILURE on the mixed window, root-caused and not papered over here
    -- see the "Fix round 1" section of task-2-report.md for the full
    investigation. Per-step comparison against the shipped
    trajectories/sequence_*.txt files shows the rollout core tracks the
    shipped trajectory to ~1e-6 at every step (proven bit-identical to the
    untouched generate_quadrotor_2d_trajectories_rl.py when run fresh); that
    tiny per-step float noise compounds over a long, high-rate rollout and
    can shift the exact termination step by one (-> a final-state mismatch
    past atol=1e-4) or, for at least one borderline near-goal trajectory,
    flip the discrete success/failure outcome entirely. This is a
    reproducibility property of the chaotic closed-loop system, not a bug in
    this file -- do not loosen atol or special-case rows to force a pass.
    '''
    if not os.path.exists(SHIPPED):
        pytest.skip('shipped dataset not mounted')
    from quad_composition.rollout2d import make_env_and_ctrl2, rollout_composite

    early_rows = np.loadtxt(SHIPPED, delimiter=',', max_rows=40)
    mixed_rows = np.loadtxt(SHIPPED, delimiter=',', skiprows=4102, max_rows=20)
    rows = np.concatenate([early_rows, mixed_rows], axis=0)
    inits, finals, labels = rows[:, :6], rows[:, 6:12], rows[:, 12].astype(int)

    assert (labels == 1).any(), 'sample must include at least one success (label 1)'
    assert (labels == 0).any(), 'sample must include at least one failure (label 0)'

    with tempfile.TemporaryDirectory() as tmp:
        env, ctrl2 = make_env_and_ctrl2(MODEL, tmp)
        try:
            for init, final, label in zip(inits, finals, labels):
                res = rollout_composite(env, None, ctrl2, None, init)
                assert res.ctrl2_success == bool(label), f'label mismatch from {init}'
                assert np.allclose(res.trajectory[-1], final, atol=1e-4), \
                    f'final state mismatch from {init}'
                assert res.handoff_index == -1, 'no handoff without ctrl1'
                assert res.flip_success is False, 'not meaningful on the baseline path'
        finally:
            # ctrl2's logger holds an open file handle in tmp (output_dir);
            # close it before the TemporaryDirectory context tries to rmtree
            # tmp, or NFS leaves a phantom .nfsXXXX file and cleanup fails
            # with "Directory not empty" despite every assertion above passing.
            env.close()
            ctrl2.close()
