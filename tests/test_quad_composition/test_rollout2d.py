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
            assert np.allclose(res.trajectory[-1], traj[-1], atol=1e-9), \
                f'final state mismatch from {init}'
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
