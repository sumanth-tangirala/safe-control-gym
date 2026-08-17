'''The supervisory guard: `quad_composition.guard3d` (feature extraction and
the frozen fitted model) and the `guard=` parameter it plugs into
`rollout3d.rollout_composite`.

Fixtures (`_quat`, `_dataset_state`, `_FakeCtrl`, `_ScriptedEnv3D`,
`_fake_set_initial_state`) are duplicated from test_composition_datasets3d.py
rather than shared via a conftest -- that file documents the same choice for
its own 2D/3D port, and no conftest.py exists in this directory. Read that
file's docstring first for why `_ScriptedEnv3D` is action-blind (`.step()`
replays a fixed script regardless of the action passed in): that property is
exactly what makes the end-to-end guard tests below strong rather than
tautological -- `handoff_index`/`flip_success`/`ctrl2_success` depend ONLY on
`rollout_composite`'s internal routing logic (which controller was
"effectively" active, and when `in_g1` fired), never on the scripted env's
physics, so a routing bug shows up as a wrong label, not just a wrong
trajectory value.
'''
import os
import sys

import numpy as np
import pybullet as p
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Fixtures -- see test_composition_datasets3d.py for the originals.
# ---------------------------------------------------------------------------

def _quat(tilt):
    return p.getQuaternionFromEuler([0.0, float(tilt), 0.0])


def _dataset_state(tilt=0.0, omega=0.0, pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0)):
    from quad_composition.rollout3d import canonical_quat_wxyz
    qx, qy, qz, qw = _quat(tilt)
    qw, qx, qy, qz = canonical_quat_wxyz([qx, qy, qz, qw])
    return [pos[0], pos[1], pos[2], qw, qx, qy, qz, vel[0], vel[1], vel[2], omega, 0.0, 0.0]


class _FakeCtrl:
    def obs_normalizer(self, obs):
        return obs

    def select_action(self, obs, info):
        return np.zeros(4)

    def close(self):
        pass


class _ScriptedEnv3D:
    '''Replays a fixed (tilt, omega, done, goal_reached) script per `.step()`
    call, IGNORING the action passed in -- see this module's docstring.
    '''

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.quat = _quat(0.0)

    def step(self, action):
        tilt, omega, done, goal_reached = self.script[self.i]
        self.i += 1
        self.quat = _quat(tilt)
        obs = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0,
                        omega, 0.0, 0.0])
        info = {'goal_reached': bool(done and goal_reached)}
        return obs, 0.0, done, info

    def close(self):
        pass


def _fake_set_initial_state(env, init_state):
    from quad_composition.rollout3d import quat_wxyz_to_pybullet
    env.i = 0
    s = np.asarray(init_state, dtype=float)
    env.quat = quat_wxyz_to_pybullet(s[3:7])
    obs = np.array([s[0], s[7], s[1], s[8], s[2], s[9],
                    0.0, 0.0, 0.0,
                    s[10], s[11], s[12]])
    return obs, {}


def _small_g1():
    from quad_composition.g1 import G1Region
    return G1Region(tilt_c=0.05, w_c=0.05)


# A script that starts OUTSIDE g1 (tilt=0.5, omega=0.5), stays outside for one
# more step, enters g1 at trajectory row 2, then finishes successfully under
# ctrl2. With ctrl1 active this is a REAL handoff (flip_success=True,
# handoff_index=2); with ctrl1 overridden to None it never even tests g1 at
# step 0 as "outside", it just latches immediately (handoff_index=-1).
_HANDOFF_SCRIPT = [
    (0.5, 0.5, False, False),
    (0.02, 0.01, False, False),   # enters g1 -> handoff_index == 2
    (0.0, 0.0, True, True),
]
_OUTSIDE_G1_INIT = _dataset_state(tilt=0.5, omega=0.5)


def _results_equal(a, b):
    return (np.allclose(a.trajectory, b.trajectory)
            and a.handoff_index == b.handoff_index
            and a.flip_success == b.flip_success
            and a.ctrl2_success == b.ctrl2_success)


# ---------------------------------------------------------------------------
# guard3d.guard_features -- pure function, no env/controller needed.
# ---------------------------------------------------------------------------

def test_guard_features_at_the_goal_state_are_all_zero():
    from quad_composition.guard3d import guard_features
    at_goal = _dataset_state(tilt=0.0, omega=0.0, pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0))
    tilt, omega, speed, dist = guard_features(at_goal)
    assert (tilt, omega, speed, dist) == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-9)


def test_guard_features_tilt_matches_tilt_from_quat_wxyz():
    from quad_composition.guard3d import guard_features
    from quad_composition.rollout3d import QUAT_SLICE, tilt_from_quat_wxyz
    state = _dataset_state(tilt=1.2, omega=3.0)
    features = guard_features(state)
    assert features[0] == pytest.approx(tilt_from_quat_wxyz(np.asarray(state)[QUAT_SLICE]))


def test_guard_features_omega_and_speed_and_dist_are_the_expected_norms():
    from quad_composition.guard3d import guard_features
    state = _dataset_state(tilt=0.0, omega=0.0, pos=(3.0, 4.0, 1.0), vel=(1.0, 0.0, 0.0))
    tilt, omega, speed, dist = guard_features(state)
    assert omega == pytest.approx(0.0)
    assert speed == pytest.approx(1.0)
    assert dist == pytest.approx(5.0)   # |(3, 4, 0)| from goal (0, 0, 1)


# ---------------------------------------------------------------------------
# guard3d.LogisticGuard -- deterministic, threshold-driven.
# ---------------------------------------------------------------------------

def test_logistic_guard_predict_is_deterministic():
    from quad_composition.guard3d import FITTED_GUARD
    state = _dataset_state(tilt=0.3, omega=1.5, pos=(0.4, -0.2, 1.1), vel=(0.5, 0.1, -0.2))
    assert FITTED_GUARD.predict(state) == FITTED_GUARD.predict(state)
    assert FITTED_GUARD.predict_proba(state) == FITTED_GUARD.predict_proba(state)


def test_logistic_guard_threshold_zero_always_predicts_true():
    from quad_composition.guard3d import LogisticGuard
    guard = LogisticGuard(mean=(0, 0, 0, 0), std=(1, 1, 1, 1), coef=(1, 1, 1, 1),
                          intercept=-100.0, threshold=0.0)
    assert guard.predict(_dataset_state(tilt=3.0, omega=20.0)) is True


def test_logistic_guard_threshold_above_one_always_predicts_false():
    from quad_composition.guard3d import LogisticGuard
    guard = LogisticGuard(mean=(0, 0, 0, 0), std=(1, 1, 1, 1), coef=(1, 1, 1, 1),
                          intercept=100.0, threshold=1.0001)
    assert guard.predict(_dataset_state(tilt=0.0, omega=0.0)) is False


def test_fitted_guard_predicts_lqr_succeeds_exactly_at_the_goal():
    '''Regression pin on the frozen fit: the goal state itself (upright,
    still, at the goal) is the easiest possible state for LQR -- if the
    fitted guard does not predict success there, something about the fit or
    the sign of its coefficients is wrong.
    '''
    from quad_composition.guard3d import FITTED_GUARD
    at_goal = _dataset_state(tilt=0.0, omega=0.0, pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0))
    assert FITTED_GUARD.predict(at_goal) is True


def test_fitted_guard_declines_far_from_the_goal_and_inverted():
    from quad_composition.guard3d import FITTED_GUARD
    hard = _dataset_state(tilt=np.pi, omega=15.0, pos=(1.7, 1.7, 2.9), vel=(2.5, 2.5, 2.5))
    assert FITTED_GUARD.predict(hard) is False


# ---------------------------------------------------------------------------
# rollout_composite(..., guard=...) -- the three end-to-end checks the spec
# calls out explicitly.
# ---------------------------------------------------------------------------

def test_guard_none_reproduces_current_behaviour_exactly(monkeypatch):
    from quad_composition import rollout3d
    monkeypatch.setattr(rollout3d, 'set_initial_state', _fake_set_initial_state)
    ctrl = _FakeCtrl()
    g1 = _small_g1()

    no_kwarg = rollout3d.rollout_composite(
        _ScriptedEnv3D(_HANDOFF_SCRIPT), ctrl, ctrl, g1, _OUTSIDE_G1_INIT)
    explicit_none = rollout3d.rollout_composite(
        _ScriptedEnv3D(_HANDOFF_SCRIPT), ctrl, ctrl, g1, _OUTSIDE_G1_INIT, guard=None)

    assert _results_equal(no_kwarg, explicit_none)
    # And this is a REAL handoff, not a trivially-idle rollout: proves the
    # comparison above isn't vacuous.
    assert no_kwarg.flip_success is True
    assert no_kwarg.handoff_index == 2


def test_guard_always_true_reproduces_the_baseline_exactly(monkeypatch):
    '''A guard that always says "LQR succeeds" must produce EXACTLY the
    ctrl1=None baseline result, even though a real (fake) controller 1 was
    passed in and, unguarded, would have taken a real handoff on this init
    (see the test above: flip_success=True, handoff_index=2 without a guard).
    '''
    from quad_composition import rollout3d
    monkeypatch.setattr(rollout3d, 'set_initial_state', _fake_set_initial_state)
    ctrl = _FakeCtrl()
    g1 = _small_g1()

    guarded = rollout3d.rollout_composite(
        _ScriptedEnv3D(_HANDOFF_SCRIPT), ctrl, ctrl, g1, _OUTSIDE_G1_INIT,
        guard=lambda state: True)
    baseline = rollout3d.rollout_composite(
        _ScriptedEnv3D(_HANDOFF_SCRIPT), None, ctrl, g1, _OUTSIDE_G1_INIT)

    assert _results_equal(guarded, baseline)
    assert guarded.handoff_index == -1
    assert guarded.flip_success is False


def test_guard_always_false_reproduces_the_unguarded_composition_exactly(monkeypatch):
    from quad_composition import rollout3d
    monkeypatch.setattr(rollout3d, 'set_initial_state', _fake_set_initial_state)
    ctrl = _FakeCtrl()
    g1 = _small_g1()

    guarded = rollout3d.rollout_composite(
        _ScriptedEnv3D(_HANDOFF_SCRIPT), ctrl, ctrl, g1, _OUTSIDE_G1_INIT,
        guard=lambda state: False)
    unguarded = rollout3d.rollout_composite(
        _ScriptedEnv3D(_HANDOFF_SCRIPT), ctrl, ctrl, g1, _OUTSIDE_G1_INIT)

    assert _results_equal(guarded, unguarded)
    assert guarded.flip_success is True
    assert guarded.handoff_index == 2


def test_guard_is_a_no_op_when_ctrl1_is_already_none(monkeypatch):
    '''There is no controller 1 to override away from on the baseline path
    (ctrl1=None): a guard, of any kind, must not change that result.
    '''
    from quad_composition import rollout3d
    monkeypatch.setattr(rollout3d, 'set_initial_state', _fake_set_initial_state)
    ctrl2 = _FakeCtrl()
    g1 = _small_g1()
    script = [(0.1, 0.1, False, False), (0.0, 0.0, True, True)]

    plain = rollout3d.rollout_composite(_ScriptedEnv3D(script), None, ctrl2, g1, _OUTSIDE_G1_INIT)
    with_guard = rollout3d.rollout_composite(
        _ScriptedEnv3D(script), None, ctrl2, g1, _OUTSIDE_G1_INIT, guard=lambda state: True)

    assert _results_equal(plain, with_guard)


def test_guard_is_consulted_exactly_once_on_the_initial_state(monkeypatch):
    '''The guard must not be re-evaluated at every step -- only the initial
    state decides whether controller 1 runs at all for this rollout.
    '''
    from quad_composition import rollout3d
    monkeypatch.setattr(rollout3d, 'set_initial_state', _fake_set_initial_state)
    ctrl = _FakeCtrl()
    g1 = _small_g1()
    calls = []

    def counting_guard(state):
        calls.append(state)
        return False

    rollout3d.rollout_composite(_ScriptedEnv3D(_HANDOFF_SCRIPT), ctrl, ctrl, g1, _OUTSIDE_G1_INIT,
                                guard=counting_guard)
    assert len(calls) == 1
    assert calls[0] == pytest.approx(_OUTSIDE_G1_INIT)


def test_lqr_success_guard_is_callable_and_matches_the_fitted_guard():
    from quad_composition.guard3d import FITTED_GUARD, lqr_success_guard
    state = _dataset_state(tilt=0.2, omega=1.0, pos=(0.3, 0.1, 1.05), vel=(0.2, 0.0, 0.1))
    assert lqr_success_guard(state) == FITTED_GUARD.predict(state)
