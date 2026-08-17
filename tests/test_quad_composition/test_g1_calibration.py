'''G1's parameters come from controller 1's exits, never from RoA2 (spec D1).'''
import ast
import inspect
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Imported after the sys.path insert above, and function-locally in every
# test below (matching test_rollout2d.py's convention): isort otherwise
# hoists a module-level `import quad_composition...` above the sys.path
# insert it depends on, since isort does not understand the runtime
# dependency between the two.


def test_fit_covers_the_requested_quantile_of_exits():
    from quad_composition.g1 import fit_from_exits

    rng = np.random.default_rng(0)
    tilts = np.abs(rng.normal(0.0, 0.1, size=5000))
    omegas = np.abs(rng.normal(0.0, 0.8, size=5000))
    g1 = fit_from_exits(tilts, omegas, quantile=0.9)
    assert g1.contains(tilts, omegas).mean() >= 0.80
    assert g1.tilt_c == np.quantile(tilts, 0.9)
    assert g1.w_c == np.quantile(omegas, 0.9)


def test_fit_is_monotone_in_the_quantile():
    from quad_composition.g1 import fit_from_exits

    rng = np.random.default_rng(1)
    tilts = np.abs(rng.normal(0.0, 0.1, size=2000))
    omegas = np.abs(rng.normal(0.0, 0.8, size=2000))
    tight = fit_from_exits(tilts, omegas, quantile=0.5)
    loose = fit_from_exits(tilts, omegas, quantile=0.95)
    assert tight.tilt_c < loose.tilt_c and tight.w_c < loose.w_c


def test_fit_rejects_an_empty_sample():
    from quad_composition.g1 import fit_from_exits

    with pytest.raises(ValueError, match='no exits'):
        fit_from_exits(np.array([]), np.array([]), quantile=0.9)


# ---------------------------------------------------------------------------
# collect_exit_attitudes: the rollout-and-collect loop, tested against fakes
# so it does not require a trained controller-1 checkpoint (task-5 ruling 3).
# ---------------------------------------------------------------------------


class _FakeCtrl1:
    '''Deterministic no-op controller: only obs_normalizer/select_action are
    ever called by collect_exit_attitudes.
    '''

    def obs_normalizer(self, obs):
        return obs

    def select_action(self, obs, info):
        return np.zeros(2)


class _ScriptedEnv:
    '''Replays a fixed list of (theta, theta_dot, done) per .step() call, in
    env order [x, x_dot, z, z_dot, theta, theta_dot]. `set_initial_state` is
    faked (below) to rewind this env's script index to 0 at the start of
    every rollout, mirroring what a real env.reset() would do.
    '''

    def __init__(self, script):
        self.script = script
        self.i = 0
        self.n_steps = 0

    def step(self, action):
        theta, theta_dot, done = self.script[self.i]
        self.i += 1
        self.n_steps += 1
        obs = np.array([0.0, 0.0, 1.0, 0.0, theta, theta_dot])
        return obs, 0.0, done, {}


def _fake_set_initial_state(env, init_state):
    env.i = 0
    return np.zeros(6), {}


def test_collect_exit_attitudes_reports_the_best_scoring_state_not_the_last(monkeypatch):
    '''"Best" must be the minimum-score state seen along the rollout, not
    wherever the rollout happens to end -- controller 1 can overshoot past
    its closest approach to upright before the episode terminates.
    '''
    import calibrate_quad2d_g1 as cal

    # score = |theta|/pi + |theta_dot|/8
    script = [
        (0.5, 4.0, False),   # score ~= 0.659
        (0.1, 0.5, False),   # score ~= 0.094 -- the minimum
        (0.3, 0.2, True),    # score ~= 0.120, higher than the minimum; done fires here
    ]
    env = _ScriptedEnv(script)
    monkeypatch.setattr(cal, 'set_initial_state', _fake_set_initial_state)
    monkeypatch.setattr(cal, 'sample_uniform_state', lambda rng: np.zeros(6))

    tilts, omegas = cal.collect_exit_attitudes(
        env, _FakeCtrl1(), np.random.default_rng(0), num_rollouts=1, settle_steps=10)

    assert tilts == pytest.approx([0.1])
    assert omegas == pytest.approx([0.5])


def test_collect_exit_attitudes_stops_stepping_once_the_rollout_finishes(monkeypatch):
    '''`done` must stop the inner loop even though settle_steps is larger
    than the number of scripted ticks -- otherwise it would index past the
    script (a real env would just keep running a finished episode).
    '''
    import calibrate_quad2d_g1 as cal

    script = [(0.5, 4.0, False), (0.4, 3.0, False), (0.1, 0.1, True)]
    env = _ScriptedEnv(script)
    monkeypatch.setattr(cal, 'set_initial_state', _fake_set_initial_state)
    monkeypatch.setattr(cal, 'sample_uniform_state', lambda rng: np.zeros(6))

    cal.collect_exit_attitudes(env, _FakeCtrl1(), np.random.default_rng(0),
                               num_rollouts=1, settle_steps=100)

    assert env.n_steps == len(script)


def test_collect_exit_attitudes_runs_num_rollouts_independent_rollouts(monkeypatch):
    '''Each rollout must reset (via set_initial_state) and contribute exactly
    one (tilt, omega) pair; num_rollouts=3 must yield 3 pairs, not 1 or 9.
    '''
    import calibrate_quad2d_g1 as cal

    script = [(0.2, 1.0, True)]
    env = _ScriptedEnv(script)
    reset_calls = {'n': 0}

    def fake_reset(env, init_state):
        reset_calls['n'] += 1
        env.i = 0
        return np.zeros(6), {}

    monkeypatch.setattr(cal, 'set_initial_state', fake_reset)
    monkeypatch.setattr(cal, 'sample_uniform_state', lambda rng: np.zeros(6))

    tilts, omegas = cal.collect_exit_attitudes(
        env, _FakeCtrl1(), np.random.default_rng(0), num_rollouts=3, settle_steps=5)

    assert reset_calls['n'] == 3
    assert len(tilts) == 3 and len(omegas) == 3
    assert tilts == pytest.approx([0.2, 0.2, 0.2])
    assert omegas == pytest.approx([1.0, 1.0, 1.0])


def test_collect_exit_attitudes_uses_magnitude_so_sign_does_not_matter(monkeypatch):
    '''G1 (and its calibration) is defined on |theta|, |theta_dot| -- a
    negative exit tilt/rate must be reported as its magnitude, matching
    G1Region.contains and fit_from_exits, both of which take abs() too.
    '''
    import calibrate_quad2d_g1 as cal

    script = [(-0.2, -1.0, True)]
    env = _ScriptedEnv(script)
    monkeypatch.setattr(cal, 'set_initial_state', _fake_set_initial_state)
    monkeypatch.setattr(cal, 'sample_uniform_state', lambda rng: np.zeros(6))

    tilts, omegas = cal.collect_exit_attitudes(
        env, _FakeCtrl1(), np.random.default_rng(0), num_rollouts=1, settle_steps=5)

    assert tilts == pytest.approx([0.2])
    assert omegas == pytest.approx([1.0])


def test_collect_exit_attitudes_drops_a_rollout_that_takes_zero_steps(monkeypatch):
    '''settle_steps=0 means the inner loop never runs, so `best` stays None
    and that rollout must contribute nothing -- not a spurious (0, 0) pair.
    '''
    import calibrate_quad2d_g1 as cal

    env = _ScriptedEnv([])
    monkeypatch.setattr(cal, 'set_initial_state', _fake_set_initial_state)
    monkeypatch.setattr(cal, 'sample_uniform_state', lambda rng: np.zeros(6))

    tilts, omegas = cal.collect_exit_attitudes(
        env, _FakeCtrl1(), np.random.default_rng(0), num_rollouts=1, settle_steps=0)

    assert len(tilts) == 0 and len(omegas) == 0


def test_calibration_script_never_imports_controller_2_or_roa2():
    '''Guards spec D1's ordering property at the source level: the script's
    actual `import` statements -- not its prose, which is free to explain in
    words why controller 2's loader is avoided -- must not name it.

    AST-parsed rather than a raw substring search over the source, because
    the module's own docstring explains this ruling and names
    `make_env_and_ctrl2` and `quadrotor2D_rl` to do so; a substring check
    would flag its own documentation as a violation.
    '''
    import calibrate_quad2d_g1 as cal

    tree = ast.parse(inspect.getsource(cal))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)

    forbidden = {'make_env_and_ctrl2', 'ALGO_CONFIG'}
    assert not (imported_names & forbidden), \
        f'calibrate_quad2d_g1.py must not import {imported_names & forbidden}'
    assert not hasattr(cal, 'make_env_and_ctrl2')
    assert not hasattr(cal, 'ALGO_CONFIG')
