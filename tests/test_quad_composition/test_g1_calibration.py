'''G1's parameters come from controller 1's exits, never from RoA2 (spec D1).'''
import ast
import inspect
import os
import sys

import numpy as np
import pybullet as p
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


def _quat(theta):
    '''Body orientation for a pure pitch of `theta` -- what a real env caches
    on `.quat`, and what `rollout2d.true_theta` reads (Finding C1).
    '''
    return p.getQuaternionFromEuler([0.0, float(theta), 0.0])


class _ScriptedEnv:
    '''Replays a fixed list of (theta, theta_dot, done) per .step() call, in
    env order [x, x_dot, z, z_dot, theta, theta_dot]. `set_initial_state` is
    faked (below) to rewind this env's script index to 0 at the start of
    every rollout, mirroring what a real env.reset() would do.

    The scripted theta is TRUE attitude: `.quat` is set from it and the
    observation carries PyBullet's own folded pitch, as a real env does
    (Finding C1). Calibration reads the exits off `.quat`, so a fake without
    one would not exercise the path at all.
    '''

    def __init__(self, script):
        self.script = script
        self.i = 0
        self.n_steps = 0
        self.quat = _quat(0.0)

    def step(self, action):
        theta, theta_dot, done = self.script[self.i]
        self.i += 1
        self.n_steps += 1
        self.quat = _quat(theta)
        folded = p.getEulerFromQuaternion(self.quat)[1]
        obs = np.array([0.0, 0.0, 1.0, 0.0, folded, theta_dot])
        return obs, 0.0, done, {}


def _fake_set_initial_state(env, init_state):
    env.i = 0
    env.quat = _quat(0.0)
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
        env.quat = _quat(0.0)
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


def test_collect_exit_attitudes_records_true_attitude_not_the_folded_observation(monkeypatch):
    '''Finding C1. An exit at true theta = pi - 0.05 OBSERVES theta = 0.05, so
    calibrating on the observation would record a near-perfect exit for a drone
    that is upside down -- and, across a whole calibration run, would fit
    `tilt_c` to a distribution structurally capped at pi/2. `_ScriptedEnv` folds
    via PyBullet, so this test sees the real fold.
    '''
    import calibrate_quad2d_g1 as cal

    inverted = np.pi - 0.05
    env = _ScriptedEnv([(inverted, 0.0, True)])
    monkeypatch.setattr(cal, 'set_initial_state', _fake_set_initial_state)
    monkeypatch.setattr(cal, 'sample_uniform_state', lambda rng: np.zeros(6))

    # The fake really is presenting an upright-looking observation.
    assert abs(env.step(None)[0][4]) == pytest.approx(0.05, abs=1e-6)
    env.i = 0

    tilts, omegas = cal.collect_exit_attitudes(
        env, _FakeCtrl1(), np.random.default_rng(0), num_rollouts=1, settle_steps=5)

    assert tilts == pytest.approx([inverted], abs=1e-6)
    assert tilts[0] > np.pi / 2, 'a folded exit distribution cannot exceed pi/2'


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


_RUINS_THE_ORDERING_PROPERTY = (
    'would let calibrate_quad2d_g1.py consult RoA2 (controller 2, its checkpoint, or the '
    "shipped quadrotor2D_rl labels). G1 must be calibrated ONLY from controller 1's measured "
    'exits (spec D1) -- if RoA2 leaks into calibration, G1 gets fitted to the answer, and the '
    "experiment's headline result becomes a construction instead of a discovery. Do not delete "
    'or weaken this check to get past it; fix the script instead.'
)


def _docstring_constant_ids(tree):
    '''Node ids of every module/function/class docstring's Constant node.

    Docstrings are prose and are allowed to name `make_env_and_ctrl2` or
    `quadrotor2D_rl` to explain why they are avoided (this file's own module
    docstring does exactly that) -- only actual imports, attribute accesses,
    and non-docstring string literals are evidence of a real reference.
    '''
    ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        is_docstring = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
        if is_docstring:
            ids.add(id(first.value))
    return ids


def test_calibration_script_never_imports_or_references_controller_2_or_roa2():
    '''Guards spec D1's ordering property at the source level, against three
    routes a routine-looking edit could reintroduce it by:

    1. Importing `make_env_and_ctrl2` / `ALGO_CONFIG` by name (the original
       check), OR importing `quad_composition.rollout2d` as a whole module
       (under any alias) -- a whole-module import is the vector that makes
       route 2 possible even when no forbidden name is imported directly.
    2. Reaching either forbidden name through attribute access instead of
       import, e.g. `import quad_composition.rollout2d as r2d; ...
       r2d.make_env_and_ctrl2(...)` -- a routine-looking refactor that the
       import-name check alone does not see.
    3. Hardcoding a path to controller 2's checkpoint or the shipped RoA2
       labels as a string literal (`quadrotor2D_rl`, `safe_explorer_ppo`,
       `eval_states.txt`), bypassing the loader helpers entirely.

    AST-parsed rather than a raw substring search over the source, because
    the module's own docstring explains this ruling in prose and legitimately
    names `make_env_and_ctrl2` and `quadrotor2D_rl` to do so; a substring
    search over the whole source (or an unfiltered string-literal scan) would
    flag the script's own documentation as a violation. `_docstring_constant_ids`
    excludes exactly those prose nodes from the string-literal check (route 3)
    without weakening it against a real hardcoded reference anywhere else.
    '''
    import calibrate_quad2d_g1 as cal

    tree = ast.parse(inspect.getsource(cal))
    docstring_ids = _docstring_constant_ids(tree)

    forbidden_names = {'make_env_and_ctrl2', 'ALGO_CONFIG'}
    forbidden_substrings = ('quadrotor2D_rl', 'safe_explorer_ppo', 'eval_states.txt')

    imported_names = set()
    whole_module_rollout2d_imports = set()
    forbidden_attr_accesses = set()
    forbidden_string_literals = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
            if node.module == 'quad_composition':
                whole_module_rollout2d_imports.update(
                    alias.name for alias in node.names if alias.name == 'rollout2d')
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
            whole_module_rollout2d_imports.update(
                alias.name for alias in node.names
                if alias.name == 'rollout2d' or alias.name.endswith('.rollout2d'))
        elif isinstance(node, ast.Attribute) and node.attr in forbidden_names:
            forbidden_attr_accesses.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstring_ids:
            hit = next((s for s in forbidden_substrings if s in node.value), None)
            if hit is not None:
                forbidden_string_literals.add(node.value)

    bad_imports = imported_names & forbidden_names
    assert not bad_imports, (
        f'calibrate_quad2d_g1.py imports {bad_imports}, which {_RUINS_THE_ORDERING_PROPERTY}')
    assert not whole_module_rollout2d_imports, (
        'calibrate_quad2d_g1.py imports the whole rollout2d module '
        f'({whole_module_rollout2d_imports}), which opens attribute access to '
        f'make_env_and_ctrl2 and {_RUINS_THE_ORDERING_PROPERTY}')
    assert not forbidden_attr_accesses, (
        f'calibrate_quad2d_g1.py accesses .{forbidden_attr_accesses} as an attribute, which '
        f'{_RUINS_THE_ORDERING_PROPERTY}')
    assert not forbidden_string_literals, (
        f'calibrate_quad2d_g1.py hardcodes {forbidden_string_literals} in a string literal, '
        f'which {_RUINS_THE_ORDERING_PROPERTY}')
    assert not hasattr(cal, 'make_env_and_ctrl2')
    assert not hasattr(cal, 'ALGO_CONFIG')
