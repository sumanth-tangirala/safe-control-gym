'''Dataset invariants for the composition (spec D3, D7, D8).

RULING D-I (quad_composition/rollout2d.py module docstring; task-2-report.md
"Fix round 2"): the archived quadrotor2D_rl dataset is not bit-reproducible
per trajectory on this machine, so generate_quadrotor_2d_composition.py adds
a third mode, --mode baseline, that regenerates a controller-2-alone
baseline through the SAME rollout core the flip/composite datasets go
through. --mode flip/composite need a trained controller-1 checkpoint that
does not exist yet (Task 4's 1M-step SAC training run was deliberately not
launched -- see the session ledger), so their dataset-assembly logic
(generate_dataset, write_outputs, build_description, main's wiring) is
exercised here against fakes; --mode baseline needs no checkpoint and is
additionally exercised for real in
test_baseline_mode_runs_for_real_end_to_end.
'''
import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Imported function-locally in every test below (matching
# test_rollout2d.py's / test_g1_calibration.py's convention): isort
# otherwise hoists a module-level import above the sys.path insert it
# depends on.

SHIPPED = ('/common/users/shared/pracsys/genMoPlan/data_trajectories/'
           'deterministic/quadrotor2D_rl')


# ---------------------------------------------------------------------------
# The brief's three pinned tests (Step 1), verbatim.
# ---------------------------------------------------------------------------

def test_impossible_label_combination_is_rejected():
    from generate_quadrotor_2d_composition import validate_labels

    # (flip_success=0, ctrl2_success=1) cannot occur: no handoff, no controller 2
    with pytest.raises(ValueError, match='impossible label'):
        validate_labels(np.array([0]), np.array([1]))
    validate_labels(np.array([1, 1, 0]), np.array([1, 0, 0]))   # all legal


def test_handoff_index_minus_one_means_flip_failed():
    from generate_quadrotor_2d_composition import labels_from_result
    from quad_composition.rollout2d import RolloutResult
    res = RolloutResult(trajectory=[[0] * 6], handoff_index=-1,
                        flip_success=False, ctrl2_success=False)
    assert labels_from_result(res) == (0, 0)


def test_eval_states_row_is_init_final_and_two_labels():
    from generate_quadrotor_2d_composition import eval_states_row
    from quad_composition.rollout2d import RolloutResult
    init = [0.1, 1.2, 0.3, 0.4, 0.5, 0.6]
    final = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    res = RolloutResult(trajectory=[init, final], handoff_index=1,
                        flip_success=True, ctrl2_success=True)
    row = eval_states_row(init, res)
    assert len(row) == 14           # 6 + 6 + 2
    assert row[:6] == pytest.approx(init)
    assert row[6:12] == pytest.approx(final)
    assert row[12:] == [1, 1]


# ---------------------------------------------------------------------------
# Ruling D-I: --mode baseline needs neither --flip_model nor --g1.
# ---------------------------------------------------------------------------

def test_baseline_mode_does_not_require_flip_model_or_g1():
    from generate_quadrotor_2d_composition import parse_args
    args = parse_args(['--mode', 'baseline', '--baseline_dir', '/x', '--output_dir', '/y'])
    assert args.flip_model is None


def test_flip_mode_requires_flip_model():
    from generate_quadrotor_2d_composition import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--mode', 'flip', '--baseline_dir', '/x', '--output_dir', '/y'])


def test_composite_mode_requires_flip_model():
    from generate_quadrotor_2d_composition import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--mode', 'composite', '--baseline_dir', '/x', '--output_dir', '/y'])


# ---------------------------------------------------------------------------
# baseline_eval_states_row / handoff_row: the two new row builders.
# ---------------------------------------------------------------------------

def test_baseline_eval_states_row_is_init_final_and_one_label():
    from generate_quadrotor_2d_composition import baseline_eval_states_row
    from quad_composition.rollout2d import RolloutResult
    init = [0.1, 1.2, 0.3, 0.4, 0.5, 0.6]
    final = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # flip_success is always False on the baseline path (ctrl1=None, Ruling
    # D-F) -- included here to prove the row builder ignores it regardless.
    res = RolloutResult(trajectory=[init, final], handoff_index=-1,
                        flip_success=False, ctrl2_success=True)
    row = baseline_eval_states_row(init, res)
    assert len(row) == 13           # 6 + 6 + 1, matching the archived format
    assert row[:6] == pytest.approx(init)
    assert row[6:12] == pytest.approx(final)
    assert row[12] == 1


def test_handoff_row_is_minus_one_when_no_handoff():
    from generate_quadrotor_2d_composition import handoff_row
    from quad_composition.rollout2d import RolloutResult
    init = [0.1, 1.2, 0.3, 0.4, 0.5, 0.6]
    res = RolloutResult(trajectory=[init], handoff_index=-1,
                        flip_success=False, ctrl2_success=False)
    row = handoff_row(init, res)
    assert row == pytest.approx(init + [-1.0] * 6)


def test_handoff_row_is_the_state_at_handoff_index():
    from generate_quadrotor_2d_composition import handoff_row
    from quad_composition.rollout2d import RolloutResult
    init = [0.1, 1.2, 0.3, 0.4, 0.5, 0.6]
    handoff_state = [0.05, 1.0, 0.02, 0.1, 0.0, 0.01]
    res = RolloutResult(trajectory=[init, handoff_state], handoff_index=1,
                        flip_success=True, ctrl2_success=False)
    row = handoff_row(init, res)
    assert row == pytest.approx(init + handoff_state)


# ---------------------------------------------------------------------------
# generate_dataset: the rollout-and-write loop, tested against fakes so
# --mode flip/composite are exercised without a trained controller-1
# checkpoint.
# ---------------------------------------------------------------------------

class _FakeCtrl:
    '''Deterministic no-op controller usable as either controller 1 or 2:
    only obs_normalizer/select_action/close are ever called by
    generate_dataset (through rollout_composite).
    '''

    def obs_normalizer(self, obs):
        return obs

    def select_action(self, obs, info):
        return np.zeros(2)

    def close(self):
        pass


class _ScriptedEnv:
    '''Replays a fixed list of (theta, theta_dot, done) per .step() call, in
    env order [x, x_dot, z, z_dot, theta, theta_dot] with x=0, z=1 fixed.
    `goal_reached` fires whenever the step both finishes and lands at
    theta=theta_dot=0 (upright and still), independent of which controller
    is "active" -- mirrors a real env, whose physics do not know which
    policy chose the action. `set_initial_state` is faked (below) to rewind
    this env's script index to 0 at the start of every rollout, mirroring
    what a real env.reset() would do.
    '''

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def step(self, action):
        theta, theta_dot, done = self.script[self.i]
        self.i += 1
        obs = np.array([0.0, 0.0, 1.0, 0.0, theta, theta_dot])
        info = {'goal_reached': bool(done and theta == 0.0 and theta_dot == 0.0)}
        return obs, 0.0, done, info

    def close(self):
        pass


class _FakeG1:
    def __init__(self, tilt_c=0.05, w_c=0.05):
        self.tilt_c, self.w_c = tilt_c, w_c

    def contains(self, tilt, omega):
        return bool(tilt < self.tilt_c and omega < self.w_c)

    def to_dict(self):
        return {'form': 'attitude_only', 'tilt_c_rad': self.tilt_c,
                'tilt_c_deg': float(np.degrees(self.tilt_c)), 'w_c_rad_s': self.w_c}


def _fake_set_initial_state(env, init_state):
    env.i = 0
    return np.zeros(6), {}


# Enters G1 (tilt_c=w_c=0.05) at step index 1, then finishes successfully at
# step index 2. Shared by several tests below.
_HANDOFF_SCRIPT = [
    (0.5, 0.5, False),    # outside G1
    (0.02, 0.01, False),  # enters G1 here -> handoff_index == 2 (trajectory row)
    (0.0, 0.0, True),     # ctrl2 takes over, reaches the goal, done
]
_INIT = [0.0, 1.0, 0.5, 0.0, 0.0, 0.5]  # outside G1 (|theta|=0.5 > tilt_c)


def test_generate_dataset_flip_mode_truncates_trajectory_at_handoff(monkeypatch, tmp_path):
    from generate_quadrotor_2d_composition import generate_dataset
    from quad_composition import rollout2d

    monkeypatch.setattr(rollout2d, 'set_initial_state', _fake_set_initial_state)
    env = _ScriptedEnv(_HANDOFF_SCRIPT)
    ctrl = _FakeCtrl()

    rows, handoffs = generate_dataset('flip', env, ctrl, ctrl, _FakeG1(), [_INIT], str(tmp_path))

    traj = np.loadtxt(tmp_path / 'trajectories' / 'sequence_0.txt', delimiter=',', ndmin=2)
    assert len(traj) == 3  # init + 2 steps, truncated at (and including) the handoff state
    assert len(rows) == 1 and len(rows[0]) == 14
    assert rows[0][12] == 1  # flip_success: controller 1 reached G1
    # final state stored is the (truncated) handoff state, not the full rollout's end
    assert rows[0][6:12] == pytest.approx([0.0, 1.0, 0.02, 0.0, 0.0, 0.01])
    assert handoffs[0][6:] == pytest.approx([0.0, 1.0, 0.02, 0.0, 0.0, 0.01])


def test_generate_dataset_composite_mode_keeps_the_full_trajectory(monkeypatch, tmp_path):
    from generate_quadrotor_2d_composition import generate_dataset
    from quad_composition import rollout2d

    monkeypatch.setattr(rollout2d, 'set_initial_state', _fake_set_initial_state)
    env = _ScriptedEnv(_HANDOFF_SCRIPT)
    ctrl = _FakeCtrl()

    rows, handoffs = generate_dataset('composite', env, ctrl, ctrl, _FakeG1(), [_INIT], str(tmp_path))

    traj = np.loadtxt(tmp_path / 'trajectories' / 'sequence_0.txt', delimiter=',', ndmin=2)
    assert len(traj) == 4  # init + all 3 steps, not truncated
    assert rows[0][12] == 1  # flip_success
    assert rows[0][13] == 1  # ctrl2_success
    assert rows[0][6:12] == pytest.approx([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # final state


def test_generate_dataset_baseline_mode_uses_ctrl2_alone_and_one_label(monkeypatch, tmp_path):
    from generate_quadrotor_2d_composition import generate_dataset
    from quad_composition import rollout2d

    monkeypatch.setattr(rollout2d, 'set_initial_state', _fake_set_initial_state)
    script = [(0.1, 0.1, False), (0.0, 0.0, True)]
    env = _ScriptedEnv(script)
    ctrl2 = _FakeCtrl()

    rows, handoffs = generate_dataset('baseline', env, None, ctrl2, None, [_INIT], str(tmp_path))

    assert len(rows) == 1 and len(rows[0]) == 13
    assert handoffs[0][6:] == pytest.approx([-1.0] * 6)  # baseline never hands off


def test_prefix_invariant_flip_trajectory_prefixes_composite_up_to_handoff(monkeypatch, tmp_path):
    '''Task 6 Step 5's manual shell check, as a real test: the flip-only
    trajectory for a given init must be a byte-for-byte prefix of the
    composite trajectory for the same init, up to and including the handoff
    state -- both modes run the identical policy sequence up to handoff
    (rollout_composite does not know which output mode is asking); only the
    STORED trajectory differs (composite keeps the post-handoff tail, flip
    drops it).
    '''
    from generate_quadrotor_2d_composition import generate_dataset
    from quad_composition import rollout2d

    monkeypatch.setattr(rollout2d, 'set_initial_state', _fake_set_initial_state)
    ctrl = _FakeCtrl()
    g1 = _FakeG1()

    flip_dir, comp_dir = tmp_path / 'flip', tmp_path / 'composite'
    # Two independent (but identically scripted) env instances, mirroring
    # the brief's two separate script invocations -- one per mode.
    generate_dataset('flip', _ScriptedEnv(_HANDOFF_SCRIPT), ctrl, ctrl, g1, [_INIT], str(flip_dir))
    generate_dataset('composite', _ScriptedEnv(_HANDOFF_SCRIPT), ctrl, ctrl, g1, [_INIT], str(comp_dir))

    flip_traj = np.loadtxt(flip_dir / 'trajectories' / 'sequence_0.txt', delimiter=',', ndmin=2)
    comp_traj = np.loadtxt(comp_dir / 'trajectories' / 'sequence_0.txt', delimiter=',', ndmin=2)

    assert len(flip_traj) < len(comp_traj), 'flip must actually be truncated for this fixture'
    assert np.allclose(flip_traj, comp_traj[:len(flip_traj)]), \
        'flip trajectory must prefix the composite trajectory'


# ---------------------------------------------------------------------------
# write_outputs / build_description: column widths, validation wiring, and
# the regenerated-baseline note (required "for all modes").
# ---------------------------------------------------------------------------

def test_write_outputs_baseline_mode_writes_thirteen_and_seven_column_files(tmp_path):
    from generate_quadrotor_2d_composition import parse_args, write_outputs

    args = parse_args(['--mode', 'baseline', '--baseline_dir', '/x',
                       '--output_dir', str(tmp_path)])
    rows = [[0.0] * 13, [0.1] * 12 + [1.0]]
    handoffs = [[0.0] * 12, [0.0] * 12]

    write_outputs(args, None, rows, handoffs)

    eval_states = np.loadtxt(tmp_path / 'eval_states.txt', delimiter=',', ndmin=2)
    roa = np.loadtxt(tmp_path / 'roa_labels.txt', delimiter=',', ndmin=2)
    assert eval_states.shape == (2, 13)
    assert roa.shape == (2, 7)

    desc = json.loads((tmp_path / 'dataset_description.json').read_text())
    assert 'regenerated_baseline_note' in desc
    assert 'g1' not in desc
    assert desc['statistics']['ctrl2_success'] == 1


def test_write_outputs_flip_mode_writes_fourteen_and_eight_column_files(tmp_path):
    from generate_quadrotor_2d_composition import parse_args, write_outputs

    args = parse_args(['--mode', 'flip', '--flip_model', 'unused.pt',
                       '--baseline_dir', '/x', '--output_dir', str(tmp_path)])
    rows = [[0.0] * 12 + [1.0, 0.0], [0.0] * 12 + [1.0, 1.0]]
    handoffs = [[0.0] * 12, [0.0] * 12]

    write_outputs(args, _FakeG1(), rows, handoffs)

    eval_states = np.loadtxt(tmp_path / 'eval_states.txt', delimiter=',', ndmin=2)
    roa = np.loadtxt(tmp_path / 'roa_labels.txt', delimiter=',', ndmin=2)
    assert eval_states.shape == (2, 14)
    assert roa.shape == (2, 8)

    desc = json.loads((tmp_path / 'dataset_description.json').read_text())
    assert 'regenerated_baseline_note' in desc
    assert desc['g1']['tilt_c_rad'] == pytest.approx(0.05)
    assert desc['statistics'] == {'total': 2, 'flip_success': 2, 'ctrl2_success': 1}


def test_write_outputs_rejects_impossible_labels_for_flip_and_composite(tmp_path):
    from generate_quadrotor_2d_composition import parse_args, write_outputs

    args = parse_args(['--mode', 'composite', '--flip_model', 'unused.pt',
                       '--baseline_dir', '/x', '--output_dir', str(tmp_path)])
    rows = [[0.0] * 12 + [0, 1]]  # impossible: flip_success=0, ctrl2_success=1
    with pytest.raises(ValueError, match='impossible label'):
        write_outputs(args, _FakeG1(), rows, [[0.0] * 12])
    # and it must not have written a partial/bad eval_states.txt on the way out
    assert not os.path.exists(tmp_path / 'eval_states.txt')


# ---------------------------------------------------------------------------
# main(): runtime enforcement that --mode baseline touches neither
# controller 1 nor the --g1 file (not just argparse-level, spec D-I).
# ---------------------------------------------------------------------------

def _write_fake_baseline_dir(tmp_path, inits):
    baseline_dir = tmp_path / 'shipped'
    baseline_dir.mkdir()
    padded = np.column_stack([inits, np.zeros((len(inits), 7))])
    np.savetxt(baseline_dir / 'eval_states.txt', padded, delimiter=',')
    return baseline_dir


def test_main_baseline_mode_never_touches_g1_file_or_ctrl1(monkeypatch, tmp_path):
    '''Runtime enforcement of "must not require --flip_model or --g1": the
    default --g1 path (models/quad2d_flip/g1.json) does not exist on disk --
    no calibration has run yet -- so if main() ever tried to open it in
    baseline mode this test would fail with FileNotFoundError instead of
    completing. load_ctrl1 must also never be called.
    '''
    import generate_quadrotor_2d_composition as gen
    from quad_composition import rollout2d

    inits = np.array([[0.0, 1.0, 0.5, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    baseline_dir = _write_fake_baseline_dir(tmp_path, inits)
    output_dir = tmp_path / 'out'

    load_ctrl1_calls = []
    monkeypatch.setattr(gen, 'load_ctrl1', lambda *a, **kw: load_ctrl1_calls.append(1))
    monkeypatch.setattr(gen, 'make_env_and_ctrl2',
                        lambda model_path, out: (_ScriptedEnv([(0.0, 0.0, True)] * 2), _FakeCtrl()))
    monkeypatch.setattr(rollout2d, 'set_initial_state', _fake_set_initial_state)
    monkeypatch.setattr(sys, 'argv', ['prog', '--mode', 'baseline',
                                      '--baseline_dir', str(baseline_dir),
                                      '--output_dir', str(output_dir)])

    gen.main()

    assert load_ctrl1_calls == []
    eval_states = np.loadtxt(output_dir / 'eval_states.txt', delimiter=',', ndmin=2)
    assert eval_states.shape == (2, 13)


def test_main_flip_mode_loads_g1_file_and_ctrl1(monkeypatch, tmp_path):
    import generate_quadrotor_2d_composition as gen
    from quad_composition import rollout2d

    inits = np.array([_INIT])
    baseline_dir = _write_fake_baseline_dir(tmp_path, inits)
    g1_path = tmp_path / 'g1.json'
    g1_path.write_text(json.dumps({'g1': {'tilt_c_rad': 0.05, 'w_c_rad_s': 0.05}}))
    output_dir = tmp_path / 'out'

    load_ctrl1_calls = []

    def fake_load_ctrl1(*a, **kw):
        load_ctrl1_calls.append(1)
        return _FakeCtrl()

    monkeypatch.setattr(gen, 'load_ctrl1', fake_load_ctrl1)
    monkeypatch.setattr(gen, 'make_env_and_ctrl2',
                        lambda model_path, out: (_ScriptedEnv(_HANDOFF_SCRIPT), _FakeCtrl()))
    monkeypatch.setattr(rollout2d, 'set_initial_state', _fake_set_initial_state)
    monkeypatch.setattr(sys, 'argv', ['prog', '--mode', 'flip', '--flip_model', 'unused.pt',
                                      '--g1', str(g1_path),
                                      '--baseline_dir', str(baseline_dir),
                                      '--output_dir', str(output_dir)])

    gen.main()

    assert load_ctrl1_calls == [1]
    desc = json.loads((output_dir / 'dataset_description.json').read_text())
    assert desc['g1']['tilt_c_rad'] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# --mode baseline needs no trained checkpoint (Ruling D-I), so unlike
# flip/composite it is exercised for real, not just against fakes.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_baseline_mode_runs_for_real_end_to_end(tmp_path):
    if not os.path.exists(os.path.join(SHIPPED, 'eval_states.txt')):
        pytest.skip('shipped dataset not mounted')
    import generate_quadrotor_2d_composition as gen

    output_dir = tmp_path / 'q2d_baseline_smoke'
    old_argv = sys.argv
    sys.argv = ['prog', '--mode', 'baseline', '--baseline_dir', SHIPPED,
                '--output_dir', str(output_dir), '--limit', '5']
    try:
        gen.main()
    finally:
        sys.argv = old_argv

    eval_states = np.loadtxt(output_dir / 'eval_states.txt', delimiter=',', ndmin=2)
    assert eval_states.shape == (5, 13)
    roa = np.loadtxt(output_dir / 'roa_labels.txt', delimiter=',', ndmin=2)
    assert roa.shape == (5, 7)
    handoffs = np.loadtxt(output_dir / 'handoff_states.txt', delimiter=',', ndmin=2)
    assert handoffs.shape == (5, 12)
    assert np.all(handoffs[:, 6:] == -1.0)  # controller 1 never runs on this path

    desc = json.loads((output_dir / 'dataset_description.json').read_text())
    assert desc['statistics']['total'] == 5
    assert 'regenerated_baseline_note' in desc
    assert 'g1' not in desc
