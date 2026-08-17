'''Dataset invariants for the 3D composition. The 3D port of
test_composition_datasets.py -- read that file's docstring first; every test
class/fixture here mirrors it, adjusted from 6-dim [x, z, theta, x_dot, z_dot,
theta_dot] states to 13-dim dataset-order states
[x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r].

WHAT IS NEW HERE, RELATIVE TO THE 2D FILE: --num_workers and resumability.
generate_quadrotor_3d_composition.py dispatches work through a spawn-context
multiprocessing.Pool (mirroring analyze_quad3d_composition.py), so monkeypatch
on the module-level fakes used for `generate_dataset` below does NOT reach a
worker subprocess (spawn re-imports the module fresh in the child; it does
not inherit the parent's patched attributes). Two consequences:

  1. `generate_dataset` (the sequential, single-process loop) is still tested
     against fakes exactly as in 2D -- it exercises the exact same per-index
     logic (`process_one`) that a worker calls, just without the process
     boundary.
  2. The multiprocessing dispatch itself (`run_parallel`, and `main()`'s
     wiring into it) is tested two ways: (a) fast, in-process tests that
     monkeypatch `run_parallel`/`snapshot_checkpoint` to verify main()'s
     wiring without ever spawning a worker, and (b) real, `@pytest.mark.slow`
     end-to-end tests that boot real PyBullet workers via --mode baseline (no
     checkpoint needed) to prove the Pool + resumability actually work.
'''
import json
import os
import sys

import numpy as np
import pybullet as p
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Imported function-locally in every test below (isort would otherwise hoist a
# module-level import above the sys.path insert it depends on).

SHIPPED = ('/common/users/shared/pracsys/genMoPlan/data_trajectories/'
          'deterministic/quadrotor3D_lqr')
CTRL1_CHECKPOINT = os.path.join(REPO_ROOT, 'models', 'quad3d_ctrl1_selected.pt')


@pytest.fixture()
def tmp_path():
    '''Shadows pytest's built-in `tmp_path`: on this machine TMPDIR resolves
    under /common/users/st1122/tmp, an NFS mount that intermittently hangs
    (see test_metrics3d.py's scratch_dir fixture, same rationale). Every test
    in this file gets a real /tmp-backed directory instead, as a
    pathlib.Path, so `tmp_path / 'x'`-style composition works exactly like
    the fixture it shadows -- a local fixture definition takes priority over
    the built-in one for every test in this module.
    '''
    import shutil
    import tempfile
    from pathlib import Path
    d = tempfile.mkdtemp(dir='/tmp', prefix='quad3d_composition_test_')
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Fixtures: a scripted fake env/controller pair, dimension-adjusted from
# test_composition_datasets.py's 2D fixtures. state_from_env/ctrl1_observation
# never read obs[6:9] (phi/theta/psi) -- attitude always comes from
# `env.quat` -- so the fake obs leaves those three zero throughout.
# ---------------------------------------------------------------------------

def _quat(tilt):
    '''PyBullet-order [x, y, z, w] quaternion for a pure pitch of `tilt`
    radians. For a pure pitch, R[2, 2] = cos(tilt), so
    rollout3d.tilt_from_quat(this) == tilt for tilt in [0, pi] -- the 3D
    analogue of the 2D fixture's `_quat`.
    '''
    return p.getQuaternionFromEuler([0.0, float(tilt), 0.0])


def _dataset_state(tilt=0.0, omega=0.0, pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0)):
    '''A 13-dim dataset-order state with a pure-pitch quaternion of `tilt`
    radians and body rate (omega, 0, 0) -- so omega_norm() == omega exactly,
    mirroring how the 2D fixtures drove (theta, theta_dot) directly.
    '''
    from quad_composition.rollout3d import canonical_quat_wxyz
    qx, qy, qz, qw = _quat(tilt)
    qw, qx, qy, qz = canonical_quat_wxyz([qx, qy, qz, qw])
    return [pos[0], pos[1], pos[2], qw, qx, qy, qz, vel[0], vel[1], vel[2], omega, 0.0, 0.0]


class _FakeCtrl:
    '''Deterministic no-op controller usable as either controller 1 or 2:
    only obs_normalizer/select_action/close are ever called by
    generate_dataset (through rollout_composite).
    '''

    def obs_normalizer(self, obs):
        return obs

    def select_action(self, obs, info):
        return np.zeros(4)   # 3D: 4 rotors

    def close(self):
        pass


class _ScriptedEnv3D:
    '''Replays a fixed list of (tilt, omega, done, goal_reached) per .step()
    call. Position is held at the goal (0, 0, 1) and translational velocity
    at zero throughout: state_from_env/ctrl1_observation only read attitude
    (env.quat) and body rates (obs[9:12]), so nothing else needs to vary for
    these tests. `goal_reached` fires whenever the script says so on a `done`
    step, independent of which controller is "active" -- mirrors a real env,
    whose physics do not know which policy chose the action.
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
    '''Rewinds the script and places the fake at `init_state`
    (dataset-order 13-dim), including the quaternion `rollout3d.tilt_from_quat`
    reads and the env-order obs a real `set_initial_state` would return.
    '''
    from quad_composition.rollout3d import quat_wxyz_to_pybullet
    env.i = 0
    s = np.asarray(init_state, dtype=float)
    env.quat = quat_wxyz_to_pybullet(s[3:7])
    obs = np.array([s[0], s[7], s[1], s[8], s[2], s[9],
                    0.0, 0.0, 0.0,
                    s[10], s[11], s[12]])
    return obs, {}


# Enters G1 (tilt_c=w_c=0.05) at step index 1, then finishes successfully at
# step index 2. Shared by several tests below.
_HANDOFF_SCRIPT = [
    (0.5, 0.5, False, False),    # outside G1
    (0.02, 0.01, False, False),  # enters G1 here -> handoff_index == 2 (trajectory row)
    (0.0, 0.0, True, True),      # ctrl2 takes over, reaches the goal, done
]
_INIT = _dataset_state(tilt=0.5, omega=0.5)  # outside G1 (|tilt|=0.5 > tilt_c)


def _small_g1():
    from quad_composition.g1 import G1Region
    return G1Region(tilt_c=0.05, w_c=0.05)


# ---------------------------------------------------------------------------
# validate_labels / labels_from_result / row builders -- ported from 2D's
# pinned tests, verbatim in spirit, 13-dim in practice.
# ---------------------------------------------------------------------------

def test_impossible_label_combination_is_rejected():
    from generate_quadrotor_3d_composition import validate_labels
    with pytest.raises(ValueError, match='impossible label'):
        validate_labels(np.array([0]), np.array([1]))
    validate_labels(np.array([1, 1, 0]), np.array([1, 0, 0]))   # all legal


def test_handoff_index_minus_one_means_flip_failed():
    from generate_quadrotor_3d_composition import labels_from_result
    from quad_composition.rollout3d import RolloutResult
    res = RolloutResult(trajectory=[[0] * 13], handoff_index=-1,
                        flip_success=False, ctrl2_success=False)
    assert labels_from_result(res) == (0, 0)


def test_eval_states_row_is_init_final_and_two_labels():
    from generate_quadrotor_3d_composition import eval_states_row
    from quad_composition.rollout3d import RolloutResult
    init = list(range(13))
    final = [float(-v) for v in range(13)]
    res = RolloutResult(trajectory=[init, final], handoff_index=1,
                        flip_success=True, ctrl2_success=True)
    row = eval_states_row(init, res)
    assert len(row) == 28           # 13 + 13 + 2
    assert row[:13] == pytest.approx(init)
    assert row[13:26] == pytest.approx(final)
    assert row[26:] == [1, 1]


def test_baseline_eval_states_row_is_init_final_and_one_label():
    from generate_quadrotor_3d_composition import baseline_eval_states_row
    from quad_composition.rollout3d import RolloutResult
    init = list(range(13))
    final = [float(-v) for v in range(13)]
    # flip_success is always False on the baseline path (ctrl1=None) --
    # included here to prove the row builder ignores it regardless.
    res = RolloutResult(trajectory=[init, final], handoff_index=-1,
                        flip_success=False, ctrl2_success=True)
    row = baseline_eval_states_row(init, res)
    assert len(row) == 27           # 13 + 13 + 1, matching the archived format
    assert row[:13] == pytest.approx(init)
    assert row[13:26] == pytest.approx(final)
    assert row[26] == 1


def test_handoff_row_is_minus_one_when_no_handoff():
    from generate_quadrotor_3d_composition import handoff_row
    from quad_composition.rollout3d import RolloutResult
    init = list(range(13))
    res = RolloutResult(trajectory=[init], handoff_index=-1,
                        flip_success=False, ctrl2_success=False)
    row = handoff_row(init, res)
    assert row == pytest.approx(init + [-1.0] * 13 + [-1.0])


def test_handoff_row_is_the_state_at_handoff_index():
    from generate_quadrotor_3d_composition import handoff_row
    from quad_composition.rollout3d import RolloutResult
    init = list(range(13))
    handoff_state = [v / 10.0 for v in range(13)]
    res = RolloutResult(trajectory=[init, handoff_state], handoff_index=1,
                        flip_success=True, ctrl2_success=False)
    row = handoff_row(init, res)
    assert row == pytest.approx(init + handoff_state + [1.0])


def test_handoff_row_persists_the_handoff_index_so_step_zero_is_distinguishable():
    '''A rollout whose INITIAL state was already inside G1 has
    handoff_index == 0 -- controller 1 never acted -- and must be
    distinguishable from a real handoff.
    '''
    from generate_quadrotor_3d_composition import handoff_row
    from quad_composition.rollout3d import RolloutResult

    init = _dataset_state(tilt=0.02, omega=0.01)   # already inside a tight G1
    started_inside = RolloutResult(trajectory=[init, [0.0] * 13], handoff_index=0,
                                   flip_success=True, ctrl2_success=True)
    real = RolloutResult(trajectory=[init, list(init)], handoff_index=1,
                         flip_success=True, ctrl2_success=True)

    assert handoff_row(init, started_inside)[26] == 0
    assert handoff_row(init, real)[26] == 1
    assert handoff_row(init, started_inside)[:26] == pytest.approx(handoff_row(init, real)[:26])


# ---------------------------------------------------------------------------
# generate_dataset (sequential): the rollout-and-write loop, tested against
# fakes so --mode flip/composite are exercised without booting a real SAC
# checkpoint. This is the SAME per-index logic (process_one) a real worker
# process runs.
# ---------------------------------------------------------------------------

def test_generate_dataset_flip_mode_truncates_trajectory_at_handoff(monkeypatch, tmp_path):
    from generate_quadrotor_3d_composition import generate_dataset
    from quad_composition import rollout3d

    monkeypatch.setattr(rollout3d, 'set_initial_state', _fake_set_initial_state)
    env = _ScriptedEnv3D(_HANDOFF_SCRIPT)
    ctrl = _FakeCtrl()

    rows, handoffs = generate_dataset('flip', env, ctrl, ctrl, _small_g1(), [_INIT], str(tmp_path))

    traj = np.loadtxt(tmp_path / 'trajectories' / 'sequence_0.txt', delimiter=',', ndmin=2)
    assert len(traj) == 3  # init + 2 steps, truncated at (and including) the handoff state
    assert len(rows) == 1 and len(rows[0]) == 28
    assert rows[0][26] == 1  # flip_success: controller 1 reached G1
    assert rows[0][27] == 1  # ctrl2_success recorded from the FULL underlying rollout
    handoff_state = _dataset_state(tilt=0.02, omega=0.01)
    assert rows[0][13:26] == pytest.approx(handoff_state)   # stored final = truncated handoff state
    assert handoffs[0][13:26] == pytest.approx(handoff_state)
    assert handoffs[0][26] == 2   # handoff_index: trajectory row 2


def test_generate_dataset_composite_mode_keeps_the_full_trajectory(monkeypatch, tmp_path):
    from generate_quadrotor_3d_composition import generate_dataset
    from quad_composition import rollout3d

    monkeypatch.setattr(rollout3d, 'set_initial_state', _fake_set_initial_state)
    env = _ScriptedEnv3D(_HANDOFF_SCRIPT)
    ctrl = _FakeCtrl()

    rows, handoffs = generate_dataset('composite', env, ctrl, ctrl, _small_g1(), [_INIT], str(tmp_path))

    traj = np.loadtxt(tmp_path / 'trajectories' / 'sequence_0.txt', delimiter=',', ndmin=2)
    assert len(traj) == 4  # init + all 3 steps, not truncated
    assert rows[0][26] == 1  # flip_success
    assert rows[0][27] == 1  # ctrl2_success
    final_state = _dataset_state(tilt=0.0, omega=0.0)
    assert rows[0][13:26] == pytest.approx(final_state)


def test_generate_dataset_baseline_mode_uses_ctrl2_alone_and_one_label(monkeypatch, tmp_path):
    from generate_quadrotor_3d_composition import generate_dataset
    from quad_composition import rollout3d

    monkeypatch.setattr(rollout3d, 'set_initial_state', _fake_set_initial_state)
    script = [(0.1, 0.1, False, False), (0.0, 0.0, True, True)]
    env = _ScriptedEnv3D(script)
    ctrl2 = _FakeCtrl()

    rows, handoffs = generate_dataset('baseline', env, None, ctrl2, None, [_INIT], str(tmp_path))

    assert len(rows) == 1 and len(rows[0]) == 27
    # baseline never hands off: handoff state and handoff_index are all -1
    assert handoffs[0][13:] == pytest.approx([-1.0] * 14)


def test_generate_dataset_records_handoff_index_zero_for_an_init_already_inside_g1(
        monkeypatch, tmp_path):
    from generate_quadrotor_3d_composition import generate_dataset
    from quad_composition import rollout3d

    monkeypatch.setattr(rollout3d, 'set_initial_state', _fake_set_initial_state)
    env = _ScriptedEnv3D([(0.0, 0.0, True, True)])
    ctrl = _FakeCtrl()

    inside_g1 = _dataset_state(tilt=0.02, omega=0.01)   # |tilt| < 0.05, |omega| < 0.05
    rows, handoffs = generate_dataset('composite', env, ctrl, ctrl, _small_g1(), [inside_g1],
                                      str(tmp_path))

    assert rows[0][26] == 1, 'flip_success is 1, exactly as for a real handoff'
    assert handoffs[0][26] == 0, 'handoff_index must record that ctrl1 never acted'
    assert handoffs[0][13:26] == pytest.approx(inside_g1)


def test_prefix_invariant_flip_trajectory_prefixes_composite_up_to_handoff(monkeypatch, tmp_path):
    '''The flip-only trajectory for a given init must be a byte-for-byte
    prefix of the composite trajectory for the same init, up to and including
    the handoff state.
    '''
    from generate_quadrotor_3d_composition import generate_dataset
    from quad_composition import rollout3d

    monkeypatch.setattr(rollout3d, 'set_initial_state', _fake_set_initial_state)
    ctrl = _FakeCtrl()
    g1 = _small_g1()

    flip_dir, comp_dir = tmp_path / 'flip', tmp_path / 'composite'
    generate_dataset('flip', _ScriptedEnv3D(_HANDOFF_SCRIPT), ctrl, ctrl, g1, [_INIT], str(flip_dir))
    generate_dataset('composite', _ScriptedEnv3D(_HANDOFF_SCRIPT), ctrl, ctrl, g1, [_INIT], str(comp_dir))

    flip_traj = np.loadtxt(flip_dir / 'trajectories' / 'sequence_0.txt', delimiter=',', ndmin=2)
    comp_traj = np.loadtxt(comp_dir / 'trajectories' / 'sequence_0.txt', delimiter=',', ndmin=2)

    assert len(flip_traj) < len(comp_traj), 'flip must actually be truncated for this fixture'
    assert np.allclose(flip_traj, comp_traj[:len(flip_traj)])


# ---------------------------------------------------------------------------
# Sidecar files: atomic writes and the resumability they enable.
# ---------------------------------------------------------------------------

def test_write_trajectory_and_label_round_trip_and_leave_no_tmp_file(tmp_path):
    from generate_quadrotor_3d_composition import (label_path, read_label, trajectory_path,
                                                    write_label, write_trajectory)
    output_dir = str(tmp_path)
    write_trajectory(output_dir, 0, [[0.0] * 13, [1.0] * 13])
    write_label(output_dir, 0, [1.0] * 28, [2.0] * 27)

    traj = np.loadtxt(trajectory_path(output_dir, 0), delimiter=',', ndmin=2)
    assert traj.shape == (2, 13)
    eval_row, ho_row = read_label(output_dir, 0)
    assert eval_row == [1.0] * 28
    assert ho_row == [2.0] * 27
    assert list(tmp_path.rglob('*.tmp*')) == []


def test_pending_indices_skips_already_done(tmp_path):
    from generate_quadrotor_3d_composition import pending_indices, write_label, write_trajectory
    output_dir = str(tmp_path)
    inits = [_INIT] * 4
    for i in (0, 2):
        write_trajectory(output_dir, i, [_INIT])
        write_label(output_dir, i, [0.0], [0.0])

    todo = pending_indices(output_dir, inits)
    assert [i for i, _ in todo] == [1, 3]


def test_pending_indices_requires_both_trajectory_and_label(tmp_path):
    '''A trajectory file with no label sidecar (or vice versa) is NOT done --
    an interrupted worker could plausibly leave either half-finished.
    '''
    from generate_quadrotor_3d_composition import (pending_indices, write_label,
                                                    write_trajectory)
    output_dir = str(tmp_path)
    write_trajectory(output_dir, 0, [_INIT])   # no label written
    write_label(output_dir, 1, [0.0], [0.0])   # no trajectory written

    todo = pending_indices(output_dir, [_INIT, _INIT])
    assert [i for i, _ in todo] == [0, 1]


def test_run_parallel_skips_all_work_and_never_spawns_a_pool_when_everything_is_done(tmp_path):
    '''Resumability's short circuit: if every index is already written,
    run_parallel must return the persisted rows without ever building a
    worker (a real spawn would try `load_ctrl1` on the bogus path below and
    fail).
    '''
    from generate_quadrotor_3d_composition import (baseline_eval_states_row, handoff_row,
                                                    run_parallel, write_label, write_trajectory)
    from quad_composition.g1 import G1Region
    from quad_composition.rollout3d import RolloutResult

    inits = [_INIT, _INIT]
    output_dir = str(tmp_path)
    expected_rows, expected_handoffs = [], []
    for i, init in enumerate(inits):
        res = RolloutResult(trajectory=[init, init], handoff_index=-1,
                            flip_success=False, ctrl2_success=bool(i))
        eval_row = baseline_eval_states_row(init, res)
        ho_row = handoff_row(init, res)
        write_trajectory(output_dir, i, res.trajectory)
        write_label(output_dir, i, eval_row, ho_row)
        expected_rows.append(eval_row)
        expected_handoffs.append(ho_row)

    rows, handoffs = run_parallel('baseline', inits, '/does/not/exist.pt',
                                  G1Region(tilt_c=0.175, w_c=4.0), output_dir, num_workers=1)
    assert rows == expected_rows
    assert handoffs == expected_handoffs


# ---------------------------------------------------------------------------
# write_outputs / build_description: column widths, validation wiring, the
# regenerated-baseline note, and G1/attitude-convention documentation.
# ---------------------------------------------------------------------------

def _make_args(mode, output_dir, ctrl1_path='unused.pt'):
    from generate_quadrotor_3d_composition import parse_args
    argv = ['--mode', mode, '--ctrl1_path', ctrl1_path,
           '--baseline_dir', '/x', '--output_dir', str(output_dir)]
    return parse_args(argv)


def test_write_outputs_baseline_mode_writes_27_and_14_column_files(tmp_path):
    from generate_quadrotor_3d_composition import write_outputs
    args = _make_args('baseline', tmp_path)
    rows = [[0.0] * 27, [0.1] * 26 + [1.0]]
    handoffs = [[0.0] * 26 + [-1.0], [0.0] * 26 + [-1.0]]

    write_outputs(args, rows, handoffs)

    eval_states = np.loadtxt(tmp_path / 'eval_states.txt', delimiter=',', ndmin=2)
    roa = np.loadtxt(tmp_path / 'roa_labels.txt', delimiter=',', ndmin=2)
    assert eval_states.shape == (2, 27)
    assert roa.shape == (2, 14)

    desc = json.loads((tmp_path / 'dataset_description.json').read_text())
    assert 'regenerated_baseline_note' in desc
    assert 'g1' not in desc
    assert desc['statistics']['ctrl2_success'] == 1


def test_write_outputs_flip_mode_writes_28_and_15_column_files(tmp_path):
    from generate_quadrotor_3d_composition import write_outputs
    args = _make_args('flip', tmp_path)
    rows = [[0.0] * 26 + [1.0, 0.0], [0.0] * 26 + [1.0, 1.0]]
    handoffs = [[0.0] * 26 + [-1.0], [0.0] * 26 + [-1.0]]

    write_outputs(args, rows, handoffs)

    eval_states = np.loadtxt(tmp_path / 'eval_states.txt', delimiter=',', ndmin=2)
    roa = np.loadtxt(tmp_path / 'roa_labels.txt', delimiter=',', ndmin=2)
    assert eval_states.shape == (2, 28)
    assert roa.shape == (2, 15)

    desc = json.loads((tmp_path / 'dataset_description.json').read_text())
    assert 'regenerated_baseline_note' in desc
    assert desc['g1']['tilt_c_rad'] == pytest.approx(0.175)
    assert desc['g1']['w_c_rad_s'] == pytest.approx(4.0)
    assert desc['statistics'] == {'total': 2, 'flip_success': 2, 'ctrl2_success': 1}


def test_write_outputs_rejects_impossible_labels_for_flip_and_composite(tmp_path):
    from generate_quadrotor_3d_composition import write_outputs
    args = _make_args('composite', tmp_path)
    rows = [[0.0] * 26 + [0, 1]]  # impossible: flip_success=0, ctrl2_success=1
    with pytest.raises(ValueError, match='impossible label'):
        write_outputs(args, rows, [[0.0] * 26 + [-1.0]])
    assert not os.path.exists(tmp_path / 'eval_states.txt')


def test_descriptions_record_the_fixed_g1_and_attitude_convention(tmp_path):
    from generate_quadrotor_3d_composition import write_outputs
    for mode, rows in (('baseline', [[0.0] * 27]),
                      ('composite', [[0.0] * 26 + [1.0, 1.0]])):
        out = tmp_path / mode
        args = _make_args(mode, out)
        write_outputs(args, rows, [[0.0] * 26 + [-1.0]])

        desc = json.loads((out / 'dataset_description.json').read_text())
        note = desc['attitude_convention']
        assert 'quadrotor3D_lqr' in note
        assert 'rotation matrix' in note.lower() or 'tilt_from_quat' in note


def test_baseline_description_documents_handoff_states_and_its_inert_columns(tmp_path):
    from generate_quadrotor_3d_composition import write_outputs
    args = _make_args('baseline', tmp_path)
    write_outputs(args, [[0.0] * 27], [[0.0] * 13 + [-1.0] * 14])

    desc = json.loads((tmp_path / 'dataset_description.json').read_text())
    handoff_doc = desc['files']['handoff_states.txt']
    assert '27 columns' in handoff_doc
    assert 'handoff_index' in handoff_doc
    assert 'ALWAYS' in handoff_doc and '-1' in handoff_doc
    assert {'eval_states.txt', 'roa_labels.txt', 'handoff_states.txt'} <= set(desc['files'])


# ---------------------------------------------------------------------------
# main(): wiring, verified WITHOUT spawning a real worker (monkeypatch
# run_parallel/snapshot_checkpoint directly -- both are looked up on their
# module at call time, so this works even though run_parallel itself uses
# multiprocessing when not faked).
# ---------------------------------------------------------------------------

def _write_fake_baseline_dir(tmp_path, inits):
    baseline_dir = tmp_path / 'shipped'
    baseline_dir.mkdir()
    padded = np.column_stack([inits, np.zeros((len(inits), 14))])   # 13 + 14 = 27, like the shipped file
    np.savetxt(baseline_dir / 'eval_states.txt', padded, delimiter=',')
    return baseline_dir


def test_main_baseline_mode_never_snapshots_checkpoint_and_forces_ctrl1_path_to_none(
        monkeypatch, tmp_path):
    import generate_quadrotor_3d_composition as gen

    inits = np.array([_INIT, _INIT])
    baseline_dir = _write_fake_baseline_dir(tmp_path, inits)
    output_dir = tmp_path / 'out'

    calls = []

    def fake_run_parallel(mode, inits_, ctrl1_path, g1, output_dir_, num_workers,
                          max_steps=None, progress=None):
        calls.append((mode, ctrl1_path))
        from quad_composition.rollout3d import RolloutResult
        rows, handoffs = [], []
        for init in inits_:
            res = RolloutResult(trajectory=[init, init], handoff_index=-1,
                                flip_success=False, ctrl2_success=True)
            rows.append(gen.baseline_eval_states_row(init, res))
            handoffs.append(gen.handoff_row(init, res))
        return rows, handoffs

    monkeypatch.setattr(gen, 'run_parallel', fake_run_parallel)
    monkeypatch.setattr(sys, 'argv', ['prog', '--mode', 'baseline',
                                      '--ctrl1_path', '/does/not/exist.pt',
                                      '--baseline_dir', str(baseline_dir),
                                      '--output_dir', str(output_dir), '--num_workers', '1'])
    gen.main()

    assert calls == [('baseline', None)]   # ctrl1_path forced to None; snapshot never imported
    eval_states = np.loadtxt(output_dir / 'eval_states.txt', delimiter=',', ndmin=2)
    assert eval_states.shape == (2, 27)


def test_main_composite_mode_snapshots_the_checkpoint_before_running(monkeypatch, tmp_path):
    import analyze_quad3d_composition as analyze
    import generate_quadrotor_3d_composition as gen

    inits = np.array([_INIT])
    baseline_dir = _write_fake_baseline_dir(tmp_path, inits)
    output_dir = tmp_path / 'out'

    snapshot_calls = []

    def fake_snapshot(src, tmp_root):
        snapshot_calls.append(src)
        return '/frozen/ctrl1.pt', {'path': src, 'mtime': 0, 'size': 0, 'warning': None}

    run_parallel_calls = []

    def fake_run_parallel(mode, inits_, ctrl1_path, g1, output_dir_, num_workers,
                          max_steps=None, progress=None):
        run_parallel_calls.append((mode, ctrl1_path, g1))
        from quad_composition.rollout3d import RolloutResult
        res = RolloutResult(trajectory=[inits_[0], inits_[0]], handoff_index=0,
                            flip_success=True, ctrl2_success=True)
        return [gen.eval_states_row(inits_[0], res)], [gen.handoff_row(inits_[0], res)]

    monkeypatch.setattr(analyze, 'snapshot_checkpoint', fake_snapshot)
    monkeypatch.setattr(gen, 'run_parallel', fake_run_parallel)
    monkeypatch.setattr(sys, 'argv', ['prog', '--mode', 'composite',
                                      '--ctrl1_path', 'unused.pt',
                                      '--baseline_dir', str(baseline_dir),
                                      '--output_dir', str(output_dir), '--num_workers', '1'])
    gen.main()

    assert snapshot_calls == ['unused.pt']
    assert run_parallel_calls[0][0] == 'composite'
    assert run_parallel_calls[0][1] == '/frozen/ctrl1.pt'   # the FROZEN path, not the raw arg
    assert run_parallel_calls[0][2] is gen.G_NOM_3D
    desc = json.loads((output_dir / 'dataset_description.json').read_text())
    assert desc['controller_1']['checkpoint_info']['path'] == 'unused.pt'


# ---------------------------------------------------------------------------
# Real, slow end-to-end tests: --mode baseline needs no checkpoint (Ruling
# D-I), so its full Pool + resumability path is exercised for real, not just
# against fakes -- mirroring test_composition_datasets.py's
# test_baseline_mode_runs_for_real_end_to_end, extended to cover parallelism
# and resume-after-interruption.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_baseline_mode_runs_for_real_with_parallel_workers_and_resumes_after_interruption(tmp_path):
    if not os.path.exists(os.path.join(SHIPPED, 'eval_states.txt')):
        pytest.skip('shipped dataset not mounted')
    import generate_quadrotor_3d_composition as gen

    output_dir = tmp_path / 'q3d_baseline_smoke'
    argv = ['prog', '--mode', 'baseline', '--baseline_dir', SHIPPED,
           '--output_dir', str(output_dir), '--limit', '4', '--num_workers', '2']
    old_argv = sys.argv
    sys.argv = argv
    try:
        gen.main()
    finally:
        sys.argv = old_argv

    eval_states = np.loadtxt(output_dir / 'eval_states.txt', delimiter=',', ndmin=2)
    assert eval_states.shape == (4, 27)
    roa = np.loadtxt(output_dir / 'roa_labels.txt', delimiter=',', ndmin=2)
    assert roa.shape == (4, 14)
    handoffs = np.loadtxt(output_dir / 'handoff_states.txt', delimiter=',', ndmin=2)
    assert handoffs.shape == (4, 27)
    assert np.all(handoffs[:, 13:] == -1.0)   # controller 1 never runs on this path

    desc = json.loads((output_dir / 'dataset_description.json').read_text())
    assert desc['statistics']['total'] == 4
    assert 'regenerated_baseline_note' in desc
    assert 'g1' not in desc

    # Resumability: delete one trajectory + its label sidecar (an interrupted
    # run) and rerun with the SAME argv -- only that index should need
    # recomputing, and the reassembled files must still be complete.
    os.remove(gen.trajectory_path(str(output_dir), 1))
    os.remove(gen.label_path(str(output_dir), 1))
    sys.argv = argv
    try:
        gen.main()
    finally:
        sys.argv = old_argv

    eval_states2 = np.loadtxt(output_dir / 'eval_states.txt', delimiter=',', ndmin=2)
    assert eval_states2.shape == (4, 27)
    # Rows untouched by the deletion must be byte-identical to the first run.
    assert eval_states2[[0, 2, 3]] == pytest.approx(eval_states[[0, 2, 3]])


@pytest.mark.slow
def test_composite_mode_runs_for_real_with_the_real_checkpoint(tmp_path):
    if not os.path.exists(os.path.join(SHIPPED, 'eval_states.txt')):
        pytest.skip('shipped dataset not mounted')
    if not os.path.exists(CTRL1_CHECKPOINT):
        pytest.skip('controller 1 checkpoint not present')
    import generate_quadrotor_3d_composition as gen

    output_dir = tmp_path / 'q3d_composite_smoke'
    old_argv = sys.argv
    sys.argv = ['prog', '--mode', 'composite', '--ctrl1_path', CTRL1_CHECKPOINT,
               '--baseline_dir', SHIPPED, '--output_dir', str(output_dir),
               '--limit', '2', '--num_workers', '1']
    try:
        gen.main()
    finally:
        sys.argv = old_argv

    eval_states = np.loadtxt(output_dir / 'eval_states.txt', delimiter=',', ndmin=2)
    assert eval_states.shape == (2, 28)
    desc = json.loads((output_dir / 'dataset_description.json').read_text())
    assert desc['g1']['tilt_c_rad'] == pytest.approx(0.175)
    assert desc['controller_1']['checkpoint_info']['path'] == os.path.abspath(CTRL1_CHECKPOINT)
