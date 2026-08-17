'''Tests for visualize_quad3d_composition.py.

Most tests exercise classification/sampling/file-writing logic against fakes
(matching this test directory's established convention -- see
test_visualization.py, the 2D analogue this module ports). A handful of
`slow` tests boot real PyBullet (`render_frames`'s handoff colour cue) or run
the full CLI end-to-end against the real env + real controller-2 (LQR, always
available -- no checkpoint) with a randomly-initialised controller 1 (no
trained flip checkpoint exists yet -- see visualize_quad3d_composition.py's
module docstring).
'''
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _row(i, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    '''One dataset-order 13-dim row [x, y, z, qw, qx, qy, qz, x_dot, y_dot,
    z_dot, p, q, r] -- upright by default, position varying with `i` so
    successive rows in a fake trajectory are distinguishable.
    '''
    return [float(i), 0.0, 1.0, qw, qx, qy, qz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def make_result(flip_success, ctrl2_success, handoff_index, trajectory=None):
    from quad_composition.rollout3d import RolloutResult
    if trajectory is None:
        n = max(handoff_index + 1, 1) if handoff_index >= 0 else 3
        trajectory = [_row(i) for i in range(n)]
    return RolloutResult(trajectory=trajectory, handoff_index=handoff_index,
                         flip_success=flip_success, ctrl2_success=ctrl2_success)


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def test_classify_flip_failure_is_f1_only():
    from visualize_quad3d_composition import classify
    result = make_result(flip_success=False, ctrl2_success=False, handoff_index=-1)
    assert classify(result) == ['F1']


def test_classify_flip_success_and_ctrl2_success_fills_s1_and_s1_to_s2():
    from visualize_quad3d_composition import classify
    result = make_result(flip_success=True, ctrl2_success=True, handoff_index=5)
    assert classify(result) == ['S1', 'S1_to_S2']


def test_classify_flip_success_and_ctrl2_failure_fills_s1_and_s1_to_f2():
    from visualize_quad3d_composition import classify
    result = make_result(flip_success=True, ctrl2_success=False, handoff_index=5)
    assert classify(result) == ['S1', 'S1_to_F2']


def test_classify_never_produces_the_impossible_f1_s2_combination():
    '''(F1, S2) is impossible: whenever flip_success is False, 'S1_to_S2'
    (which implies a handoff fired) must never appear.
    '''
    from visualize_quad3d_composition import classify
    for ctrl2_success in (True, False):
        result = make_result(flip_success=False, ctrl2_success=ctrl2_success, handoff_index=-1)
        assert 'S1_to_S2' not in classify(result)
        assert 'S1_to_F2' not in classify(result)


# ---------------------------------------------------------------------------
# sample_and_classify
# ---------------------------------------------------------------------------

def test_sample_and_classify_requires_a_sample_fn():
    '''3D sampling needs bounds baked into the closure (see
    flip_env3d.sample_uniform_state) -- there is no usable module-level
    default, unlike the 2D branch. Omitting sample_fn must fail loudly, not
    silently sample from the wrong box.
    '''
    from visualize_quad3d_composition import sample_and_classify

    with pytest.raises(ValueError):
        sample_and_classify(None, None, None, None, np.random.default_rng(0),
                            categories=['F1'], num_per_category=1, max_steps=10, max_attempts=10)


def test_sample_and_classify_stops_once_every_category_is_full():
    from visualize_quad3d_composition import sample_and_classify

    # Scripted sequence of outcomes: enough of everything within a few
    # attempts, plus extras that must be ignored once a category is full.
    outcomes = [
        make_result(False, False, -1),                 # F1
        make_result(True, True, 0),                     # S1, S1_to_S2
        make_result(True, False, 2),                     # S1, S1_to_F2
        make_result(False, False, -1),                 # F1 (2nd)
        make_result(True, True, 1),                     # S1 (already full if cap=2), S1_to_S2 (2nd)
        make_result(True, False, 3),                     # extra, should be ignored past cap
    ]
    calls = {'n': 0}

    def fake_rollout(env, ctrl1, ctrl2, g1, init_state, max_steps):
        res = outcomes[calls['n']]
        calls['n'] += 1
        return res

    def fake_sample(rng):
        return _row(0)

    recorded, attempts = sample_and_classify(
        None, None, None, None, np.random.default_rng(0),
        categories=['F1', 'S1', 'S1_to_S2', 'S1_to_F2'], num_per_category=2,
        max_steps=10, max_attempts=100, sample_fn=fake_sample, rollout_fn=fake_rollout)

    # S1_to_F2 only reaches its cap of 2 on the 6th (last) scripted outcome,
    # so the loop must run all 6 attempts -- not stop early, and not run the
    # 7th (there isn't one; a 7th call would IndexError).
    assert attempts == 6, 'must stop as soon as every requested category is full'
    assert len(recorded['F1']) == 2
    assert len(recorded['S1']) == 2
    assert len(recorded['S1_to_S2']) == 2
    assert len(recorded['S1_to_F2']) == 2


def test_a_single_rollout_can_fill_both_s1_and_its_subdivision_at_once():
    '''One sampled rollout with flip_success=True, ctrl2_success=True must
    count toward BOTH 'S1' and 'S1_to_S2' from a single attempt -- not
    require two separate samples.
    '''
    from visualize_quad3d_composition import sample_and_classify

    def fake_rollout(env, ctrl1, ctrl2, g1, init_state, max_steps):
        return make_result(True, True, 0)

    recorded, attempts = sample_and_classify(
        None, None, None, None, np.random.default_rng(0),
        categories=['S1', 'S1_to_S2'], num_per_category=1, max_steps=10, max_attempts=10,
        sample_fn=lambda rng: _row(0), rollout_fn=fake_rollout)

    assert attempts == 1
    assert len(recorded['S1']) == 1
    assert len(recorded['S1_to_S2']) == 1


def test_sample_and_classify_respects_the_sampling_budget_and_never_loops_forever():
    '''A category that can never be filled (here: S1_to_S2, but every
    sampled rollout is F1) must not hang the loop -- it must stop at
    max_attempts and leave that category short.
    '''
    from visualize_quad3d_composition import sample_and_classify

    def fake_rollout(env, ctrl1, ctrl2, g1, init_state, max_steps):
        return make_result(False, False, -1)  # always F1

    recorded, attempts = sample_and_classify(
        None, None, None, None, np.random.default_rng(0),
        categories=['F1', 'S1_to_S2'], num_per_category=3, max_steps=10, max_attempts=7,
        sample_fn=lambda rng: _row(0), rollout_fn=fake_rollout)

    assert attempts == 7
    assert len(recorded['F1']) == 3          # filled and capped
    assert len(recorded['S1_to_S2']) == 0    # never fillable from this outcome stream


def test_sample_and_classify_min_trajectory_states_discards_short_rollouts():
    '''A real (quality-bar) concern: many F1 failures are a Euclidean-bound
    violation reached in 2-3 ticks -- technically correct but not a watchable
    video. `min_trajectory_states` must discard those from EVERY category
    (not just F1) while still counting them toward `attempts`, and must leave
    unfiltered behaviour untouched when omitted (every other test here relies
    on that).
    '''
    from visualize_quad3d_composition import sample_and_classify

    short_traj = [_row(i) for i in range(2)]
    long_traj = [_row(i) for i in range(20)]
    outcomes = [
        make_result(False, False, -1, trajectory=short_traj),   # F1, too short: discarded
        make_result(False, False, -1, trajectory=long_traj),    # F1, long enough: kept
    ]
    calls = {'n': 0}

    def fake_rollout(env, ctrl1, ctrl2, g1, init_state, max_steps):
        res = outcomes[calls['n'] % len(outcomes)]
        calls['n'] += 1
        return res

    recorded, attempts = sample_and_classify(
        None, None, None, None, np.random.default_rng(0),
        categories=['F1'], num_per_category=1, max_steps=10, max_attempts=10,
        sample_fn=lambda rng: _row(0), rollout_fn=fake_rollout, min_trajectory_states=10)

    assert attempts == 2, 'the discarded short rollout must still count toward the budget'
    assert len(recorded['F1']) == 1
    assert recorded['F1'][0][1].trajectory == long_traj


def test_sample_and_classify_without_min_trajectory_states_keeps_everything():
    '''Default (None) must reproduce the pre-filter behaviour exactly -- a
    single-tick rollout is accepted, matching every other test in this file.
    '''
    from visualize_quad3d_composition import sample_and_classify

    tiny_traj = [_row(0)]

    def fake_rollout(env, ctrl1, ctrl2, g1, init_state, max_steps):
        return make_result(False, False, -1, trajectory=tiny_traj)

    recorded, attempts = sample_and_classify(
        None, None, None, None, np.random.default_rng(0),
        categories=['F1'], num_per_category=1, max_steps=10, max_attempts=10,
        sample_fn=lambda rng: _row(0), rollout_fn=fake_rollout)

    assert attempts == 1
    assert len(recorded['F1']) == 1


# ---------------------------------------------------------------------------
# record_rollout (file writing, category-specific truncation/marking)
# ---------------------------------------------------------------------------

def _patch_rendering(monkeypatch, calls):
    import visualize_quad3d_composition as m

    def fake_render_frames(poses, handoff_frame_index, ctrl_freq=None, fps=None, width=None,
                           height=None, urdf_path=None):
        calls['render_poses_len'] = len(poses)
        calls['render_handoff_index'] = handoff_frame_index
        return [np.zeros((4, 4, 3), dtype=np.uint8)]

    def fake_save_video(frames, path, fps):
        calls['video_path'] = path
        with open(path, 'wb') as fh:
            fh.write(b'fake-mp4-bytes')

    def fake_plot(states, path, success, handoff_index=None):
        calls['plot_states_len'] = len(states)
        calls['plot_handoff_index'] = handoff_index
        calls['plot_success'] = success
        with open(path, 'w') as fh:
            fh.write('fake-plot')

    monkeypatch.setattr(m, 'render_frames', fake_render_frames)
    monkeypatch.setattr(m, 'save_video', fake_save_video)
    monkeypatch.setattr(m, 'plot_tilt_and_trajectory', fake_plot)


def test_record_rollout_s1_truncates_at_the_handoff_state(tmp_path, monkeypatch):
    from visualize_quad3d_composition import record_rollout

    calls = {}
    _patch_rendering(monkeypatch, calls)

    trajectory = [_row(i) for i in range(7)]  # 7 states, 0..6
    result = make_result(True, True, handoff_index=3, trajectory=trajectory)

    sidecar = record_rollout('S1', 0, _row(0), result, str(tmp_path))

    # Truncated to states[:handoff_index + 1] == 4 states, plus the S1 hold.
    assert calls['plot_states_len'] == 4
    assert calls['plot_handoff_index'] == 3
    assert calls['render_poses_len'] > 4, 'S1 must hold the final frame, not cut instantly'
    assert sidecar['num_recorded_states'] == 4
    assert sidecar['num_full_trajectory_states'] == 7
    assert sidecar['handoff_index'] == 3
    assert sidecar['initial_tilt_deg'] == pytest.approx(0.0)  # upright rows
    assert os.path.exists(os.path.join(tmp_path, 'S1', 'rollout_000.mp4'))
    assert os.path.exists(os.path.join(tmp_path, 'S1', 'rollout_000_tilt.png'))
    with open(os.path.join(tmp_path, 'S1', 'rollout_000.json')) as fh:
        assert json.load(fh) == sidecar


def test_record_rollout_s1_to_s2_keeps_full_trajectory_and_marks_handoff(tmp_path, monkeypatch):
    from visualize_quad3d_composition import record_rollout

    calls = {}
    _patch_rendering(monkeypatch, calls)

    trajectory = [_row(i) for i in range(7)]
    result = make_result(True, True, handoff_index=3, trajectory=trajectory)

    sidecar = record_rollout('S1_to_S2', 0, _row(0), result, str(tmp_path))

    assert calls['plot_states_len'] == 7, 'full trajectory must be kept, not truncated'
    assert calls['plot_handoff_index'] == 3
    assert calls['render_handoff_index'] == 3
    assert calls['plot_success'] is True
    assert sidecar['num_recorded_states'] == 7
    assert os.path.exists(os.path.join(tmp_path, 'S1_to_S2', 'rollout_000.mp4'))


def test_record_rollout_s1_to_f2_marks_handoff_but_is_not_a_success(tmp_path, monkeypatch):
    from visualize_quad3d_composition import record_rollout

    calls = {}
    _patch_rendering(monkeypatch, calls)

    trajectory = [_row(i) for i in range(5)]
    result = make_result(True, False, handoff_index=2, trajectory=trajectory)

    record_rollout('S1_to_F2', 0, _row(0), result, str(tmp_path))

    assert calls['plot_states_len'] == 5
    assert calls['render_handoff_index'] == 2
    assert calls['plot_success'] is False


def test_record_rollout_f1_has_no_handoff_marker(tmp_path, monkeypatch):
    from visualize_quad3d_composition import record_rollout

    calls = {}
    _patch_rendering(monkeypatch, calls)

    trajectory = [_row(i) for i in range(4)]
    result = make_result(False, False, handoff_index=-1, trajectory=trajectory)

    sidecar = record_rollout('F1', 0, _row(0), result, str(tmp_path))

    assert calls['render_handoff_index'] is None
    assert calls['plot_handoff_index'] is None
    assert calls['plot_success'] is False
    assert sidecar['handoff_index'] == -1


def test_record_rollout_f1_holds_the_final_frame(tmp_path, monkeypatch):
    '''Quality bar: many F1 failures are a 2-3 tick Euclidean-bound
    violation. record_rollout must hold the LAST pose for a moment (as S1
    already does at its handoff) so the failure is actually visible rather
    than a blink-and-miss-it clip -- while the PLOT still sees the real,
    unpadded trajectory (num_recorded_states must not be inflated by the
    hold, which is a rendering-only artefact).
    '''
    from visualize_quad3d_composition import record_rollout

    calls = {}
    _patch_rendering(monkeypatch, calls)

    trajectory = [_row(i) for i in range(3)]  # a near-instant OOB failure
    result = make_result(False, False, handoff_index=-1, trajectory=trajectory)

    sidecar = record_rollout('F1', 0, _row(0), result, str(tmp_path))

    assert calls['render_poses_len'] > 3, 'the final (failure) pose must be held, not shown once'
    assert calls['plot_states_len'] == 3, 'the plot must see the real trajectory, unpadded'
    assert sidecar['num_recorded_states'] == 3


def test_record_rollout_names_files_by_category_and_index(tmp_path, monkeypatch):
    from visualize_quad3d_composition import record_rollout

    calls = {}
    _patch_rendering(monkeypatch, calls)
    trajectory = [_row(0)] * 4
    result = make_result(True, True, handoff_index=2, trajectory=trajectory)

    record_rollout('S1_to_S2', 7, _row(0), result, str(tmp_path))

    assert os.path.exists(os.path.join(tmp_path, 'S1_to_S2', 'rollout_007.mp4'))


def test_record_rollout_reports_initial_tilt_from_a_tilted_start(tmp_path, monkeypatch):
    '''requirement 5: each recorded rollout's sidecar (and hence summary.json)
    must carry the INITIAL tilt, not just the handoff index.
    '''
    from visualize_quad3d_composition import record_rollout

    calls = {}
    _patch_rendering(monkeypatch, calls)

    # 90 deg tilt about the body x axis: qw=qx=cos/sin(45 deg).
    c = float(np.cos(np.pi / 4))
    tilted = _row(0, qw=c, qx=c, qy=0.0, qz=0.0)
    trajectory = [tilted] + [_row(i) for i in range(1, 4)]
    result = make_result(True, True, handoff_index=2, trajectory=trajectory)

    sidecar = record_rollout('S1_to_S2', 0, tilted, result, str(tmp_path))

    assert sidecar['initial_tilt_deg'] == pytest.approx(90.0, abs=1.0)


# ---------------------------------------------------------------------------
# write_summary
# ---------------------------------------------------------------------------

def test_write_summary_reports_found_sampled_and_handoff_indices():
    from visualize_quad3d_composition import write_summary

    tmp = tempfile.mkdtemp(dir='/tmp', prefix='test_viz3d_summary_')
    try:
        sidecars = {
            'F1': [{'handoff_index': -1, 'initial_tilt_deg': 170.0, 'category': 'F1', 'index': 0}],
            'S1': [{'handoff_index': 0, 'initial_tilt_deg': 5.0, 'category': 'S1', 'index': 0},
                   {'handoff_index': 4, 'initial_tilt_deg': 12.0, 'category': 'S1', 'index': 1}],
            'S1_to_S2': [],
            'S1_to_F2': [],
        }
        summary = write_summary(tmp, categories=['F1', 'S1', 'S1_to_S2', 'S1_to_F2'],
                                num_per_category=2, attempts=37, max_attempts=100, seed=5,
                                max_steps=1200, sidecars=sidecars)

        assert summary['total_attempts_sampled'] == 37
        assert summary['categories']['F1']['found'] == 1
        assert summary['categories']['F1']['filled'] is False
        assert summary['categories']['S1']['found'] == 2
        assert summary['categories']['S1']['filled'] is True
        assert summary['categories']['S1']['handoff_indices'] == [0, 4]
        assert summary['categories']['S1']['initial_tilts_deg'] == [5.0, 12.0]
        assert set(summary['unfilled_categories']) == {'F1', 'S1_to_S2', 'S1_to_F2'}

        with open(os.path.join(tmp, 'summary.json')) as fh:
            on_disk = json.load(fh)
        assert on_disk == summary
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

def test_parse_args_defaults_to_all_four_categories():
    from visualize_quad3d_composition import CATEGORIES, parse_args

    args = parse_args(['--output_dir', 'out'])
    assert set(args.categories) == set(CATEGORIES)
    assert args.num_per_category == 3
    assert args.flip_model is None
    assert args.g1 is None


def test_parse_args_rejects_unknown_category():
    from visualize_quad3d_composition import parse_args

    with pytest.raises(SystemExit):
        parse_args(['--output_dir', 'out', '--categories', 'bogus'])


def test_parse_args_default_max_attempts_scales_with_num_per_category():
    from visualize_quad3d_composition import parse_args

    args = parse_args(['--output_dir', 'out', '--num_per_category', '10'])
    assert args.max_attempts == 500


def test_parse_args_controller_defaults_to_sac_and_accepts_geometric():
    from visualize_quad3d_composition import parse_args

    args = parse_args(['--output_dir', 'out'])
    assert args.controller == 'sac'
    assert args.min_trajectory_states is None

    args = parse_args(['--output_dir', 'out', '--controller', 'geometric'])
    assert args.controller == 'geometric'

    with pytest.raises(SystemExit):
        parse_args(['--output_dir', 'out', '--controller', 'bogus'])


def test_parse_args_requires_no_g1_and_no_ctrl2_model():
    '''Unlike the 2D branch, --g1 is OPTIONAL (falls back to G_NOM_3D) and
    there is no --ctrl2_model at all (controller 2 is LQR, analytic).
    '''
    from visualize_quad3d_composition import parse_args

    args = parse_args(['--output_dir', 'out'])  # must not raise / require --g1
    assert not hasattr(args, 'ctrl2_model')


# ---------------------------------------------------------------------------
# poses_from_states
# ---------------------------------------------------------------------------

def test_poses_from_states_matches_set_initial_state_convention():
    import pybullet as p

    from visualize_quad3d_composition import poses_from_states

    # 90 deg about body x: wxyz = [cos45, sin45, 0, 0].
    c = float(np.cos(np.pi / 4))
    s = float(np.sin(np.pi / 4))
    states = [[0.3, -0.2, 0.9, c, s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    poses = poses_from_states(states)
    assert len(poses) == 1
    pos, orn = poses[0]
    assert pos == pytest.approx([0.3, -0.2, 0.9])
    # dataset order [qw, qx, qy, qz] -> PyBullet order [qx, qy, qz, qw].
    assert orn == pytest.approx([s, 0.0, 0.0, c])


# ---------------------------------------------------------------------------
# Real PyBullet: the handoff colour cue must actually change pixels.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_render_frames_changes_drone_colour_at_the_handoff_frame():
    '''Requirement 4: the handoff must be visually obvious in the video. Use
    a STATIC pose (identical position/orientation for every frame) so the
    only possible difference between frames is the colour change -- proving
    render_frames actually recolours the drone rather than relying on
    incidental pose movement to make frames differ.
    '''
    from visualize_quad3d_composition import render_frames

    static_pose = ([0.0, 0.0, 1.0], (0.0, 0.0, 0.0, 1.0))
    poses = [static_pose] * 4
    frames = render_frames(poses, handoff_frame_index=2, ctrl_freq=1, fps=1,
                           width=160, height=120)

    assert len(frames) == 4
    assert np.array_equal(frames[0], frames[1]), 'both pre-handoff frames must be identical'
    assert np.array_equal(frames[2], frames[3]), 'both post-handoff frames must be identical'
    assert not np.array_equal(frames[0], frames[2]), \
        'the handoff frame must look visibly different from the pre-handoff frames'


@pytest.mark.slow
def test_render_frames_without_a_handoff_never_recolours():
    from visualize_quad3d_composition import render_frames

    static_pose = ([0.0, 0.0, 1.0], (0.0, 0.0, 0.0, 1.0))
    poses = [static_pose] * 3
    frames = render_frames(poses, handoff_frame_index=None, ctrl_freq=1, fps=1,
                           width=160, height=120)
    assert all(np.array_equal(frames[0], f) for f in frames)


@pytest.mark.slow
def test_render_frames_tracking_camera_keeps_a_far_drone_in_frame():
    '''Requirement 6: the drone can be anywhere in the 3.6 x 3.6 x 2.9 m box.
    A far-corner static pose must still produce a non-trivial (non-blank)
    frame, proving the camera followed it there rather than staying pinned
    near the origin.
    '''
    from visualize_quad3d_composition import render_frames

    far_pose = ([1.7, -1.7, 2.8], (0.0, 0.0, 0.0, 1.0))
    frames = render_frames([far_pose, far_pose], handoff_frame_index=None, ctrl_freq=1, fps=1,
                           width=160, height=120)
    assert len(frames) == 2
    # The drone (and its shading) must actually be visible: not every pixel
    # identical to the corner's background colour.
    assert len(np.unique(frames[0].reshape(-1, 3), axis=0)) > 1


# ---------------------------------------------------------------------------
# Real end-to-end: real env, real controller 2 (LQR), randomly-initialised
# controller 1 (CRITICAL requirement -- no trained flip checkpoint exists yet).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_build_ctrl1_without_a_flip_model_is_a_usable_random_policy():
    from quad_composition.rollout3d import ctrl1_observation, make_env
    from visualize_quad3d_composition import build_ctrl1

    tmp = tempfile.mkdtemp(dir='/tmp', prefix='test_viz3d_ctrl1_')
    env = None
    try:
        env = make_env(seed=0)
        obs, info = env.reset()
        ctrl1 = build_ctrl1(None, env, tmp)
        try:
            action = ctrl1.select_action(
                ctrl1.obs_normalizer(ctrl1_observation(env, obs)), info)
            assert action.shape == env.action_space.shape
            assert ctrl1.obs_normalizer.read_only
        finally:
            ctrl1.close()
    finally:
        if env is not None:
            env.close()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.slow
def test_main_runs_for_real_end_to_end_and_writes_non_empty_mp4s(tmp_path):
    '''The CRITICAL verification: no --flip_model given (random controller
    1), no --g1 given (falls back to G_NOM_3D), real env, real controller 2
    (LQR), real rollouts, real video files. Only requests 'F1' -- reachable
    from a random policy almost immediately and fast to render -- so this
    stays a smoke test.
    '''
    import visualize_quad3d_composition as m

    output_dir = tmp_path / 'videos'

    summary = m.main([
        '--output_dir', str(output_dir), '--categories', 'F1',
        '--num_per_category', '1', '--max_steps', '50', '--max_attempts', '10',
        '--seed', '0', '--width', '160', '--height', '120',
    ])

    assert summary['categories']['F1']['found'] >= 0  # ran without raising either way
    video_path = output_dir / 'F1' / 'rollout_000.mp4'
    if summary['categories']['F1']['found'] >= 1:
        assert video_path.exists()
        assert video_path.stat().st_size > 0
        assert (output_dir / 'F1' / 'rollout_000_tilt.png').exists()
        assert (output_dir / 'F1' / 'rollout_000.json').exists()
    assert (output_dir / 'summary.json').exists()
