#!/usr/bin/env python3
'''Visualize the two-stage quadrotor-3D controller composition.

Controller 1 (a "flip" recovery policy) runs until the state first enters the
attitude-only handoff region G1, then control latches PERMANENTLY to
controller 2 (LQR) which flies to the goal. This script samples initial
states over the closed 3D state space (full SO(3) attitude coverage), rolls
each through `quad_composition.rollout3d.rollout_composite`, classifies the
outcome into one of four categories, and records an MP4 + a tilt-vs-time /
3D-position plot + a sidecar JSON for each recorded rollout, until
`--num_per_category` examples of every requested category are found or a
sampling budget is exhausted.

CATEGORIES
    F1         controller 1 never reached G1 (flip failed).
    S1         controller 1 reached G1 (flip succeeded). The clip is
               TRUNCATED to controller 1's own portion of the rollout, ending
               at (and including) the handoff state -- so this shows the flip
               succeeding, not what controller 2 does afterwards.
    S1_to_S2   handoff fired, then controller 2 (LQR) reached the goal. Full
               trajectory; the handoff is marked (drone colour change + a
               marker bead at the handoff position, plus a vertical line on
               the tilt-vs-time plot).
    S1_to_F2   handoff fired, then controller 2 failed. Same as S1_to_S2 but
               the flip's continuation did not reach the goal.

S1 and F1 are properties of the controller-1 phase; S1_to_S2 and S1_to_F2
subdivide S1, so a single sampled rollout is ELIGIBLE for both the 'S1' slot
and (depending on ctrl2's outcome) the 'S1_to_S2' or 'S1_to_F2' slot -- see
`classify`. (F1, S2) is impossible by construction: controller 2 never runs
unless a handoff fired.

TWO DISJOINT VIDEO SETS. Eligible for both is not the same as recorded in
both. The deliverable is "a video visualization S1 and F1, and then a separate
set of videos showcasing S1->S2 and S1->F2", and letting one rollout fill a
slot in each set makes the two sets show the SAME flight twice -- which is
what the first cut did (its three S1 clips were the opening seconds of its
three S1_to_F2 clips). `assign_rollouts` gives each rollout to at most one set
(`DISJOINT_SETS`), and `disjointness_report` re-checks the result.

CLIPS MUST BE LONG ENOUGH TO WATCH. Selection filters on the length of the
clip a category actually RENDERS (`clip_for_category`, `MIN_CLIP_STATES`) --
not on the rollout's total length, which for 'S1' overstates it by a factor of
30 -- and prefers the highest initial tilt among what survives, a bigger flip
being both longer and the better picture. Categories that cannot reach the
`MIN_CLIP_SECONDS` floor at real time (F1 ends when the drone leaves the box,
often within a handful of ticks) are captured at the full simulation rate and
played back slower; `playback_plan` decides that and summary.json records it.

CONTROLLER 2 IS ANALYTIC (LQR): unlike the 2D branch there is no
`--ctrl2_model` -- `quad_composition.rollout3d.make_env_and_ctrl2` builds LQR
directly, exactly as `generate_quadrotor_3d_trajectories.py` does.

NO TRAINED CONTROLLER-1 CHECKPOINT REQUIRED: pass no `--flip_model` and
controller 1 is built as a randomly-initialised SAC policy -- the exact same
construction path as `quad_composition.rollout3d.load_ctrl1`, minus loading a
checkpoint (see `build_ctrl1`). This lets the whole pipeline (real env, real
rollouts, real classification, real videos) be exercised before training
finishes.

Ports `visualize_quad2d_composition.py` (same CLI shape, category logic,
summary.json, and handoff marking) to the 3D system. Reuses `save_video` and
lifts the PyBullet DIRECT-mode camera/frame-capture setup from
`visualize_quadrotor_3d_rollout.py`, with one change: the camera TRACKS the
drone (target = current drone position, recomputed every frame) rather than
staying fixed, since the drone can be anywhere in the 3.6 x 3.6 x 2.9 m box
and is often tumbling.
'''

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import pybullet as p
import pybullet_data

from quad_composition.flip_env3d import G_NOM_3D, sample_uniform_state, sampling_bounds_from_env
from quad_composition.g1 import G1Region
from quad_composition.geometric_flip3d import GeometricFlipController3D
from quad_composition.rollout3d import (ENV_CONFIG, GOAL_TOLERANCE, MAX_STEPS, QUAT_SLICE, SAC_CONFIG,
                                        TERMINATION, _Ctrl1ObservationSpaceEnv, load_ctrl1,
                                        make_env_and_ctrl2, quat_wxyz_to_pybullet, rollout_composite,
                                        tilt_from_quat_wxyz)
from safe_control_gym.utils.registration import make
from visualize_quadrotor_3d_rollout import save_video

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

CTRL_FREQ = ENV_CONFIG['ctrl_freq']

CATEGORIES = ['S1', 'F1', 'S1_to_S2', 'S1_to_F2']

# Rendering (matches visualize_quadrotor_3d_rollout.py's defaults).
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
FPS = 30
DRONE_VISUAL_SCALE = 5.0

# Tracking camera (requirement 6): the drone can be anywhere in the closed
# state-space box (3.6 x 3.6 x 2.9 m) and is often tumbling under controller
# 1. A FIXED camera (as in visualize_quadrotor_3d_rollout.py) risks losing it;
# re-targeting the camera at the drone's current position every frame cannot.
CAMERA_DISTANCE = 1.6
CAMERA_YAW = 45.0
CAMERA_PITCH = -30.0

# Handoff visual cue (requirement 4): the drone's own body colour changes at
# the handoff frame, plus a marker bead is left at the handoff position.
PRE_HANDOFF_COLOR = [0.1, 0.1, 0.1, 1.0]
POST_HANDOFF_COLOR = [1.0, 0.55, 0.0, 1.0]
HANDOFF_MARKER_COLOR = [0.9, 0.0, 0.9, 1.0]

# 'S1' clips are truncated right at the handoff frame; hold that last frame
# for a moment so the colour change is actually visible rather than a single
# instantaneous frame.  Counted in OUTPUT frames (see `record_rollout`): the
# hold lasts this long in the finished video regardless of the capture rate.
S1_HOLD_SECONDS = 0.6

# The two video SETS the deliverable asks for: "a video visualization S1 and
# F1, and then a separate set of videos showcasing S1->S2 and S1->F2".  They
# must not share underlying rollouts -- a rollout that fills an 'S1' slot is
# then unavailable to 'S1_to_S2'/'S1_to_F2', and vice versa.  Enforced in
# `assign_rollouts`, not left to luck of sampling, and re-checked afterwards by
# `disjointness_report`.
DISJOINT_SETS = [['S1', 'F1'], ['S1_to_S2', 'S1_to_F2']]

# Minimum length, in trajectory states, of the CLIP each category actually
# renders (NOT of the whole rollout -- 'S1' renders only up to the handoff, so
# a 500-state S1->S2 rollout can still yield a 12-state 'S1' clip; that is
# exactly how the first cut produced 10-16 frame videos).  Measured against the
# selected controller over 600 rollouts, these keep a usable fraction of the
# sampled pool: S1 clip >= 80 states is the top ~9% of handoffs, F1 >= 40 the
# top ~3% of failures, S1->F2 >= 120 the top ~24%, and every S1->S2 already
# runs 324+ states.
MIN_CLIP_STATES = {'S1': 80, 'F1': 40, 'S1_to_S2': 300, 'S1_to_F2': 120}

# Playback floor.  A clip shorter than this at real-time speed is re-timed
# (see `playback_plan`): captured at the full simulation rate and played back
# slower, which is SLOW MOTION of real frames, not padding.  Which categories
# needed it is recorded in summary.json and the README.
MIN_CLIP_SECONDS = 3.0
MIN_PLAYBACK_FPS = 15
TREATED_HOLD_SECONDS = 1.0


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(result):
    '''Map a `RolloutResult` to the category slot(s) it fills.

    F1 iff the flip never reached G1. Otherwise the rollout always fills 'S1'
    (controller 1 succeeded) AND exactly one of 'S1_to_S2'/'S1_to_F2'
    (controller 2's outcome after the handoff) -- never both, and never
    neither, since `rollout_composite` always latches to controller 2 once
    G1 is entered.
    '''
    if not result.flip_success:
        return ['F1']
    return ['S1', 'S1_to_S2' if result.ctrl2_success else 'S1_to_F2']


def clip_for_category(category, result):
    '''The states a category's video actually shows, and the pose index its
    handoff marker belongs on (None when there is no handoff to mark).

    THE SINGLE DEFINITION of "how long is this clip", shared by the selection
    filter (`sample_and_classify`), the re-timing decision (`playback_plan`)
    and the renderer (`record_rollout`).  Keeping one definition is the point:
    selecting on `len(result.trajectory)` while rendering `trajectory[:handoff
    + 1]` is precisely what let 500-state S1->S2 rollouts through as 12-state
    'S1' clips.

        S1                  controller 1's own portion only, ending at (and
                            including) the handoff state -- this shows the flip
                            succeeding, not what controller 2 does next.
        S1_to_S2/S1_to_F2   the full trajectory, handoff marked in place.
        F1                  the full trajectory; there is no handoff.
    '''
    if category == 'S1':
        assert result.handoff_index >= 0, "'S1' requires a fired handoff"
        return result.trajectory[:result.handoff_index + 1], result.handoff_index
    if category in ('S1_to_S2', 'S1_to_F2'):
        assert result.handoff_index >= 0, f"'{category}' requires a fired handoff"
        return result.trajectory, result.handoff_index
    return result.trajectory, None


# ---------------------------------------------------------------------------
# Controller 1: real checkpoint, or a randomly-initialised stand-in
# ---------------------------------------------------------------------------

def build_ctrl1(flip_model_path, env, output_dir):
    '''Build controller 1. If `flip_model_path` is given, defers to
    `rollout3d.load_ctrl1` unchanged. If omitted, builds the SAME network
    (`make('sac', ...)` against `_Ctrl1ObservationSpaceEnv(env)` with
    `SAC_CONFIG`) but never calls `.load()`, so the weights stay at
    `make`'s random initialisation -- a randomly-initialised SAC policy
    that exercises the entire pipeline end-to-end without a trained
    checkpoint.
    '''
    if flip_model_path:
        return load_ctrl1(flip_model_path, env, output_dir)
    ctrl1_env = _Ctrl1ObservationSpaceEnv(env)
    ctrl1 = make('sac', lambda **kw: ctrl1_env, **SAC_CONFIG, output_dir=output_dir)
    ctrl1.obs_normalizer.set_read_only()
    return ctrl1


# ---------------------------------------------------------------------------
# Sampling loop
# ---------------------------------------------------------------------------

def initial_tilt_deg(result):
    '''Initial tilt of a rollout, in degrees, from the state the env actually
    started in (`trajectory[0]`) rather than the requested one -- the same
    number `record_rollout` writes to the sidecar.
    '''
    return float(np.degrees(tilt_from_quat_wxyz(np.asarray(result.trajectory[0])[QUAT_SLICE])))


def assign_rollouts(pool, categories, num_per_category, disjoint_sets=None,
                    prefer_high_tilt=False):
    '''Hand out pooled rollouts to categories, at most one category each.

    DISJOINTNESS.  `disjoint_sets` partitions the categories into video SETS
    (see `DISJOINT_SETS`).  A rollout committed to a category in one set is
    struck off for every category in every OTHER set, so the two sets of videos
    cannot share underlying rollouts.  Within a set the categories are mutually
    exclusive by construction ('S1' vs 'F1', 'S1_to_S2' vs 'S1_to_F2' -- see
    `classify`), so in practice this is "each rollout is used at most once",
    which is the strongest reading of the requirement and the easiest to check
    (`disjointness_report`).  Categories not named in `disjoint_sets` each form
    their own singleton set.

    SCARCEST CATEGORY FIRST.  Categories compete for the same pool ('S1' and
    'S1_to_F2' can be filled by the same rollout), so allocating in pool-size
    order stops an abundant category from consuming the only candidates a rare
    one had.  Ties are broken by the caller's category order, so the assignment
    is deterministic.

    `prefer_high_tilt` takes the highest initial tilt first: a bigger flip is
    both longer and the more interesting picture.  Without it, pool order (i.e.
    sampling order) wins, which is the historical behaviour.
    '''
    group_of = {}
    for index, group in enumerate(disjoint_sets or []):
        for category in group:
            group_of[category] = index
    next_group = len(disjoint_sets or [])
    for category in categories:
        if category not in group_of:
            group_of[category] = next_group
            next_group += 1

    claimed = {}                      # pool index -> group index that took it
    recorded = {c: [] for c in categories}
    scarcity = {c: sum(1 for e in pool if c in e['categories']) for c in categories}
    for category in sorted(categories, key=lambda c: (scarcity[c], categories.index(c))):
        group = group_of[category]
        candidates = [i for i, e in enumerate(pool)
                      if category in e['categories'] and claimed.get(i, group) == group]
        if prefer_high_tilt:
            candidates.sort(key=lambda i: pool[i]['initial_tilt_deg'], reverse=True)
        for i in candidates[:num_per_category]:
            claimed[i] = group
            recorded[category].append((pool[i]['init_state'], pool[i]['result']))
    return recorded


def disjointness_report(recorded, disjoint_sets):
    '''Which rollouts each video set used, and whether the sets overlap.

    Identity is the rollout's own trajectory object, so this checks the actual
    recorded rollouts rather than re-deriving anything from the initial states
    (two rollouts CAN share an initial state -- the same state sampled twice --
    and comparing states would then report a false overlap).
    '''
    per_set = []
    for group in disjoint_sets:
        ids = set()
        for category in group:
            for _, result in recorded.get(category, []):
                ids.add(id(result.trajectory))
        per_set.append(ids)
    overlaps = []
    for i in range(len(per_set)):
        for j in range(i + 1, len(per_set)):
            shared = per_set[i] & per_set[j]
            if shared:
                overlaps.append({'sets': [disjoint_sets[i], disjoint_sets[j]],
                                 'shared_rollouts': len(shared)})
    return {'sets': [{'categories': g, 'num_rollouts': len(ids)}
                     for g, ids in zip(disjoint_sets, per_set)],
            'overlaps': overlaps,
            'disjoint': not overlaps}


def sample_and_classify(env, ctrl1, ctrl2, g1, rng, categories, num_per_category,
                        max_steps, max_attempts,
                        sample_fn=None, rollout_fn=rollout_composite,
                        min_trajectory_states=None, min_clip_states=None,
                        disjoint_sets=None, prefer_high_tilt=False,
                        pool_per_category=None):
    '''Sample initial states and roll them out until every category in
    `categories` has `num_per_category` recorded rollouts, or `max_attempts`
    total rollouts have been sampled (the budget is SHARED across
    categories, not per-category) -- whichever comes first. Never loops
    forever: `max_attempts` is a hard cap.

    `sample_fn` is REQUIRED (no usable module-level default): 3D sampling
    needs the env's own closed-state-space bounds baked in (see
    `flip_env3d.sample_uniform_state`), so callers must supply a closure --
    `main` does via `sampling_bounds_from_env(env)`. Tests supply a fake.

    `min_trajectory_states`, if given, discards a rollout from EVERY category
    it would otherwise fill when `len(result.trajectory) < min_trajectory_states`
    -- it still counts toward `attempts`. This exists because a real (and
    physically legitimate) chunk of F1 failures are initial states already
    at a Euclidean bound with outward velocity, which terminate in 2-3 ticks:
    technically correct but not a watchable video (quality bar, not a bug in
    the rollout itself). Default `None` preserves the old unfiltered
    behaviour exactly, matching every existing caller/test.

    TWO MODES. With `min_clip_states`, `disjoint_sets` and `prefer_high_tilt`
    all left at their defaults this streams: the first rollout to fill a slot
    takes it, one rollout can fill several categories at once, and sampling
    stops the moment every category is full. That is the historical behaviour
    and is preserved bit for bit.

    Given ANY of the three, it instead POOLS: it keeps sampling (up to
    `max_attempts`) until every category has `pool_per_category` acceptable
    candidates -- default `num_per_category`, raise it to give the tilt
    preference something to choose from -- and only then allocates, via
    `assign_rollouts`. Pooling is what makes the other two options mean
    anything: neither "the highest-tilt three" nor "disjoint from the other
    set" can be decided while looking at one rollout at a time.

    `min_clip_states` maps category -> minimum length of THAT category's clip
    (`clip_for_category`), not of the whole rollout; a rollout too short for
    one category can still be recorded for another.

    Returns (recorded, attempts): `recorded` maps category -> ordered list of
    (init_state, result) tuples; `attempts` is the total number of rollouts
    sampled (categories not filled to `num_per_category` are a reportable
    shortfall, not an error -- see `write_summary`).
    '''
    if sample_fn is None:
        raise ValueError('sample_and_classify requires a sample_fn (rng) -> init_state')

    if not (min_clip_states or disjoint_sets or prefer_high_tilt):
        recorded = {c: [] for c in categories}
        attempts = 0
        while attempts < max_attempts and any(len(recorded[c]) < num_per_category
                                              for c in categories):
            attempts += 1
            init_state = list(sample_fn(rng))
            result = rollout_fn(env, ctrl1, ctrl2, g1, init_state, max_steps=max_steps)
            if min_trajectory_states and len(result.trajectory) < min_trajectory_states:
                continue
            for cat in classify(result):
                if cat in recorded and len(recorded[cat]) < num_per_category:
                    recorded[cat].append((init_state, result))
        return recorded, attempts

    min_clip_states = min_clip_states or {}
    target = pool_per_category if pool_per_category is not None else num_per_category
    pool = []
    counts = {c: 0 for c in categories}
    attempts = 0
    while attempts < max_attempts and any(counts[c] < target for c in categories):
        attempts += 1
        init_state = list(sample_fn(rng))
        result = rollout_fn(env, ctrl1, ctrl2, g1, init_state, max_steps=max_steps)
        if min_trajectory_states and len(result.trajectory) < min_trajectory_states:
            continue
        cats = []
        for cat in classify(result):
            if cat not in counts:
                continue
            if len(clip_for_category(cat, result)[0]) < min_clip_states.get(cat, 0):
                continue
            cats.append(cat)
        if not cats:
            continue
        for cat in cats:
            counts[cat] += 1
        pool.append({'init_state': init_state, 'result': result, 'categories': cats,
                     'initial_tilt_deg': initial_tilt_deg(result)})
    return assign_rollouts(pool, categories, num_per_category,
                           disjoint_sets=disjoint_sets,
                           prefer_high_tilt=prefer_high_tilt), attempts


# ---------------------------------------------------------------------------
# Rendering (lifted from visualize_quadrotor_3d_rollout.py, with a tracking
# camera in place of its fixed one -- see CAMERA_DISTANCE/YAW/PITCH above).
# ---------------------------------------------------------------------------

def poses_from_states(states):
    '''Dataset-order rows [x, y, z, qw, qx, qy, qz, ...] -> [(position,
    quaternion), ...], using the exact same position / PyBullet-order-quat
    convention `rollout3d.set_initial_state` places the drone with.
    '''
    poses = []
    for row in states:
        row = np.asarray(row, dtype=float)
        pos = [float(row[0]), float(row[1]), float(row[2])]
        orn = quat_wxyz_to_pybullet(row[QUAT_SLICE])
        poses.append((pos, orn))
    return poses


def _draw_goal_ring(client, center, radius, color=(0.2, 0.75, 0.2), n_points=48, bead_radius=0.005,
                    n_rings=3):
    '''Goal region border as rings of small spheres. Lifted from
    visualize_quadrotor_3d_rollout.py's `_draw_goal_ring`.
    '''
    cx, cy, cz = center
    rgba = list(color) + [1.0]
    bead_vis = p.createVisualShape(p.GEOM_SPHERE, radius=bead_radius, rgbaColor=rgba,
                                   physicsClientId=client)
    for ring_i in range(n_rings):
        phi = np.pi * (ring_i + 1) / (n_rings + 1)
        r = radius * np.sin(phi)
        z = cz + radius * np.cos(phi)
        for j in range(n_points):
            th = 2 * np.pi * j / n_points
            pos = [cx + r * np.cos(th), cy + r * np.sin(th), z]
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=bead_vis, basePosition=pos,
                              physicsClientId=client)


def _draw_handoff_marker(client, position, color=HANDOFF_MARKER_COLOR, radius=0.03):
    '''A single bright bead left at the state where control latched to
    controller 2 -- requirement 4's "obvious in the video" handoff cue,
    alongside the drone's own colour change in `render_frames`.
    '''
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=list(color),
                              physicsClientId=client)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=list(position),
                      physicsClientId=client)


def render_frames(poses, handoff_frame_index, ctrl_freq=CTRL_FREQ, fps=FPS,
                  width=RENDER_WIDTH, height=RENDER_HEIGHT, urdf_path=None):
    '''Replay `poses` with a scaled drone and capture TRACKING-camera frames.

    Lifted from visualize_quadrotor_3d_rollout.py's Phase 2 (DIRECT-mode
    client, scaled URDF, ER_TINY_RENDERER), except the camera's target is
    recomputed every frame to the drone's CURRENT position (requirement 6),
    so the drone stays framed however far it drifts within the 3.6 x 3.6 x
    2.9 m box or however it tumbles.

    If `handoff_frame_index` is not None, the drone's body colour switches
    from PRE_HANDOFF_COLOR to POST_HANDOFF_COLOR starting at that pose index,
    and a marker bead is drawn at the handoff position -- requirement 4's
    visual cue. `handoff_frame_index` indexes into `poses`, not into
    ctrl_freq-rate simulation steps.
    '''
    frame_skip = max(1, ctrl_freq // fps)
    client = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.resetSimulation(physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)

        if urdf_path is None:
            probe_env = make('quadrotor', **ENV_CONFIG)
            urdf_path = probe_env.URDF_PATH
            probe_env.close()

        drone_id = p.loadURDF(urdf_path, poses[0][0], poses[0][1],
                              globalScaling=DRONE_VISUAL_SCALE, physicsClientId=client)
        p.changeVisualShape(drone_id, -1, rgbaColor=PRE_HANDOFF_COLOR, physicsClientId=client)

        _draw_goal_ring(client, [0, 0, 1], GOAL_TOLERANCE)
        if handoff_frame_index is not None and 0 <= handoff_frame_index < len(poses):
            _draw_handoff_marker(client, poses[handoff_frame_index][0])

        cam_pro = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=float(width) / height, nearVal=0.01, farVal=1000.0)

        frames = []
        colored = False
        for i, (pos_i, orn_i) in enumerate(poses):
            if i != 0 and (i - 1) % frame_skip != 0:
                continue
            if handoff_frame_index is not None and not colored and i >= handoff_frame_index:
                p.changeVisualShape(drone_id, -1, rgbaColor=POST_HANDOFF_COLOR, physicsClientId=client)
                colored = True
            p.resetBasePositionAndOrientation(drone_id, pos_i, orn_i, physicsClientId=client)
            cam_view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=pos_i, distance=CAMERA_DISTANCE, yaw=CAMERA_YAW,
                pitch=CAMERA_PITCH, roll=0, upAxisIndex=2, physicsClientId=client)
            (w, h, rgb, _, _) = p.getCameraImage(
                width=width, height=height, shadow=1, renderer=p.ER_TINY_RENDERER,
                viewMatrix=cam_view, projectionMatrix=cam_pro, physicsClientId=client)
            frames.append(np.reshape(rgb, (h, w, 4))[:, :, :3])
        return frames
    finally:
        p.disconnect(client)


# ---------------------------------------------------------------------------
# Trajectory plot: tilt-vs-time (the point of the exercise) + 3D position
# trace, both with the handoff marked (requirement 3, requirement 4).
# ---------------------------------------------------------------------------

def plot_tilt_and_trajectory(states, path, success, handoff_index=None):
    '''Two-panel plot: tilt (deg) vs step index, and the 3D x-y-z position
    trace, both sharing a colour-graded time axis and a distinct marker at
    `handoff_index` if given.
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    states = np.asarray(states, dtype=float)
    n = len(states)
    tilts_deg = np.degrees([tilt_from_quat_wxyz(row[QUAT_SLICE]) for row in states])
    t = np.arange(n)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    ax1.plot(t, tilts_deg, color='tab:blue', linewidth=1.5)
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.4)
    ax1.axhline(180, color='gray', linestyle=':', alpha=0.4)
    has_handoff = handoff_index is not None and 0 <= handoff_index < n
    if has_handoff:
        ax1.axvline(handoff_index, color='darkorange', linestyle='--', linewidth=2,
                    label='Handoff (ctrl1 -> ctrl2)')
        ax1.plot(handoff_index, tilts_deg[handoff_index], marker='*', color='darkorange',
                 markersize=18, markeredgecolor='black', markeredgewidth=0.5, zorder=6)
    ax1.set_xlabel('step', fontsize=11)
    ax1.set_ylabel('tilt (deg)', fontsize=11)
    ax1.set_ylim(-5, 185)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Tilt vs time', fontsize=12)
    if has_handoff:
        ax1.legend(fontsize=9, loc='best')

    x_vals, y_vals, z_vals = states[:, 0], states[:, 1], states[:, 2]
    for i in range(n - 1):
        tt = i / max(n - 1, 1)
        color = (0.2 + 0.6 * tt, 0.3 * (1 - tt), 0.8 * (1 - tt))
        ax2.plot(x_vals[i:i + 2], y_vals[i:i + 2], z_vals[i:i + 2], color=color, linewidth=1.5)
    ax2.scatter(*states[0, :3], color='blue', s=80, label='Start', zorder=5)
    ax2.scatter(*states[-1, :3], color='red', s=80, marker='s', label='End', zorder=5)
    if handoff_index is not None and 0 <= handoff_index < n:
        ax2.scatter(*states[handoff_index, :3], color='darkorange', s=160, marker='*',
                    label='Handoff', zorder=6, edgecolor='black', linewidth=0.5)

    x_lo, x_hi = TERMINATION[0]
    y_lo, y_hi = TERMINATION[2]
    z_lo, z_hi = TERMINATION[4]
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(y_lo, y_hi)
    ax2.set_zlim(z_lo, z_hi)
    ax2.set_xlabel('x (m)', fontsize=10)
    ax2.set_ylabel('y (m)', fontsize=10)
    ax2.set_zlabel('z (m)', fontsize=10)
    ax2.set_title('Position trace', fontsize=12)
    ax2.legend(fontsize=8, loc='best')

    result_str = 'SUCCESS' if success else 'FAIL'
    fig.suptitle(f'3D Quadrotor Composition ({result_str}, {n - 1} steps)', fontsize=13)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Recording one rollout to disk
# ---------------------------------------------------------------------------

def _category_success(category, result):
    '''Whether THIS category's clip should be captioned/coloured as a
    success in its plot -- not always the same as `result.ctrl2_success`,
    since 'S1' is about controller 1's own (successful, by construction)
    portion of the rollout.
    '''
    if category == 'F1':
        return False
    if category == 'S1':
        return True
    return category == 'S1_to_S2'


def playback_plan(category, results, ctrl_freq=CTRL_FREQ, fps=FPS,
                  min_clip_seconds=MIN_CLIP_SECONDS, min_playback_fps=MIN_PLAYBACK_FPS,
                  hold_seconds=S1_HOLD_SECONDS, treated_hold_seconds=TREATED_HOLD_SECONDS):
    '''How this category's clips should be timed, given the rollouts selected
    for it. Returns a dict consumed by `record_rollout` and recorded verbatim
    in summary.json, so any re-timing is stated rather than silent.

    Real time is fixed by physics: a clip of `n` states lasts `(n - 1) /
    ctrl_freq` seconds however it is rendered, because the capture stride
    (`ctrl_freq // fps`) and the playback rate cancel. Some categories cannot
    produce a long clip AT ALL -- F1 failures are usually a Euclidean-bound
    violation a handful of ticks after reset, and controller 1 reaches G1 in
    well under a second -- so for those the only honest options are to show
    fewer real seconds or to show them slower.

    This picks SLOW MOTION: capture at the full simulation rate (stride 1, so
    every simulated state becomes a frame -- no interpolation, no duplication)
    and play back at a lower rate. The slowdown is `ctrl_freq /
    playback_fps`. A category whose SHORTEST selected clip already runs
    `min_clip_seconds` at real time is left alone.

    The final frame is additionally held (longer for a re-timed category),
    which IS padding -- it is reported as `hold_seconds` for exactly that
    reason.
    '''
    lengths = [len(clip_for_category(category, result)[0]) for _, result in results]
    shortest = min(lengths) if lengths else 0
    real_seconds = max(shortest - 1, 0) / float(ctrl_freq)
    if not lengths or real_seconds >= min_clip_seconds:
        return {'category': category, 'treated': False, 'capture_fps': fps, 'playback_fps': fps,
                'hold_seconds': hold_seconds, 'slowdown': 1.0,
                'shortest_clip_states': shortest,
                'shortest_clip_real_seconds': real_seconds, 'reason': None}
    # Capturing at ctrl_freq makes the frame count equal the state count, so
    # the playback rate that hits the target duration is frames / target.
    playback_fps = int(max(min_playback_fps, min(fps, shortest // max(min_clip_seconds, 1e-9))))
    return {'category': category, 'treated': True, 'capture_fps': ctrl_freq,
            'playback_fps': playback_fps, 'hold_seconds': treated_hold_seconds,
            'slowdown': ctrl_freq / float(playback_fps),
            'shortest_clip_states': shortest,
            'shortest_clip_real_seconds': real_seconds,
            'reason': (f'shortest selected clip is {shortest} states '
                       f'({real_seconds:.2f} s real time), under the {min_clip_seconds:.1f} s '
                       f'floor; captured at {ctrl_freq} Hz and played at {playback_fps} fps')}


def record_rollout(category, idx, init_state, result, output_dir, fps=FPS,
                   ctrl_freq=CTRL_FREQ, width=RENDER_WIDTH, height=RENDER_HEIGHT, urdf_path=None,
                   playback_fps=None, hold_seconds=S1_HOLD_SECONDS):
    '''Write `<output_dir>/<category>/rollout_<idx>.mp4` + `..._tilt.png` +
    `...json` for one recorded rollout. Returns the sidecar dict that was
    written (also used to build summary.json).

    `fps` is the CAPTURE rate: `render_frames` samples one frame every
    `ctrl_freq // fps` simulated states. `playback_fps` (default: the same) is
    the rate written into the MP4. Splitting the two is what makes slow motion
    possible -- with a single rate the two cancel and every clip plays at real
    time no matter what is passed (see `playback_plan`).
    '''
    cat_dir = os.path.join(output_dir, category)
    os.makedirs(cat_dir, exist_ok=True)
    stem = f'rollout_{idx:03d}'

    trajectory = result.trajectory
    handoff_index = result.handoff_index
    playback_fps = fps if playback_fps is None else playback_fps

    states, marker_index = clip_for_category(category, result)
    if category == 'S1':
        # The marker belongs on the LAST pose of the truncated clip, which is
        # the handoff state itself.
        marker_index = len(states) - 1

    poses = poses_from_states(states)
    if category in ('S1', 'F1'):
        # S1: hold the final (handoff) pose so the colour change that marks
        # it is actually visible, not a single instantaneous frame. F1: many
        # failures are a Euclidean-bound violation reached in a handful of
        # ticks (quality bar) -- holding the final (out-of-bounds) pose gives
        # the viewer time to register where and how it failed instead of a
        # blink-and-you-miss-it clip.
        #
        # Counted in OUTPUT frames, then scaled by the capture stride: appending
        # `hold_seconds * playback_fps` POSES (the old behaviour) survives only
        # one in `frame_skip` of them, so a nominal 0.6 s hold rendered at the
        # default 30 fps against a 100 Hz sim actually lasted 0.2 s.
        frame_skip = max(1, ctrl_freq // fps)
        hold_frames = max(1, int(round(hold_seconds * playback_fps)))
        poses = poses + [poses[-1]] * (hold_frames * frame_skip)

    frames = render_frames(poses, marker_index, ctrl_freq=ctrl_freq, fps=fps,
                           width=width, height=height, urdf_path=urdf_path)
    video_path = os.path.join(cat_dir, f'{stem}.mp4')
    save_video(frames, video_path, playback_fps)

    plot_path = os.path.join(cat_dir, f'{stem}_tilt.png')
    plot_tilt_and_trajectory(states, plot_path, _category_success(category, result),
                             handoff_index=marker_index)

    initial_tilt_deg = float(np.degrees(tilt_from_quat_wxyz(np.asarray(trajectory[0])[QUAT_SLICE])))

    sidecar = {
        'category': category,
        'index': idx,
        'init_state': [float(v) for v in init_state],
        'handoff_index': int(handoff_index),
        'initial_tilt_deg': initial_tilt_deg,
        'flip_success': bool(result.flip_success),
        'ctrl2_success': bool(result.ctrl2_success),
        'num_recorded_states': len(states),
        'num_full_trajectory_states': len(trajectory),
        'num_video_frames': len(frames),
        'capture_fps': int(fps),
        'playback_fps': int(playback_fps),
        'hold_seconds': float(hold_seconds) if category in ('S1', 'F1') else 0.0,
        'video_seconds': len(frames) / float(playback_fps),
        'real_time_seconds': max(len(states) - 1, 0) / float(ctrl_freq),
        'video': os.path.basename(video_path),
        'plot': os.path.basename(plot_path),
    }
    with open(os.path.join(cat_dir, f'{stem}.json'), 'w') as fh:
        json.dump(sidecar, fh, indent=2)
    return sidecar


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(output_dir, categories, num_per_category, attempts, max_attempts,
                  seed, max_steps, sidecars, controller=None, g1_source=None,
                  playback_plans=None, disjointness=None, min_clip_states=None):
    '''Per requirement 5: per category, how many were found, how many were
    sampled (shared attempt budget across all categories), and each recorded
    rollout's handoff index and initial tilt.

    `controller`/`g1_source` are optional provenance strings (which controller
    1 and which G1 produced this batch) recorded for traceability; omitted by
    callers (tests) that do not care. `playback_plans`, `disjointness` and
    `min_clip_states` likewise record the two quality guarantees -- which
    categories had to be re-timed to be watchable, and that the two video sets
    share no rollouts -- so both are auditable from the output directory alone.
    '''
    categories_summary = {}
    unfilled = []
    for cat in categories:
        rollouts = sidecars.get(cat, [])
        found = len(rollouts)
        filled = found >= num_per_category
        if not filled:
            unfilled.append(cat)
        categories_summary[cat] = {
            'requested': num_per_category,
            'found': found,
            'filled': filled,
            'handoff_indices': [r['handoff_index'] for r in rollouts],
            'initial_tilts_deg': [r['initial_tilt_deg'] for r in rollouts],
            'rollouts': rollouts,
        }

    summary = {
        'seed': seed,
        'max_steps': max_steps,
        'num_per_category': num_per_category,
        'categories_requested': categories,
        'total_attempts_sampled': attempts,
        'max_attempts': max_attempts,
        'sampling_budget_exhausted': attempts >= max_attempts,
        'categories': categories_summary,
        'unfilled_categories': unfilled,
        'controller': controller,
        'g1_source': g1_source,
        'min_clip_states': min_clip_states,
        'playback_plans': playback_plans,
        'playback_treated_categories': sorted(c for c, plan in (playback_plans or {}).items()
                                              if plan.get('treated')),
        'disjointness': disjointness,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--controller', choices=['sac', 'geometric'], default='sac',
                        help='Controller 1: a SAC policy (default -- pass --flip_model for a '
                             'trained checkpoint, otherwise a randomly-initialised network, see '
                             'build_ctrl1) or the hand-coded geometric controller '
                             '(quad_composition.geometric_flip3d), which needs no --flip_model.')
    parser.add_argument('--flip_model', default=None,
                        help='SAC checkpoint for controller 1 (the flip/recovery policy). '
                             'If omitted, controller 1 is a RANDOMLY-INITIALISED SAC policy '
                             '(see build_ctrl1) -- use this before a trained checkpoint exists. '
                             'Ignored (and unnecessary) when --controller geometric.')
    parser.add_argument('--g1', default=None,
                        help='Path to a g1.json (quad_composition.g1.G1Region). '
                             'If omitted, uses flip_env3d.G_NOM_3D (the training-time nominal '
                             'attitude target) as the handoff region.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_per_category', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    parser.add_argument('--categories', default=','.join(CATEGORIES),
                        help=f'Comma list from {CATEGORIES}.')
    parser.add_argument('--max_attempts', type=int, default=None,
                        help='Sampling budget cap, shared across all categories. '
                             'Default: max(200, 50 * num_per_category).')
    parser.add_argument('--fps', type=int, default=FPS)
    parser.add_argument('--width', type=int, default=RENDER_WIDTH)
    parser.add_argument('--height', type=int, default=RENDER_HEIGHT)
    parser.add_argument('--min_trajectory_states', type=int, default=None,
                        help='Discard a rollout (from every category) whose full trajectory has '
                             'fewer than this many recorded states -- see sample_and_classify. '
                             'Guards against near-instant (2-3 tick) Euclidean-bound-violation '
                             'F1 clips that are technically real but not watchable. Default: no '
                             'filtering, matching the old behaviour.')
    parser.add_argument('--min_clip_states', default=None,
                        help='Per-category minimum CLIP length, as "S1=80,F1=40,..." -- the '
                             'length of what that category actually renders (clip_for_category), '
                             'not of the whole rollout. Default: MIN_CLIP_STATES. Pass "" to '
                             'disable and take whatever is sampled.')
    parser.add_argument('--pool_per_category', type=int, default=None,
                        help='Sample until every category has this many acceptable candidates, '
                             'then pick the highest-tilt ones. Must exceed --num_per_category to '
                             'give the tilt preference anything to choose from. '
                             'Default: 12 * num_per_category.')
    parser.add_argument('--no_prefer_high_tilt', action='store_true',
                        help='Take candidates in sampling order instead of preferring the '
                             'highest initial tilt (a bigger flip is both longer and the more '
                             'interesting picture).')
    parser.add_argument('--no_disjoint_sets', action='store_true',
                        help='Allow the {S1, F1} and {S1_to_S2, S1_to_F2} video sets to reuse '
                             'the same rollouts (the old behaviour, which produced identical '
                             'S1 and S1_to_F2 clips). Off by default.')
    parser.add_argument('--min_clip_seconds', type=float, default=MIN_CLIP_SECONDS,
                        help='Categories whose shortest selected clip runs less than this at '
                             'real time are captured at the simulation rate and played back '
                             'slower (see playback_plan). 0 disables re-timing.')
    args = parser.parse_args(argv)

    args.categories = [c.strip() for c in args.categories.split(',') if c.strip()]
    unknown = sorted(set(args.categories) - set(CATEGORIES))
    if unknown:
        parser.error(f'unknown --categories {unknown}; choose from {CATEGORIES}')
    if not args.categories:
        parser.error('--categories must name at least one category')
    if args.max_attempts is None:
        args.max_attempts = max(200, 50 * args.num_per_category)
    if args.min_clip_states is None:
        args.min_clip_states = {c: MIN_CLIP_STATES[c] for c in args.categories
                                if c in MIN_CLIP_STATES}
    else:
        try:
            args.min_clip_states = {k.strip(): int(v)
                                    for k, v in (item.split('=') for item
                                                 in args.min_clip_states.split(',') if item.strip())}
        except ValueError:
            parser.error('--min_clip_states must look like "S1=80,F1=40"')
        unknown = sorted(set(args.min_clip_states) - set(CATEGORIES))
        if unknown:
            parser.error(f'unknown --min_clip_states categories {unknown}')
    if args.pool_per_category is None:
        args.pool_per_category = 12 * args.num_per_category
    return args


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    # NFS temp dirs intermittently hang; use /tmp explicitly and best-effort
    # cleanup, matching visualize_quad2d_composition.py.
    tmp = tempfile.mkdtemp(dir='/tmp', prefix='quad3d_composition_viz_')
    env = ctrl1 = ctrl2 = None
    try:
        env, ctrl2 = make_env_and_ctrl2(tmp)
        ctrl1 = (GeometricFlipController3D(env) if args.controller == 'geometric'
                 else build_ctrl1(args.flip_model, env, tmp))
        if args.g1:
            with open(args.g1) as fh:
                g1 = G1Region.from_dict(json.load(fh)['g1'])
            g1_source = args.g1
        else:
            g1 = G_NOM_3D
            g1_source = 'G_NOM_3D'

        bounds = sampling_bounds_from_env(env)

        def sample_fn(rng):
            return sample_uniform_state(rng, bounds)

        os.makedirs(args.output_dir, exist_ok=True)

        # Only the sets whose categories were actually requested, so
        # `--categories S1` does not carry an empty S1_to_S2/S1_to_F2 set into
        # the disjointness check.
        disjoint_sets = ([[c for c in group if c in args.categories] for group in DISJOINT_SETS]
                         if not args.no_disjoint_sets else None)
        if disjoint_sets:
            disjoint_sets = [g for g in disjoint_sets if g]

        recorded, attempts = sample_and_classify(
            env, ctrl1, ctrl2, g1, rng, args.categories, args.num_per_category,
            args.max_steps, args.max_attempts, sample_fn=sample_fn, rollout_fn=rollout_composite,
            min_trajectory_states=args.min_trajectory_states,
            min_clip_states=args.min_clip_states, disjoint_sets=disjoint_sets,
            prefer_high_tilt=not args.no_prefer_high_tilt,
            pool_per_category=args.pool_per_category)

        plans = {cat: playback_plan(cat, recorded[cat], ctrl_freq=CTRL_FREQ, fps=args.fps,
                                    min_clip_seconds=args.min_clip_seconds)
                 for cat in args.categories}

        urdf_path = env.URDF_PATH
        sidecars = {cat: [] for cat in args.categories}
        for cat in args.categories:
            plan = plans[cat]
            for idx, (init_state, result) in enumerate(recorded[cat]):
                sidecars[cat].append(record_rollout(
                    cat, idx, init_state, result, args.output_dir,
                    fps=plan['capture_fps'], ctrl_freq=CTRL_FREQ, width=args.width,
                    height=args.height, urdf_path=urdf_path,
                    playback_fps=plan['playback_fps'], hold_seconds=plan['hold_seconds']))

        disjointness = (disjointness_report(recorded, disjoint_sets) if disjoint_sets
                        else {'disjoint': None, 'sets': [], 'overlaps': []})
        summary = write_summary(args.output_dir, args.categories, args.num_per_category,
                                attempts, args.max_attempts, args.seed, args.max_steps, sidecars,
                                controller=args.controller, g1_source=g1_source,
                                playback_plans=plans, disjointness=disjointness,
                                min_clip_states=args.min_clip_states)
    finally:
        for obj in (ctrl1, ctrl2, env):
            if obj is not None:
                obj.close()
        shutil.rmtree(tmp, ignore_errors=True)

    print(f'Sampled {attempts}/{args.max_attempts} rollout(s).')
    for cat in args.categories:
        c = summary['categories'][cat]
        plan = summary['playback_plans'][cat]
        status = 'OK' if c['filled'] else 'SHORTFALL'
        timing = (f'{plan["slowdown"]:.1f}x slow @ {plan["playback_fps"]}fps'
                  if plan['treated'] else 'real time')
        seconds = [f'{r["video_seconds"]:.1f}s' for r in c['rollouts']]
        print(f'  {cat}: {c["found"]}/{c["requested"]} [{status}]  {timing}  {" ".join(seconds)}')
    if summary['playback_treated_categories']:
        print(f'Re-timed (slow motion): {summary["playback_treated_categories"]}')
    if summary['unfilled_categories']:
        print(f'Could not fill: {summary["unfilled_categories"]}')
    print(f'Video sets disjoint: {summary["disjointness"]["disjoint"]}')
    print(f'Output: {args.output_dir}')
    return summary


if __name__ == '__main__':
    main()
