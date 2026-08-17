#!/usr/bin/env python3
'''Visualize the two-stage quadrotor-2D controller composition.

Controller 1 (a "flip" recovery policy) runs until the state first enters the
attitude-only handoff region G1, then control latches PERMANENTLY to
controller 2 (an RL checkpoint) which flies to the goal. This script samples
initial states, rolls each through `quad_composition.rollout2d.rollout_composite`,
classifies the outcome into one of four categories, and records an MP4 + an
x-z trajectory plot + a sidecar JSON for each recorded rollout, until
`--num_per_category` examples of every requested category are found or a
sampling budget is exhausted.

CATEGORIES
    F1         controller 1 never reached G1 (flip failed).
    S1         controller 1 reached G1 (flip succeeded). The clip is
               TRUNCATED to controller 1's own portion of the rollout, ending
               at (and including) the handoff state -- mirroring
               generate_quadrotor_2d_composition.py's --mode flip truncation
               -- so this shows the flip succeeding, not what controller 2
               does afterwards.
    S1_to_S2   handoff fired, then controller 2 reached the goal. Full
               trajectory; the handoff is marked (drone colour change + a
               marker bead at the handoff position).
    S1_to_F2   handoff fired, then controller 2 failed. Same as S1_to_S2 but
               the flip's continuation did not reach the goal.

S1 and F1 are properties of the controller-1 phase; S1_to_S2 and S1_to_F2
subdivide S1. A single sampled rollout can therefore fill BOTH the 'S1' slot
and (depending on ctrl2's outcome) the 'S1_to_S2' or 'S1_to_F2' slot at once
-- see `classify`. (F1, S2) is impossible by construction: controller 2 never
runs unless a handoff fired.

NO TRAINED CONTROLLER-1 CHECKPOINT REQUIRED: pass no `--flip_model` and
controller 1 is built as a randomly-initialised SAC policy -- the exact same
construction path as `quad_composition.rollout2d.load_ctrl1`, minus loading a
checkpoint (see `build_ctrl1`). This lets the whole pipeline (real env, real
rollouts, real classification, real videos) be exercised before training
finishes. An untrained policy will mostly produce F1; some sampled initial
states start already inside G1, which still yields immediate-handoff S1 (and
possibly S1_to_S2/S1_to_F2) rollouts.

Reuses `save_video` from visualize_quadrotor_2d_rollout.py verbatim, and lifts
its PyBullet DIRECT-mode camera/replay setup for `render_frames` below.
'''

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import pybullet as p
import pybullet_data

from quad_composition.flip_env2d import sample_uniform_state
from quad_composition.g1 import G1Region
from quad_composition.rollout2d import (ENV_CONFIG, GOAL_TOLERANCE, MAX_STEPS, SAC_CONFIG, TERMINATION,
                                        _Ctrl1ObservationSpaceEnv, load_ctrl1, make_env_and_ctrl2,
                                        rollout_composite)
from safe_control_gym.utils.registration import make
from visualize_quadrotor_2d_rollout import save_video

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(REPO_ROOT, 'safe_control_gym', 'envs', 'gym_pybullet_drones',
                         'assets', 'quad2d_visual.urdf')

CTRL2_MODEL_DEFAULT = ('examples/rl/models/safe_explorer_ppo/'
                       'safe_explorer_ppo_model_quadrotor_2D_stab.pt')

CTRL_FREQ = ENV_CONFIG['ctrl_freq']

CATEGORIES = ['S1', 'F1', 'S1_to_S2', 'S1_to_F2']

# Rendering (matches visualize_quadrotor_2d_rollout.py's defaults).
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
FPS = 30
DRONE_VISUAL_SCALE = 5.0

# Handoff visual cue (requirement 4): the drone's own body colour changes at
# the handoff frame, plus a marker bead is left at the handoff position.
PRE_HANDOFF_COLOR = [0.1, 0.1, 0.1, 1.0]
POST_HANDOFF_COLOR = [1.0, 0.55, 0.0, 1.0]
HANDOFF_MARKER_COLOR = [0.9, 0.0, 0.9, 1.0]

# 'S1' clips are truncated right at the handoff frame; hold that last frame
# for a moment so the colour change is actually visible rather than a single
# instantaneous frame.
S1_HOLD_SECONDS = 0.6


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


# ---------------------------------------------------------------------------
# Controller 1: real checkpoint, or a randomly-initialised stand-in
# ---------------------------------------------------------------------------

def build_ctrl1(flip_model_path, env, output_dir):
    '''Build controller 1. If `flip_model_path` is given, defers to
    `rollout2d.load_ctrl1` unchanged. If omitted, builds the SAME network
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

def sample_and_classify(env, ctrl1, ctrl2, g1, rng, categories, num_per_category,
                        max_steps, max_attempts,
                        sample_fn=sample_uniform_state, rollout_fn=rollout_composite):
    '''Sample initial states and roll them out until every category in
    `categories` has `num_per_category` recorded rollouts, or `max_attempts`
    total rollouts have been sampled (the budget is SHARED across
    categories, not per-category) -- whichever comes first. Never loops
    forever: `max_attempts` is a hard cap.

    Returns (recorded, attempts): `recorded` maps category -> ordered list of
    (init_state, result) tuples; `attempts` is the total number of rollouts
    sampled (categories not filled to `num_per_category` are a reportable
    shortfall, not an error -- see `write_summary`).
    '''
    recorded = {c: [] for c in categories}
    attempts = 0
    while attempts < max_attempts and any(len(recorded[c]) < num_per_category for c in categories):
        attempts += 1
        init_state = list(sample_fn(rng))
        result = rollout_fn(env, ctrl1, ctrl2, g1, init_state, max_steps=max_steps)
        for cat in classify(result):
            if cat in recorded and len(recorded[cat]) < num_per_category:
                recorded[cat].append((init_state, result))
    return recorded, attempts


# ---------------------------------------------------------------------------
# Rendering (lifted from visualize_quadrotor_2d_rollout.py's Phase 2)
# ---------------------------------------------------------------------------

def poses_from_states(states):
    '''dataset-order rows [x, z, theta, ...] -> [(position, quaternion), ...],
    using the exact same [x, 0, z] / euler-[0, theta, 0] convention
    `rollout2d.set_initial_state` places the drone with.
    '''
    poses = []
    for row in states:
        x, z, theta = float(row[0]), float(row[1]), float(row[2])
        poses.append(([x, 0.0, z], p.getQuaternionFromEuler([0.0, theta, 0.0])))
    return poses


def _draw_goal_circle(client, center, radius, color=(0.2, 0.75, 0.2), n_points=48, bead_radius=0.005):
    '''Goal region border as a ring of small spheres in the x-z plane.
    Lifted from visualize_quadrotor_2d_rollout.py's `_draw_goal_circle`.
    '''
    cx, _, cz = center
    rgba = list(color) + [1.0]
    bead_vis = p.createVisualShape(p.GEOM_SPHERE, radius=bead_radius, rgbaColor=rgba,
                                   physicsClientId=client)
    for j in range(n_points):
        th = 2 * np.pi * j / n_points
        pos = [cx + radius * np.cos(th), 0, cz + radius * np.sin(th)]
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
                  width=RENDER_WIDTH, height=RENDER_HEIGHT):
    '''Replay `poses` with a scaled drone and capture camera frames.

    Lifted from visualize_quadrotor_2d_rollout.py's Phase 2 (DIRECT-mode
    client, scaled URDF, angled fixed camera, ER_TINY_RENDERER). If
    `handoff_frame_index` is not None, the drone's body colour switches from
    PRE_HANDOFF_COLOR to POST_HANDOFF_COLOR starting at that pose index, and
    a marker bead is drawn at the handoff position -- requirement 4's visual
    cue. `handoff_frame_index` indexes into `poses`, not into ctrl_freq-rate
    simulation steps.
    '''
    frame_skip = max(1, ctrl_freq // fps)
    client = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.resetSimulation(physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)

        drone_id = p.loadURDF(URDF_PATH, poses[0][0], poses[0][1],
                              globalScaling=DRONE_VISUAL_SCALE, physicsClientId=client)
        p.changeVisualShape(drone_id, -1, rgbaColor=PRE_HANDOFF_COLOR, physicsClientId=client)

        _draw_goal_circle(client, [0, 0, 1], GOAL_TOLERANCE)
        if handoff_frame_index is not None and 0 <= handoff_frame_index < len(poses):
            _draw_handoff_marker(client, poses[handoff_frame_index][0])

        cam_center_z = (TERMINATION[2][0] + TERMINATION[2][1]) / 2.0
        cam_view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, cam_center_z], distance=1.8, yaw=0, pitch=0, roll=0,
            upAxisIndex=2, physicsClientId=client)
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
            (w, h, rgb, _, _) = p.getCameraImage(
                width=width, height=height, shadow=1, renderer=p.ER_TINY_RENDERER,
                viewMatrix=cam_view, projectionMatrix=cam_pro, physicsClientId=client)
            frames.append(np.reshape(rgb, (h, w, 4))[:, :, :3])
        return frames
    finally:
        p.disconnect(client)


# ---------------------------------------------------------------------------
# Trajectory plot (adapted from visualize_quadrotor_2d_rollout.py's
# plot_xz_trajectory, plus a handoff marker)
# ---------------------------------------------------------------------------

def plot_xz_trajectory(states, path, success, handoff_index=None):
    '''x vs z trajectory plot. Same styling as
    visualize_quadrotor_2d_rollout.py's `plot_xz_trajectory`, plus: if
    `handoff_index` is given, a distinct star marker is drawn at that state
    (requirement 4's "mark the handoff point distinctly" in the plot).
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    states = np.asarray(states, dtype=float)
    n = len(states)
    x_vals, z_vals = states[:, 0], states[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n - 1):
        t = i / max(n - 1, 1)
        color = (0.2 + 0.6 * t, 0.3 * (1 - t), 0.8 * (1 - t))
        ax.plot(x_vals[i:i + 2], z_vals[i:i + 2], color=color, linewidth=1.5)

    ax.plot(x_vals[0], z_vals[0], 'bo', markersize=10, label='Start', zorder=5)
    ax.plot(x_vals[-1], z_vals[-1], 'rs', markersize=10, label='End', zorder=5)

    if handoff_index is not None and 0 <= handoff_index < n:
        ax.plot(x_vals[handoff_index], z_vals[handoff_index], marker='*', color='darkorange',
                markersize=22, linestyle='None', label='Handoff (ctrl1 -> ctrl2)', zorder=6,
                markeredgecolor='black', markeredgewidth=0.5)

    goal_circle = plt.Circle((0, 1), GOAL_TOLERANCE, color='green', alpha=0.2,
                             label=f'Goal (r={GOAL_TOLERANCE})')
    ax.add_patch(goal_circle)

    x_lo, x_hi = TERMINATION[0]
    z_lo, z_hi = TERMINATION[2]
    ax.axvline(x_lo, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x_hi, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(z_lo, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(z_hi, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(z_lo, z_hi)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('z (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    result_str = 'SUCCESS' if success else 'FAIL'
    ax.set_title(f'2D Quadrotor Trajectory ({result_str}, {n - 1} steps)', fontsize=13)

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


def record_rollout(category, idx, init_state, result, output_dir, fps=FPS,
                   ctrl_freq=CTRL_FREQ, width=RENDER_WIDTH, height=RENDER_HEIGHT):
    '''Write `<output_dir>/<category>/rollout_<idx>.mp4` + `..._xz.png` +
    `...json` for one recorded rollout. Returns the sidecar dict that was
    written (also used to build summary.json).
    '''
    cat_dir = os.path.join(output_dir, category)
    os.makedirs(cat_dir, exist_ok=True)
    stem = f'rollout_{idx:03d}'

    trajectory = result.trajectory
    handoff_index = result.handoff_index

    if category == 'S1':
        # Truncate to controller 1's own portion, ending at (and including)
        # the handoff state -- mirrors --mode flip in
        # generate_quadrotor_2d_composition.py.
        assert handoff_index >= 0, "'S1' requires a fired handoff"
        states = trajectory[:handoff_index + 1]
        marker_index = len(states) - 1
    elif category in ('S1_to_S2', 'S1_to_F2'):
        assert handoff_index >= 0, f"'{category}' requires a fired handoff"
        states = trajectory
        marker_index = handoff_index
    else:  # F1: no handoff to mark.
        states = trajectory
        marker_index = None

    poses = poses_from_states(states)
    if category == 'S1':
        # Hold the final (handoff) pose for a moment so the colour change
        # that marks it is actually visible, not a single instantaneous frame.
        hold_frames = max(1, int(round(S1_HOLD_SECONDS * fps)))
        poses = poses + [poses[-1]] * hold_frames

    frames = render_frames(poses, marker_index, ctrl_freq=ctrl_freq, fps=fps,
                           width=width, height=height)
    video_path = os.path.join(cat_dir, f'{stem}.mp4')
    save_video(frames, video_path, fps)

    plot_path = os.path.join(cat_dir, f'{stem}_xz.png')
    plot_xz_trajectory(states, plot_path, _category_success(category, result),
                       handoff_index=marker_index)

    sidecar = {
        'category': category,
        'index': idx,
        'init_state': [float(v) for v in init_state],
        'handoff_index': int(handoff_index),
        'flip_success': bool(result.flip_success),
        'ctrl2_success': bool(result.ctrl2_success),
        'num_recorded_states': len(states),
        'num_full_trajectory_states': len(trajectory),
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
                  seed, max_steps, sidecars):
    '''Per requirement 5: per category, how many were found, how many were
    sampled (shared attempt budget across all categories), and the handoff
    index of every recorded rollout.
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
    parser.add_argument('--flip_model', default=None,
                        help='SAC checkpoint for controller 1 (the flip/recovery policy). '
                             'If omitted, controller 1 is a RANDOMLY-INITIALISED SAC policy '
                             '(see build_ctrl1) -- use this before a trained checkpoint exists.')
    parser.add_argument('--ctrl2_model', default=CTRL2_MODEL_DEFAULT,
                        help='safe_explorer_ppo checkpoint for controller 2.')
    parser.add_argument('--g1', required=True, help='Path to a g1.json (quad_composition.g1.G1Region).')
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
    args = parser.parse_args(argv)

    args.categories = [c.strip() for c in args.categories.split(',') if c.strip()]
    unknown = sorted(set(args.categories) - set(CATEGORIES))
    if unknown:
        parser.error(f'unknown --categories {unknown}; choose from {CATEGORIES}')
    if not args.categories:
        parser.error('--categories must name at least one category')
    if args.max_attempts is None:
        args.max_attempts = max(200, 50 * args.num_per_category)
    return args


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    # NFS temp dirs intermittently hang; use /tmp explicitly and best-effort
    # cleanup, matching calibrate_quad2d_g1.py / generate_quadrotor_2d_composition.py.
    tmp = tempfile.mkdtemp(dir='/tmp', prefix='quad2d_composition_viz_')
    env = ctrl1 = ctrl2 = None
    try:
        env, ctrl2 = make_env_and_ctrl2(args.ctrl2_model, tmp)
        ctrl1 = build_ctrl1(args.flip_model, env, tmp)
        with open(args.g1) as fh:
            g1 = G1Region.from_dict(json.load(fh)['g1'])

        os.makedirs(args.output_dir, exist_ok=True)

        recorded, attempts = sample_and_classify(
            env, ctrl1, ctrl2, g1, rng, args.categories, args.num_per_category,
            args.max_steps, args.max_attempts)

        sidecars = {cat: [] for cat in args.categories}
        for cat in args.categories:
            for idx, (init_state, result) in enumerate(recorded[cat]):
                sidecars[cat].append(record_rollout(
                    cat, idx, init_state, result, args.output_dir,
                    fps=args.fps, ctrl_freq=CTRL_FREQ, width=args.width, height=args.height))

        summary = write_summary(args.output_dir, args.categories, args.num_per_category,
                                attempts, args.max_attempts, args.seed, args.max_steps, sidecars)
    finally:
        for obj in (ctrl1, ctrl2, env):
            if obj is not None:
                obj.close()
        shutil.rmtree(tmp, ignore_errors=True)

    print(f'Sampled {attempts}/{args.max_attempts} rollout(s).')
    for cat in args.categories:
        c = summary['categories'][cat]
        status = 'OK' if c['filled'] else 'SHORTFALL'
        print(f'  {cat}: {c["found"]}/{c["requested"]} [{status}]')
    if summary['unfilled_categories']:
        print(f'Could not fill: {summary["unfilled_categories"]}')
    print(f'Output: {args.output_dir}')
    return summary


if __name__ == '__main__':
    main()
