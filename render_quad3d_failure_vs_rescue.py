#!/usr/bin/env python3
'''Render "baseline fails / composition rescues" side-by-side videos.

The 40k paired composition eval datasets
(`quadrotor3D_lqr_regenerated/eval_states.txt` and
`quadrotor3D_flip_to_lqr/eval_states.txt`, both under
`/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/`)
identify 11116 initial states where controller 2 (LQR) alone fails and the
composition (controller 1 "flip" until G1, then latch to LQR) succeeds -- the
21.2% -> 46.4% headline number, made visual. `CANDIDATES` below are the 3
highest-initial-tilt states from that 11116-row win set (166-179 deg, i.e.
near-full inversion), selected by `find_candidates.py`'s tilt sort and
re-verified by `verify_candidates.py` against the SHIPPED controller-1
checkpoint (`models/quad3d_ctrl1_selected.pt`, the 300k policy) rather than
the 75k checkpoint the eval datasets were generated with -- the datasets'
labels only bind the checkpoint that produced them, and the deliverable must
reproduce under what actually ships. (Both scripts are not part of this
script's dependency graph; `CANDIDATES` embeds their verified output so this
script is self-contained and re-runnable without the /tmp scratch files.)

For each candidate this renders ONE combined side-by-side MP4: controller 2
(LQR) ALONE on the left (`rollout_composite(..., ctrl1=None, ...)`, always
fails for these states -- that is the filter), the full composition on the
right (`rollout_composite(..., ctrl1=<shipped>, ...)`, always succeeds), both
from the exact same initial state, captured at the NATIVE simulation rate
(one frame per 100 Hz physics tick, no frame-skipping) so frame i means the
same simulated instant on both sides -- true lock-step sync, not merely "both
videos happen to start together". Playing that back at 30 fps is therefore
~3.3x slow motion throughout, which is a deliberate choice per the brief
("slow the playback rather than picking tamer states"): these are the
dramatic near-full-inversion states, and the baseline's crash is a handful of
physics ticks (0.26-0.49s of real time) that would be a blink at real-time
frame rates.

Panels are independently timed only in how much they are held static:
  - Both sides open on a frozen INTRO_HOLD_SECONDS of the (identical) start
    pose, so the dramatic initial tilt is unmissable before anything moves.
  - The shorter side (always the baseline, which crashes in tens of ticks
    against the composite's several hundred) is padded with repeats of its
    own final (out-of-bounds) pose once its real trajectory ends, so both
    sides run for the same total core length.
  - Both sides close on a frozen OUTRO_HOLD_SECONDS of their own final pose
    (baseline: crashed; composition: at the goal).
No frame is fabricated for the ACTIVE portion of either trajectory -- holds
only repeat a real captured frame.

Reuses (does not reimplement) `visualize_quad3d_composition.render_frames`
for the PyBullet DIRECT-mode tracking-camera capture, handoff colour change
and magenta handoff bead, and `visualize_quadrotor_3d_rollout.save_video` for
MP4 writing. Text overlays (arm labels, live FAILED/HANDOFF/SUCCESS captions,
tilt banner) are composited in with Pillow after rendering, since neither
reused helper draws text.
'''

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from quad_composition.flip_env3d import G_NOM_3D
from quad_composition.rollout3d import (ENV_CONFIG, MAX_STEPS, load_ctrl1, make_env_and_ctrl2,
                                        rollout_composite)
from visualize_quad3d_composition import poses_from_states, render_frames
from visualize_quadrotor_3d_rollout import save_video

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CTRL1_MODEL = os.path.join(REPO_ROOT, 'models', 'quad3d_ctrl1_selected.pt')
CTRL_FREQ = ENV_CONFIG['ctrl_freq']

# The 3 highest-initial-tilt rows of the 11116-row "composite wins, baseline
# loses" set (quadrotor3D_lqr_regenerated x quadrotor3D_flip_to_lqr,
# eval_states.txt, paired on the same init(13) columns), re-verified against
# the shipped 300k controller-1 checkpoint. dataset_row indexes the paired
# eval_states.txt files (0-based); init_state is dataset order
# [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r].
CANDIDATES = [
    {
        'dataset_row': 31023,
        'initial_tilt_deg': 178.66072053840003,
        'init_state': [1.121586, 0.087067, 0.677027, 0.0027, 0.553665, 0.832658, -0.011371,
                       -0.890958, 0.277192, 1.211701, 2.657883, -13.195398, 6.149283],
    },
    {
        'dataset_row': 12931,
        'initial_tilt_deg': 177.58482067832813,
        'init_state': [1.098764, -1.094421, 2.830047, 0.020691, -0.193266, -0.98092, 0.004004,
                       1.292781, -1.469455, 0.087698, -9.939236, -1.220909, -14.703852],
    },
    {
        'dataset_row': 9579,
        'initial_tilt_deg': 176.86890577241687,
        'init_state': [1.454324, 1.540585, 0.806257, 0.022255, -0.884155, 0.466394, -0.015847,
                       0.645705, -1.680427, 0.597, 23.486769, -12.158788, 11.910695],
    },
]

PANEL_WIDTH = 960
PANEL_HEIGHT = 1080
PLAYBACK_FPS = 30
INTRO_HOLD_SECONDS = 1.2
OUTRO_HOLD_SECONDS = 2.5

TOP_BANNER_H = 64
LABEL_BAR_H = 56
STATUS_BAR_H = 56

FONT_DIR = None
for cand in (
    '/common/users/st1122/conda/envs/scg/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf',
):
    if os.path.isdir(cand):
        FONT_DIR = cand
        break


def _font(size, bold=True):
    if FONT_DIR:
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        path = os.path.join(FONT_DIR, name)
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = _font(30)
FONT_LABEL = _font(26)
FONT_STATUS = _font(28)


def _draw_bar(draw, xy, fill, alpha=170):
    x0, y0, x1, y1 = xy
    draw.rectangle([x0, y0, x1, y1], fill=(*fill, alpha))


def _text_centered(draw, box, text, font, color):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x0 + (x1 - x0 - tw) / 2 - bbox[0]
    ty = y0 + (y1 - y0 - th) / 2 - bbox[1]
    draw.text((tx, ty), text, font=font, fill=color)


def compose_frame(left_rgb, right_rgb, title_text, left_label, right_label,
                  left_status, right_status):
    '''Hstack the two panels and burn in the title banner, per-panel arm
    labels (always visible) and per-panel live status captions (only once
    that arm's caption becomes non-None for this frame).
    '''
    combined = np.hstack([left_rgb, right_rgb])
    h, w = combined.shape[:2]
    img = Image.fromarray(combined).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    _draw_bar(draw, (0, 0, w, TOP_BANNER_H), (10, 10, 10), alpha=190)
    _text_centered(draw, (0, 0, w, TOP_BANNER_H), title_text, FONT_TITLE, (255, 255, 255, 255))

    half = w // 2
    label_y0 = TOP_BANNER_H
    label_y1 = label_y0 + LABEL_BAR_H
    _draw_bar(draw, (0, label_y0, half, label_y1), (60, 20, 20), alpha=170)
    _text_centered(draw, (0, label_y0, half, label_y1), left_label, FONT_LABEL,
                  (255, 210, 210, 255))
    _draw_bar(draw, (half, label_y0, w, label_y1), (20, 40, 20), alpha=170)
    _text_centered(draw, (half, label_y0, w, label_y1), right_label, FONT_LABEL,
                  (210, 255, 210, 255))

    status_y1 = h
    status_y0 = status_y1 - STATUS_BAR_H
    if left_status:
        _draw_bar(draw, (0, status_y0, half, status_y1), (120, 0, 0), alpha=200)
        _text_centered(draw, (0, status_y0, half, status_y1), left_status, FONT_STATUS,
                      (255, 255, 255, 255))
    if right_status:
        _draw_bar(draw, (half, status_y0, w, status_y1), (0, 100, 0), alpha=200)
        _text_centered(draw, (half, status_y0, w, status_y1), right_status, FONT_STATUS,
                      (255, 255, 255, 255))

    draw.line([(half, TOP_BANNER_H), (half, h)], fill=(255, 255, 255, 255), width=3)

    out = Image.alpha_composite(img, overlay).convert('RGB')
    return np.asarray(out)


def render_pair(env, ctrl1, ctrl2, g1, candidate, pair_idx, num_pairs, output_dir, urdf_path):
    init_state = candidate['init_state']
    tilt_deg = candidate['initial_tilt_deg']

    baseline_res = rollout_composite(env, None, ctrl2, g1, init_state, max_steps=MAX_STEPS)
    composite_res = rollout_composite(env, ctrl1, ctrl2, g1, init_state, max_steps=MAX_STEPS)

    if baseline_res.ctrl2_success:
        raise RuntimeError(f'pair {pair_idx}: baseline unexpectedly SUCCEEDED at render time '
                           f'-- candidate no longer reproduces, discard it')
    if not (composite_res.flip_success and composite_res.ctrl2_success):
        raise RuntimeError(f'pair {pair_idx}: composition unexpectedly FAILED at render time '
                           f'-- candidate no longer reproduces, discard it')

    left_poses = poses_from_states(baseline_res.trajectory)
    right_poses = poses_from_states(composite_res.trajectory)
    n_left, n_right = len(left_poses), len(right_poses)
    core_len = max(n_left, n_right)

    left_core = left_poses + [left_poses[-1]] * (core_len - n_left)
    right_core = right_poses + [right_poses[-1]] * (core_len - n_right)

    intro_frames = round(INTRO_HOLD_SECONDS * PLAYBACK_FPS)
    outro_frames = round(OUTRO_HOLD_SECONDS * PLAYBACK_FPS)

    left_final = [left_poses[0]] * intro_frames + left_core + [left_core[-1]] * outro_frames
    right_final = [right_poses[0]] * intro_frames + right_core + [right_core[-1]] * outro_frames
    total = len(left_final)
    assert len(right_final) == total

    right_handoff_frame = intro_frames + composite_res.handoff_index

    print(f'  [pair {pair_idx}] rendering left panel ({total} frames)...')
    left_frames = render_frames(left_final, None, ctrl_freq=CTRL_FREQ, fps=CTRL_FREQ,
                                width=PANEL_WIDTH, height=PANEL_HEIGHT, urdf_path=urdf_path)
    print(f'  [pair {pair_idx}] rendering right panel ({total} frames)...')
    right_frames = render_frames(right_final, right_handoff_frame, ctrl_freq=CTRL_FREQ,
                                 fps=CTRL_FREQ, width=PANEL_WIDTH, height=PANEL_HEIGHT,
                                 urdf_path=urdf_path)

    left_crash_frame = intro_frames + n_left - 1
    right_success_frame = intro_frames + n_right - 1

    title = (f'PAIR {pair_idx + 1}/{num_pairs} -- SAME INITIAL STATE -- '
            f'INITIAL TILT {tilt_deg:.1f} deg (near-full inversion)')
    left_label = 'BASELINE: LQR ALONE'
    right_label = 'COMPOSITION: FLIP -> LQR (shipped, 300k)'

    combined_frames = []
    for i in range(total):
        left_status = 'FAILED -- LEFT THE BOX' if i >= left_crash_frame else None
        if i >= right_success_frame:
            right_status = 'SUCCESS -- GOAL REACHED'
        elif i >= right_handoff_frame:
            right_status = 'HANDOFF -> CTRL2 (LQR)'
        else:
            right_status = None
        combined_frames.append(compose_frame(left_frames[i], right_frames[i], title,
                                             left_label, right_label, left_status,
                                             right_status))

    pair_dir = os.path.join(output_dir, f'pair_{pair_idx:02d}')
    os.makedirs(pair_dir, exist_ok=True)
    video_path = os.path.join(pair_dir, 'side_by_side.mp4')
    save_video(combined_frames, video_path, PLAYBACK_FPS)

    sidecar = {
        'dataset_row': candidate['dataset_row'],
        'initial_tilt_deg': tilt_deg,
        'init_state': [float(v) for v in init_state],
        'baseline': {
            'ctrl2_success': bool(baseline_res.ctrl2_success),
            'num_states': n_left,
            'real_time_seconds': (n_left - 1) / float(CTRL_FREQ),
        },
        'composite': {
            'flip_success': bool(composite_res.flip_success),
            'ctrl2_success': bool(composite_res.ctrl2_success),
            'handoff_index': int(composite_res.handoff_index),
            'num_states': n_right,
            'real_time_seconds': (n_right - 1) / float(CTRL_FREQ),
        },
        'video': {
            'path': video_path,
            'num_frames': total,
            'playback_fps': PLAYBACK_FPS,
            'video_seconds': total / float(PLAYBACK_FPS),
            'intro_hold_seconds': INTRO_HOLD_SECONDS,
            'outro_hold_seconds': OUTRO_HOLD_SECONDS,
            'left_crash_frame': left_crash_frame,
            'right_handoff_frame': right_handoff_frame,
            'right_success_frame': right_success_frame,
        },
    }
    return sidecar


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output_dir', default=os.path.join(
        REPO_ROOT, 'rollout_visualizations', 'quad3d_failure_vs_rescue'))
    parser.add_argument('--flip_model', default=CTRL1_MODEL)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    tmp = tempfile.mkdtemp(dir='/tmp', prefix='quad3d_failure_vs_rescue_')
    env = ctrl1 = ctrl2 = None
    sidecars = []
    try:
        env, ctrl2 = make_env_and_ctrl2(tmp)
        ctrl1 = load_ctrl1(args.flip_model, env, tmp)
        g1 = G_NOM_3D
        urdf_path = env.URDF_PATH

        for idx, candidate in enumerate(CANDIDATES):
            print(f'Pair {idx}: dataset_row={candidate["dataset_row"]} '
                 f'tilt={candidate["initial_tilt_deg"]:.2f} deg')
            sidecar = render_pair(env, ctrl1, ctrl2, g1, candidate, idx, len(CANDIDATES),
                                  args.output_dir, urdf_path)
            sidecars.append(sidecar)
            print(f'  -> {sidecar["video"]["path"]} '
                 f'({sidecar["video"]["num_frames"]} frames, '
                 f'{sidecar["video"]["video_seconds"]:.1f}s)')
    finally:
        for obj in (ctrl1, ctrl2, env):
            if obj is not None:
                obj.close()
        shutil.rmtree(tmp, ignore_errors=True)

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as fh:
        json.dump(sidecars, fh, indent=2)
    print(f'\nWrote {len(sidecars)} pair(s) to {args.output_dir}')


if __name__ == '__main__':
    main()
