#!/usr/bin/env python3
'''Check that a `visualize_quad3d_composition.py` output directory is actually
watchable, not merely well-formed.

A technically-valid MP4 of an off-screen drone, or of three frames, passes
every "does the file exist / is it non-empty" test and is still not a
deliverable.  So this decodes the pixels:

  CONTAINER   every clip is h264, has a positive duration, and runs at least
              `--min_seconds` with at least `--min_frames` frames.

  DRONE IN    the drone is dark grey before the handoff and orange after
  FRAME       (`PRE_HANDOFF_COLOR` / `POST_HANDOFF_COLOR`).  Every frame must
              contain drone-coloured pixels inside the central crop -- the
              camera tracks the drone, so if it is not near the middle of the
              frame the tracking is broken and the clip is unusable.

  HANDOFF     for S1_to_S2 / S1_to_F2, the magenta marker bead
  VISIBLE     (`HANDOFF_MARKER_COLOR`) must appear in some frame, and the
              drone's own colour must go dark -> orange across the clip.  For
              S1 (truncated at the handoff) only the colour change is
              required; F1 has no handoff and must show NEITHER.

Colour tests are hue-band tests, not exact RGB matches: the renderer applies
lighting and shadows, so the drone's grey and orange arrive darkened.

Exits non-zero if any clip fails, and prints one line per clip either way.
'''

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

from visualize_quad3d_composition import CATEGORIES

# Fraction of the frame (centred) the drone must appear within.  The camera
# targets the drone's current position every frame, so anything outside this is
# a tracking failure, not a framing preference.
CENTER_CROP = 0.5


def ffprobe(path):
    '''Codec, frame count and duration, straight from the container.

    `nb_read_frames` (from `-count_frames`) rather than `nb_frames`: the
    latter is metadata and is frequently absent or wrong in files written by
    imageio/ffmpeg, and a frame count is exactly the thing being checked.
    '''
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
         '-show_entries', 'stream=codec_name,nb_read_frames,r_frame_rate,width,height',
         '-show_entries', 'format=duration', '-of', 'json', path],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    stream = (data.get('streams') or [{}])[0]
    num, _, den = (stream.get('r_frame_rate') or '0/1').partition('/')
    return {
        'codec': stream.get('codec_name'),
        'frames': int(stream.get('nb_read_frames') or 0),
        'width': int(stream.get('width') or 0),
        'height': int(stream.get('height') or 0),
        'fps': (float(num) / float(den)) if float(den or 0) else 0.0,
        'duration': float((data.get('format') or {}).get('duration') or 0.0),
    }


def _masks(frame):
    '''(dark drone, orange drone, magenta marker) boolean masks for one RGB frame.'''
    rgb = frame.astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mx = rgb.max(axis=2)
    dark = mx < 70
    orange = (r > 90) & (r > g * 3 // 2) & (g > b * 3 // 2) & (b < 90)
    magenta = (r > 80) & (b > 80) & (g * 2 < r) & (g * 2 < b)
    return dark, orange, magenta


def _center(mask):
    h, w = mask.shape
    dh, dw = int(h * (1 - CENTER_CROP) / 2), int(w * (1 - CENTER_CROP) / 2)
    return mask[dh:h - dh, dw:w - dw]


def inspect_frames(path):
    '''Per-frame pixel counts for the three colour cues, centre-cropped for
    the drone (which the camera keeps centred) and full-frame for the marker
    (which is left behind and drifts off as the drone flies away).
    '''
    import imageio.v2 as imageio

    dark_counts, orange_counts, magenta_counts = [], [], []
    reader = imageio.get_reader(path)
    try:
        for frame in reader:
            dark, orange, magenta = _masks(np.asarray(frame)[:, :, :3])
            dark_counts.append(int(_center(dark).sum()))
            orange_counts.append(int(_center(orange).sum()))
            magenta_counts.append(int(magenta.sum()))
    finally:
        reader.close()
    return (np.array(dark_counts), np.array(orange_counts), np.array(magenta_counts))


def check_clip(category, path, sidecar, min_seconds, min_frames, min_drone_pixels):
    '''Every check for one clip.  Returns (ok, probe, list of failure strings).'''
    failures = []
    probe = ffprobe(path)
    if probe['codec'] != 'h264':
        failures.append(f'codec is {probe["codec"]!r}, not h264')
    if probe['duration'] <= 0:
        failures.append(f'duration is {probe["duration"]}')
    if probe['frames'] < min_frames:
        failures.append(f'only {probe["frames"]} frames (< {min_frames})')
    if probe['duration'] < min_seconds:
        failures.append(f'only {probe["duration"]:.2f}s (< {min_seconds}s)')

    dark, orange, magenta = inspect_frames(path)
    if len(dark) == 0:
        failures.append('no decodable frames')
        return False, probe, failures
    if probe['frames'] and len(dark) != probe['frames']:
        failures.append(f'decoded {len(dark)} frames, container reports {probe["frames"]}')

    visible = (dark + orange) >= min_drone_pixels
    if not visible.all():
        missing = int((~visible).sum())
        failures.append(f'drone not visible in the central {int(CENTER_CROP * 100)}% '
                        f'of {missing}/{len(visible)} frames')

    has_handoff = category in ('S1', 'S1_to_S2', 'S1_to_F2')
    if has_handoff:
        if orange.max() < min_drone_pixels:
            failures.append('drone never turns orange -- the handoff colour cue is missing')
        if dark.max() < min_drone_pixels:
            failures.append('drone is never dark -- there is no pre-handoff phase to see')
        elif int(np.argmax(orange >= min_drone_pixels)) <= int(np.argmax(dark >= min_drone_pixels)):
            failures.append('orange appears before dark -- the colour change is not a handoff')
    else:
        if orange.max() >= min_drone_pixels:
            failures.append('F1 clip shows the post-handoff colour, but no handoff fired')
        if magenta.max() >= min_drone_pixels:
            failures.append('F1 clip shows a handoff marker, but no handoff fired')

    if category in ('S1_to_S2', 'S1_to_F2'):
        if magenta.max() < 1:
            failures.append('handoff marker bead never appears')
        elif sidecar is not None:
            # The bead is drawn at the handoff position and the camera flies
            # away with the drone, so require it around the handoff, not
            # throughout.
            frames_per_state = len(dark) / max(sidecar['num_recorded_states'], 1)
            frame_of_handoff = int(sidecar['handoff_index'] * frames_per_state)
            window = magenta[max(frame_of_handoff - 5, 0):frame_of_handoff + 15]
            if window.size and window.max() < 1:
                failures.append('handoff marker bead is not visible around the handoff frame')

    return not failures, probe, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--min_seconds', type=float, default=2.5)
    parser.add_argument('--min_frames', type=int, default=40)
    parser.add_argument('--min_drone_pixels', type=int, default=25,
                        help='pixels of drone colour needed to call it visible in a frame')
    parser.add_argument('--categories', default=','.join(CATEGORIES))
    args = parser.parse_args(argv)

    categories = [c.strip() for c in args.categories.split(',') if c.strip()]
    all_ok = True
    total = 0
    for category in categories:
        for path in sorted(glob.glob(os.path.join(args.output_dir, category, '*.mp4'))):
            total += 1
            sidecar_path = path[:-4] + '.json'
            sidecar = None
            if os.path.exists(sidecar_path):
                with open(sidecar_path) as fh:
                    sidecar = json.load(fh)
            ok, probe, failures = check_clip(category, path, sidecar, args.min_seconds,
                                             args.min_frames, args.min_drone_pixels)
            all_ok &= ok
            tilt = f'{sidecar["initial_tilt_deg"]:5.1f}deg' if sidecar else '     -    '
            handoff = f'ho={sidecar["handoff_index"]:<4d}' if sidecar else ''
            print(f'{"PASS" if ok else "FAIL"}  {category:<9} {os.path.basename(path):<16} '
                  f'{probe["codec"]:<5} {probe["frames"]:>4}f {probe["fps"]:>5.1f}fps '
                  f'{probe["duration"]:>5.2f}s {probe["width"]}x{probe["height"]} '
                  f'tilt={tilt} {handoff}')
            for failure in failures:
                print(f'        - {failure}')

    print(f'\n{total} clip(s) checked: {"ALL PASS" if all_ok else "FAILURES ABOVE"}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
