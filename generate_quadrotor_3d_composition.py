#!/usr/bin/env python3
'''Generate the flip-only, composite, and regenerated-baseline datasets for
quadrotor-3D. Ported from generate_quadrotor_2d_composition.py -- read that
file first; the design (three modes, paired initial states, a locally
regenerated baseline, the impossible-label guard) is unchanged here. What is
NEW in this port: --num_workers (multiprocessing, spawn context, mirroring
analyze_quad3d_composition.py's pattern) and resumability (skip trajectories
already written) -- the 2D generator had neither, which was flagged as a
defect.

--mode flip       controller 1 alone, truncated at first G1 entry
--mode composite  controller 1 then controller 2, latching on first G1 entry.
                  EVALUATION ONLY.
--mode baseline   controller 2 (LQR) alone -- no controller 1, no G1.
                  `quad_composition.rollout3d.rollout_composite` (what the
                  design brief calls rollout_composite_3d) with ctrl1=None.

G1 IS FIXED, NOT CALIBRATED (unlike 2D): `flip_env3d.G_NOM_3D` --
tilt_c=0.175 rad (10 deg), w_c=4.0 rad/s -- the SAME region controller 1 was
trained to reach and analyze_quad3d_composition.py measures against. There is
no --g1 flag and no calibration file to load; --mode baseline therefore has
nothing G1-shaped to avoid touching in the first place.

CONTROLLER 1: SAC, loaded via `quad_composition.rollout3d.load_ctrl1` from
models/quad3d_ctrl1_selected.pt (the checkpoint promoted as controller 1 --
see the session ledger commit "Promote the hand-coded 3D flip controller to
controller 1"). CONTROLLER 2: LQR, analytic, built by
`quad_composition.rollout3d.make_env_and_ctrl2` exactly as
generate_quadrotor_3d_trajectories.py builds it.

WHY --mode baseline EXISTS (Ruling D-I, carried over unchanged from 2D --
generate_quadrotor_2d_composition.py's module docstring): the 2D
investigation found the untouched original 2D generator reproducing only
19/20 labels and 12/20 final states against its OWN shipped dataset, while
the shared rollout core matched that generator exactly at atol=1e-9. That is
chaotic PyBullet/library/hardware divergence, not a defect in any particular
script, and the same simulation stack (PyBullet, the same env class, the same
machine class) underlies the 3D path too. So comparing a locally generated
composite against the SHIPPED quadrotor3D_lqr labels would attribute
numerical-environment artifacts to controller behaviour. --mode baseline
reruns controller 2 alone over the SAME initial states as --mode
flip/composite, through the SAME rollout core, so any composite-vs-baseline
comparison is apples to apples. It needs neither a controller-1 checkpoint
nor G1.

INITIAL STATES: the first 13 columns of the shipped quadrotor3D_lqr
eval_states.txt (dataset order [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot,
p, q, r]), so every dataset produced here is PAIRED with the published one --
same rows, same order, exactly as the 2D version pairs against
quadrotor2D_rl.

COLUMN WIDTHS DIFFER BY MODE -- do not assume a common width (the 2D bug this
port must not repeat): baseline eval_states.txt rows are init(13) + final(13)
+ ctrl2_success(1) = 27 columns; flip/composite rows are init(13) + final(13)
+ flip_success + ctrl2_success = 28 columns.
'''

import argparse
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time

import numpy as np

from quad_composition.flip_env3d import G_NOM_3D
from quad_composition.rollout3d import (GOAL_TOLERANCE, LQR_CONFIG, MAX_STEPS, load_ctrl1,
                                        make_env_and_ctrl2, rollout_composite)

CTRL1_MODEL = 'models/quad3d_ctrl1_selected.pt'
BASELINE_DIR = ('/common/users/shared/pracsys/genMoPlan/data_trajectories/'
                'deterministic/quadrotor3D_lqr')

INIT_DIM = 13  # dataset-order state width: x,y,z,qw,qx,qy,qz,x_dot,y_dot,z_dot,p,q,r

REGENERATED_BASELINE_NOTE = (
    'Ruling D-I, carried over from the 2D composition experiment '
    '(generate_quadrotor_2d_composition.py): the untouched 2D reference '
    'generator did not even reproduce its OWN shipped labels on this machine '
    '(19/20 label agreement, 12/20 final-state agreement at atol=1e-4 on a '
    'mixed-class window) while the shared rollout core matched that generator '
    'exactly at atol=1e-9 -- chaotic PyBullet/library/hardware divergence, not '
    'a script defect, and the same simulation stack underlies this 3D path. '
    'Any baseline-vs-composition comparison must therefore use a baseline '
    'regenerated locally by THIS rollout core (--mode baseline), never the '
    'archived quadrotor3D_lqr file, or numerical drift rather than controller '
    "composition would dominate the comparison's win/loss counts."
)

ACTION_SPACE = {
    'normalized_rl_action_space': False,
    'note': ('Physical actuator space (thrust-to-weight ratio 2.24), matching '
             'quadrotor3D_lqr generation exactly -- controller 1 must have exactly '
             "controller 2's authority. See quad_composition/rollout3d.py module "
             'docstring, item 4. Unlike the 2D composition dataset, this is NOT the '
             'normalized RL action space.'),
}

ATTITUDE_CONVENTION_NOTE = (
    'The quaternion column (qw, qx, qy, qz; canonicalized qw >= 0) uses the SAME '
    'convention as the shipped quadrotor3D_lqr dataset -- no format drift here, '
    "unlike the 2D composition dataset's theta column (Finding C1 there). G1 "
    'membership, and every stored attitude value, come from the body->world '
    'rotation matrix (quad_composition.rollout3d.tilt_from_quat), never from Euler '
    "angles -- see rollout3d.py's module docstring, item 1."
)


def validate_labels(flip_success, ctrl2_success):
    '''(flip_success=0, ctrl2_success=1) cannot occur -- no handoff, no ctrl 2.'''
    bad = (np.asarray(flip_success) == 0) & (np.asarray(ctrl2_success) == 1)
    if bad.any():
        raise ValueError(f'impossible label combination in {int(bad.sum())} rows')


def labels_from_result(result):
    return int(result.flip_success), int(result.ctrl2_success)


def eval_states_row(init, result):
    '''28-column row for --mode flip/composite: init(13) + final(13) +
    [flip_success, ctrl2_success]. Used for BOTH modes (not just composite):
    even --mode flip's stored trajectory is truncated at handoff, but this row
    records the labels from the FULL underlying rollout (rollout_composite
    always runs the full composite internally), matching
    generate_quadrotor_2d_composition.py.
    '''
    flip, ctrl2 = labels_from_result(result)
    return list(map(float, init)) + list(map(float, result.trajectory[-1])) + [flip, ctrl2]


def baseline_eval_states_row(init, result):
    '''27-column row for --mode baseline, matching the archived
    quadrotor3D_lqr eval_states.txt format exactly: init(13) + final(13) +
    [ctrl2_success].

    flip_success is never written here: it is not meaningful on the baseline
    path (ctrl1=None), and a meaningless placeholder column would make this
    file indistinguishable (by column count) from a real 28-column
    flip/composite file, inviting a downstream reader to misparse it.
    '''
    return list(map(float, init)) + list(map(float, result.trajectory[-1])) + [int(result.ctrl2_success)]


def handoff_row(init, result):
    '''27 columns: init(13) + the state at handoff_index (13) + handoff_index.

    The handoff state is [-1]*13 when no handoff fired (flip failure, or
    --mode baseline where it never can), in which case handoff_index is -1.

    handoff_index is PERSISTED for the same reason as in 2D: without it, a row
    whose INITIAL state was already inside G1 (handoff_index == 0, controller
    1 never acted) cannot be told apart from a real handoff -- and they
    measure something different (see analyze_quad3d_composition.py's
    real-vs-all-handoffs split).
    '''
    handoff = (result.trajectory[result.handoff_index]
              if result.handoff_index >= 0 else [-1.0] * INIT_DIM)
    row = list(map(float, init)) + list(map(float, handoff))
    return row + [float(result.handoff_index)]


def row_builder(mode):
    return baseline_eval_states_row if mode == 'baseline' else eval_states_row


# ---------------------------------------------------------------------------
# Per-trajectory unit of work, and its on-disk sidecar (resumability).
#
# Every rollout is expensive (a real PyBullet simulation up to MAX_STEPS
# steps); the trajectory file alone is not enough to resume from, because
# --mode flip's stored trajectory is TRUNCATED at handoff and cannot recover
# the full rollout's ctrl2_success. So each index also gets a small JSON
# sidecar carrying exactly the two rows write_outputs needs. Both files are
# written to a *.tmp path and atomically os.replace'd into place, so a killed
# worker never leaves a half-written file that a resumed run would mistake
# for "done" (the atomicity trap: partial writes masquerading as complete
# ones).
# ---------------------------------------------------------------------------

def trajectory_path(output_dir, idx):
    return os.path.join(output_dir, 'trajectories', f'sequence_{idx}.txt')


def label_path(output_dir, idx):
    return os.path.join(output_dir, 'partial', f'{idx}.json')


def is_done(output_dir, idx):
    return os.path.exists(trajectory_path(output_dir, idx)) and os.path.exists(label_path(output_dir, idx))


def pending_indices(output_dir, inits):
    '''(idx, init) pairs not yet fully written -- what a (re)run must still do.'''
    return [(i, init) for i, init in enumerate(inits) if not is_done(output_dir, i)]


def _atomic_write(path, write_fn):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    write_fn(tmp)
    os.replace(tmp, path)


def write_trajectory(output_dir, idx, trajectory):
    _atomic_write(trajectory_path(output_dir, idx),
                 lambda tmp: np.savetxt(tmp, np.array(trajectory), delimiter=',', fmt='%.6f'))


def write_label(output_dir, idx, eval_row, ho_row):
    def _write(tmp):
        with open(tmp, 'w') as fh:
            json.dump({'eval_row': eval_row, 'handoff_row': ho_row}, fh)
    _atomic_write(label_path(output_dir, idx), _write)


def read_label(output_dir, idx):
    with open(label_path(output_dir, idx)) as fh:
        d = json.load(fh)
    return d['eval_row'], d['handoff_row']


def process_one(mode, env, ctrl1, ctrl2, g1, idx, init, output_dir, max_steps=MAX_STEPS):
    '''Roll one init through rollout_composite and persist its two output
    rows. This is the SAME function every sequential test call and every
    parallel worker call go through -- one code path, so a test against fakes
    (below) exercises exactly what a worker process runs for real.
    '''
    res = rollout_composite(env, ctrl1, ctrl2, g1, init, max_steps=max_steps)
    if mode == 'flip' and res.handoff_index >= 0:
        # Keep only controller 1's own portion, up to and including the state
        # that entered G1. The simulated rollout is identical to --mode
        # composite's; only the STORED trajectory differs.
        res.trajectory = res.trajectory[:res.handoff_index + 1]

    eval_row = row_builder(mode)(init, res)
    ho_row = handoff_row(init, res)
    write_trajectory(output_dir, idx, res.trajectory)
    write_label(output_dir, idx, eval_row, ho_row)
    return eval_row, ho_row


def generate_dataset(mode, env, ctrl1, ctrl2, g1, inits, output_dir, max_steps=MAX_STEPS):
    '''Sequential rollout-and-write loop over one shared env/ctrl1/ctrl2 --
    used directly by tests (against fakes, so --mode flip/composite are
    exercised without booting a real SAC checkpoint) and available as a
    single-process fallback. `run_parallel` below dispatches the identical
    per-index work (`process_one`) across worker-local env/ctrl1/ctrl2
    instances instead.

    Returns (rows, handoffs): rows are 28-column (flip/composite) or
    27-column (baseline) plain lists ready for np.array; handoffs are always
    27-column.
    '''
    os.makedirs(os.path.join(output_dir, 'trajectories'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'partial'), exist_ok=True)
    rows, handoffs = [], []
    for idx, init in enumerate(inits):
        eval_row, ho_row = process_one(mode, env, ctrl1, ctrl2, g1, idx, init, output_dir, max_steps)
        rows.append(eval_row)
        handoffs.append(ho_row)
    return rows, handoffs


# ---------------------------------------------------------------------------
# Parallel driver (new relative to the 2D port): spawn-context Pool, one
# env/ctrl1/ctrl2 built per worker in an initializer, mirroring
# analyze_quad3d_composition.py's pattern.
# ---------------------------------------------------------------------------

_WORKER = {}


def _init_worker(mode, ctrl1_path, g1_dict, output_dir, max_steps, tmp_root):
    from quad_composition.g1 import G1Region  # local import: picklable across spawn
    tmp = tempfile.mkdtemp(dir=tmp_root)
    env, ctrl2 = make_env_and_ctrl2(tmp)
    ctrl1 = load_ctrl1(ctrl1_path, env, tmp) if mode != 'baseline' else None
    _WORKER.update(mode=mode, env=env, ctrl1=ctrl1, ctrl2=ctrl2,
                   g1=G1Region.from_dict(g1_dict), output_dir=output_dir, max_steps=max_steps)


def _run_one(job):
    idx, init = job
    w = _WORKER
    process_one(w['mode'], w['env'], w['ctrl1'], w['ctrl2'], w['g1'], idx, init,
               w['output_dir'], w['max_steps'])
    return idx


def run_parallel(mode, inits, ctrl1_path, g1, output_dir, num_workers, max_steps=MAX_STEPS,
                 progress=None):
    '''Run every (idx, init) not already written, across `num_workers` worker
    processes, then re-read every index's sidecar (old and new alike) in
    dataset order and return (rows, handoffs) -- ready for `write_outputs`.

    Resumable: indices whose trajectory + label sidecar already exist are
    skipped entirely, so a second invocation over the same --output_dir picks
    up only what is missing. `ctrl1_path`/`g1` are never touched for
    --mode baseline (`_init_worker` only calls `load_ctrl1` when mode !=
    'baseline').
    '''
    os.makedirs(os.path.join(output_dir, 'trajectories'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'partial'), exist_ok=True)

    todo = pending_indices(output_dir, inits)
    n_total = len(inits)
    if todo:
        tmp_root = tempfile.mkdtemp(dir='/tmp', prefix='quad3d_composition_')
        try:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=num_workers, initializer=_init_worker,
                         initargs=(mode, ctrl1_path, g1.to_dict(), output_dir, max_steps,
                                   tmp_root)) as pool:
                for done_idx in pool.imap_unordered(_run_one, todo, chunksize=1):
                    if progress is not None:
                        progress(done_idx)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    rows, handoffs = [], []
    for i in range(n_total):
        eval_row, ho_row = read_label(output_dir, i)
        rows.append(eval_row)
        handoffs.append(ho_row)
    return rows, handoffs


def load_initial_states(baseline_dir):
    path = os.path.join(baseline_dir, 'eval_states.txt')
    # ndmin=2: a baseline_dir with exactly one row (e.g. a tiny --limit smoke
    # fixture) would otherwise come back 1-D and break the [:, :13] slice.
    return np.loadtxt(path, delimiter=',', ndmin=2)[:, :INIT_DIM]


def build_description(args, rows, elapsed, ckpt_info=None):
    stats = {'total': int(len(rows))}
    success_criteria = {'type': 'radius', 'threshold': GOAL_TOLERANCE}
    controller_2 = {'type': 'lqr', 'q_lqr': LQR_CONFIG['q_lqr'], 'r_lqr': LQR_CONFIG['r_lqr']}

    if args.mode == 'baseline':
        stats['ctrl2_success'] = int(rows[:, 26].sum())
        return {
            'dataset_name': 'Quadrotor-3D baseline trajectories (regenerated)',
            'purpose': ('locally regenerated controller-2-alone baseline; use this, not '
                       'the archived quadrotor3D_lqr dataset, wherever a comparison against '
                       'the flip/composite datasets is needed -- see '
                       'regenerated_baseline_note'),
            'regenerated_baseline_note': REGENERATED_BASELINE_NOTE,
            'controller_2': controller_2,
            'action_space': ACTION_SPACE,
            'labels': {'ctrl2_success': '1 if controller 2 alone reached the goal ball'},
            'files': {
                'eval_states.txt': '27 columns: init(13), final(13), ctrl2_success',
                'roa_labels.txt': '14 columns: init(13), ctrl2_success',
                'handoff_states.txt': (
                    '27 columns: init(13), handoff state(13), handoff_index. Written for '
                    'format parity with the flip/composite datasets, but INERT on this '
                    'path: --mode baseline runs controller 2 alone, with no controller 1 '
                    'and no G1, so no handoff can ever fire and columns 13..25 are ALWAYS '
                    '-1 in every row. Nothing downstream should read them from a baseline '
                    'directory.'),
                'trajectories/sequence_<i>.txt': 'full state trajectory, dataset order',
            },
            'attitude_convention': ATTITUDE_CONVENTION_NOTE,
            'success_criteria': success_criteria,
            'max_steps': args.max_steps,
            'statistics': stats,
            'elapsed_sec': elapsed,
        }

    stats['flip_success'] = int(rows[:, 26].sum())
    stats['ctrl2_success'] = int(rows[:, 27].sum())
    return {
        'dataset_name': f'Quadrotor-3D {args.mode} trajectories',
        'purpose': ('EVALUATION ONLY' if args.mode == 'composite'
                   else 'controller 1 alone, truncated at first G1 entry'),
        'regenerated_baseline_note': REGENERATED_BASELINE_NOTE,
        'g1': G_NOM_3D.to_dict(),
        'controller_1': {'type': 'sac', 'model': args.ctrl1_path, 'objective': 'attitude-only',
                         'checkpoint_info': ckpt_info},
        'controller_2': controller_2,
        'handoff': {'operator': 'sequential latch on first entry into G1, permanent '
                                'once fired (g1 is never consulted again after handoff)'},
        'action_space': ACTION_SPACE,
        'labels': {
            'flip_success': '1 if controller 1 reached G1',
            'ctrl2_success': '1 if the composite reached the goal ball',
            'note': '(flip_success=0, ctrl2_success=1) cannot occur'},
        'files': {
            'eval_states.txt': '28 columns: init(13), final(13), flip_success, ctrl2_success',
            'roa_labels.txt': '15 columns: init(13), flip_success, ctrl2_success',
            'handoff_states.txt': (
                '27 columns: init(13), handoff state(13), handoff_index. handoff_index is '
                '-1 when no handoff fired (columns 13..25 are then all -1), 0 when the '
                'INITIAL state was already inside G1 (controller 1 never acted), and > 0 '
                'for a real handoff at that trajectory row.'),
            'trajectories/sequence_<i>.txt': (
                'full state trajectory, dataset order; --mode flip truncates it at (and '
                'including) the handoff state'),
        },
        'attitude_convention': ATTITUDE_CONVENTION_NOTE,
        'success_criteria': success_criteria,
        'max_steps': args.max_steps,
        'statistics': stats,
        'elapsed_sec': elapsed,
    }


def write_outputs(args, rows, handoffs, elapsed=None, ckpt_info=None):
    rows = np.array(rows)
    if args.mode != 'baseline':
        validate_labels(rows[:, 26], rows[:, 27])

    os.makedirs(args.output_dir, exist_ok=True)
    np.savetxt(os.path.join(args.output_dir, 'eval_states.txt'), rows,
              delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(args.output_dir, 'roa_labels.txt'),
              np.column_stack([rows[:, :INIT_DIM], rows[:, 26:]]), delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(args.output_dir, 'handoff_states.txt'),
              np.array(handoffs), delimiter=',', fmt='%.6f')

    with open(os.path.join(args.output_dir, 'dataset_description.json'), 'w') as fh:
        json.dump(build_description(args, rows, elapsed, ckpt_info), fh, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['flip', 'composite', 'baseline'], required=True)
    parser.add_argument('--ctrl1_path', default=CTRL1_MODEL,
                        help='SAC checkpoint for controller 1. Never opened for --mode '
                             'baseline, regardless of this value.')
    parser.add_argument('--baseline_dir', default=BASELINE_DIR,
                        help='Directory whose eval_states.txt supplies the shared initial '
                             'states (first 13 columns) for every mode -- default: the '
                             'shipped quadrotor3D_lqr directory.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None,
                        help='default: CPUs visible via os.sched_getaffinity(0)')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    inits = load_initial_states(args.baseline_dir)
    if args.limit:
        inits = inits[:args.limit]

    num_workers = args.num_workers or len(os.sched_getaffinity(0))
    ckpt_info = None
    ctrl1_path = None

    tmp_root = tempfile.mkdtemp(dir='/tmp', prefix='quad3d_composition_ckpt_')
    try:
        if args.mode != 'baseline':
            # Freeze the checkpoint once, up front: models/quad3d_ctrl1_selected.pt
            # may be rewritten concurrently by a re-selection pass running
            # elsewhere (same operational constraint analyze_quad3d_composition.py
            # already documents). Every worker loads the frozen copy, never the
            # live path, so the run is attributable to one specific checkpoint even
            # if the source file changes mid-run -- and this script must not
            # itself write to that path.
            from analyze_quad3d_composition import snapshot_checkpoint
            ctrl1_path, ckpt_info = snapshot_checkpoint(args.ctrl1_path, tmp_root)

        print(f'{len(inits)} initial states from {args.baseline_dir} (--mode {args.mode}, '
             f'{num_workers} workers)')
        t0 = time.time()
        n_done = [0]

        def _progress(_idx):
            n_done[0] += 1
            if n_done[0] % 2000 == 0:
                print(f'  {n_done[0]} trajectories completed '
                     f'({time.time() - t0:.1f}s elapsed)')

        rows, handoffs = run_parallel(args.mode, inits, ctrl1_path, G_NOM_3D, args.output_dir,
                                      num_workers, max_steps=args.max_steps, progress=_progress)
        elapsed = time.time() - t0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    write_outputs(args, rows, handoffs, elapsed=elapsed, ckpt_info=ckpt_info)
    print(f'{len(rows)} trajectories -> {args.output_dir} ({elapsed:.1f}s)')


if __name__ == '__main__':
    main()
