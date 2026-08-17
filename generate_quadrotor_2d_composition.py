#!/usr/bin/env python3
'''Generate the flip-only, composite, and regenerated-baseline datasets for
quadrotor-2D.

--mode flip       controller 1 alone, truncated at first G1 entry
--mode composite  controller 1 then controller 2, latching on first G1 entry
--mode baseline   controller 2 alone -- no controller 1, no G1 (Ruling D-I)

All three share the initial states of the archived quadrotor2D_rl dataset
(via --baseline_dir) so all comparisons are paired (spec D7).

RULING D-D (task-2-report.md): controller 1 is SAC, not safe_explorer_ppo, so
it is built through `quad_composition.rollout2d.load_ctrl1`. Passing
rollout2d.ALGO_CONFIG (safe_explorer_ppo's config, with different
hyperparameters -- hidden_dim, batch size, max_env_steps all differ) to
`make('sac', ...)` -- as an earlier draft of this script did -- would
silently construct the wrong controller.

RULING D-I (quad_composition/rollout2d.py module docstring; task-2-report.md
"Fix round 2"): the archived quadrotor2D_rl dataset is NOT bit-reproducible
per trajectory on this machine -- confirmed by running the untouched
generate_quadrotor_2d_trajectories_rl.run_trajectory against its own shipped
eval_states.txt (19/20 labels, 12/20 final states agree at atol=1e-4 on a
mixed-class window; one row's discrete outcome flips). That is chaotic
PyBullet/library/hardware divergence, not a defect in any script. So any
baseline-vs-composition comparison must use a baseline REGENERATED LOCALLY by
this same rollout core rather than the archived file -- otherwise numerical
drift, not controller composition, would dominate the comparison's "lost"
count (expected to be near zero; see REGENERATED_BASELINE_NOTE below).
--mode baseline exists for exactly this: it reruns controller 2 alone
(`rollout_composite` with `ctrl1=None`) over the SAME initial states as
--mode flip/composite and writes a dataset in the archived dataset's own
13-column eval_states.txt / 7-column roa_labels.txt format, so a later
comparison script can point --baseline_dir at it as a drop-in replacement for
the archived quadrotor2D_rl directory. --mode baseline needs neither
--flip_model nor --g1 -- it never constructs controller 1 or G1.
'''

import argparse
import json
import os
import shutil
import tempfile

import numpy as np

from quad_composition.g1 import G1Region
from quad_composition.rollout2d import (GOAL_TOLERANCE, MAX_STEPS, load_ctrl1, make_env_and_ctrl2,
                                        rollout_composite)

CTRL2_MODEL = ('examples/rl/models/safe_explorer_ppo/'
               'safe_explorer_ppo_model_quadrotor_2D_stab.pt')

REGENERATED_BASELINE_NOTE = (
    'Ruling D-I: the archived quadrotor2D_rl dataset is not bit-reproducible '
    'per trajectory on this machine (chaotic PyBullet/library/hardware '
    'divergence -- the untouched reference generator does not even reproduce '
    'its OWN shipped labels here: 19/20 label agreement, 12/20 final-state '
    'agreement at atol=1e-4 on a mixed-class window; see '
    "quad_composition/rollout2d.py's module docstring for the full "
    'investigation). Any baseline-vs-composition comparison must use a '
    'baseline regenerated locally by THIS rollout core (--mode baseline, '
    'e.g. quadrotor2D_rl_regenerated/), not the archived file -- otherwise '
    'numerical drift, not controller composition, would dominate the '
    "comparison's 'lost' count, which is expected to be near zero."
)

THETA_CONVENTION_NOTE = (
    'The theta column is TRUE attitude on [-pi, pi], recovered from the '
    "drone's rotation matrix (quad_composition.rollout2d.true_theta), NOT the "
    "env's own observed pitch. PyBullet's getEulerFromQuaternion returns the "
    'branch with pitch clamped to [-pi/2, pi/2], so the observation reports a '
    'nearly-inverted drone (true theta 3.0) as nearly upright (0.1416). The '
    'archived quadrotor2D_rl dataset stores that folded value in its '
    'final-state theta column (measured: it spans exactly [-pi/2, pi/2]); '
    'these datasets do not. Initial-state theta is unaffected -- it was '
    'sampled, never observed, and is true in both. Controller 2 still '
    'receives the folded observation it was trained on; only the stored '
    'column and the supervisory decisions changed. See Finding C1 in '
    "quad_composition/rollout2d.py's module docstring."
)


def validate_labels(flip_success, ctrl2_success):
    '''(flip_success=0, ctrl2_success=1) cannot occur -- no handoff, no ctrl 2.'''
    bad = (np.asarray(flip_success) == 0) & (np.asarray(ctrl2_success) == 1)
    if bad.any():
        raise ValueError(f'impossible label combination in {int(bad.sum())} rows')


def labels_from_result(result):
    return int(result.flip_success), int(result.ctrl2_success)


def eval_states_row(init, result):
    '''14-column row for --mode flip/composite: init(6) + final(6) +
    [flip_success, ctrl2_success].
    '''
    flip, ctrl2 = labels_from_result(result)
    return list(map(float, init)) + list(map(float, result.trajectory[-1])) + [flip, ctrl2]


def baseline_eval_states_row(init, result):
    '''13-column row for --mode baseline, matching the archived
    quadrotor2D_rl eval_states.txt format exactly: init(6) + final(6) +
    [ctrl2_success].

    flip_success is never written here: it is not meaningful on the baseline
    path (ctrl1=None; Ruling D-F in quad_composition/rollout2d.py), and a
    meaningless placeholder column would make this file indistinguishable
    (by column count) from a real 14-column flip/composite file, inviting a
    downstream reader to misparse it.
    '''
    return list(map(float, init)) + list(map(float, result.trajectory[-1])) + [int(result.ctrl2_success)]


def handoff_row(init, result):
    '''13 columns: init(6) + the state at handoff_index (6) + handoff_index.

    The handoff state is [-1]*6 when no handoff fired (flip failure, or
    --mode baseline where it never can), in which case handoff_index is -1.

    handoff_index is PERSISTED (it was not, before) because without it the
    rows where the INITIAL state was already inside G1 -- handoff_index == 0,
    controller 1 never acted -- cannot be told apart from real handoffs, and
    they measure something different: a subsumption figure computed over them
    is partly a statement about the sampling grid, not about controller 1.
    This is not hypothetical here. G1's shape is attitude-only and the
    archived initial-state grid's smallest |theta| is 0.158407 rad (9.08 deg),
    which sits inside G_NOM's 0.175 rad and may well sit inside a calibrated
    G1, so a real fraction of rows can hand off at step 0.
    analyze_quad2d_composition.py reports non-subsumption both ways.
    '''
    handoff = (result.trajectory[result.handoff_index]
               if result.handoff_index >= 0 else [-1.0] * 6)
    row = list(map(float, init)) + list(map(float, handoff))
    return row + [float(result.handoff_index)]


def load_initial_states(baseline_dir):
    path = os.path.join(baseline_dir, 'eval_states.txt')
    # ndmin=2: a baseline_dir with exactly one row (e.g. a tiny --limit smoke
    # fixture) would otherwise come back 1-D and break the [:, :6] slice.
    return np.loadtxt(path, delimiter=',', ndmin=2)[:, :6]


def generate_dataset(mode, env, ctrl1, ctrl2, g1, inits, output_dir, max_steps=MAX_STEPS):
    '''Roll every init through `rollout_composite` and write trajectory files.

    Pure with respect to env/ctrl1/ctrl2/g1 beyond the calls `rollout_composite`
    itself makes on them, so callers can pass fakes -- this is how --mode
    flip/composite are exercised without a trained controller-1 checkpoint
    (none exists yet: the 1M-step SAC training run was deliberately not
    launched; see the session ledger). ctrl1=None (the --mode baseline path)
    is passed straight through to `rollout_composite`, which already handles
    it (Ruling D-F).

    Returns (rows, handoffs): rows are 14-column (flip/composite, via
    `eval_states_row`) or 13-column (baseline, via `baseline_eval_states_row`)
    plain lists ready for np.array; handoffs are always 13-column
    (`handoff_row`: init(6) + handoff state(6) + handoff_index).
    '''
    os.makedirs(os.path.join(output_dir, 'trajectories'), exist_ok=True)
    row_fn = baseline_eval_states_row if mode == 'baseline' else eval_states_row

    rows, handoffs = [], []
    for idx, init in enumerate(inits):
        res = rollout_composite(env, ctrl1, ctrl2, g1, init, max_steps=max_steps)
        if mode == 'flip' and res.handoff_index >= 0:
            # Keep only controller 1's own portion of the rollout, up to and
            # including the state that entered G1. The underlying simulated
            # rollout is identical to --mode composite's (rollout_composite
            # does not know which output mode is asking) -- only the STORED
            # trajectory differs, which is exactly the prefix invariant
            # verified in test_composition_datasets.py.
            res.trajectory = res.trajectory[:res.handoff_index + 1]
        np.savetxt(os.path.join(output_dir, 'trajectories', f'sequence_{idx}.txt'),
                   np.array(res.trajectory), delimiter=',', fmt='%.6f')
        rows.append(row_fn(init, res))
        handoffs.append(handoff_row(init, res))
    return rows, handoffs


def build_description(args, g1, rows):
    '''rows must already be a 2D numpy array (post np.array(rows) in
    write_outputs), 13-column for --mode baseline or 14-column otherwise.
    '''
    stats = {'total': int(len(rows))}
    action_space = {'normalized_rl_action_space': True, 'norm_act_scale': 0.1,
                    'twr_max': 1.100, 'alpha_max_rad_s2': 53.1}
    success_criteria = {'type': 'radius', 'threshold': GOAL_TOLERANCE}
    theta_convention = THETA_CONVENTION_NOTE

    if args.mode == 'baseline':
        stats['ctrl2_success'] = int(rows[:, 12].sum())
        return {
            'dataset_name': 'Quadrotor-2D baseline trajectories (regenerated)',
            'purpose': ('locally regenerated controller-2-alone baseline; use this, not '
                        'the archived quadrotor2D_rl dataset, wherever a comparison against '
                        'the flip/composite datasets is needed -- see '
                        'regenerated_baseline_note'),
            'regenerated_baseline_note': REGENERATED_BASELINE_NOTE,
            'controller_2': {'type': 'safe_explorer_ppo', 'model': CTRL2_MODEL},
            'action_space': action_space,
            'labels': {'ctrl2_success': '1 if controller 2 alone reached the goal ball'},
            'files': {
                'eval_states.txt': '13 columns: init(6), final(6), ctrl2_success',
                'roa_labels.txt': '7 columns: init(6), ctrl2_success',
                'handoff_states.txt': (
                    '13 columns: init(6), handoff state(6), handoff_index. Written for '
                    'format parity with the flip/composite datasets, but INERT on this '
                    'path: --mode baseline runs controller 2 alone, with no controller 1 '
                    'and no G1, so no handoff can ever fire and columns 6..12 are ALWAYS '
                    '-1 in every row. Nothing downstream should read them from a baseline '
                    'directory.'),
                'trajectories/sequence_<i>.txt': 'full state trajectory, dataset order',
            },
            'theta_convention': theta_convention,
            'success_criteria': success_criteria,
            'statistics': stats,
        }

    stats['flip_success'] = int(rows[:, 12].sum())
    stats['ctrl2_success'] = int(rows[:, 13].sum())
    return {
        'dataset_name': f'Quadrotor-2D {args.mode} trajectories',
        'purpose': ('EVALUATION ONLY' if args.mode == 'composite'
                    else 'controller 1 alone, truncated at first G1 entry'),
        'regenerated_baseline_note': REGENERATED_BASELINE_NOTE,
        'g1': g1.to_dict(),
        'controller_1': {'type': 'sac', 'model': args.flip_model, 'objective': 'attitude-only'},
        'controller_2': {'type': 'safe_explorer_ppo', 'model': CTRL2_MODEL},
        'handoff': {'operator': 'sequential latch on first entry into G1'},
        'action_space': action_space,
        'labels': {
            'flip_success': '1 if controller 1 reached G1',
            'ctrl2_success': '1 if the composite reached the goal ball',
            'note': '(flip_success=0, ctrl2_success=1) cannot occur'},
        'files': {
            'eval_states.txt': '14 columns: init(6), final(6), flip_success, ctrl2_success',
            'roa_labels.txt': '8 columns: init(6), flip_success, ctrl2_success',
            'handoff_states.txt': (
                '13 columns: init(6), handoff state(6), handoff_index. handoff_index is '
                '-1 when no handoff fired (columns 6..12 are then all -1), 0 when the '
                'INITIAL state was already inside G1 (controller 1 never acted), and > 0 '
                'for a real handoff at that trajectory row. Separating index 0 from '
                'index > 0 matters: subsumption measured over index-0 rows is partly a '
                'statement about the initial-state grid rather than about controller 1.'),
            'trajectories/sequence_<i>.txt': (
                'full state trajectory, dataset order; --mode flip truncates it at (and '
                'including) the handoff state'),
        },
        'theta_convention': theta_convention,
        'success_criteria': success_criteria,
        'statistics': stats,
    }


def write_outputs(args, g1, rows, handoffs):
    rows = np.array(rows)
    if args.mode != 'baseline':
        validate_labels(rows[:, 12], rows[:, 13])

    os.makedirs(args.output_dir, exist_ok=True)
    np.savetxt(os.path.join(args.output_dir, 'eval_states.txt'), rows,
               delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(args.output_dir, 'roa_labels.txt'),
               np.column_stack([rows[:, :6], rows[:, 12:]]), delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(args.output_dir, 'handoff_states.txt'),
               np.array(handoffs), delimiter=',', fmt='%.6f')

    with open(os.path.join(args.output_dir, 'dataset_description.json'), 'w') as fh:
        json.dump(build_description(args, g1, rows), fh, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['flip', 'composite', 'baseline'], required=True)
    parser.add_argument('--flip_model', default=None,
                        help='SAC checkpoint for controller 1. Required for --mode '
                             'flip/composite; must be omitted for --mode baseline, which '
                             'never constructs controller 1.')
    parser.add_argument('--g1', default='models/quad2d_flip/g1.json',
                        help='G1Region JSON from calibrate_quad2d_g1.py. Required for '
                             '--mode flip/composite; unused for --mode baseline.')
    parser.add_argument('--baseline_dir', required=True,
                        help='Directory whose eval_states.txt supplies the shared initial '
                             'states (spec D7) for every mode -- typically the archived '
                             'quadrotor2D_rl directory.')
    parser.add_argument('--output_dir', required=True,
                        help='e.g. quadrotor2D_flip, quadrotor2D_flip_to_rl, or '
                             'quadrotor2D_rl_regenerated for --mode baseline.')
    parser.add_argument('--limit', type=int, default=None)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode in ('flip', 'composite') and not args.flip_model:
        parser.error(f'--flip_model is required for --mode {args.mode}')
    return args


def main():
    args = parse_args()

    # NFS temp dirs intermittently hang; use /tmp explicitly and best-effort
    # cleanup rather than TemporaryDirectory's strict (and occasionally
    # failing) teardown -- matches calibrate_quad2d_g1.py's convention.
    tmp = tempfile.mkdtemp(dir='/tmp', prefix='quad2d_composition_')
    env = ctrl1 = ctrl2 = None
    g1 = None
    try:
        env, ctrl2 = make_env_and_ctrl2(CTRL2_MODEL, tmp)
        if args.mode != 'baseline':
            with open(args.g1) as fh:
                g1 = G1Region.from_dict(json.load(fh)['g1'])
            ctrl1 = load_ctrl1(args.flip_model, env, tmp)

        inits = load_initial_states(args.baseline_dir)
        if args.limit:
            inits = inits[:args.limit]

        rows, handoffs = generate_dataset(args.mode, env, ctrl1, ctrl2, g1, inits,
                                          args.output_dir)
        write_outputs(args, g1, rows, handoffs)
    finally:
        for obj in (ctrl1, ctrl2, env):
            if obj is not None:
                obj.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
