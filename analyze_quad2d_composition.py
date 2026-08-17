#!/usr/bin/env python3
'''Measure the primary result: is G1 subsumed by RoA2?

non_subsumption = 1 - P(ctrl2_success = 1 | flip_success = 1), measured over
REAL handoffs.  The claim under test is that it is bounded away from both 0
(G1 subsumed, composition trivially sound) and 1 (G1 disjoint, composition
useless).

RULING D-I -- --baseline_dir MUST be a regenerated baseline, never the
shipped dataset
--------------------------------------------------------------------------
An earlier version of this plan pointed --baseline_dir at the archived/
shipped quadrotor2D_rl dataset directly. That is wrong and this script
refuses it (validate_baseline_dir, called first thing in main()).

Why: direct measurement (task-2-report.md "Fix round 2";
quad_composition/rollout2d.py's module docstring) showed the archived
dataset is not bit-reproducible per trajectory on this machine -- running
the UNTOUCHED reference generator against its OWN shipped eval_states.txt
(rows 4102..4121) reproduces only 19/20 labels and 12/20 final states at
atol=1e-4. quad_composition/rollout2d.py's own rollout core matches that
same reference generator exactly (20/20 at atol=1e-9) -- the rollout logic
is right, the shipped file simply is not reproducible in this environment
(chaotic PyBullet/library/hardware divergence).

Comparing the composite dataset against the shipped file would therefore
attribute a chunk of label disagreement (order 5-10%, per the 19/20
measurement) to numerical artifacts rather than to controller behaviour.
non_subsumption's 'lost' count (see composed_gain) is expected to be near
zero -- exactly the regime where that contamination would dominate the
number people scrutinise most.

So: --baseline_dir must point at a `--mode baseline` output directory
produced by generate_quadrotor_2d_composition.py (a LOCAL regeneration of
controller 2 alone, through the same rollout core the composite dataset
went through), never at the archived quadrotor2D_rl directory.
validate_baseline_dir enforces this by reading dataset_description.json and
refusing anything that is not recognizably a `--mode baseline` output --
the shipped dataset (different dataset_name, no regenerated_baseline_note
key) and flip/composite outputs (have g1/controller_1 keys) are both
rejected with a clear error, not silently accepted.

Column counts differ between the two input files -- do not assume a common
width:
  composite (--mode composite): 14 columns -- init(6), final(6),
      flip_success, ctrl2_success. Labels at indices 12 and 13.
  baseline  (--mode baseline):  13 columns -- init(6), final(6),
      ctrl2_success. Label at index 12. There is no flip_success column:
      with no controller 1 there is no handoff.

BOTH input directories are validated by dataset_description.json, not by
column count. --composite_dir gets the same fail-closed treatment as
--baseline_dir (validate_composite_dir) because a `--mode flip` output is
ALSO 14 columns, in the same order: it would parse silently and produce a
plausible but wrong headline number. Flip-only trajectories stop at the
handoff, so their ctrl2_success is 0 in every row and non-subsumption would
read exactly 1.0 -- one of the two conclusions this experiment exists to
distinguish.

TWO NON-SUBSUMPTION FIGURES ARE REPORTED, not one (Finding I6). A row with
handoff_index == 0 was ALREADY inside G1 at its initial state, so controller
1 never acted; measuring subsumption over those rows is partly a statement
about the initial-state grid, not about the composition. The grid's smallest
|theta| is 0.158407 rad (9.08 deg), inside G_NOM's 0.175 rad, so this is a
real fraction of rows rather than an edge case. handoff_index comes from
column 12 of the composite directory's handoff_states.txt, which is required.
'''

import argparse
import json
import math
import os

import numpy as np

from generate_quadrotor_2d_composition import validate_labels

BASELINE_DATASET_NAME = 'Quadrotor-2D baseline trajectories (regenerated)'
COMPOSITE_DATASET_NAME = 'Quadrotor-2D composite trajectories'


# Two-sided 95% normal quantile, for the Wilson interval below.
Z_95 = 1.959963984540054


def non_subsumption(flip_success, ctrl2_success, z=Z_95):
    '''(point estimate, lo, hi) of 1 - P(ctrl2 succeeds | handoff fired).

    The interval is a WILSON SCORE interval, not a bootstrap. This quantity is
    a plain Bernoulli proportion over the handoff rows, so a closed form is
    both exact enough and free; the bootstrap this replaced allocated an
    (n_boot, n_handoffs) index array, which at the real scale (10,000 draws
    over ~245,000 handoffs) is ~20 GB and would have OOMed at the final step
    of a multi-week pipeline. It is O(1) memory now, whatever the sample size.

    Wilson rather than the textbook normal (Wald) interval because the
    interesting regimes here are exactly the ones Wald handles worst: the
    experiment's two failure modes are p near 0 (G1 subsumed by RoA2) and p
    near 1 (G1 disjoint from it), where Wald's interval is badly miscentered
    and can even run outside [0, 1]. Wilson stays inside [0, 1] and remains
    sensible at k = 0 and k = n.
    '''
    flip = np.asarray(flip_success).astype(bool)
    ctrl2 = np.asarray(ctrl2_success).astype(bool)
    handed = ctrl2[flip]
    n = int(handed.size)
    if n == 0:
        raise ValueError('no handoffs: cannot measure non-subsumption')

    point = 1.0 - float(handed.mean())
    denom = 1.0 + z * z / n
    center = (point + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(point * (1.0 - point) / n + z * z / (4 * n * n))
    return point, max(0.0, center - half), min(1.0, center + half)


def composed_gain(baseline_labels, composite_labels):
    '''Paired comparison over shared initial states.'''
    base = np.asarray(baseline_labels).astype(bool)
    comp = np.asarray(composite_labels).astype(bool)
    if base.shape != comp.shape:
        raise ValueError('paired comparison needs identical initial states')
    return {
        'baseline_rate': float(base.mean()),
        'composed_rate': float(comp.mean()),
        'won': int((comp & ~base).sum()),
        'lost': int((~comp & base).sum()),
    }


def validate_baseline_dir(baseline_dir):
    '''Refuse --baseline_dir unless it is recognizably a `--mode baseline`
    output (RULING D-I -- see this module's docstring for the full
    investigation). Returns the parsed dataset_description.json on success.

    This is deliberately the FIRST thing main() does: the archived/shipped
    quadrotor2D_rl directory and a --mode flip/composite output directory
    both have an eval_states.txt that np.loadtxt will happily parse, so a
    column-count or NaN check alone would not catch either mistake. Reading
    dataset_description.json's self-identification is the only reliable
    signal.
    '''
    desc_path = os.path.join(baseline_dir, 'dataset_description.json')
    if not os.path.isfile(desc_path):
        raise ValueError(
            f'--baseline_dir {baseline_dir!r} has no dataset_description.json -- cannot '
            'verify it is a regenerated baseline (a `--mode baseline` output of '
            'generate_quadrotor_2d_composition.py). Refusing to guess; see RULING D-I in '
            "this module's docstring."
        )
    with open(desc_path) as fh:
        desc = json.load(fh)

    dataset_name = desc.get('dataset_name')
    if dataset_name != BASELINE_DATASET_NAME:
        raise ValueError(
            f'--baseline_dir {baseline_dir!r} is not a regenerated baseline '
            f"(dataset_description.json['dataset_name'] = {dataset_name!r}, expected "
            f'{BASELINE_DATASET_NAME!r}). RULING D-I: comparing against anything else -- in '
            'particular the archived/shipped quadrotor2D_rl dataset -- is WRONG: it is not '
            'per-trajectory reproducible on this machine (19/20 labels, 12/20 final states '
            "agree against its OWN shipped file; see this module's docstring). Regenerate a "
            'baseline with `generate_quadrotor_2d_composition.py --mode baseline` and point '
            '--baseline_dir at that output directory instead.'
        )
    if 'g1' in desc or 'controller_1' in desc:
        # Defense in depth: dataset_name already rules this out, but a
        # tampered/hand-edited description file could match the name while
        # still carrying these keys, which only --mode flip/composite write.
        raise ValueError(
            f'--baseline_dir {baseline_dir!r} has a dataset_description.json with a '
            "'g1'/'controller_1' key -- that means it is a --mode flip/composite output, not "
            'a --mode baseline output. Point --baseline_dir at a regenerated baseline '
            'directory instead.'
        )
    return desc


def validate_composite_dir(composite_dir):
    '''Refuse --composite_dir unless it is a `--mode composite` output.

    A column-count check is not enough, and this is not a theoretical gap: a
    `--mode flip` output is ALSO 14 columns with flip_success/ctrl2_success in
    the same places, so it parses silently and yields a plausible-looking
    headline number that means something else entirely. Flip-only rollouts are
    truncated at the handoff, so their ctrl2_success is 0 in every row and
    non-subsumption would come out at exactly 1.0 -- "G1 barely intersects
    RoA2", which is one of the two results this experiment is set up to
    distinguish. Mirrors validate_baseline_dir; returns the parsed
    dataset_description.json on success.
    '''
    desc_path = os.path.join(composite_dir, 'dataset_description.json')
    if not os.path.isfile(desc_path):
        raise ValueError(
            f'--composite_dir {composite_dir!r} has no dataset_description.json -- cannot '
            'verify it is a `--mode composite` output of '
            'generate_quadrotor_2d_composition.py. Refusing to guess: a `--mode flip` '
            'output has the same 14-column shape and would parse silently.'
        )
    with open(desc_path) as fh:
        desc = json.load(fh)

    dataset_name = desc.get('dataset_name')
    if dataset_name != COMPOSITE_DATASET_NAME:
        raise ValueError(
            f'--composite_dir {composite_dir!r} is not a composite dataset '
            f"(dataset_description.json['dataset_name'] = {dataset_name!r}, expected "
            f'{COMPOSITE_DATASET_NAME!r}). A --mode flip output is also 14 columns and '
            'would parse silently into a wrong headline number -- its trajectories stop at '
            'the handoff, so ctrl2_success is 0 everywhere and non-subsumption would read '
            '1.0. Point --composite_dir at a `--mode composite` output directory.'
        )
    return desc


def load_handoff_indices(composite_dir, n_expected):
    '''handoff_index per row, from the composite dataset's handoff_states.txt.

    Required, not optional (Finding I6): without it, rows whose INITIAL state
    was already inside G1 -- handoff_index == 0, controller 1 never acted --
    are indistinguishable from real handoffs, and they answer a different
    question. The archived initial-state grid's smallest |theta| is 0.158407
    rad (9.08 deg), inside G_NOM's 0.175 rad, so index-0 rows are expected to
    be a real fraction of the dataset rather than a curiosity. Failing loudly
    on an old 12-column file beats silently reporting one number as if it were
    both.
    '''
    path = os.path.join(composite_dir, 'handoff_states.txt')
    if not os.path.isfile(path):
        raise ValueError(
            f'{path} is missing. It carries handoff_index (column 12), which is needed to '
            'separate real handoffs from rows whose initial state was already inside G1. '
            'Regenerate the composite dataset with the current '
            'generate_quadrotor_2d_composition.py.'
        )
    arr = np.loadtxt(path, delimiter=',', ndmin=2)
    if arr.shape[1] != 13:
        raise ValueError(
            f'{path} has {arr.shape[1]} columns, expected 13 (init(6), handoff state(6), '
            'handoff_index). A 12-column file predates handoff_index being persisted and '
            'cannot distinguish a real handoff from one at step 0; regenerate the composite '
            'dataset.'
        )
    if len(arr) != n_expected:
        raise ValueError(
            f'{path} has {len(arr)} rows but eval_states.txt has {n_expected} -- they are '
            'row-aligned by construction, so this directory is inconsistent.'
        )
    return arr[:, 12].astype(int)


def assert_handoff_indices_agree_with_labels(handoff_index, flip_success):
    '''flip_success is defined as handoff_index >= 0 (rollout2d.RolloutResult),
    so a disagreement means the two files came from different runs.
    '''
    expected = (np.asarray(handoff_index) >= 0)
    actual = np.asarray(flip_success).astype(bool)
    bad = expected != actual
    if bad.any():
        raise ValueError(
            f'handoff_states.txt and eval_states.txt disagree on {int(bad.sum())} rows '
            f'(first at row {int(np.flatnonzero(bad)[0])}): flip_success must equal '
            '(handoff_index >= 0). The two files are not from the same run.'
        )


def load_eval_states(path, expected_cols):
    '''Load eval_states.txt and enforce its expected column count.

    The composite (14 cols) and baseline (13 cols) files are NOT
    interchangeable -- wiring the wrong file to the wrong flag would
    silently misalign every downstream column index (e.g. reading
    ctrl2_success out of what is actually a final-state column). Fail loudly
    instead.
    '''
    arr = np.loadtxt(path, delimiter=',', ndmin=2)
    if arr.shape[1] != expected_cols:
        raise ValueError(
            f'{path} has {arr.shape[1]} columns, expected {expected_cols}. The composite '
            'dataset (--mode composite) is 14 columns (init(6), final(6), flip_success, '
            'ctrl2_success); the regenerated baseline (--mode baseline) is 13 columns '
            '(init(6), final(6), ctrl2_success) -- there is no flip_success column, because '
            'with no controller 1 there is no handoff. Check --composite_dir/--baseline_dir '
            'are not swapped.'
        )
    return arr


def assert_paired_initial_states(base_init, comp_init, atol=1e-5):
    '''composed_gain is only a valid PAIRED comparison if both datasets were
    rolled out from the SAME initial states, row for row (spec D7). Equal
    length is not sufficient evidence of that: two same-sized datasets drawn
    from different --limit windows or different source files would pass a
    length check and silently produce a meaningless won/lost count.
    '''
    if base_init.shape != comp_init.shape:
        raise ValueError(
            f'baseline and composite initial-state blocks have different shapes '
            f'({base_init.shape} vs {comp_init.shape}) -- composed_gain requires a paired '
            'comparison over IDENTICAL initial states (spec D7).'
        )
    mismatched = ~np.all(np.isclose(base_init, comp_init, atol=atol), axis=1)
    if mismatched.any():
        bad_rows = np.flatnonzero(mismatched)
        raise ValueError(
            f'baseline and composite initial states differ in {mismatched.sum()} of '
            f'{len(base_init)} rows (first mismatch at row {int(bad_rows[0])}) -- '
            'composed_gain requires a paired comparison over IDENTICAL initial states '
            '(spec D7). Regenerate --baseline_dir and --composite_dir from the same source '
            '--baseline_dir/--limit so their row order matches.'
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--composite_dir', required=True,
                        help='--mode composite output of generate_quadrotor_2d_composition.py '
                             '(14-column eval_states.txt plus a 13-column '
                             'handoff_states.txt). Validated automatically; a --mode flip '
                             'output has the same column count and is refused rather than '
                             'silently misread.')
    parser.add_argument('--baseline_dir', required=True,
                        help='--mode baseline output of generate_quadrotor_2d_composition.py '
                             '(13-column eval_states.txt). MUST NOT be the archived/shipped '
                             'quadrotor2D_rl dataset -- see RULING D-I in this module '
                             'docstring. Validated automatically; non-baseline directories '
                             'are refused.')
    parser.add_argument('--output', default='results/quad2d_composition.json')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    validate_composite_dir(args.composite_dir)
    validate_baseline_dir(args.baseline_dir)

    comp = load_eval_states(os.path.join(args.composite_dir, 'eval_states.txt'), 14)
    base = load_eval_states(os.path.join(args.baseline_dir, 'eval_states.txt'), 13)

    validate_labels(comp[:, 12], comp[:, 13])

    handoff_index = load_handoff_indices(args.composite_dir, len(comp))
    assert_handoff_indices_agree_with_labels(handoff_index, comp[:, 12])

    n = min(len(comp), len(base))
    # non_subsumption is a property of the composite dataset alone (it never
    # touches baseline) -- compute it over the FULL composite array, not a
    # subset shrunk to match a possibly-shorter baseline (e.g. a smaller
    # --limit smoke run). Only composed_gain needs the paired/truncated rows.
    point, lo, hi = non_subsumption(comp[:, 12], comp[:, 13])

    # Finding I6: report the same quantity over REAL handoffs only. A row with
    # handoff_index == 0 started inside G1, so controller 1 never acted and the
    # rollout is just controller 2 from an initial state that happened to
    # satisfy G1 -- informative about the initial-state grid, not about the
    # composition. Both figures are reported; neither is silently substituted
    # for the other.
    real = handoff_index > 0
    n_real = int(real.sum())
    n_at_zero = int((handoff_index == 0).sum())
    if n_real:
        point_r, lo_r, hi_r = non_subsumption(real, comp[:, 13])
        real_result = {'point': point_r, 'ci95': [lo_r, hi_r], 'n_handoffs': n_real}
    else:
        point_r = lo_r = hi_r = None
        real_result = {'point': None, 'ci95': None, 'n_handoffs': 0,
                       'note': 'no handoff had handoff_index > 0'}

    if len(comp) != len(base):
        print(f'NOTE: composite has {len(comp)} rows, baseline has {len(base)} -- '
              f'composed_gain (a paired comparison) uses the shared first {n}.')
    paired_comp, paired_base = comp[:n], base[:n]
    assert_paired_initial_states(paired_base[:, :6], paired_comp[:, :6])
    gain = composed_gain(paired_base[:, 12], paired_comp[:, 13])

    result = {
        'non_subsumption': {'point': point, 'ci95': [lo, hi],
                            'n_handoffs': int(comp[:, 12].sum()),
                            'scope': 'all handoffs, including handoff_index == 0'},
        'non_subsumption_real_handoffs': dict(
            real_result, scope='handoff_index > 0 only (controller 1 actually acted)'),
        'composed_gain': gain,
        'handoff_rate': float(comp[:, 12].mean()),
        'n_handoffs_at_index_zero': n_at_zero,
        'n_paired': int(n),
        'ci_method': 'wilson_score_95',
    }
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump(result, fh, indent=2)

    print(f'non-subsumption (all)  : {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]'
          f'  over {int(comp[:, 12].sum())} handoffs')
    if point_r is None:
        print('non-subsumption (real) : n/a -- no handoff had handoff_index > 0; every '
              'handoff row started inside G1, so controller 1 never acted.')
    else:
        print(f'non-subsumption (real) : {point_r:.4f}  95% CI [{lo_r:.4f}, {hi_r:.4f}]'
              f'  over {n_real} handoffs with handoff_index > 0')
    print(f'  ({n_at_zero} handoffs were at index 0: the initial state was already inside '
          'G1, so controller 1 never acted)')
    print(f"baseline        : {gain['baseline_rate']:.4f}")
    print(f"composed        : {gain['composed_rate']:.4f}"
          f"  (+{gain['won']} won, -{gain['lost']} lost)")
    headline = point if point_r is None else point_r
    if headline < 0.02:
        print('WARNING: G1 is effectively subsumed by RoA2 -- the primary claim fails.')
    if headline > 0.98:
        print('WARNING: G1 barely intersects RoA2 -- handoffs almost never succeed.')

    return result


if __name__ == '__main__':
    main()
