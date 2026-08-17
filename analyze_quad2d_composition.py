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
'''

import argparse
import json
import math
import os

import numpy as np

from generate_quadrotor_2d_composition import validate_labels

BASELINE_DATASET_NAME = 'Quadrotor-2D baseline trajectories (regenerated)'


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
                             '(14-column eval_states.txt).')
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

    validate_baseline_dir(args.baseline_dir)

    comp = load_eval_states(os.path.join(args.composite_dir, 'eval_states.txt'), 14)
    base = load_eval_states(os.path.join(args.baseline_dir, 'eval_states.txt'), 13)

    validate_labels(comp[:, 12], comp[:, 13])

    n = min(len(comp), len(base))
    # non_subsumption is a property of the composite dataset alone (it never
    # touches baseline) -- compute it over the FULL composite array, not a
    # subset shrunk to match a possibly-shorter baseline (e.g. a smaller
    # --limit smoke run). Only composed_gain needs the paired/truncated rows.
    point, lo, hi = non_subsumption(comp[:, 12], comp[:, 13])

    if len(comp) != len(base):
        print(f'NOTE: composite has {len(comp)} rows, baseline has {len(base)} -- '
              f'composed_gain (a paired comparison) uses the shared first {n}.')
    paired_comp, paired_base = comp[:n], base[:n]
    assert_paired_initial_states(paired_base[:, :6], paired_comp[:, :6])
    gain = composed_gain(paired_base[:, 12], paired_comp[:, 13])

    result = {
        'non_subsumption': {'point': point, 'ci95': [lo, hi],
                            'n_handoffs': int(comp[:, 12].sum())},
        'composed_gain': gain,
        'handoff_rate': float(comp[:, 12].mean()),
        'n_paired': int(n),
    }
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump(result, fh, indent=2)

    print(f'non-subsumption : {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]'
          f'  over {int(comp[:, 12].sum())} handoffs')
    print(f"baseline        : {gain['baseline_rate']:.4f}")
    print(f"composed        : {gain['composed_rate']:.4f}"
          f"  (+{gain['won']} won, -{gain['lost']} lost)")
    if point < 0.02:
        print('WARNING: G1 is effectively subsumed by RoA2 -- the primary claim fails.')
    if point > 0.98:
        print('WARNING: G1 barely intersects RoA2 -- handoffs almost never succeed.')

    return result


if __name__ == '__main__':
    main()
