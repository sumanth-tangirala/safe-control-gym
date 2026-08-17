'''The primary metric: what fraction of G1 falls outside RoA2.

RULING D-I (generate_quadrotor_2d_composition.py's module docstring;
task-6-report.md): the archived quadrotor2D_rl dataset is not
per-trajectory reproducible on this machine (19/20 labels, 12/20 final
states agree against its OWN shipped file). --baseline_dir here MUST be a
--mode baseline output (generate_quadrotor_2d_composition.py), never the
archived/shipped dataset -- validate_baseline_dir enforces this, and most of
the tests below build small synthetic fixture directories in /tmp (never
pytest's tmp_path fixture: TMPDIR is set to the NFS mount that intermittently
hangs, see the operational constraints for this task) to exercise that
enforcement without needing a real composite dataset (no trained controller-1
checkpoint exists yet).
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

# analyze_quad2d_composition is imported function-locally in every test below
# (matching test_composition_datasets.py's convention): isort otherwise
# hoists a module-level import above the sys.path insert it depends on --
# confirmed the hard way, by pre-commit doing exactly that on a first draft
# of this file.


# ---------------------------------------------------------------------------
# Fixture scaffolding. Explicitly /tmp, never tmp_path (see module docstring)
# and never TemporaryDirectory() as a context manager (per this task's
# operational constraints).
# ---------------------------------------------------------------------------

@pytest.fixture()
def scratch_dir():
    d = tempfile.mkdtemp(dir='/tmp', prefix='quad2d_metrics_test_')
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


BASELINE_DESC = {
    'dataset_name': 'Quadrotor-2D baseline trajectories (regenerated)',
    'purpose': ('locally regenerated controller-2-alone baseline; use this, not '
                'the archived quadrotor2D_rl dataset, wherever a comparison against '
                'the flip/composite datasets is needed -- see regenerated_baseline_note'),
    'regenerated_baseline_note': 'Ruling D-I: the archived quadrotor2D_rl dataset is not '
                                 'bit-reproducible per trajectory on this machine.',
    'controller_2': {'type': 'safe_explorer_ppo', 'model': 'x.pt'},
    'labels': {'ctrl2_success': '1 if controller 2 alone reached the goal ball'},
}

COMPOSITE_DESC = {
    'dataset_name': 'Quadrotor-2D composite trajectories',
    'purpose': 'EVALUATION ONLY',
    'regenerated_baseline_note': 'Ruling D-I: the archived quadrotor2D_rl dataset is not '
                                 'bit-reproducible per trajectory on this machine.',
    'g1': {'form': 'attitude_only', 'tilt_c_rad': 0.1, 'tilt_c_deg': 5.7, 'w_c_rad_s': 0.5},
    'controller_1': {'type': 'sac', 'model': 'x.pt', 'objective': 'attitude-only'},
    'controller_2': {'type': 'safe_explorer_ppo', 'model': 'x.pt'},
    'handoff': {'operator': 'sequential latch on first entry into G1'},
}

FLIP_DESC = dict(COMPOSITE_DESC, dataset_name='Quadrotor-2D flip trajectories',
                 purpose='controller 1 alone, truncated at first G1 entry')

# The archived/shipped dataset's real dataset_description.json shape (verified
# against /common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/
# quadrotor2D_rl/dataset_description.json): no regenerated_baseline_note, a
# completely different dataset_name, no g1/controller_1 either.
SHIPPED_DESC = {
    'dataset_name': '2D Quadrotor RL Stabilization Trajectories',
    'description': 'Dataset of 2D quadrotor stabilization trajectories ...',
    'generation_parameters': {'controller': {'type': 'safe_explorer_ppo'}},
}


def _write_dir(root, rows, desc, handoffs=None):
    '''Write a fixture dataset directory. `handoffs` (13-column rows) is
    written as handoff_states.txt; composite directories need one, since
    analyze_quad2d_composition requires handoff_index to separate real
    handoffs from rows that started inside G1 (Finding I6).
    '''
    os.makedirs(root, exist_ok=True)
    np.savetxt(os.path.join(root, 'eval_states.txt'), np.array(rows, dtype=float),
               delimiter=',', fmt='%.6f')
    if handoffs is not None:
        np.savetxt(os.path.join(root, 'handoff_states.txt'), np.array(handoffs, dtype=float),
                   delimiter=',', fmt='%.6f')
    with open(os.path.join(root, 'dataset_description.json'), 'w') as fh:
        json.dump(desc, fh)
    return root


def _init(seed_offset=0.0):
    return [0.1 + seed_offset, 1.2, 0.3, 0.4, 0.5, 0.6]


def _final():
    return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def _composite_row(seed_offset, flip, ctrl2):
    return _init(seed_offset) + _final() + [flip, ctrl2]


def _baseline_row(seed_offset, ctrl2):
    return _init(seed_offset) + _final() + [ctrl2]


def _handoff_row(seed_offset, handoff_index):
    '''13 columns: init(6) + handoff state(6) + handoff_index.'''
    state = _final() if handoff_index >= 0 else [-1.0] * 6
    return _init(seed_offset) + state + [handoff_index]


def _handoffs_for(comp_rows, indices=None):
    '''Handoff rows consistent with `comp_rows`' flip_success column.

    Default: every handoff is a real one (index 5). Pass `indices` to place
    some of them at 0 (initial state already inside G1).
    '''
    out = []
    for i, row in enumerate(comp_rows):
        if indices is not None:
            idx = indices[i]
        else:
            idx = 5 if row[12] else -1
        out.append(_handoff_row(row[0] - _init()[0], idx))
    return out


# ---------------------------------------------------------------------------
# The brief's four pinned tests (Step 1), verbatim.
# ---------------------------------------------------------------------------

def test_non_subsumption_is_measured_only_over_actual_handoffs():
    '''Rows where the flip never reached G1 say nothing about RoA2.'''
    from analyze_quad2d_composition import non_subsumption
    flip = np.array([1, 1, 1, 1, 0, 0])
    ctrl2 = np.array([1, 1, 0, 0, 0, 0])
    point, lo, hi = non_subsumption(flip, ctrl2)
    assert point == pytest.approx(0.5)      # 2 of 4 handoffs failed
    assert lo <= point <= hi


def test_non_subsumption_is_zero_when_g1_is_subsumed():
    from analyze_quad2d_composition import non_subsumption
    flip = np.array([1, 1, 1])
    ctrl2 = np.array([1, 1, 1])
    point, _, _ = non_subsumption(flip, ctrl2)
    assert point == pytest.approx(0.0)


def test_non_subsumption_needs_at_least_one_handoff():
    from analyze_quad2d_composition import non_subsumption
    with pytest.raises(ValueError, match='no handoffs'):
        non_subsumption(np.array([0, 0]), np.array([0, 0]))


def test_non_subsumption_scales_to_the_real_sample_size_without_a_huge_allocation():
    '''Finding I1: the interval used to be a bootstrap, drawing an
    (n_boot, n_handoffs) index array. At this experiment's real scale --
    10,000 draws over ~245,000 handoffs -- that is ~20 GB of int64, i.e. an
    OOM at the very last step of a multi-week pipeline. The Wilson score
    interval is closed-form and O(1) in memory.

    tracemalloc is the assertion rather than a timing bound because it fails
    for the right reason: a reintroduced bootstrap would blow the peak, not
    merely be slow. The 50 MB budget is generous next to the handful of
    megabytes the boolean mask itself needs, and tiny next to 20 GB.
    '''
    import tracemalloc

    from analyze_quad2d_composition import non_subsumption

    n = 500_000
    rng = np.random.default_rng(0)
    flip = np.ones(n, dtype=bool)
    ctrl2 = rng.random(n) < 0.7        # built before tracing starts

    tracemalloc.start()
    try:
        point, lo, hi = non_subsumption(flip, ctrl2)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < 50 * 1024 * 1024, f'allocated {peak / 1e6:.0f} MB for n={n}'
    assert point == pytest.approx(1.0 - ctrl2.mean())
    assert lo <= point <= hi
    # At half a million samples the interval is narrow, and Wilson is
    # essentially centred on the point estimate.
    assert hi - lo < 0.01


def test_non_subsumption_interval_stays_inside_the_unit_interval_at_the_extremes():
    '''The two regimes this experiment is designed to distinguish are p ~ 0
    (G1 subsumed) and p ~ 1 (G1 disjoint) -- exactly where a Wald interval
    misbehaves and can run outside [0, 1]. Wilson must not.
    '''
    from analyze_quad2d_composition import non_subsumption

    all_succeeded = np.ones(40, dtype=bool)
    point, lo, hi = non_subsumption(np.ones(40, dtype=bool), all_succeeded)
    assert point == pytest.approx(0.0)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi > 0.0, 'a zero-count interval must still have width'

    none_succeeded = np.zeros(40, dtype=bool)
    point, lo, hi = non_subsumption(np.ones(40, dtype=bool), none_succeeded)
    assert point == pytest.approx(1.0)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < 1.0


def test_composed_gain_is_paired_over_shared_initial_states():
    from analyze_quad2d_composition import composed_gain
    baseline = np.array([0, 0, 1, 0])
    composite = np.array([1, 0, 1, 1])
    gain = composed_gain(baseline, composite)
    assert gain['baseline_rate'] == pytest.approx(0.25)
    assert gain['composed_rate'] == pytest.approx(0.75)
    assert gain['won'] == 2      # states the composition rescued
    assert gain['lost'] == 0     # states the composition broke


# ---------------------------------------------------------------------------
# validate_baseline_dir: RULING D-I's hard enforcement. This is the single
# easiest way for someone to silently produce a wrong headline number, so it
# gets the most tests.
# ---------------------------------------------------------------------------

def test_validate_baseline_dir_accepts_a_regenerated_baseline(scratch_dir):
    from analyze_quad2d_composition import validate_baseline_dir
    root = _write_dir(os.path.join(scratch_dir, 'base'), [_baseline_row(0, 1)], BASELINE_DESC)
    desc = validate_baseline_dir(root)
    assert desc['dataset_name'] == 'Quadrotor-2D baseline trajectories (regenerated)'


def test_validate_baseline_dir_refuses_missing_description(scratch_dir):
    from analyze_quad2d_composition import validate_baseline_dir
    root = os.path.join(scratch_dir, 'nodesc')
    os.makedirs(root)
    np.savetxt(os.path.join(root, 'eval_states.txt'), np.array([_baseline_row(0, 1)]),
               delimiter=',', fmt='%.6f')
    with pytest.raises(ValueError, match='dataset_description.json'):
        validate_baseline_dir(root)


def test_validate_baseline_dir_refuses_the_shipped_dataset(scratch_dir):
    '''The shipped/archived dataset is not a --mode baseline output (RULING D-I):
    comparing against it would let numerical drift, not controller behaviour,
    dominate the headline number.
    '''
    from analyze_quad2d_composition import validate_baseline_dir
    root = _write_dir(os.path.join(scratch_dir, 'shipped'), [_baseline_row(0, 1)], SHIPPED_DESC)
    with pytest.raises(ValueError, match='RULING D-I|regenerated baseline'):
        validate_baseline_dir(root)


def test_validate_baseline_dir_refuses_a_composite_dir(scratch_dir):
    from analyze_quad2d_composition import validate_baseline_dir
    root = _write_dir(os.path.join(scratch_dir, 'composite_as_baseline'),
                      [_composite_row(0, 1, 1)], COMPOSITE_DESC)
    with pytest.raises(ValueError, match='regenerated baseline'):
        validate_baseline_dir(root)


def test_validate_baseline_dir_refuses_a_flip_dir(scratch_dir):
    from analyze_quad2d_composition import validate_baseline_dir
    root = _write_dir(os.path.join(scratch_dir, 'flip_as_baseline'),
                      [_composite_row(0, 1, 0)], FLIP_DESC)
    with pytest.raises(ValueError, match='regenerated baseline'):
        validate_baseline_dir(root)


# ---------------------------------------------------------------------------
# validate_composite_dir: the mirror of validate_baseline_dir. A --mode flip
# output is ALSO 14 columns, so a column-count check would let it through --
# and it would yield a plausible, entirely wrong headline number.
# ---------------------------------------------------------------------------

def test_validate_composite_dir_accepts_a_composite_dataset(scratch_dir):
    from analyze_quad2d_composition import validate_composite_dir
    rows = [_composite_row(0, 1, 1)]
    root = _write_dir(os.path.join(scratch_dir, 'comp'), rows, COMPOSITE_DESC,
                      _handoffs_for(rows))
    desc = validate_composite_dir(root)
    assert desc['dataset_name'] == 'Quadrotor-2D composite trajectories'


def test_validate_composite_dir_refuses_a_flip_dir(scratch_dir):
    '''The dangerous one: a --mode flip directory has the same 14 columns in
    the same order, so it parses. Its trajectories stop at the handoff, so
    ctrl2_success is 0 in every row and non-subsumption would come out at 1.0
    -- "G1 barely intersects RoA2", a real (and wrong) conclusion.
    '''
    from analyze_quad2d_composition import validate_composite_dir
    rows = [_composite_row(0, 1, 0)]
    root = _write_dir(os.path.join(scratch_dir, 'flip_as_composite'), rows, FLIP_DESC,
                      _handoffs_for(rows))
    with pytest.raises(ValueError, match='not a composite dataset|--mode flip'):
        validate_composite_dir(root)


def test_validate_composite_dir_refuses_a_baseline_dir(scratch_dir):
    from analyze_quad2d_composition import validate_composite_dir
    root = _write_dir(os.path.join(scratch_dir, 'baseline_as_composite'),
                      [_baseline_row(0, 1)], BASELINE_DESC)
    with pytest.raises(ValueError, match='not a composite dataset'):
        validate_composite_dir(root)


def test_validate_composite_dir_refuses_missing_description(scratch_dir):
    from analyze_quad2d_composition import validate_composite_dir
    root = os.path.join(scratch_dir, 'nodesc')
    os.makedirs(root)
    np.savetxt(os.path.join(root, 'eval_states.txt'), np.array([_composite_row(0, 1, 1)]),
               delimiter=',', fmt='%.6f')
    with pytest.raises(ValueError, match='dataset_description.json'):
        validate_composite_dir(root)


def test_main_refuses_a_flip_dir_as_composite_dir(scratch_dir):
    from analyze_quad2d_composition import main
    rows = [_composite_row(0, 1, 0)]
    comp_dir = _write_dir(os.path.join(scratch_dir, 'flip'), rows, FLIP_DESC,
                          _handoffs_for(rows))
    base_dir = _write_dir(os.path.join(scratch_dir, 'baseline'), [_baseline_row(0, 1)],
                          BASELINE_DESC)
    out_path = os.path.join(scratch_dir, 'result.json')
    with pytest.raises(ValueError, match='not a composite dataset|--mode flip'):
        main(['--composite_dir', comp_dir, '--baseline_dir', base_dir, '--output', out_path])
    assert not os.path.exists(out_path)


# ---------------------------------------------------------------------------
# load_handoff_indices: Finding I6. Without handoff_index, a row whose INITIAL
# state was already inside G1 is indistinguishable from a real handoff.
# ---------------------------------------------------------------------------

def test_load_handoff_indices_reads_the_index_column(scratch_dir):
    from analyze_quad2d_composition import load_handoff_indices
    rows = [_composite_row(0, 1, 1), _composite_row(1, 1, 0), _composite_row(2, 0, 0)]
    root = _write_dir(os.path.join(scratch_dir, 'comp'), rows, COMPOSITE_DESC,
                      _handoffs_for(rows, indices=[0, 7, -1]))
    np.testing.assert_array_equal(load_handoff_indices(root, 3), [0, 7, -1])


def test_load_handoff_indices_refuses_a_missing_file(scratch_dir):
    from analyze_quad2d_composition import load_handoff_indices
    root = _write_dir(os.path.join(scratch_dir, 'comp'), [_composite_row(0, 1, 1)],
                      COMPOSITE_DESC)
    with pytest.raises(ValueError, match='handoff_states.txt is missing'):
        load_handoff_indices(root, 1)


def test_load_handoff_indices_refuses_the_old_twelve_column_format(scratch_dir):
    '''A pre-Finding-I6 dataset cannot answer the question, so it must fail
    rather than be silently reported as if every handoff were real.
    '''
    from analyze_quad2d_composition import load_handoff_indices
    root = _write_dir(os.path.join(scratch_dir, 'comp'), [_composite_row(0, 1, 1)],
                      COMPOSITE_DESC)
    np.savetxt(os.path.join(root, 'handoff_states.txt'),
               np.array([_init() + _final()]), delimiter=',', fmt='%.6f')
    with pytest.raises(ValueError, match='12 columns|expected 13'):
        load_handoff_indices(root, 1)


def test_load_handoff_indices_refuses_a_row_count_mismatch(scratch_dir):
    from analyze_quad2d_composition import load_handoff_indices
    rows = [_composite_row(0, 1, 1), _composite_row(1, 1, 0)]
    root = _write_dir(os.path.join(scratch_dir, 'comp'), rows, COMPOSITE_DESC,
                      [_handoff_row(0, 3)])
    with pytest.raises(ValueError, match='rows but eval_states.txt has'):
        load_handoff_indices(root, 2)


def test_main_reports_non_subsumption_over_all_and_over_real_handoffs(scratch_dir):
    '''Finding I6. Rows 0 and 1 hand off at index 0 -- their initial state was
    already inside G1, so controller 1 never acted -- and both succeed under
    controller 2. Rows 2 and 3 are real handoffs and both fail. Over ALL
    handoffs non-subsumption is 0.5; over REAL handoffs it is 1.0. Reporting
    only the first would badly understate the result, and reporting only the
    second would overstate it; both are recorded.
    '''
    from analyze_quad2d_composition import main

    comp_rows = [
        _composite_row(0, 1, 1),
        _composite_row(1, 1, 1),
        _composite_row(2, 1, 0),
        _composite_row(3, 1, 0),
        _composite_row(4, 0, 0),
    ]
    base_rows = [_baseline_row(i, 0) for i in range(5)]
    comp_dir = _write_dir(os.path.join(scratch_dir, 'composite'), comp_rows, COMPOSITE_DESC,
                          _handoffs_for(comp_rows, indices=[0, 0, 9, 11, -1]))
    base_dir = _write_dir(os.path.join(scratch_dir, 'baseline'), base_rows, BASELINE_DESC)
    out_path = os.path.join(scratch_dir, 'result.json')

    main(['--composite_dir', comp_dir, '--baseline_dir', base_dir, '--output', out_path])

    with open(out_path) as fh:
        result = json.load(fh)

    assert result['non_subsumption']['point'] == pytest.approx(0.5)
    assert result['non_subsumption']['n_handoffs'] == 4
    assert result['non_subsumption_real_handoffs']['point'] == pytest.approx(1.0)
    assert result['non_subsumption_real_handoffs']['n_handoffs'] == 2
    assert result['n_handoffs_at_index_zero'] == 2


def test_main_reports_no_real_handoffs_rather_than_crashing(scratch_dir):
    '''If every handoff is at index 0, the real-handoff figure is undefined --
    it must be reported as absent, not raised as "no handoffs" (which would
    kill the run even though the all-handoffs figure is perfectly valid).
    '''
    from analyze_quad2d_composition import main

    comp_rows = [_composite_row(0, 1, 1), _composite_row(1, 1, 0)]
    base_rows = [_baseline_row(0, 0), _baseline_row(1, 0)]
    comp_dir = _write_dir(os.path.join(scratch_dir, 'composite'), comp_rows, COMPOSITE_DESC,
                          _handoffs_for(comp_rows, indices=[0, 0]))
    base_dir = _write_dir(os.path.join(scratch_dir, 'baseline'), base_rows, BASELINE_DESC)
    out_path = os.path.join(scratch_dir, 'result.json')

    result = main(['--composite_dir', comp_dir, '--baseline_dir', base_dir,
                   '--output', out_path])

    assert result['non_subsumption']['point'] == pytest.approx(0.5)
    assert result['non_subsumption_real_handoffs']['point'] is None
    assert result['non_subsumption_real_handoffs']['n_handoffs'] == 0


def test_main_rejects_handoff_indices_that_disagree_with_flip_success(scratch_dir):
    '''flip_success is defined as handoff_index >= 0, so a disagreement means
    the two files came from different runs -- and every downstream figure
    would be mixing them.
    '''
    from analyze_quad2d_composition import main

    comp_rows = [_composite_row(0, 1, 1), _composite_row(1, 0, 0)]
    base_rows = [_baseline_row(0, 0), _baseline_row(1, 0)]
    comp_dir = _write_dir(os.path.join(scratch_dir, 'composite'), comp_rows, COMPOSITE_DESC,
                          _handoffs_for(comp_rows, indices=[-1, 4]))   # both inverted
    base_dir = _write_dir(os.path.join(scratch_dir, 'baseline'), base_rows, BASELINE_DESC)
    out_path = os.path.join(scratch_dir, 'result.json')

    with pytest.raises(ValueError, match='disagree'):
        main(['--composite_dir', comp_dir, '--baseline_dir', base_dir, '--output', out_path])


# ---------------------------------------------------------------------------
# load_eval_states: the two datasets have DIFFERENT column counts. Wiring the
# wrong file to the wrong flag must fail loudly, not silently misparse.
# ---------------------------------------------------------------------------

def test_load_eval_states_accepts_correct_column_count(scratch_dir):
    from analyze_quad2d_composition import load_eval_states
    root = _write_dir(os.path.join(scratch_dir, 'comp'), [_composite_row(0, 1, 1)],
                      COMPOSITE_DESC)
    arr = load_eval_states(os.path.join(root, 'eval_states.txt'), 14)
    assert arr.shape == (1, 14)


def test_load_eval_states_rejects_wrong_column_count(scratch_dir):
    from analyze_quad2d_composition import load_eval_states

    # A baseline (13-col) file handed to the composite (14-col) reader.
    root = _write_dir(os.path.join(scratch_dir, 'base'), [_baseline_row(0, 1)], BASELINE_DESC)
    with pytest.raises(ValueError, match='13.*14|14.*13'):
        load_eval_states(os.path.join(root, 'eval_states.txt'), 14)


# ---------------------------------------------------------------------------
# assert_paired_initial_states: equal length is not equal identity.
# ---------------------------------------------------------------------------

def test_assert_paired_initial_states_accepts_matching_rows():
    from analyze_quad2d_composition import assert_paired_initial_states
    base_init = np.array([_init(0), _init(1)])
    comp_init = np.array([_init(0), _init(1)])
    assert_paired_initial_states(base_init, comp_init)   # must not raise


def test_assert_paired_initial_states_rejects_mismatched_values():
    from analyze_quad2d_composition import assert_paired_initial_states
    base_init = np.array([_init(0), _init(1)])
    comp_init = np.array([_init(0), _init(99)])           # row 1 does not match
    with pytest.raises(ValueError, match='IDENTICAL|identical|paired'):
        assert_paired_initial_states(base_init, comp_init)


def test_assert_paired_initial_states_rejects_mismatched_shape():
    '''Equal-length arrays can still be unpaired -- but unequal length is the
    most obvious case a naive length check would (correctly) also catch. The
    real bar this must clear is the mismatched-values test above.
    '''
    from analyze_quad2d_composition import assert_paired_initial_states
    base_init = np.array([_init(0), _init(1), _init(2)])
    comp_init = np.array([_init(0), _init(1)])
    with pytest.raises(ValueError):
        assert_paired_initial_states(base_init, comp_init)


# ---------------------------------------------------------------------------
# main(): end-to-end over synthetic fixtures only (no real dataset exists).
# ---------------------------------------------------------------------------

def test_main_end_to_end_with_synthetic_fixtures(scratch_dir):
    from analyze_quad2d_composition import main

    comp_rows = [
        _composite_row(0, 1, 1),
        _composite_row(1, 1, 1),
        _composite_row(2, 1, 0),
        _composite_row(3, 1, 0),
        _composite_row(4, 0, 0),
    ]
    base_rows = [
        _baseline_row(0, 0),
        _baseline_row(1, 1),
        _baseline_row(2, 1),
        _baseline_row(3, 0),
        _baseline_row(4, 0),
    ]
    comp_dir = _write_dir(os.path.join(scratch_dir, 'composite'), comp_rows, COMPOSITE_DESC,
                          _handoffs_for(comp_rows))
    base_dir = _write_dir(os.path.join(scratch_dir, 'baseline'), base_rows, BASELINE_DESC)
    out_path = os.path.join(scratch_dir, 'result.json')

    main(['--composite_dir', comp_dir, '--baseline_dir', base_dir, '--output', out_path])

    with open(out_path) as fh:
        result = json.load(fh)

    # 4 handoffs (flip=1 rows 0-3), of which 2 succeeded under ctrl2 -> 0.5.
    assert result['non_subsumption']['point'] == pytest.approx(0.5)
    assert result['non_subsumption']['n_handoffs'] == 4
    assert result['composed_gain']['baseline_rate'] == pytest.approx(0.4)
    assert result['composed_gain']['composed_rate'] == pytest.approx(0.4)


def test_main_non_subsumption_uses_full_composite_even_when_baseline_is_shorter(scratch_dir):
    '''non_subsumption is a property of the composite dataset alone. If the
    baseline happens to be shorter (e.g. a smaller smoke-test --limit), that
    must not shrink the sample non_subsumption is measured over -- only
    composed_gain (which needs pairing) should be restricted to the shared
    prefix.
    '''
    from analyze_quad2d_composition import main

    comp_rows = [
        _composite_row(0, 1, 1),
        _composite_row(1, 1, 0),
        _composite_row(2, 1, 0),
        _composite_row(3, 1, 1),
        _composite_row(4, 1, 0),
    ]
    # Baseline is a strict prefix -- only 2 rows -- of composite's init states.
    base_rows = [_baseline_row(0, 1), _baseline_row(1, 0)]

    comp_dir = _write_dir(os.path.join(scratch_dir, 'composite'), comp_rows, COMPOSITE_DESC,
                          _handoffs_for(comp_rows))
    base_dir = _write_dir(os.path.join(scratch_dir, 'baseline'), base_rows, BASELINE_DESC)
    out_path = os.path.join(scratch_dir, 'result.json')

    main(['--composite_dir', comp_dir, '--baseline_dir', base_dir, '--output', out_path])

    with open(out_path) as fh:
        result = json.load(fh)

    # All 5 composite rows are handoffs; non_subsumption must see all 5, not
    # just the 2 that overlap with the shorter baseline.
    assert result['non_subsumption']['n_handoffs'] == 5
    assert result['non_subsumption']['point'] == pytest.approx(3 / 5)
    assert result['n_paired'] == 2


def test_main_refuses_shipped_style_baseline_dir(scratch_dir):
    from analyze_quad2d_composition import main
    comp_rows = [_composite_row(0, 1, 1)]
    comp_dir = _write_dir(os.path.join(scratch_dir, 'composite'), comp_rows,
                          COMPOSITE_DESC, _handoffs_for(comp_rows))
    base_dir = _write_dir(os.path.join(scratch_dir, 'shipped'), [_baseline_row(0, 1)],
                          SHIPPED_DESC)
    out_path = os.path.join(scratch_dir, 'result.json')
    with pytest.raises(ValueError, match='RULING D-I|regenerated baseline'):
        main(['--composite_dir', comp_dir, '--baseline_dir', base_dir, '--output', out_path])
    assert not os.path.exists(out_path)


def test_main_rejects_impossible_label_combination(scratch_dir):
    from analyze_quad2d_composition import main
    bad_rows = [_composite_row(0, 0, 1)]        # flip=0, ctrl2=1: impossible
    comp_dir = _write_dir(os.path.join(scratch_dir, 'composite'), bad_rows, COMPOSITE_DESC,
                          _handoffs_for(bad_rows))
    base_dir = _write_dir(os.path.join(scratch_dir, 'baseline'), [_baseline_row(0, 1)],
                          BASELINE_DESC)
    out_path = os.path.join(scratch_dir, 'result.json')
    with pytest.raises(ValueError, match='impossible'):
        main(['--composite_dir', comp_dir, '--baseline_dir', base_dir, '--output', out_path])


def test_main_rejects_unpaired_initial_states(scratch_dir):
    from analyze_quad2d_composition import main
    comp_rows = [_composite_row(0, 1, 1), _composite_row(1, 1, 0)]
    comp_dir = _write_dir(os.path.join(scratch_dir, 'composite'), comp_rows, COMPOSITE_DESC,
                          _handoffs_for(comp_rows))
    base_dir = _write_dir(os.path.join(scratch_dir, 'baseline'),
                          [_baseline_row(0, 1), _baseline_row(99, 0)], BASELINE_DESC)
    out_path = os.path.join(scratch_dir, 'result.json')
    with pytest.raises(ValueError, match='IDENTICAL|identical|paired'):
        main(['--composite_dir', comp_dir, '--baseline_dir', base_dir, '--output', out_path])
