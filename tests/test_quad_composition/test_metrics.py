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


def _write_dir(root, rows, desc):
    os.makedirs(root, exist_ok=True)
    np.savetxt(os.path.join(root, 'eval_states.txt'), np.array(rows, dtype=float),
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
    comp_dir = _write_dir(os.path.join(scratch_dir, 'composite'), comp_rows, COMPOSITE_DESC)
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


def test_main_refuses_shipped_style_baseline_dir(scratch_dir):
    from analyze_quad2d_composition import main
    comp_dir = _write_dir(os.path.join(scratch_dir, 'composite'), [_composite_row(0, 1, 1)],
                          COMPOSITE_DESC)
    base_dir = _write_dir(os.path.join(scratch_dir, 'shipped'), [_baseline_row(0, 1)],
                          SHIPPED_DESC)
    out_path = os.path.join(scratch_dir, 'result.json')
    with pytest.raises(ValueError, match='RULING D-I|regenerated baseline'):
        main(['--composite_dir', comp_dir, '--baseline_dir', base_dir, '--output', out_path])
    assert not os.path.exists(out_path)


def test_main_rejects_impossible_label_combination(scratch_dir):
    from analyze_quad2d_composition import main
    bad_rows = [_composite_row(0, 0, 1)]        # flip=0, ctrl2=1: impossible
    comp_dir = _write_dir(os.path.join(scratch_dir, 'composite'), bad_rows, COMPOSITE_DESC)
    base_dir = _write_dir(os.path.join(scratch_dir, 'baseline'), [_baseline_row(0, 1)],
                          BASELINE_DESC)
    out_path = os.path.join(scratch_dir, 'result.json')
    with pytest.raises(ValueError, match='impossible'):
        main(['--composite_dir', comp_dir, '--baseline_dir', base_dir, '--output', out_path])


def test_main_rejects_unpaired_initial_states(scratch_dir):
    from analyze_quad2d_composition import main
    comp_dir = _write_dir(os.path.join(scratch_dir, 'composite'),
                          [_composite_row(0, 1, 1), _composite_row(1, 1, 0)], COMPOSITE_DESC)
    base_dir = _write_dir(os.path.join(scratch_dir, 'baseline'),
                          [_baseline_row(0, 1), _baseline_row(99, 0)], BASELINE_DESC)
    out_path = os.path.join(scratch_dir, 'result.json')
    with pytest.raises(ValueError, match='IDENTICAL|identical|paired'):
        main(['--composite_dir', comp_dir, '--baseline_dir', base_dir, '--output', out_path])
