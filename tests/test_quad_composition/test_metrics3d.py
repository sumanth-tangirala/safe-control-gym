'''The primary metric, ported to 3D: what fraction of G1 falls outside RoA2.

non_subsumption() itself is unchanged from analyze_quad2d_composition.py (a
pure function of two boolean arrays, indifferent to dimensionality), so the
five pinned tests below mirror test_metrics.py's non_subsumption tests
verbatim. The rest of this file exercises the 3D-specific helpers this
script adds: the by-initial-tilt bucket assignment and the mechanism-check
quintile split, both pure functions with no PyBullet/env dependency, so they
run fast and need no fixtures.
'''
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# non_subsumption -- mirrors test_metrics.py's pinned tests.
# ---------------------------------------------------------------------------

def test_non_subsumption_is_measured_only_over_actual_handoffs():
    '''Rows where the flip never reached G1 say nothing about RoA2.'''
    from analyze_quad3d_composition import non_subsumption
    flip = np.array([1, 1, 1, 1, 0, 0])
    ctrl2 = np.array([1, 1, 0, 0, 0, 0])
    point, lo, hi = non_subsumption(flip, ctrl2)
    assert point == pytest.approx(0.5)      # 2 of 4 handoffs failed
    assert lo <= point <= hi


def test_non_subsumption_is_zero_when_g1_is_subsumed():
    from analyze_quad3d_composition import non_subsumption
    flip = np.array([1, 1, 1])
    ctrl2 = np.array([1, 1, 1])
    point, _, _ = non_subsumption(flip, ctrl2)
    assert point == pytest.approx(0.0)


def test_non_subsumption_is_one_when_g1_is_disjoint_from_roa2():
    '''The other pole this experiment is designed to distinguish: every
    handoff fires but ctrl2 never succeeds afterward.
    '''
    from analyze_quad3d_composition import non_subsumption
    flip = np.array([1, 1, 1, 0])
    ctrl2 = np.array([0, 0, 0, 0])
    point, lo, hi = non_subsumption(flip, ctrl2)
    assert point == pytest.approx(1.0)
    assert 0.0 <= lo <= hi <= 1.0


def test_non_subsumption_needs_at_least_one_handoff():
    from analyze_quad3d_composition import non_subsumption
    with pytest.raises(ValueError, match='no handoffs'):
        non_subsumption(np.array([0, 0]), np.array([0, 0]))


def test_non_subsumption_scales_to_the_real_sample_size_without_a_huge_allocation():
    '''Same finding as the 2D script: a bootstrap over ~245,000 handoffs
    allocated ~20 GB and OOMed. The Wilson score interval is closed-form and
    O(1) in memory.
    '''
    import tracemalloc

    from analyze_quad3d_composition import non_subsumption

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
    assert hi - lo < 0.01


def test_non_subsumption_interval_stays_inside_the_unit_interval_at_the_extremes():
    '''p ~ 0 (G1 subsumed) and p ~ 1 (G1 disjoint) are exactly where a Wald
    interval misbehaves and can run outside [0, 1]. Wilson must not.
    '''
    from analyze_quad3d_composition import non_subsumption

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


# ---------------------------------------------------------------------------
# safe_non_subsumption -- the bucket-table wrapper that must not raise on an
# empty subset.
# ---------------------------------------------------------------------------

def test_safe_non_subsumption_returns_none_point_for_empty_subset():
    from analyze_quad3d_composition import safe_non_subsumption
    result = safe_non_subsumption(np.zeros(5, dtype=bool), np.zeros(5, dtype=bool))
    assert result == {'point': None, 'ci95': None, 'n_handoffs': 0}


def test_safe_non_subsumption_matches_non_subsumption_on_a_nonempty_subset():
    from analyze_quad3d_composition import non_subsumption, safe_non_subsumption
    flip = np.array([1, 1, 0, 0], dtype=bool)
    ctrl2 = np.array([1, 0, 0, 0], dtype=bool)
    expected = non_subsumption(flip, ctrl2)
    result = safe_non_subsumption(flip, ctrl2)
    assert result['point'] == pytest.approx(expected[0])
    assert result['ci95'] == pytest.approx([expected[1], expected[2]])
    assert result['n_handoffs'] == 2


# ---------------------------------------------------------------------------
# assign_tilt_bucket
# ---------------------------------------------------------------------------

def test_assign_tilt_bucket_covers_the_six_thirty_degree_bands():
    from analyze_quad3d_composition import assign_tilt_bucket
    tilts = np.array([0.0, 29.9, 30.0, 89.9, 90.0, 179.9, 180.0])
    idx = assign_tilt_bucket(tilts)
    assert idx.tolist() == [0, 0, 1, 2, 3, 5, 5]


def test_assign_tilt_bucket_is_half_open_except_the_last_bucket():
    '''30.0 belongs to [30, 60), not [0, 30); 180.0 belongs to the closed
    final bucket since no bucket starts above it.
    '''
    from analyze_quad3d_composition import assign_tilt_bucket
    assert assign_tilt_bucket(np.array([30.0]))[0] == 1
    assert assign_tilt_bucket(np.array([60.0]))[0] == 2
    assert assign_tilt_bucket(np.array([180.0]))[0] == 5


# ---------------------------------------------------------------------------
# quintile_success / quintile_spread
# ---------------------------------------------------------------------------

def test_quintile_success_splits_into_five_equal_count_groups():
    from analyze_quad3d_composition import quintile_success
    values = np.arange(10, dtype=float)          # 10 rows -> 5 groups of 2
    success = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=bool)
    quintiles = quintile_success(values, success, n_bins=5)
    assert len(quintiles) == 5
    assert [q['n'] for q in quintiles] == [2, 2, 2, 2, 2]
    assert quintiles[0]['success_rate'] == pytest.approx(0.0)
    assert quintiles[-1]['success_rate'] == pytest.approx(1.0)
    assert quintiles[0]['value_lo'] == pytest.approx(0.0)
    assert quintiles[0]['value_hi'] == pytest.approx(1.0)


def test_quintile_success_handles_a_count_not_divisible_by_five():
    from analyze_quad3d_composition import quintile_success
    values = np.arange(7, dtype=float)
    success = np.zeros(7, dtype=bool)
    quintiles = quintile_success(values, success, n_bins=5)
    assert sum(q['n'] for q in quintiles) == 7
    assert len(quintiles) == 5


def test_quintile_success_rejects_mismatched_shapes():
    from analyze_quad3d_composition import quintile_success
    with pytest.raises(ValueError):
        quintile_success(np.zeros(3), np.zeros(4, dtype=bool))


def test_quintile_spread_is_max_minus_min_success_rate():
    from analyze_quad3d_composition import quintile_spread
    quintiles = [{'success_rate': 0.1}, {'success_rate': 0.9}, {'success_rate': 0.5}]
    assert quintile_spread(quintiles) == pytest.approx(0.8)


def test_quintile_spread_ignores_empty_bins():
    from analyze_quad3d_composition import quintile_spread
    quintiles = [{'success_rate': None}, {'success_rate': 0.2}, {'success_rate': 0.6}]
    assert quintile_spread(quintiles) == pytest.approx(0.4)


def test_quintile_spread_is_none_when_every_bin_is_empty():
    from analyze_quad3d_composition import quintile_spread
    assert quintile_spread([{'success_rate': None}]) is None


# ---------------------------------------------------------------------------
# snapshot_checkpoint -- exercised against a real /tmp file, never the shared
# model checkpoint (this test must not touch models/quad3d_ctrl1_selected.pt,
# which another process may be writing concurrently, and must not use
# tmp_path -- TMPDIR is the NFS mount that hangs).
# ---------------------------------------------------------------------------

@pytest.fixture()
def scratch_dir():
    import shutil
    import tempfile
    d = tempfile.mkdtemp(dir='/tmp', prefix='quad3d_metrics3d_test_')
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_snapshot_checkpoint_records_matching_stat_and_copies_bytes(scratch_dir):
    from analyze_quad3d_composition import snapshot_checkpoint
    src = os.path.join(scratch_dir, 'src.pt')
    with open(src, 'wb') as fh:
        fh.write(b'fake checkpoint bytes')
    dst_root = os.path.join(scratch_dir, 'dst')
    os.makedirs(dst_root)

    frozen_path, info = snapshot_checkpoint(src, dst_root)

    assert os.path.isfile(frozen_path)
    with open(frozen_path, 'rb') as fh:
        assert fh.read() == b'fake checkpoint bytes'
    assert info['path'] == os.path.abspath(src)
    assert info['size'] == os.path.getsize(src)
    assert info['warning'] is None


def test_build_parser_rejects_fewer_than_1500_states():
    '''The measurement requires >= 1500 initial states; main() checks this
    directly (not through argparse) so the error message is specific.
    '''
    from analyze_quad3d_composition import build_parser, main
    args = build_parser().parse_args(['--num_states', '10'])
    assert args.num_states == 10
    with pytest.raises(ValueError, match='1500'):
        main(['--num_states', '10'])
