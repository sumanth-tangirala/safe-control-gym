'''The corridor reducer: merges collector shards into the publication layout.

Uses synthetic shards shaped exactly like q2_corridor_collect.py's real output
(same keys, same filename grammar) so these tests do not depend on a real
collection run landing. See q2_corridor_collect.py's shard_eval for why
summing hits/trials_used across k-windows is correct: only the trial_lo=0
window ever sets a non-reachable state's trials_used, so later windows
contribute 0 to it and summing recovers the single-probe shortcut exactly.
'''
import importlib.util
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    'q2_corridor_reduce', os.path.join(ROOT, 'q2_corridor_reduce.py'))
q2cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q2cr)

LEVEL = 0.08
AMBIENT = 0.06
MODEL = 'sine+ambient'


def _write_eval_shard(shard_dir, shard, klo, khi, lo, hi, starts, hits, used, det,
                       trial_lo=None, level=LEVEL, ambient=AMBIENT, model=MODEL):
    trial_lo = klo if trial_lo is None else trial_lo
    amb_tag = 'none' if ambient is None else f'{ambient}'
    name = f'eval_L{level}_A{amb_tag}_k{klo}-{khi}_s{shard}.npz'
    np.savez(os.path.join(shard_dir, name),
             starts=np.asarray(starts, dtype=np.float64),
             hits=np.asarray(hits, dtype=np.int32),
             trials_used=np.asarray(used, dtype=np.int32),
             det_labels=np.asarray(det, dtype=np.int64),
             lo=lo, hi=hi, f_max=level, model=model, trial_lo=trial_lo, trials=khi,
             ambient=-1.0 if ambient is None else ambient)
    return name


def _write_train_shard(shard_dir, shard, lo, hi, starts, labels, level=LEVEL,
                        ambient=AMBIENT, model=MODEL):
    amb_tag = 'none' if ambient is None else f'{ambient}'
    name = f'train_L{level}_A{amb_tag}_k0-1_s{shard}.npz'
    starts = np.asarray(starts, dtype=np.float64)
    n = len(starts)
    states = starts.astype(np.float32)          # one "step" per trajectory: enough for the contract
    offsets = np.arange(0, n + 1, dtype=np.int64)
    seeds = np.arange(n, dtype=np.int64) + shard * 1000
    np.savez(os.path.join(shard_dir, name), states=states, offsets=offsets,
             starts=starts, labels=np.asarray(labels, dtype=np.uint8), seeds=seeds,
             lo=lo, hi=hi, f_max=level, model=model,
             ambient=-1.0 if ambient is None else ambient)
    return name


@pytest.fixture
def shard_dir(tmp_path):
    d = tmp_path / 'shards'
    d.mkdir()
    return str(d)


def _make_three_eval_shards(shard_dir, extra_window_on_shard0=True):
    '''3 shards x 2 states = 6 states. Shard 0 gets a second k-window that
    tops up its non-shortcut state and leaves its shortcut state untouched --
    the property under test.'''
    # shard 0: state 0 is a normal (reachable) state, state 1 is a shortcut.
    _write_eval_shard(shard_dir, 0, 0, 2, 0, 2,
                       starts=[[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]],
                       hits=[1, 0], used=[2, 1], det=[1, 0])
    if extra_window_on_shard0:
        # Top-up window: state 0 gains 2 more trials (1 hit); state 1 is a
        # shortcut so the collector never touches it here (hits=0, used=0).
        _write_eval_shard(shard_dir, 0, 2, 4, 0, 2,
                           starts=[[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]],
                           hits=[1, 0], used=[2, 0], det=[1, 0])
    _write_eval_shard(shard_dir, 1, 0, 2, 2, 4,
                       starts=[[2, 2, 0, 0, 0, 0], [3, 3, 0, 0, 0, 0]],
                       hits=[2, 1], used=[2, 2], det=[1, 1])
    _write_eval_shard(shard_dir, 2, 0, 2, 4, 6,
                       starts=[[4, 4, 0, 0, 0, 0], [5, 5, 0, 0, 0, 0]],
                       hits=[0, 2], used=[1, 2], det=[0, 1])


def _make_two_train_shards(shard_dir):
    _write_train_shard(shard_dir, 0, 0, 2,
                       starts=[[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], labels=[1, 0])
    _write_train_shard(shard_dir, 1, 2, 4,
                       starts=[[2, 2, 0, 0, 0, 0], [3, 3, 0, 0, 0, 0]], labels=[1, 1])


# ---------------------------------------------------------------------------
# find_shard_files / missing_shards
# ---------------------------------------------------------------------------

def test_find_shard_files_groups_windows_by_shard(shard_dir):
    _make_three_eval_shards(shard_dir)
    by_idx = q2cr.find_shard_files(shard_dir, 'eval', LEVEL, AMBIENT)
    assert set(by_idx) == {0, 1, 2}
    assert len(by_idx[0]) == 2   # two k-windows for shard 0
    assert len(by_idx[1]) == 1
    # windows are sorted by klo
    assert [w[1] for w in by_idx[0]] == [0, 2]


def test_find_shard_files_only_matches_requested_level_and_ambient(shard_dir):
    _make_three_eval_shards(shard_dir)
    _write_eval_shard(shard_dir, 0, 0, 1, 0, 2, starts=[[0] * 6], hits=[0], used=[1],
                      det=[0], level=0.0, ambient=None, model='sine')
    by_level = q2cr.find_shard_files(shard_dir, 'eval', 0.0, None)
    assert set(by_level) == {0}
    by_sharp = q2cr.find_shard_files(shard_dir, 'eval', LEVEL, AMBIENT)
    assert set(by_sharp) == {0, 1, 2}


def test_missing_shards_detects_gap(shard_dir):
    _make_three_eval_shards(shard_dir)
    by_idx = q2cr.find_shard_files(shard_dir, 'eval', LEVEL, AMBIENT)
    assert q2cr.missing_shards(by_idx, 3) == []
    assert q2cr.missing_shards(by_idx, 5) == [3, 4]


# ---------------------------------------------------------------------------
# reduce_eval: window merging
# ---------------------------------------------------------------------------

def test_reduce_eval_sums_hits_and_trials_across_windows(tmp_path, shard_dir):
    _make_three_eval_shards(shard_dir)
    by_idx = q2cr.find_shard_files(shard_dir, 'eval', LEVEL, AMBIENT)
    out_dir = str(tmp_path / 'out')
    os.makedirs(out_dir)
    stats = q2cr.reduce_eval(by_idx, LEVEL, AMBIENT, MODEL, out_dir)

    d = np.load(os.path.join(out_dir, 'eval_success_prob.npz'))
    # 6 states total, merged in shard order.
    assert len(d['starts']) == 6
    np.testing.assert_array_equal(d['successes'], [2, 0, 2, 1, 0, 2])
    np.testing.assert_array_equal(d['trials'], [4, 1, 2, 2, 1, 2])
    np.testing.assert_allclose(d['p_success'], [0.5, 0.0, 1.0, 0.5, 0.0, 1.0])

    # Spot-check p == hits / trials_used for every state.
    np.testing.assert_allclose(d['p_success'], d['successes'] / d['trials'])

    assert stats['num_states'] == 6
    assert stats['shards'] == 3
    # state 1 (index 1) is the only merged-shortcut state (used==1) besides
    # shard-2's first state, which was also never reached -> two shortcuts.
    assert stats['trials_shortcut'] == 2
    assert stats['trials_full'] == 4
    assert stats['num_windows'] == 2      # k0-2 and k2-4 both appear
    assert stats['k_total'] == 4


def test_reduce_eval_without_the_extra_window_matches_single_window_baseline(tmp_path, shard_dir):
    _make_three_eval_shards(shard_dir, extra_window_on_shard0=False)
    by_idx = q2cr.find_shard_files(shard_dir, 'eval', LEVEL, AMBIENT)
    out_dir = str(tmp_path / 'out')
    os.makedirs(out_dir)
    stats = q2cr.reduce_eval(by_idx, LEVEL, AMBIENT, MODEL, out_dir)
    d = np.load(os.path.join(out_dir, 'eval_success_prob.npz'))
    np.testing.assert_array_equal(d['trials'], [2, 1, 2, 2, 1, 2])
    assert stats['num_windows'] == 1
    assert stats['k_total'] == 2


def test_reduce_eval_writes_contract_files(tmp_path, shard_dir):
    _make_three_eval_shards(shard_dir)
    by_idx = q2cr.find_shard_files(shard_dir, 'eval', LEVEL, AMBIENT)
    out_dir = str(tmp_path / 'out')
    os.makedirs(out_dir)
    q2cr.reduce_eval(by_idx, LEVEL, AMBIENT, MODEL, out_dir)
    for name in ('eval_success_prob.npz', 'roa_labels.txt', 'eval_states.txt',
                 'cal_set.txt', 'test_set.txt'):
        assert os.path.exists(os.path.join(out_dir, name)), name
    with open(os.path.join(out_dir, 'roa_labels.txt')) as fh:
        lines = fh.read().strip().split('\n')
    assert len(lines) == 6
    assert lines[0].count(',') == 6   # 6 state cols + p_success


# ---------------------------------------------------------------------------
# reduce_train
# ---------------------------------------------------------------------------

def test_reduce_train_concatenates_shards(tmp_path, shard_dir):
    _make_two_train_shards(shard_dir)
    by_idx = q2cr.find_shard_files(shard_dir, 'train', LEVEL, AMBIENT)
    out_dir = str(tmp_path / 'out')
    os.makedirs(out_dir)
    stats = q2cr.reduce_train(by_idx, LEVEL, AMBIENT, MODEL, out_dir)
    d = np.load(os.path.join(out_dir, 'train.npz'))
    assert len(d['labels']) == 4
    assert len(d['starts']) == 4
    assert d['offsets'][-1] == len(d['states'])
    assert stats['num_trajectories'] == 4
    assert stats['success_count'] == 3
    assert os.path.exists(os.path.join(out_dir, 'train_test_splits', 'shuffled_indices_0.txt'))
    assert os.path.exists(os.path.join(out_dir, 'train_test_splits', 'shuffled_labels_0.txt'))


def test_reduce_train_rejects_a_parameter_mismatch(tmp_path, shard_dir):
    _make_two_train_shards(shard_dir)
    by_idx = q2cr.find_shard_files(shard_dir, 'train', LEVEL, AMBIENT)
    out_dir = str(tmp_path / 'out')
    os.makedirs(out_dir)
    with pytest.raises(ValueError):
        q2cr.reduce_train(by_idx, 0.13, AMBIENT, MODEL, out_dir)   # wrong level


# ---------------------------------------------------------------------------
# main(): CLI, completeness gate, description contents
# ---------------------------------------------------------------------------

def test_main_refuses_on_missing_shard_without_allow_partial(tmp_path, shard_dir, capsys):
    _make_three_eval_shards(shard_dir)   # only shards 0,1,2 -- n_eval below asks for 5
    _make_two_train_shards(shard_dir)
    out_dir = str(tmp_path / 'out')
    with pytest.raises(SystemExit) as exc:
        q2cr.main(['--level', str(LEVEL), '--ambient', str(AMBIENT), '--model', MODEL,
                  '--n_train', '2', '--n_eval', '5',
                   '--shard_dir', shard_dir, '--out_dir', out_dir])
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert '3' in err and '4' in err   # the missing indices, listed
    assert not os.path.exists(os.path.join(out_dir, 'dataset_description.json'))


def test_main_allow_partial_writes_available_states(tmp_path, shard_dir):
    _make_three_eval_shards(shard_dir)
    _make_two_train_shards(shard_dir)
    out_dir = str(tmp_path / 'out')
    q2cr.main(['--level', str(LEVEL), '--ambient', str(AMBIENT), '--model', MODEL,
              '--n_train', '2', '--n_eval', '5', '--allow_partial',
               '--shard_dir', shard_dir, '--out_dir', out_dir])
    d = np.load(os.path.join(out_dir, 'eval_success_prob.npz'))
    assert len(d['starts']) == 6   # only the 3 present shards' states


def test_main_produces_the_q2_reduce_contract(tmp_path, shard_dir):
    _make_three_eval_shards(shard_dir)
    _make_two_train_shards(shard_dir)
    out_dir = str(tmp_path / 'out')
    q2cr.main(['--level', str(LEVEL), '--ambient', str(AMBIENT), '--model', MODEL,
              '--n_train', '2', '--n_eval', '3',
               '--shard_dir', shard_dir, '--out_dir', out_dir])
    expected = {'train.npz', 'eval_success_prob.npz', 'eval_states.txt', 'roa_labels.txt',
                'cal_set.txt', 'test_set.txt', 'dataset_description.json',
                'train_description.json', 'eval_description.json', 'train_test_splits'}
    assert expected <= set(os.listdir(out_dir))


def test_description_contains_noise_model_and_corridor_blocks(tmp_path, shard_dir):
    _make_three_eval_shards(shard_dir)
    _make_two_train_shards(shard_dir)
    out_dir = str(tmp_path / 'out')
    q2cr.main(['--level', str(LEVEL), '--ambient', str(AMBIENT), '--model', MODEL,
              '--n_train', '2', '--n_eval', '3',
               '--shard_dir', shard_dir, '--out_dir', out_dir])
    with open(os.path.join(out_dir, 'dataset_description.json')) as fh:
        desc = json.load(fh)
    gp = desc['generation_parameters']
    assert gp['noise_model']['model'] == MODEL
    assert gp['noise_model']['ambient'] == pytest.approx(AMBIENT)
    c = gp['corridor']
    assert c['profile'] == 'gaussian'
    assert c['centre'] == pytest.approx(0.55)
    assert c['width'] == pytest.approx(0.12)
    assert c['band_1pct'] == [0.186, 0.914]
    assert c['draw_law'] == MODEL
    assert c['period_s'] == pytest.approx(2.0)
    assert c['ambient_std'] == pytest.approx(AMBIENT)
    assert c['skip_margin_m'] == pytest.approx(0.10)
    assert c['sigma_at_goal'] < 1e-4
    assert desc['eval_statistics']['trials_shortcut'] == 2
    assert desc['eval_statistics']['num_windows'] == 2
    assert desc['eval_statistics']['k_total'] == 4


def test_baseline_config_maps_to_level_zero_no_ambient(tmp_path, shard_dir):
    _write_eval_shard(shard_dir, 0, 0, 1, 0, 2,
                      starts=[[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]],
                      hits=[1, 0], used=[1, 1], det=[1, 0],
                      level=0.0, ambient=None, model='sine')
    _write_train_shard(shard_dir, 0, 0, 2,
                       starts=[[0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], labels=[1, 0],
                       level=0.0, ambient=None, model='sine')
    out_dir = str(tmp_path / 'out')
    q2cr.main(['--config', 'baseline', '--n_train', '1', '--n_eval', '1',
              '--shard_dir', shard_dir, '--out_dir', out_dir])
    with open(os.path.join(out_dir, 'dataset_description.json')) as fh:
        desc = json.load(fh)
    assert desc['generation_parameters']['corridor']['f_max'] == 0.0
    assert desc['generation_parameters']['noise_model']['model'] == 'sine'


def test_draw_law_reports_the_sinusoid_not_the_falsified_uniform_draw():
    '''describe() used to hardcode the per-step uniform draw for every model.

    That draw was measured degenerate and replaced by the per-rollout sinusoid
    (see AltitudeGatedSineNoise); NOISE_MODELS keeps 'uniform' only for
    reproduction. Emitting it for a sine+ambient set told the consumer the
    wrong law.
    '''
    law = q2cr.draw_law('sine+ambient')
    assert 'sinusoid' in law['distribution']
    assert 'per-rollout' in law['distribution']
    assert 'sin(' in law['formula']
    assert 'N(0, ambient_std)' in law['formula']
    # The corridor gust is held for the episode; only the ambient term redraws.
    assert 'deterministic function of (z, t)' in law['hold']
    assert 'ambient term IS redrawn every step' in law['hold']


def test_draw_law_keeps_the_uniform_model_describable_for_reproduction():
    law = q2cr.draw_law('uniform')
    assert law['distribution'] == 'uniform'
    assert law['hold'] == 'zero-order, redrawn each control step (100 Hz)'
    assert law['formula'] == 'F_x = U(0, sigma(z))'
    assert 'N(0, ambient_std)' not in law['formula']


def test_draw_law_marks_the_ambient_only_model_as_having_no_corridor():
    law = q2cr.draw_law('ambient')
    assert law['formula'] == 'F_x = N(0, ambient_std)'
    assert law['low'] is None and law['high'] is None
    assert 'two-sided' in law['applied_as']


def test_draw_law_rejects_an_unknown_model():
    with pytest.raises(ValueError, match='unknown noise model'):
        q2cr.draw_law('brownian')


def test_mechanism_block_agrees_with_the_resolved_noise_model(tmp_path, shard_dir):
    '''Regression: the same JSON must not state two different disturbance laws.'''
    _make_three_eval_shards(shard_dir)
    _make_two_train_shards(shard_dir)
    out_dir = str(tmp_path / 'out')
    q2cr.main(['--level', str(LEVEL), '--ambient', str(AMBIENT), '--model', MODEL,
               '--n_train', '2', '--n_eval', '3',
               '--shard_dir', shard_dir, '--out_dir', out_dir])
    with open(os.path.join(out_dir, 'dataset_description.json')) as fh:
        desc = json.load(fh)
    mech = desc['mechanism']
    assert desc['generation_parameters']['noise_model']['model'] == MODEL
    assert mech['distribution'] != 'uniform'
    assert 'sin(' in mech['formula']
    # mechanism and the corridor block are two views of one law.
    assert mech['formula'] == desc['generation_parameters']['corridor']['formula']
    # The ambient term is zero-mean, so the stack is no longer purely one-sided.
    assert 'two-sided' in mech['applied_as']
