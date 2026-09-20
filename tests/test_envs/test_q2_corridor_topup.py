'''The top-up window that repairs shards collected with the unsound shortcut.

The property that matters: original k0-N window plus the k1-N top-up must equal
what one uninterrupted no-shortcut run at N trials would have produced. That
holds because rollout_seed is a pure function of (base, split, state index,
trial), so trial k of a state draws the same noise whichever process rolls it.
'''
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


topup = _load('q2_corridor_topup')

LEVEL, AMBIENT, MODEL, TRIALS = 0.08, 0.06, 'sine+ambient', 20
LO, HI = 0, 4


def _write_first_window(path, used, hits, starts=None, ambient=AMBIENT, trial_lo=0):
    n = len(used)
    starts = np.zeros((n, 6)) if starts is None else starts
    np.savez(path, starts=starts, hits=np.asarray(hits, dtype=np.int32),
             trials_used=np.asarray(used, dtype=np.int32),
             det_labels=np.zeros(n, dtype=np.int64), lo=LO, hi=LO + n,
             f_max=LEVEL, model=MODEL, trial_lo=trial_lo, trials=TRIALS,
             ambient=ambient)


def test_topup_refuses_a_window_that_is_not_the_first(tmp_path):
    p = str(tmp_path / 'in.npz')
    _write_first_window(p, used=[1, 20], hits=[1, 9], trial_lo=1)
    with pytest.raises(ValueError, match='trial_lo is 1'):
        topup.topup(p, str(tmp_path / 'out.npz'), 20260817)


def test_topup_refuses_a_run_whose_shortcut_was_sound(tmp_path):
    '''ambient == 0 means every disturbance term is altitude-gated, so the
    shortcut held and there is nothing to repair.'''
    p = str(tmp_path / 'in.npz')
    _write_first_window(p, used=[1, 20], hits=[1, 9], ambient=0.0)
    with pytest.raises(ValueError, match='shortcut was sound'):
        topup.topup(p, str(tmp_path / 'out.npz'), 20260817)


def test_topup_refuses_when_starts_disagree_with_the_grid(tmp_path):
    '''Cheap guard: the shard names its state range, so check the range still
    means the same states before spending an hour rolling against it.'''
    p = str(tmp_path / 'in.npz')
    _write_first_window(p, used=[1, 20], hits=[1, 9], starts=np.full((2, 6), 99.0))
    with pytest.raises(ValueError, match='do not match grid_states'):
        topup.topup(p, str(tmp_path / 'out.npz'), 20260817)


def test_topup_of_a_shard_with_nothing_shortcut_is_all_zeros(tmp_path):
    p, q = str(tmp_path / 'in.npz'), str(tmp_path / 'out.npz')
    _write_first_window(p, used=[20, 20, 20], hits=[3, 20, 0])
    n, flights = topup.topup(p, q, 20260817)
    assert (n, flights) == (0, 0)
    d = np.load(q)
    assert d['trials_used'].tolist() == [0, 0, 0]
    assert d['hits'].tolist() == [0, 0, 0]
    # The window still records which slice it covers, so the reducer can merge
    # it like any other.
    assert int(d['trial_lo']) == 1 and int(d['trials']) == TRIALS


def test_summed_windows_reach_the_full_trial_count(tmp_path):
    '''The reducer sums windows; the sum must land on `trials` for every state,
    whichever path a state took.'''
    p, q = str(tmp_path / 'in.npz'), str(tmp_path / 'out.npz')
    used = np.array([1, 20, 1, 20])
    _write_first_window(p, used=used, hits=[1, 7, 0, 20],
                        starts=topup.grid_states(LO, HI)[0])
    topup.topup(p, q, 20260817)
    add = np.load(q)['trials_used']
    assert add.tolist() == [19, 0, 19, 0]
    assert (used + add).tolist() == [TRIALS] * 4


def test_topped_up_shard_equals_an_uninterrupted_no_shortcut_run(tmp_path):
    '''The load-bearing one. Roll a state's trials 0..19 directly, then check
    the shard-plus-top-up sum reproduces it exactly.'''
    starts, _ = topup.grid_states(LO, HI)
    i = 0
    env, ctrl = topup.build(LEVEL, model=MODEL, ambient=AMBIENT)
    try:
        direct = [int(topup.roll(env, ctrl, starts[i],
                                 topup.rollout_seed(20260817, topup.EVAL_SPLIT_ID, LO + i, k))[0])
                  for k in range(TRIALS)]
    finally:
        env.close()

    # A shard that shortcut state i: it kept trial 0 only.
    p, q = str(tmp_path / 'in.npz'), str(tmp_path / 'out.npz')
    used = np.array([1] + [TRIALS] * (len(starts) - 1))
    hits = np.array([direct[0]] + [0] * (len(starts) - 1))
    _write_first_window(p, used=used, hits=hits, starts=starts)
    topup.topup(p, q, 20260817)

    d = np.load(q)
    assert hits[i] + d['hits'][i] == sum(direct)
    assert used[i] + d['trials_used'][i] == TRIALS
