'''Torque-noise collection path (docs/superpowers/specs/2026-08-06-...).

The load-bearing properties are that the noise reaches the plant through the
action channel, and that a rollout stops at the first state inside the box --
so `terminal state in the box` and `label 1` are the same statement, in both
directions.
'''
import importlib.util
import os

import numpy as np
import pytest

SPEC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'generate_inverted_pendulum_trajectories.py')
_spec = importlib.util.spec_from_file_location('pendulum_generator', SPEC_PATH)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _env_and_ctrl(tau):
    cfg = {'ctrl_freq': 100, 'pyb_freq': 100, 'episode_len_sec': 6,
           'max_steps': 500, 'noise': None, 'invariant': False,
           'torque_noise': tau}
    env_func = gen.make_env_func(cfg)
    return env_func(), gen.make_controller('lqr', env_func)


def test_torque_noise_reaches_the_action_channel():
    '''Not the observation or dynamics channel -- the whole point of the fix.'''
    env, _ = _env_and_ctrl(0.15)
    try:
        assert 'action' in env.disturbances
        assert 'dynamics' not in env.disturbances
    finally:
        env.close()


def test_env_l2_termination_is_disabled_under_the_box_rule():
    '''Otherwise the env would cut trajectories on the criterion we replaced.'''
    env, _ = _env_and_ctrl(0.0)
    try:
        assert env.goal_threshold == 0.0
    finally:
        env.close()


@pytest.mark.parametrize('start', [[0.2, 0.0], [-0.3, 0.4], [0.05, -0.2]])
def test_successful_trajectory_ends_inside_the_box(start):
    '''The entry-cut invariant: a label-1 trajectory's LAST state is in the box.'''
    env, ctrl = _env_and_ctrl(0.0)
    try:
        traj, success, _ = gen.run_trajectory(env, ctrl, start, 500, seed=7, box_rule=True)
    finally:
        env.close()
    assert success, 'start chosen to succeed without noise'
    assert np.all(np.abs(np.array(traj[-1])) < gen.BOX_TOL)


def test_rollout_stops_at_first_entry_and_stores_that_state():
    """No dwell: the rollout ends at the first in-box state, which is stored last.

    Checked against the uncut rollout of the same seeded trajectory, so it fails
    if the cut lands off by one -- the bug the entry-cut arithmetic invites.
    """
    start, seed, horizon = [0.2, 0.0], 7, 500
    env, ctrl = _env_and_ctrl(0.0)
    try:
        cut, success, _ = gen.run_trajectory(env, ctrl, start, horizon, seed=seed, box_rule=True)
        full, _, _ = gen.run_trajectory(env, ctrl, start, horizon, seed=seed, box_rule=False)
    finally:
        env.close()
    assert success
    full = np.array(full)
    entry = len(cut) - 1
    assert np.array_equal(np.array(cut), full[:entry + 1]), 'cut is a prefix of the full rollout'
    assert np.all(np.abs(full[entry]) < gen.BOX_TOL), 'the stored last state is inside the box'
    # And it is the FIRST such state: nothing earlier qualified.
    earlier = np.all(np.abs(full[:entry]) < gen.BOX_TOL, axis=1)
    assert not earlier.any(), 'an earlier state was already inside the box'


def test_failure_can_never_end_inside_the_box():
    """The invariant the no-dwell rule buys: label is a function of the terminal state.

    Under the previous 10-step dwell this was false -- a rollout could visit the
    box without holding it and be stored ending inside it with label 0 (9,863 of
    100,000 trajectories at tau=0.50).
    """
    env, ctrl = _env_and_ctrl(0.30)
    try:
        for k in range(40):
            traj, success, _ = gen.run_trajectory(env, ctrl, [2.6, 5.4], 300,
                                                  seed=900 + k, box_rule=True)
            inbox = bool(np.all(np.abs(np.array(traj[-1])) < gen.BOX_TOL))
            assert inbox == success, (
                f'seed {900 + k}: terminal in box {inbox} but label {success}')
    finally:
        env.close()


def test_deterministic_torque_run_is_reproducible():
    '''tau=0 is a real dataset in the sweep, not a special case -- it must repeat.'''
    out = []
    for _ in range(2):
        env, ctrl = _env_and_ctrl(0.0)
        try:
            traj, _, _ = gen.run_trajectory(env, ctrl, [0.4, -0.3], 300, seed=11, box_rule=True)
        finally:
            env.close()
        out.append(np.array(traj))
    assert np.array_equal(out[0], out[1])


def test_same_seed_couples_noise_across_tau_levels():
    '''rollout_seed omits tau on purpose: common random numbers across levels.'''
    assert gen.rollout_seed(42, gen.EVAL_SPLIT_ID, 17, 3) == \
        gen.rollout_seed(42, gen.EVAL_SPLIT_ID, 17, 3)
    assert gen.rollout_seed(42, gen.EVAL_SPLIT_ID, 17, 3) != \
        gen.rollout_seed(42, gen.EVAL_SPLIT_ID, 17, 4)


def _fake_shard(tmp_path, lo, hi, n_cells=4, succ=1):
    p = gen.shard_path(str(tmp_path), lo, hi)
    np.savez(p, successes=np.full(n_cells, succ * (hi - lo), np.int32),
             trials=np.full(n_cells, hi - lo, np.int32),
             batch_lo=np.int64(lo), batch_hi=np.int64(hi))
    return p


def test_merge_refuses_a_gap_in_the_batch_range(tmp_path):
    '''A gap silently under-reports `trials`, and the result looks fine after.'''
    for lo, hi in [(0, 3), (6, 9)]:
        _fake_shard(tmp_path, lo, hi)
    with pytest.raises(ValueError, match='not contiguous'):
        gen.merge_eval_shards(str(tmp_path), np.zeros((4, 2)), np.zeros(2), np.zeros(2))


def test_merge_refuses_an_overlap_in_the_batch_range(tmp_path):
    '''An overlap double-counts those batches into successes AND trials.'''
    for lo, hi in [(0, 3), (1, 4), (3, 6)]:
        _fake_shard(tmp_path, lo, hi)
    with pytest.raises(ValueError, match='not contiguous'):
        gen.merge_eval_shards(str(tmp_path), np.zeros((4, 2)), np.zeros(2), np.zeros(2))


def test_merge_refuses_when_no_shards_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        gen.merge_eval_shards(str(tmp_path), np.zeros((4, 2)), np.zeros(2), np.zeros(2))


def test_shard_batch_index_is_global_so_draws_do_not_move(tmp_path):
    '''The property that makes sharding sound: a cell's seed depends on the
    GLOBAL batch number, never on which shard happened to run it.'''
    assert gen.rollout_seed(42, gen.EVAL_SPLIT_ID, 100, 7) == \
        gen.rollout_seed(42, gen.EVAL_SPLIT_ID, 100, 7)
    # Distinct batches must not collide, or shards would duplicate rollouts.
    seeds = {gen.rollout_seed(42, gen.EVAL_SPLIT_ID, 100, b) for b in range(50)}
    assert len(seeds) == 50


def test_output_dir_family_is_separate_from_the_preset_levels():
    '''tau_0.10 must not land beside `high`; they are not comparable.'''
    d = gen.default_output_dir('lqr', None, 0.1)
    assert d.endswith(os.path.join('noisy_torque', 'pendulum', 'lqr', 'tau_0.10'))
    assert gen.default_output_dir('lqr', None, 0.0).endswith('tau_0.00')
    # The preset path is untouched.
    assert gen.default_output_dir('lqr', 'control_proportional_med').endswith(
        os.path.join('noisy', 'pendulum', 'lqr', 'med'))


def test_each_disturbance_gets_its_own_rng_stream():
    """Disturbances must not draw from env.np_random directly.

    Sharing it makes every other draw on that generator -- initial-state
    randomisation, inertial-property randomisation -- depend on whether noise
    happens to be configured and how many samples it consumed, so two runs with
    the same seed differ in their STARTING conditions. That breaks the purity
    rollout_seed exists to guarantee.

    Note this changes the noise stream: datasets collected before this fix do
    not replay against it. The shipped tau_* datasets record the producing
    commit in dataset_description.json['provenance'] for exactly that reason.
    """
    env, _ = _env_and_ctrl(0.15)
    try:
        env.reset(seed=4242)
        d = env.disturbances['action'].disturbances[0]
        assert d.np_random is not env.np_random, 'disturbance shares the env generator'
        assert d.np_random.bit_generator.seed_seq.spawn_key != (), 'not a spawned child'
    finally:
        env.close()


def test_disturbance_stream_is_a_pure_function_of_the_env_seed():
    """Same seed -> same noise, so a resumed run draws what an uninterrupted one would."""
    draws = []
    for _ in range(2):
        env, _ = _env_and_ctrl(0.15)
        try:
            env.reset(seed=99)
            d = env.disturbances['action'].disturbances[0]
            draws.append(d.np_random.uniform(-1, 1, size=4))
        finally:
            env.close()
    assert np.array_equal(draws[0], draws[1])
