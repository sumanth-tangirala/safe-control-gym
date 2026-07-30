'''Evaluation must be reproducible, and must compare like with like.

A verdict is only worth quoting if re-running produces it again, and only
meaningful if both controllers faced the same problem. Neither is obvious from
reading the code: the policy runs on the shaped env and the baseline on the bare
one, so they share initial states only because they share per-episode seeds --
a property that a wrapper change could break silently.
'''
import json
import os
import subprocess
import sys

import pytest

from safe_control_gym.experiments.eval_policy import baseline_id, episode_seeds

REPO = os.path.join(os.path.dirname(__file__), '..', '..')

TRAIN = ['--kv_overrides', 'sb3_config.total_timesteps=256',
         'sb3_config.save_freq=256', 'sb3_config.eval_freq=256',
         'sb3_config.n_eval_episodes=1']


@pytest.fixture(scope='module')
def trained_run(tmp_path_factory):
    '''One short cartpole run, shared by every test here.'''
    out = tmp_path_factory.mktemp('logs')
    result = subprocess.run(
        [sys.executable, '-m', 'safe_control_gym.experiments.train_sb3',
         '--env_id', 'cartpole_stabilization', '--algo', 'sac', '--seed', '1',
         '--output_dir', str(out)] + TRAIN,
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    runs = sorted((out / 'sac').glob('cartpole_stabilization_*'))
    assert len(runs) == 1
    return runs[0]


def _evaluate(run, seed, n_episodes=4, out_name='eval.json', eval_bounds=None):
    destination = os.path.join(str(run), out_name)
    command = [sys.executable, '-m', 'safe_control_gym.experiments.eval_policy',
               '--run', str(run), '--n_episodes', str(n_episodes), '--seed', str(seed),
               '--out', destination]
    if eval_bounds:
        command += ['--eval_bounds', eval_bounds]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    with open(destination) as handle:
        return json.load(handle)


def test_evaluation_is_deterministic(trained_run):
    '''Same seed, same numbers -- otherwise a verdict says nothing.'''
    first = _evaluate(trained_run, seed=0, out_name='eval_a.json')
    second = _evaluate(trained_run, seed=0, out_name='eval_b.json')
    assert first['policy'] == second['policy']
    assert first['baseline'] == second['baseline']
    assert first['verdict'] == second['verdict']


def test_report_is_complete(trained_run):
    '''Absolute numbers for both controllers must always be present.

    The LQR-relative bar is vacuous where LQR is itself weak, and the agreed
    mitigation is disclosure: a report carrying only a verdict would hide
    exactly the case the design is worried about.
    '''
    report = _evaluate(trained_run, seed=0)
    assert report['verdict'] in ('PASS', 'FAIL', 'NON_DISCRIMINATING')
    assert report['baseline_id'] == 'lqr'
    assert report['goal_tolerance'] > 0
    for block in ('policy', 'baseline'):
        for key in ('success_rate', 'mean_return', 'mean_episode_length',
                    'out_of_bounds_rate', 'constraint_violation_rate',
                    'terminated_frac', 'truncated_frac'):
            assert key in report[block], f'{block} is missing {key}'


def test_saturated_baseline_is_not_a_pass(trained_run):
    '''A baseline at 0 or 1 ranks nothing, and must not be reported as PASS.

    This is not hypothetical: cartpole's default init box is +/-0.05 on all four
    dimensions, and LQR solves all of it. Measured over 30 episodes there, LQR
    scores exactly 1.0000 -- so the bar becomes "within 0.05 of perfect" and any
    adequate policy clears it. Over cartpole's collection box the same LQR
    scores 0.0667, which does rank things.

    The fixture trains on the default box precisely so this path is exercised.
    '''
    report = _evaluate(trained_run, seed=0, n_episodes=6, out_name='eval_sat.json')
    if report['baseline']['success_rate'] in (0.0, 1.0):
        assert report['verdict'] == 'NON_DISCRIMINATING', (
            'a saturated baseline was reported as '
            f"{report['verdict']}, which reads as a real result")
    else:
        assert report['verdict'] in ('PASS', 'FAIL')


def test_eval_bounds_override_the_training_region(trained_run):
    '''--eval_bounds must actually move where episodes start.

    The point of the flag is that a policy is judged on the region the dataset
    covers rather than the one it trained in. If the override silently failed,
    every number would look plausible and be answering the easier question.
    '''
    bounds = os.path.join(REPO, 'configs/collection/cartpole.yaml')
    report = _evaluate(trained_run, seed=0, n_episodes=6,
                       out_name='eval_bounds.json', eval_bounds=bounds)
    assert report['eval_bounds'] == bounds
    assert report['reference_success'] == pytest.approx(0.1797)

    # The reference is a reach number -- every shipped dataset was collected
    # before terminate_on_goal existed, when entering the goal ball ended the
    # episode. This run is cartpole_stabilization, which must hold position
    # there instead, so the two are not the same problem and the comparison is
    # deliberately withheld rather than made misleadingly.
    assert report['reference_task'] == 'reach'
    assert report['terminate_on_goal'] is False
    assert report['beats_reference'] is None, (
        'a stabilization run was compared against a reach reference')

    # cartpole's default box is +/-0.05; the collection box is +/-6 in x and
    # +/-pi in theta. Starts must reflect that, which also proves the
    # env_attributes raising x_threshold from 2.4 and theta from pi/2 took
    # effect -- without them these states terminate at step one.
    default_report = _evaluate(trained_run, seed=0, n_episodes=6,
                               out_name='eval_default.json')
    assert report['policy']['mean_episode_length'] != \
        default_report['policy']['mean_episode_length']


def test_reach_run_is_compared_against_the_reach_reference(tmp_path):
    '''The mirror of the above: matching tasks, so the comparison IS made.

    Without this, "beats_reference is None" would pass whether the withholding
    logic worked or the field were simply never populated.
    '''
    result = subprocess.run(
        [sys.executable, '-m', 'safe_control_gym.experiments.train_sb3',
         '--env_id', 'cartpole_reach', '--algo', 'sac', '--seed', '1',
         '--output_dir', str(tmp_path)] + TRAIN,
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    run = sorted((tmp_path / 'sac').glob('cartpole_reach_*'))[0]

    bounds = os.path.join(REPO, 'configs/collection/cartpole.yaml')
    report = _evaluate(run, seed=0, n_episodes=6, eval_bounds=bounds)
    assert report['terminate_on_goal'] is True
    assert report['reference_task'] == 'reach'
    assert report['beats_reference'] is not None, (
        'a reach run was not compared against the reach reference')


def test_skip_baseline_yields_no_verdict(trained_run):
    '''Without a baseline there is nothing to compare against, and the report
    must say so rather than defaulting to PASS.'''
    destination = os.path.join(str(trained_run), 'eval_nobase.json')
    result = subprocess.run(
        [sys.executable, '-m', 'safe_control_gym.experiments.eval_policy',
         '--run', str(trained_run), '--n_episodes', '2', '--seed', '0',
         '--skip_baseline', '--out', destination],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    with open(destination) as handle:
        report = json.load(handle)
    assert report['verdict'] == 'NO_BASELINE'
    assert report['baseline_id'] is None


def test_evaluates_a_wrapped_env(tmp_path):
    '''The pendulum trains wrapped, and wrappers do not forward what eval reads.

    AttributeForwardingMixin's allowlist contains neither `goal_threshold` nor
    `TASK_INFO`, so resolving the goal tolerance on the wrapper raised
    AttributeError for the one system whose config uses AngleObservation and
    ActionRepeat. Every other test here uses cartpole, which is unwrapped, so
    the suite was green while the pendulum path could not run at all.
    '''
    result = subprocess.run(
        [sys.executable, '-m', 'safe_control_gym.experiments.train_sb3',
         '--env_id', 'inverted_pendulum_stabilization', '--algo', 'sac', '--seed', '1',
         '--output_dir', str(tmp_path),
         '--overrides', 'configs/sb3/inverted_pendulum_stabilization_sac.yaml'] + TRAIN,
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    run = sorted((tmp_path / 'sac').glob('inverted_pendulum_stabilization_*'))[0]

    report = _evaluate(run, seed=0, n_episodes=2)
    assert report['baseline_id'] == 'pendulum_lqr'
    # goal_threshold from the yaml, not TASK_INFO's copy of it.
    assert report['goal_tolerance'] == pytest.approx(0.075)
    assert report['verdict'] in ('PASS', 'FAIL')


def test_episode_seeds_are_a_pure_function_of_the_seed():
    '''The shared-seed scheme is what makes the comparison fair.'''
    assert episode_seeds(7, 5) == episode_seeds(7, 5)
    assert episode_seeds(7, 5) != episode_seeds(8, 5)
    assert len(episode_seeds(7, 5)) == 5


def test_pendulum_uses_its_own_lqr():
    '''pendulum_lqr, not the generic lqr, for the inverted pendulum.'''
    assert baseline_id('inverted_pendulum_stabilization') == 'pendulum_lqr'
    assert baseline_id('inverted_pendulum') == 'pendulum_lqr'
    assert baseline_id('cartpole_stabilization') == 'lqr'
    assert baseline_id('quadrotor3d_stabilization') == 'lqr'
