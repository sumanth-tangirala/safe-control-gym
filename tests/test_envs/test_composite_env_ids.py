'''Composite (system, task) ids must be a rename, not a behaviour change.

Each composite id is its base entry_point plus a yaml with the system and task
axes pinned. That only holds while the yaml stays a faithful copy of its base --
and nothing stops someone editing `cartpole_stabilization.yaml` to raise a
learning rate or widen the init distribution, at which point runs labelled with
the composite id silently stop being comparable to everything collected under
the base id.

So the property asserted here is the one the design rests on: building the
composite id and building its base id with the *same* config produces the same
environment, observation for observation. A training-specific value added to a
composite yaml does not fail this test -- the same value reaches both envs --
but a value that changes what the env *is* relative to its base does, and those
are the edits that break comparability. Training values belong in configs/sb3/.
'''
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from safe_control_gym.utils.registration import get_config, make

# composite id -> the base id it must remain equivalent to.
COMPOSITE = {
    'inverted_pendulum_stabilization': 'inverted_pendulum',
    'cartpole_stabilization': 'cartpole',
    'quadrotor2d_stabilization': 'quadrotor',
    'quadrotor3d_stabilization': 'quadrotor',
}

SEED = 1234
STEPS = 20


@pytest.mark.parametrize('idx', sorted(COMPOSITE))
def test_composite_id_conforms(idx):
    '''SB3's checker validates the composite ids, not only the base ones.'''
    env = make(idx, **get_config(idx))
    check_env(env, warn=True, skip_render_check=True)
    env.close()


@pytest.mark.parametrize('idx,base', sorted(COMPOSITE.items()))
def test_composite_matches_base(idx, base):
    '''Same config, two ids, one environment.'''
    config = get_config(idx)
    composite, plain = make(idx, **config), make(base, **config)
    try:
        assert composite.observation_space == plain.observation_space
        assert composite.action_space == plain.action_space
        np.testing.assert_array_equal(composite.state_space.low, plain.state_space.low)
        np.testing.assert_array_equal(composite.state_space.high, plain.state_space.high)
        np.testing.assert_array_equal(composite.X_GOAL, plain.X_GOAL)

        # Identical actions from the same seed must give identical observations.
        # Spaces comparing equal says nothing about the dynamics behind them.
        obs_a, _ = composite.reset(seed=SEED)
        obs_b, _ = plain.reset(seed=SEED)
        np.testing.assert_allclose(obs_a, obs_b, atol=0,
                                   err_msg=f'{idx} and {base} disagree at reset')

        rng = np.random.default_rng(SEED)
        for step in range(STEPS):
            action = rng.uniform(composite.action_space.low,
                                 composite.action_space.high).astype(np.float32)
            a = composite.step(action)
            b = plain.step(action)
            np.testing.assert_allclose(a[0], b[0], atol=0,
                                       err_msg=f'{idx} and {base} diverge at step {step}')
            assert a[2:4] == b[2:4], f'{idx} and {base} disagree on done flags at step {step}'
            if a[2] or a[3]:
                break
    finally:
        composite.close()
        plain.close()


@pytest.mark.parametrize('idx', sorted(COMPOSITE))
def test_goal_signal_can_fire(idx):
    '''A zero goal tolerance makes success_rate identically zero.

    quadrotor3d's collector sets stabilization_goal_tolerance to 0.0 under
    --invariant_terminal_sets, where success is ellipsoid membership and the goal
    ball must never fire. Evaluation reads info['goal_reached'], so inheriting
    that 0.0 would silently report every policy as a total failure.
    '''
    env = make(idx, **get_config(idx))
    try:
        tolerance = getattr(env, 'GOAL_THRESHOLD', None)
        if tolerance is None:
            tolerance = env.TASK_INFO['stabilization_goal_tolerance']
        assert tolerance > 0, f'{idx} cannot ever report goal_reached'
    finally:
        env.close()
