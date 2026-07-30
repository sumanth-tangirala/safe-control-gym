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
    'inverted_pendulum_reach': 'inverted_pendulum',
    'cartpole_reach': 'cartpole',
    'quadrotor2d_reach': 'quadrotor',
    'quadrotor3d_reach': 'quadrotor',
}

# The ONLY keys each composite may differ from its base on. Everything else must
# match value for value.
#
# quadrotor3d pins two: quad_type, and task_info -- THREE_D needs a 3-element
# stabilization_goal, and task_info replaces the dict rather than merging, so
# the whole thing is restated in the yaml.
#
# quadrotor2d pins nothing, because quad_type: 2 is already quadrotor.yaml's
# default. It restates it for self-description, which is a match, not a diff.
#
# terminate_on_goal is pinned by every composite, in both directions: the reach
# ids set it True and the stabilization ids False, while the base yamls omit it
# entirely and inherit the True default. That default is what every shipped
# dataset was collected under, which is why the base yamls are left alone.
PINNED = {
    'inverted_pendulum_stabilization': {'terminate_on_goal'},
    'cartpole_stabilization': {'terminate_on_goal'},
    'quadrotor2d_stabilization': {'terminate_on_goal'},
    'quadrotor3d_stabilization': {'quad_type', 'task_info', 'terminate_on_goal'},
    'inverted_pendulum_reach': {'terminate_on_goal'},
    'cartpole_reach': {'terminate_on_goal'},
    'quadrotor2d_reach': {'terminate_on_goal'},
    'quadrotor3d_reach': {'quad_type', 'task_info', 'terminate_on_goal'},
}

SEED = 1234
STEPS = 20

# Sentinel for "the base does not have this key at all", distinct from a key
# whose value is legitimately None -- `task_info: null` is exactly that.
_ABSENT = object()


@pytest.mark.parametrize('idx', sorted(COMPOSITE))
def test_composite_id_conforms(idx):
    '''SB3's checker validates the composite ids, not only the base ones.'''
    env = make(idx, **get_config(idx))
    check_env(env, warn=True, skip_render_check=True)
    env.close()


@pytest.mark.parametrize('idx,base', sorted(COMPOSITE.items()))
def test_composite_yaml_tracks_its_base(idx, base):
    '''The composite yaml must stay a copy of its base but for the pinned keys.

    This is the check that actually protects the invariant, and it is a CONFIG
    comparison rather than a rollout one. test_composite_matches_base builds
    both ids from the *composite's* config, so it cannot see the failure that
    matters: someone edits cartpole.yaml, leaves cartpole_stabilization.yaml
    alone, and the two silently diverge while every rollout test still passes.

    Both directions are checked. A key added to the base and not to the
    composite is drift; so is a key whose value the composite changed without
    declaring it in PINNED. Adding a training value to a composite yaml fails
    here, which is the point -- those belong in configs/sb3/.
    '''
    composite_config, base_config = get_config(idx), get_config(base)

    missing = sorted(set(base_config) - set(composite_config))
    assert not missing, (
        f'{idx}.yaml is missing keys its base gained: {missing}. '
        f'Copy them across, or pin them deliberately in PINNED.')

    differs = {key for key, value in composite_config.items()
               if base_config.get(key, _ABSENT) != value}
    undeclared = sorted(differs - PINNED[idx])
    assert not undeclared, (
        f'{idx}.yaml diverges from {base}.yaml on undeclared keys: '
        f'{ {k: (base_config.get(k, "<absent>"), composite_config[k]) for k in undeclared} }. '
        f'Training values belong in configs/sb3/, not in an env yaml.')

    unused = sorted(PINNED[idx] - differs)
    assert not unused, (
        f'{idx} declares {unused} as pinned but they now match the base. '
        f'Remove them from PINNED so the allowance stays honest.')


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
