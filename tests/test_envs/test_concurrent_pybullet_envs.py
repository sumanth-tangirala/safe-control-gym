'''Two PyBullet envs alive in one process must not interfere.

Every PyBullet call in base_aviary.py passes physicsClientId except one, and
omitting it silently targets client 0. A second env's reset() then applied its
own DRONE_ID's damping to the FIRST env's client, corrupting that env and
leaving itself at PyBullet's default damping rather than 0. Measured before the
fix: two concurrent quadrotor-2D envs diverged by 0.34 in state within five
steps, while sequential envs agreed exactly -- because with one env the only
client is 0 and the omission is harmless.

That is reachable from ordinary training, not just from tests: SB3's
EvalCallback holds an eval env open alongside the training env, and DummyVecEnv
holds n_envs of them. Neither would have raised; both would have quietly
produced wrong dynamics.

The oracle is a sequential rollout, which was correct throughout. A concurrent
rollout must reproduce it exactly.
'''
import numpy as np
import pytest

from safe_control_gym.utils.registration import get_config, make

QUADROTORS = ['quadrotor2d_stabilization', 'quadrotor3d_stabilization']

SEED = 1234
STEPS = 5


def _rollout(env, seed=SEED, steps=STEPS):
    '''Observations from a fixed seed and a fixed action sequence.'''
    obs, _ = env.reset(seed=seed)
    trace = [obs.copy()]
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        trace.append(env.step(action)[0].copy())
    return np.array(trace)


@pytest.mark.parametrize('idx', QUADROTORS)
def test_concurrent_envs_match_sequential(idx):
    config = get_config(idx)

    # Oracle: one env alive at a time. Correct before and after the fix.
    solo = make(idx, **config)
    expected = _rollout(solo)
    solo.close()

    first, second = make(idx, **config), make(idx, **config)
    try:
        got_first = _rollout(first)
        got_second = _rollout(second)
    finally:
        first.close()
        second.close()

    # The second env is the one that mis-targets client 0, so it corrupts the
    # first. Both are asserted: which one shows the damage depends on ordering.
    np.testing.assert_allclose(
        got_first, expected, atol=0,
        err_msg=f'{idx}: env 1 was perturbed by a second env in the same process')
    np.testing.assert_allclose(
        got_second, expected, atol=0,
        err_msg=f'{idx}: env 2 did not get its own physics settings')
