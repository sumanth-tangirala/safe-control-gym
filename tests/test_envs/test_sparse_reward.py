'''Cost.SPARSE must pay out on outcomes, and only on outcomes.

The dense reward makes success irrational. `_get_done` ends an episode when the
goal ball is entered, and `rl_reward` is strictly positive, so reaching the goal
forfeits every remaining step. Measured on cartpole: tightening the terminal
error from 0.143 to 0.050 was worth +4.46 return, while the early termination it
causes cost -181.2. SAC learned the correct policy for that reward -- hover just
outside the ball, 3.8x the LQR return at 0.000 success.

Cost.SPARSE removes the inversion by making the goal the only positive term.
These tests pin that it fires on the right event, that its values are
configuration rather than constants, and that adding it did not disturb the two
existing cost functions.
'''
import numpy as np
import pytest

from safe_control_gym.envs.benchmark_env import Cost
from safe_control_gym.utils.registration import get_config, make

SYSTEMS = ['cartpole_stabilization', 'inverted_pendulum_stabilization',
           'quadrotor2d_stabilization']

DEFAULTS = {'goal': 1.0, 'oob': -1.0, 'step': -0.01}


def _sparse_env(env_id, **overrides):
    return make(env_id, **{**get_config(env_id), 'cost': 'sparse', **overrides})


# The goal state of each system, for starting an episode already inside the
# goal ball. Random actions will not find it: the pendulum's ball is 0.075 wide
# over the whole state space, and 40 random episodes never entered it.
#
# ndarray, not list -- the envs accept an ndarray or a dict and raise on a list.
AT_GOAL = {
    'cartpole': np.zeros(4),
    'inverted_pendulum': np.zeros(2),
    'quadrotor2d': np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
}

# Systems whose episodes can end out-of-bounds at all. The inverted pendulum is
# absent deliberately: its _get_done ends only on reaching the goal, and
# theta_dot is clipped at theta_dot_max rather than terminated, so
# sparse_oob_reward is unreachable there however it is configured.
HAS_OUT_OF_BOUNDS = ['cartpole_stabilization', 'quadrotor2d_stabilization']


@pytest.mark.parametrize('env_id', SYSTEMS)
def test_steps_outside_the_goal_pay_the_step_cost(env_id):
    '''Every step that resolves nothing costs exactly sparse_step_reward.

    Conditioned on being outside the goal ball, which matters under
    stabilization: there entering the ball pays +1 and does NOT end the episode,
    so a non-terminal step can legitimately pay the goal bonus. Under reach the
    same step would terminate. Asserting "all non-terminal steps cost -0.01"
    would therefore be asserting reach semantics on both tasks.
    '''
    env = _sparse_env(env_id)
    outside, inside = set(), set()
    try:
        for seed in range(20):
            env.reset(seed=seed)
            done = False
            while not done:
                _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                done = terminated or truncated
                if done:
                    continue
                if env.unwrapped.goal_reached:
                    inside.add(reward)
                else:
                    outside.add(reward)
    finally:
        env.close()
    assert outside == {DEFAULTS['step']}, f'steps outside the goal paid {outside}'
    assert inside <= {DEFAULTS['goal']}, f'steps inside the goal paid {inside}'


def _held_at_goal(system, task, steps):
    '''Start inside the goal ball and hold the equilibrium action.

    U_GOAL, not a zero action: for the quadrotors zero thrust is free fall, so
    the drone leaves the ball within one step and the branch under test never
    fires. Driven deliberately rather than stumbled upon, so every system covers
    it instead of only those random play happens to reach.
    '''
    env = _sparse_env(f'{system}_{task}', randomized_init=False,
                      init_state=AT_GOAL[system])
    rewards, terminated = [], False
    try:
        env.reset(seed=0)
        hover = np.asarray(env.unwrapped.U_GOAL, dtype=np.float32)
        for _ in range(steps):
            _, reward, terminated, truncated, _ = env.step(hover)
            rewards.append(reward)
            if terminated or truncated:
                break
    finally:
        env.close()
    return rewards, terminated


@pytest.mark.parametrize('system', sorted(AT_GOAL))
def test_reach_ends_on_first_contact(system):
    '''reach: touch the ball, collect +1, episode over.'''
    rewards, terminated = _held_at_goal(system, 'reach', steps=5)
    assert terminated, f'{system}_reach did not end on reaching the goal'
    assert rewards == [DEFAULTS['goal']], rewards


@pytest.mark.parametrize('system', sorted(AT_GOAL))
def test_stabilization_requires_holding(system):
    '''stabilization: the ball does not end the episode, and holding keeps paying.

    This is the distinction between the two tasks. Before terminate_on_goal
    existed, Task.STABILIZATION ended the episode on first contact -- which is
    reach -- and every shipped dataset was collected that way.
    '''
    rewards, terminated = _held_at_goal(system, 'stabilization', steps=5)
    assert not terminated, f'{system}_stabilization ended on reaching the goal'
    assert rewards == [DEFAULTS['goal']] * 5, rewards


@pytest.mark.parametrize('env_id', HAS_OUT_OF_BOUNDS)
def test_going_out_of_bounds_pays_the_penalty(env_id):
    env = _sparse_env(env_id)
    oob = set()
    try:
        for seed in range(40):
            env.reset(seed=seed)
            done = False
            while not done:
                _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                done = terminated or truncated
                if done and getattr(env.unwrapped, 'out_of_bounds', False):
                    oob.add(reward)
    finally:
        env.close()
    assert oob == {DEFAULTS['oob']}, f'out-of-bounds steps paid {oob}'


@pytest.mark.parametrize('env_id', SYSTEMS)
def test_values_are_configuration(env_id):
    '''All three are tunable, because they trade off through the horizon.

    With the defaults, timing out costs -0.01 * H -- on cartpole's 250-step
    horizon that is -2.50, worse than crashing at step 10 for -1.10. Tuning
    out of that regime has to be possible without editing code.
    '''
    env = _sparse_env(env_id, sparse_goal_reward=10.0,
                      sparse_oob_reward=-5.0, sparse_step_reward=-0.002)
    try:
        assert env.unwrapped.sparse_goal_reward == 10.0
        assert env.unwrapped.sparse_oob_reward == -5.0
        assert env.unwrapped.sparse_step_reward == -0.002
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert reward in (10.0, -5.0, -0.002)
    finally:
        env.close()


@pytest.mark.parametrize('env_id', SYSTEMS)
def test_dense_costs_are_undisturbed(env_id):
    '''Adding SPARSE, and reordering _get_done before _get_reward, must not
    move rl_reward.

    The reorder is what makes goal_reached and out_of_bounds current when the
    reward is computed; previously they were a step stale. No dense branch reads
    those flags, so it should be neutral -- but "should be" is why the golden
    rollout fixtures exist, and this asserts the reward stream directly.
    '''
    env = make(env_id, **{**get_config(env_id), 'cost': 'rl_reward'})
    try:
        env.reset(seed=7)
        rng = np.random.default_rng(7)
        rewards = []
        for _ in range(15):
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        # rl_reward is exp(-dist), strictly positive and bounded by 1 -- the
        # property that made success irrational in the first place.
        assert all(0.0 < r <= 1.0 for r in rewards), rewards
    finally:
        env.close()


def test_sparse_is_a_registered_cost():
    assert Cost.SPARSE == 'sparse'
    assert Cost('sparse') is Cost.SPARSE


def test_quadrotor3d_observes_a_rotation_matrix():
    '''quadrotor3d must feed R, not Euler angles.

    No representation of SO(3) in four or fewer dimensions is continuous
    (Zhou et al. 2019), and this env's Euler readback is the worst case: its
    chart is singular at pitch +/-pi/2, which is why measured pitch topped out
    at 1.54 while roll ran to +/-pi.

    Asserted as a rotation, not merely as nine numbers: orthonormal with
    determinant +1. A transposed or mis-ordered construction still produces nine
    plausible-looking values.
    '''
    import munch
    import yaml

    from safe_control_gym.experiments.train_sb3 import build_env
    over = yaml.safe_load(open('configs/sb3/quadrotor3d_reach_sac.yaml'))
    cfg = {'task': 'quadrotor3d_reach',
           'task_config': {**get_config('quadrotor3d_reach'), **over.get('task_config', {})},
           'sb3_config': over['sb3_config']}
    env = build_env(munch.munchify(cfg))
    try:
        # 12 state channels, three of them Euler angles, replaced by nine.
        assert env.observation_space.shape == (18,), env.observation_space.shape
        for seed in range(25):
            obs, _ = env.reset(seed=seed)
            # Layout: x, x_dot, y, y_dot, z, z_dot, [R 3x3], p, q, r
            matrix = np.asarray(obs[6:15]).reshape(3, 3)
            np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-9,
                                       err_msg=f'seed {seed}: R is not orthonormal')
            assert abs(np.linalg.det(matrix) - 1.0) < 1e-9, (
                f'seed {seed}: det(R) = {np.linalg.det(matrix)}, not a rotation')
    finally:
        env.close()


def test_rotation_matrix_matches_pybullet():
    '''The convention must be PyBullet's, since that is what produced the state.'''
    import pybullet as pb

    from safe_control_gym.envs.env_wrappers.shaping import rotation_matrix_from_rpy
    rng = np.random.default_rng(0)
    for _ in range(200):
        rpy = rng.uniform(-np.pi, np.pi, 3)
        theirs = np.array(pb.getMatrixFromQuaternion(
            pb.getQuaternionFromEuler(rpy))).reshape(3, 3)
        np.testing.assert_allclose(rotation_matrix_from_rpy(*rpy), theirs, atol=1e-12)
