'''The corridor disturbance as the env actually applies it.

Two things the unit tests cannot see: that the config key resolves, and that a
2-D dynamics disturbance masked to [1, 0] moves the vehicle in +x only.
'''
import numpy as np

from safe_control_gym.utils.registration import make

TASK_INFO = {'stabilization_goal': [0, 1], 'stabilization_goal_tolerance': 0.2}


def _env(f_max, centre=0.55, width=0.12):
    return make('quadrotor', quad_type=2, task='stabilization', task_info=TASK_INFO,
                ctrl_freq=100, pyb_freq=5000, episode_len_sec=10, cost='quadratic',
                done_on_out_of_bound=False, gui=False, randomized_init=False,
                disturbances={'dynamics': [{'disturbance_func': 'altitude_gated',
                                            'f_max': f_max, 'profile': 'gaussian',
                                            'centre': centre, 'width': width,
                                            'mask': [1, 0]}]})


def test_config_key_resolves_to_the_class():
    from safe_control_gym.envs.disturbances import AltitudeGatedNoise
    env = _env(0.05)
    try:
        dist = env.disturbances['dynamics'].disturbances[0]
        assert isinstance(dist, AltitudeGatedNoise)
        assert dist.dim == 2
    finally:
        env.close()


def test_hover_inside_the_band_drifts_positive_x():
    '''A one-sided force has a mean, so the drift has a sign.'''
    env = _env(0.05)
    try:
        env.reset(seed=0)
        import pybullet as pb
        pb.resetBasePositionAndOrientation(
            env.DRONE_ID, [0, 0, 0.55], pb.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=env.PYB_CLIENT)
        env._update_and_store_kinematic_information()
        env._get_observation()   # refresh env.state after the teleport
        hover = env.U_GOAL.copy()   # per-pair thrust mg/2, physical units
        for _ in range(100):
            env.step(hover)
        assert env.state[0] > 0.01      # x has moved +
        assert env.state[1] > 0.0       # and is still moving +
    finally:
        env.close()


def test_hover_at_the_goal_altitude_does_not_drift():
    '''sigma(1.0) is ~0.09% of peak, so the goal is calm by construction.'''
    env = _env(0.05)
    try:
        env.reset(seed=0)
        import pybullet as pb
        pb.resetBasePositionAndOrientation(
            env.DRONE_ID, [0, 0, 1.0], pb.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=env.PYB_CLIENT)
        env._update_and_store_kinematic_information()
        env._get_observation()   # refresh env.state after the teleport
        hover = env.U_GOAL.copy()   # per-pair thrust mg/2, physical units
        for _ in range(100):
            env.step(hover)
        assert abs(env.state[0]) < 1e-3
    finally:
        env.close()


def test_f_max_zero_is_bit_identical_to_no_disturbance():
    '''The baseline level must be the same code path, not a different env.'''
    import pybullet as pb
    finals = []
    for dist in (True, False):
        env = _env(0.0) if dist else make(
            'quadrotor', quad_type=2, task='stabilization', task_info=TASK_INFO,
            ctrl_freq=100, pyb_freq=5000, episode_len_sec=10, cost='quadratic',
            done_on_out_of_bound=False, gui=False, randomized_init=False)
        try:
            env.reset(seed=3)
            pb.resetBasePositionAndOrientation(
                env.DRONE_ID, [0, 0, 0.55], pb.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=env.PYB_CLIENT)
            env._update_and_store_kinematic_information()
            env._get_observation()   # refresh env.state after the teleport
            hover = env.U_GOAL.copy()   # per-pair thrust mg/2, physical units
            for _ in range(50):
                env.step(hover)
            finals.append(env.state.copy())
        finally:
            env.close()
    assert np.allclose(finals[0], finals[1], atol=0, rtol=0)
