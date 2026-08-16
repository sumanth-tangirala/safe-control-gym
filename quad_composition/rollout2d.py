'''Shared rollout core for the quadrotor-2D composition experiments.

All three datasets (baseline, flip-only, composite) go through
`rollout_composite` so the baseline is reproduced by the same code path that
produces the composition.  The env configuration below is copied from
generate_quadrotor_2d_trajectories_rl.py and must not drift from it.
'''

import math
from dataclasses import dataclass
from functools import partial

import numpy as np
import pybullet as p

from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
from safe_control_gym.utils.registration import make

# Copied verbatim from generate_quadrotor_2d_trajectories_rl.py so the composed
# system is the system the baseline was generated on.
SAFE_EXPLORER_CONSTRAINTS = [
    {'constraint_form': 'default_constraint', 'constrained_variable': 'state',
     'upper_bounds': [2, 1, 2, 1, 0.2, 1.5],
     'lower_bounds': [-2, -1, 0, -1, -0.2, -1.5]},
    {'constraint_form': 'default_constraint', 'constrained_variable': 'input',
     'upper_bounds': [0.29, 0.29], 'lower_bounds': [0.06, 0.06]},
]

GOAL_TOLERANCE = 0.2
MAX_STEPS = 1200

# norm_act_scale is deliberately absent: it must inherit the 0.1 default so
# controller 1 gets exactly controller 2's authority (spec D6).
ENV_CONFIG = {
    'quad_type': QuadType.TWO_D,
    'task': 'stabilization',
    'ctrl_freq': 100,
    'pyb_freq': 5000,
    'episode_len_sec': 1000,
    'done_on_out_of_bound': True,
    'cost': 'quadratic',
    'normalized_rl_action_space': True,
    'gui': False,
    'randomized_init': False,
    'constraints': SAFE_EXPLORER_CONSTRAINTS,
    'done_on_violation': False,
    'task_info': {'stabilization_goal': [0, 1],
                  'stabilization_goal_tolerance': GOAL_TOLERANCE},
}

# Termination thresholds, env state order [x, x_dot, z, z_dot, theta, theta_dot].
# theta is periodic and gets an infinite bound.
TERMINATION = {
    0: (-1.0, 1.0),        # x
    1: (-1.0, 1.0),        # x_dot
    2: (0.1, 1.5),         # z
    3: (-1.0, 1.0),        # z_dot
    4: (-np.inf, np.inf),  # theta
    5: (-8.0, 8.0),        # theta_dot
}

# Controller 2 (evaluation controller): safe_explorer_ppo, the exact model
# that generated quadrotor2D_rl.  Copied verbatim from
# generate_quadrotor_2d_trajectories_rl.py's ALGO_CONFIGS['safe_explorer_ppo'].
ALGO_CONFIG = {
    'hidden_dim': 128, 'norm_obs': False, 'norm_reward': False,
    'clip_obs': 10.0, 'clip_reward': 10.0,
    'pretraining': False, 'pretrained': None,
    'constraint_hidden_dim': 150, 'constraint_lr': 0.0001,
    'constraint_batch_size': 256, 'constraint_steps_per_epoch': 6000,
    'constraint_epochs': 25, 'constraint_eval_steps': 1500,
    'constraint_eval_interval': 5, 'constraint_buffer_size': 1000000,
    'constraint_slack': [0.05, 0.05, 0.05, 0.05, 0.01, 0.01,
                         0.05, 0.05, 0.05, 0.05, 0.01, 0.01],
    'gamma': 0.99, 'use_gae': True, 'gae_lambda': 0.95,
    'use_clipped_value': False, 'clip_param': 0.2, 'target_kl': 0.01,
    'entropy_coef': 0.01, 'opt_epochs': 20, 'mini_batch_size': 250,
    'actor_lr': 0.001, 'critic_lr': 0.001, 'max_grad_norm': 0.5,
    'max_env_steps': 500000, 'num_workers': 1, 'rollout_batch_size': 4,
    'rollout_steps': 250, 'deque_size': 10, 'eval_batch_size': 10,
    'log_interval': 0, 'save_interval': 0, 'num_checkpoints': 0,
    'eval_interval': 0, 'eval_save_best': False, 'tensorboard': False,
    'training': False,
}

# Controller 1 (flip/recovery controller): SAC.  Ruling D-D -- this is a
# *different* config from ALGO_CONFIG above (which is safe_explorer_ppo's);
# reusing ALGO_CONFIG to build a SAC controller would silently pass it the
# wrong hyperparameters (e.g. hidden_dim, batch size, max_env_steps all
# differ).  Copied verbatim from safe_control_gym/controllers/sac/sac.yaml,
# with training=False added since this is eval-only, never a training run.
SAC_CONFIG = {
    'hidden_dim': 256, 'activation': 'relu', 'norm_obs': False, 'norm_reward': False,
    'clip_obs': 10.0, 'clip_reward': 10.0,
    'gamma': 0.99, 'tau': 0.005, 'init_temperature': 0.2,
    'use_entropy_tuning': False, 'target_entropy': None,
    'train_interval': 100, 'train_batch_size': 64,
    'actor_lr': 0.001, 'critic_lr': 0.001, 'entropy_lr': 0.001,
    'max_env_steps': 1000000, 'warm_up_steps': 1000, 'rollout_batch_size': 4,
    'num_workers': 1, 'max_buffer_size': 1000000, 'deque_size': 10,
    'eval_batch_size': 10,
    'log_interval': 0, 'save_interval': 0, 'num_checkpoints': 0,
    'eval_interval': 0, 'eval_save_best': False, 'tensorboard': False,
    'training': False,
}

GOAL_STATE = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # dataset order


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def state_from_obs(obs):
    '''env order [x, x_dot, z, z_dot, theta, theta_dot] -> dataset order.'''
    x, x_dot, z, z_dot, theta, theta_dot = obs[:6]
    return [float(x), float(z), float(normalize_angle(theta)),
            float(x_dot), float(z_dot), float(theta_dot)]


def make_env_and_ctrl2(model_path, output_dir):
    '''Build the env and load controller 2 (safe_explorer_ppo) unmodified.'''
    env_func = partial(make, 'quadrotor', **ENV_CONFIG)
    ctrl2 = make('safe_explorer_ppo', env_func, **ALGO_CONFIG, output_dir=output_dir)
    ctrl2.load(model_path)
    ctrl2.obs_normalizer.set_read_only()
    env = env_func()
    for idx, (lo, hi) in TERMINATION.items():
        env.state_space.low[idx] = lo
        env.state_space.high[idx] = hi
    return env, ctrl2


def load_ctrl1(flip_model_path, env, output_dir):
    '''Build and load controller 1 (SAC) against an already-built, shared env.

    `lambda **kw: env` is used instead of an env-constructing partial because
    SAC's runner may call `env_func(seed=...)`; the lambda swallows any
    kwargs and always hands back the one shared env instance (Ruling D-D).
    '''
    ctrl1 = make('sac', lambda **kw: env, **SAC_CONFIG, output_dir=output_dir)
    ctrl1.load(flip_model_path)
    ctrl1.obs_normalizer.set_read_only()
    return ctrl1


def set_initial_state(env, init_state):
    '''Place the drone at a dataset-order state and return (obs, info).'''
    obs, info = env.reset()
    x, z, theta, x_dot, z_dot, theta_dot = init_state
    p.resetBasePositionAndOrientation(
        env.DRONE_ID, [x, 0, z], p.getQuaternionFromEuler([0, theta, 0]),
        physicsClientId=env.PYB_CLIENT)
    p.resetBaseVelocity(
        env.DRONE_ID, [x_dot, 0, z_dot], [0, theta_dot, 0],
        physicsClientId=env.PYB_CLIENT)
    env._update_and_store_kinematic_information()
    obs = env._get_observation()
    if getattr(env, 'constraints', None) is not None:
        info['constraint_values'] = env.constraints.get_values(env, only_state=True)
    return obs, info


@dataclass
class RolloutResult:
    '''flip_success is NOT meaningful when ctrl1 is None (the baseline path):
    it is always False there, since there is no controller 1 whose success it
    could report.
    '''
    trajectory: list
    handoff_index: int      # -1 when no handoff fired (always, when ctrl1 is None)
    flip_success: bool      # reached G1 during the rollout
    ctrl2_success: bool     # composite reached the goal ball under ctrl2


def _act(ctrl, obs, info):
    return ctrl.select_action(ctrl.obs_normalizer(obs), info)


def rollout_composite(env, ctrl1, ctrl2, g1, init_state, max_steps=MAX_STEPS):
    '''Run ctrl1 until the first state inside g1, then latch to ctrl2 forever.

    ctrl1=None runs ctrl2 from the start, reproducing the baseline: no handoff
    ever fires, so handoff_index=-1 and flip_success=False unconditionally
    (Ruling D-F -- flip_success is not a meaningful signal on the baseline
    path, since there is no controller 1 whose success it could report). The
    latch is permanent: once handed off, g1 is never consulted again (spec
    D3).

    `done` is evaluated exactly once per iteration, after the latch update
    (Ruling D-E). This matters for a step that both enters g1 and satisfies
    the env's own termination on the same tick: the latch must take effect
    and the resulting done must still be handled on that same tick, rather
    than being missed and the (now-finished) env stepped again next
    iteration.
    '''
    obs, info = set_initial_state(env, init_state)
    trajectory = [list(map(float, init_state))]

    if ctrl1 is None:
        handoff_index, latched = -1, True
    else:
        tilt, omega = abs(normalize_angle(init_state[2])), abs(init_state[5])
        latched = bool(g1.contains(tilt, omega))
        handoff_index = 0 if latched else -1

    ctrl2_success = False
    for step in range(max_steps):
        action = _act(ctrl2 if latched else ctrl1, obs, info)
        obs, _, done, info = env.step(action)
        state = state_from_obs(obs)
        trajectory.append(state)

        if not latched and bool(g1.contains(abs(state[2]), abs(state[5]))):
            latched = True
            handoff_index = len(trajectory) - 1

        if done:
            if latched:
                ctrl2_success = bool(info.get('goal_reached', False))
            break

    return RolloutResult(trajectory=trajectory,
                         handoff_index=handoff_index,
                         flip_success=handoff_index >= 0,
                         ctrl2_success=ctrl2_success)
