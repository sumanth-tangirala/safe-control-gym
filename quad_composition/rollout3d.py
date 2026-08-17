'''Shared rollout core for the quadrotor-3D composition experiments.

The 3D analogue of `rollout2d.py`.  WHY 3D EXISTS: quadrotor-2D cannot be
flipped from inversion by ANY controller under its shipped limits -- a
hand-coded bang-bang flip scores 0.000 above 90 deg, matching the RL
baseline cliff and the analytic ~107 deg ceiling.  A feasibility probe on
the 3D system (a hand-tuned geometric attitude controller, run against the
real PyBullet dynamics under the quadrotor3D_lqr limits below) recovers
from FULL inversion at 35-42%, and at 100% in an isolated attitude-only
test.  So the composition experiment moves to 3D.

The env configuration below is copied from
generate_quadrotor_3d_trajectories.py and must not drift from it.

FOUR THINGS THAT BIT US IN 2D, AND HOW THIS MODULE AVOIDS THEM
--------------------------------------------------------------------------
1. ATTITUDE COMES FROM THE ROTATION MATRIX, NEVER FROM EULER ANGLES.
   PyBullet's `getEulerFromQuaternion` returns the branch with pitch folded
   into [-pi/2, pi/2], so in 2D an INVERTED drone read as UPRIGHT and the
   attitude-only reward trained controller 1 to STAY inverted (Finding C1).
   The 3D quantity is `tilt = arccos(clip(R[2, 2], -1, 1))`, the angle
   between the body z axis and world z, with `R` from
   `p.getMatrixFromQuaternion`.  It is monotone, has no branch cut, and is
   exactly 0 upright / pi inverted.  See `tilt_from_quat`.

2. CONTROLLER 1'S OBSERVATION CARRIES NO EULER ANGLES (spec D6).  The env's
   native 12-dim obs has `[phi, theta, psi]` at indices 6, 7, 8; those three
   elements are REPLACED by the nine entries of `R`, giving 18 dims
   (12 - 3 + 9).  Euler angles are discontinuous exactly on the region of
   interest, so a policy trained on them is being asked to learn a flip on a
   POMDP whose two most important states alias.  See `ctrl1_observation`.

   CONTROLLER 2 (LQR) KEEPS ITS NATIVE 12-DIM STATE, UNCHANGED.  Changing it
   would redefine its region of attraction and destroy comparability with the
   shipped `quadrotor3D_lqr` dataset.  See `_act_ctrl2`.

3. THE STATE SPACE IS CLOSED, AND ITS BOUNDS ARE READ BACK OFF THE ENV.
   `flip_env3d.sampling_bounds_from_env` derives the initial-state sampling
   box from `env.state_space` rather than hardcoding it, so sampling can
   never desynchronise from termination.  Attitude samples the full SO(3).

4. THE ACTION SPACE IS THE PHYSICAL ONE.  `normalized_rl_action_space` is
   LEFT AT ITS DEFAULT (False) -- unlike 2D, the shipped 3D dataset was
   generated with the physical actuator (TWR 2.24), and controller 1 must
   have exactly controller 2's authority (spec D6).  Do not add it here.
'''

import math
from dataclasses import dataclass
from functools import partial

import gymnasium as gym
import numpy as np
import pybullet as p

from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
from safe_control_gym.utils.registration import make

GOAL_TOLERANCE = 0.05
MAX_STEPS = 1200

# Copied from generate_quadrotor_3d_trajectories.py.  No `constraints` entry
# (the 3D generation path uses none), and NO `normalized_rl_action_space`:
# it must inherit its False default so controller 1 acts on the same physical
# thrust space controller 2 (LQR) does -- see this module's docstring, item 4.
ENV_CONFIG = {
    'quad_type': QuadType.THREE_D,
    'task': 'stabilization',
    'ctrl_freq': 100,
    'pyb_freq': 5000,
    'episode_len_sec': 1000,
    'done_on_out_of_bound': True,
    'cost': 'quadratic',
    'gui': False,
    'randomized_init': False,
    'task_info': {'stabilization_goal': [0, 0, 1],
                  'stabilization_goal_tolerance': GOAL_TOLERANCE},
}

# Termination thresholds, ENV state order
# [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
# phi/theta/psi are periodic, get infinite bounds, and are additionally masked
# out of `Quadrotor._get_done()`'s out-of-bounds test.
TERMINATION = {
    0: (-1.8, 1.8),        # x
    1: (-3.0, 3.0),        # x_dot
    2: (-1.8, 1.8),        # y
    3: (-3.0, 3.0),        # y_dot
    4: (0.1, 3.0),         # z
    5: (-3.0, 3.0),        # z_dot
    6: (-np.inf, np.inf),  # phi
    7: (-np.inf, np.inf),  # theta
    8: (-np.inf, np.inf),  # psi
    9: (-24.0, 24.0),      # p
    10: (-24.0, 24.0),     # q
    11: (-24.0, 24.0),     # r
}

# Controller 2 (evaluation controller): LQR, exactly as
# generate_quadrotor_3d_trajectories.py builds it.
LQR_CONFIG = {
    'q_lqr': [1] * 12,
    'r_lqr': [0.1] * 4,
    'discrete_dynamics': True,
}

# Controller 1 (flip/recovery controller): SAC.  Verified copy of
# safe_control_gym/controllers/sac/sac.yaml, with training=False added since
# this module is eval-only; train_quadrotor_3d_flip.py overrides it to True.
# Do NOT substitute another algorithm's config -- SACAgent.__init__ reads
# `activation`, `use_entropy_tuning` and `target_entropy` unconditionally.
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

# Dataset order (13-dim, quaternion scalar-first, canonicalised qw >= 0):
# [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r]
DATASET_LABELS = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz',
                  'x_dot', 'y_dot', 'z_dot', 'p', 'q', 'r']
QUAT_SLICE = slice(3, 7)
RATE_SLICE = slice(10, 13)

GOAL_STATE = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def rotation_matrix_from_quat(quat):
    '''3x3 body->world rotation from a PYBULLET-order quaternion [x, y, z, w].'''
    return np.asarray(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)


def tilt_from_quat(quat):
    '''Tilt angle in [0, pi] from a PYBULLET-order quaternion [x, y, z, w].

    THIS IS THE ONLY ADMISSIBLE ATTITUDE SCALAR ON THIS BRANCH.  It is the
    angle between the drone's body z axis and world z:

        tilt = arccos(clip(R[2, 2], -1, 1))

    0 upright, pi/2 on its side, pi fully inverted; monotone in "how far from
    upright", and continuous everywhere including through inversion.

    NEVER derive this from Euler angles.  `p.getEulerFromQuaternion` folds
    pitch into [-pi/2, pi/2] and pushes the remainder into roll/yaw, so an
    inverted drone reads as upright -- in 2D that made the attitude-only
    reward pay its G_nom bonus to an INVERTED drone and trained controller 1
    to stay there (Finding C1).  `normalize_angle` cannot undo the fold: the
    folded value is already inside [-pi, pi], so normalising is a no-op.
    '''
    return float(math.acos(min(1.0, max(-1.0, rotation_matrix_from_quat(quat)[2, 2]))))


def tilt_from_quat_wxyz(quat_wxyz):
    '''Tilt from a DATASET-order (scalar-first) quaternion [qw, qx, qy, qz].'''
    qw, qx, qy, qz = (float(v) for v in quat_wxyz)
    return tilt_from_quat([qx, qy, qz, qw])


def tilt(env):
    '''Tilt of the drone in `env`, read off `env.quat` -- the orientation
    `BaseAviary._update_and_store_kinematic_information` caches on every reset
    and every step, so it can never drift from the observation within a tick.
    '''
    return tilt_from_quat(env.quat)


def canonical_quat_wxyz(quat):
    '''PyBullet-order [x, y, z, w] -> dataset-order [qw, qx, qy, qz] with
    qw >= 0.

    q and -q are the same rotation; canonicalising the sign makes the stored
    representation single-valued, matching the shipped quadrotor3D dataset.
    '''
    qx, qy, qz, qw = (float(v) for v in quat)
    if qw < 0.0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    return [qw, qx, qy, qz]


def quat_wxyz_to_pybullet(quat_wxyz):
    '''Dataset-order [qw, qx, qy, qz] -> PyBullet-order [x, y, z, w].'''
    qw, qx, qy, qz = (float(v) for v in quat_wxyz)
    return [qx, qy, qz, qw]


def omega_norm(state):
    '''|omega| (rad/s) from a dataset-order 13-dim state -- the body-rate
    magnitude that, together with `tilt`, defines G1 membership.
    '''
    return float(np.linalg.norm(np.asarray(state, dtype=float)[RATE_SLICE]))


def state_from_env(env, obs):
    '''Dataset-order 13-dim state with attitude as a canonical quaternion.

    env obs order  [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]
    dataset order  [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r]

    Attitude comes from `env.quat` -- the raw orientation quaternion -- and
    NOT from the obs's `[phi, theta, psi]`, which are PyBullet's folded Euler
    branch (see `tilt_from_quat`).  Body rates are taken from the obs, which
    `Quadrotor._get_observation` has already rotated into the body frame.
    '''
    o = np.asarray(obs, dtype=float)
    qw, qx, qy, qz = canonical_quat_wxyz(env.quat)
    return [float(o[0]), float(o[2]), float(o[4]),
            qw, qx, qy, qz,
            float(o[1]), float(o[3]), float(o[5]),
            float(o[9]), float(o[10]), float(o[11])]


def ctrl1_observation(env, obs):
    '''Controller 1's observation (spec D6): 18-dim, EULER-FREE.

    The env's native 12-dim obs is
    `[x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r]`.  The three
    Euler elements at indices 6, 7, 8 are REPLACED by the nine row-major
    entries of the body->world rotation matrix `R`, giving

        [x, x_dot, y, y_dot, z, z_dot,
         R00, R01, R02, R10, R11, R12, R20, R21, R22,
         p, q, r]

    18 dims (12 - 3 + 9).  Every other element keeps its value and its
    relative position.

    WHY: Euler angles are discontinuous exactly on the region controller 1
    exists to handle.  PyBullet's branch folds pitch into [-pi/2, pi/2] and
    absorbs the rest into roll/yaw, so an upright and an inverted drone can
    present near-identical `[phi, theta, psi]` while their dynamics are
    opposite (thrust up vs thrust down).  `R` is a smooth, global, injective
    encoding of SO(3) with no branch cut anywhere.

    CONTROLLER 1 ONLY -- see `_act_ctrl1`.  Controller 2 (LQR) must keep
    receiving `obs` untouched (`_act_ctrl2`), or its region of attraction is
    redefined and comparability with the shipped quadrotor3D_lqr dataset is
    destroyed.
    '''
    o = np.asarray(obs, dtype=np.float32)
    rot = rotation_matrix_from_quat(env.quat).reshape(9)
    return np.concatenate([o[0:6], rot.astype(np.float32), o[9:12]]).astype(np.float32)


def ctrl1_observation_space(env):
    '''`gymnasium.spaces.Box` matching `ctrl1_observation`'s shape (18,) and
    layout: the env's own observation_space bounds for every element carried
    through unchanged, with the three Euler bounds replaced by the rotation
    matrix's exact `[-1, 1]` range on all nine entries.

    Used both to build `flip_env3d.FlipTrainingEnv3D.observation_space` (so
    SAC sizes controller 1's TRAINING-time network correctly) and, at
    inference time, by `load_ctrl1`.  A mismatch between the two is a
    `load_state_dict` size error at best and a silent misinterpretation of the
    checkpoint's weights at worst.
    '''
    low, high = env.observation_space.low, env.observation_space.high
    obs_low = np.concatenate([low[0:6], -np.ones(9), low[9:12]]).astype(np.float32)
    obs_high = np.concatenate([high[0:6], np.ones(9), high[9:12]]).astype(np.float32)
    return gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)


def apply_termination(env, overrides=None):
    '''Write TERMINATION (optionally overridden) onto `env.state_space`.

    Every path that builds a 3D env for this experiment must call this: the
    env's own defaults are much wider, so skipping it breaks both
    out-of-bounds termination and the closed state space (CLAUDE.md).
    '''
    bounds = dict(TERMINATION)
    if overrides:
        bounds.update(overrides)
    for idx, (lo, hi) in bounds.items():
        env.state_space.low[idx] = lo
        env.state_space.high[idx] = hi
    return env


def make_env(seed=None, termination_overrides=None):
    '''Build the 3D quadrotor env alone, with TERMINATION applied and no
    controller loaded.
    '''
    return apply_termination(make('quadrotor', seed=seed, **ENV_CONFIG),
                             termination_overrides)


def make_env_and_ctrl2(output_dir=None):
    '''Build the env and controller 2 (LQR), exactly as
    generate_quadrotor_3d_trajectories.py does.

    LQR is analytic -- there is no checkpoint to load, so unlike the 2D path
    this takes no model_path.
    '''
    env_func = partial(make, 'quadrotor', **ENV_CONFIG)
    kwargs = {'output_dir': output_dir} if output_dir is not None else {}
    ctrl2 = make('lqr', env_func, **LQR_CONFIG, **kwargs)
    return make_env(), ctrl2


class _Ctrl1ObservationSpaceEnv(gym.Wrapper):
    '''Advertises `ctrl1_observation_space` (18-dim, spec D6) over an
    otherwise-untouched env, ONLY so `make('sac', ...)` sizes controller 1's
    networks and normalizer to the shape it was trained on.

    Never reset or stepped on the eval path: `rollout_composite` steps the
    shared underlying env directly and computes `ctrl1_observation` itself.
    Note that `.reset()`/`.step()` here return the env's NATIVE 12-dim obs --
    this wrapper advertises the 18-dim shape, it does not produce it.
    '''

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = ctrl1_observation_space(env)


def load_ctrl1(flip_model_path, env, output_dir):
    '''Build and load controller 1 (SAC) against an already-built, shared env.

    `lambda **kw: ctrl1_env` rather than an env-constructing partial because
    SAC's runner may call `env_func(seed=...)`; the lambda swallows any kwargs
    and always hands back the one shared, wrapped instance.
    '''
    ctrl1_env = _Ctrl1ObservationSpaceEnv(env)
    ctrl1 = make('sac', lambda **kw: ctrl1_env, **SAC_CONFIG, output_dir=output_dir)
    ctrl1.load(flip_model_path)
    ctrl1.obs_normalizer.set_read_only()
    return ctrl1


def set_initial_state(env, init_state):
    '''Place the drone at a dataset-order 13-dim state and return (obs, info).

    Attitude is set from the state's QUATERNION directly -- never by round
    tripping through Euler angles, which would silently fold any attitude past
    90 deg of pitch onto a different rotation.
    '''
    obs, info = env.reset()
    s = np.asarray(init_state, dtype=float)
    p.resetBasePositionAndOrientation(
        env.DRONE_ID, [s[0], s[1], s[2]], quat_wxyz_to_pybullet(s[QUAT_SLICE]),
        physicsClientId=env.PYB_CLIENT)
    p.resetBaseVelocity(
        env.DRONE_ID, [s[7], s[8], s[9]], [s[10], s[11], s[12]],
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


def _act_ctrl1(env, ctrl1, obs, info):
    '''Feed controller 1 the SAME 18-dim euler-free observation it saw during
    training (spec D6; see `ctrl1_observation`, `flip_env3d.FlipTrainingEnv3D`).
    '''
    return ctrl1.select_action(ctrl1.obs_normalizer(ctrl1_observation(env, obs)), info)


def _act_ctrl2(ctrl2, obs, info):
    '''Feed controller 2 (LQR) its native, UNCHANGED 12-dim observation --
    exactly what generated the shipped quadrotor3D_lqr dataset.  Deliberately
    does not take `env`, so this path cannot accidentally grow a rotation
    matrix computation.
    '''
    return ctrl2.select_action(obs, info)


def in_g1(g1, state):
    '''G1 membership for a dataset-order 13-dim state.

    `G1Region` is dimension-agnostic (it takes tilt and |omega| scalars), so
    2D and 3D share it verbatim -- do not fork it.  Here tilt comes from the
    rotation matrix and |omega| from the body-rate norm.
    '''
    return bool(g1.contains(tilt_from_quat_wxyz(np.asarray(state)[QUAT_SLICE]),
                            omega_norm(state)))


def rollout_composite(env, ctrl1, ctrl2, g1, init_state, max_steps=MAX_STEPS):
    '''Run ctrl1 until the first state inside g1, then latch to ctrl2 forever.

    ctrl1=None runs ctrl2 from the start, reproducing the baseline: no handoff
    ever fires, so handoff_index=-1 and flip_success=False unconditionally.
    The latch is permanent: once handed off, g1 is never consulted again
    (spec D3).

    `done` is evaluated exactly once per iteration, AFTER the latch update, so
    a step that both enters g1 and terminates the env is handled on that same
    tick rather than being missed.

    Every g1 test -- step 0 and step >= 1 alike -- goes through `in_g1`, i.e.
    tilt from the rotation matrix and |omega| from the body rates, never the
    env's Euler observation.
    '''
    obs, info = set_initial_state(env, init_state)
    trajectory = [state_from_env(env, obs)]

    if ctrl1 is None:
        handoff_index, latched = -1, True
    else:
        latched = in_g1(g1, trajectory[0])
        handoff_index = 0 if latched else -1

    ctrl2_success = False
    for _ in range(max_steps):
        # Two separate, differently-shaped helpers rather than one branch:
        # _act_ctrl1 REQUIRES env (it computes the rotation matrix);
        # _act_ctrl2 does not accept one.  Swapping them is a TypeError, not
        # a silent bug.
        action = (_act_ctrl2(ctrl2, obs, info) if latched
                  else _act_ctrl1(env, ctrl1, obs, info))
        obs, _, done, info = env.step(action)
        state = state_from_env(env, obs)
        trajectory.append(state)

        if not latched and in_g1(g1, state):
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
