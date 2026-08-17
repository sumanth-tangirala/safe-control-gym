'''Shared rollout core for the quadrotor-2D composition experiments.

All three datasets (baseline, flip-only, composite) go through
`rollout_composite` so the baseline is reproduced by the same code path that
produces the composition.  The env configuration below is copied from
generate_quadrotor_2d_trajectories_rl.py and must not drift from it.

RULING D-I (see task-2-report.md, "Fix round 2"): the shipped
`quadrotor2D_rl` dataset is NOT bit-reproducible on this machine, per
trajectory -- confirmed by running the untouched, unmodified
generate_quadrotor_2d_trajectories_rl.run_trajectory against its own shipped
eval_states.txt (label agreement 19/20, final-state agreement 12/20 at
atol=1e-4 on a mixed-class window; one row's discrete outcome flips). This
is chaotic divergence from whatever PyBullet/library/hardware state
generated the dataset, amplified by this system's known sensitivity near
recovery/failure boundaries -- not a defect in this module or in the
original script. What IS exact is that this module's rollout core matches
the reference generation script when both run in the same process on the
same machine (verified at atol=1e-9; see
test_rollout_core_matches_the_reference_implementation).

CONSEQUENCE FOR LATER TASKS: any baseline-vs-composition comparison built on
top of this module must use a baseline REGENERATED LOCALLY (on the same
machine, in the same process/session) rather than the archived
`quadrotor2D_rl` dataset directly, or the comparison will be confounded by
this same environmental drift rather than by the actual controller
composition being studied. Do not rediscover this by re-running the
equivalence test against the shipped file and getting confused when it
disagrees with itself.

FINDING C1 -- THE STORED `theta` COLUMN IS TRUE ATTITUDE, NOT THE ENV'S
FOLDED OBSERVATION
--------------------------------------------------------------------------
PyBullet's `getEulerFromQuaternion` returns the euler branch with pitch in
[-pi/2, pi/2], so `quadrotor.py`'s TWO_D observation (`rpy[1]`) is
GIMBAL-FOLDED: a drone at true pitch 3.0 rad -- nearly inverted -- observes
theta = 0.1416, i.e. upright. `true_theta`/`true_theta_from_quat` recover the
unfolded value from the drone's rotation matrix, and EVERY supervisory
decision on this branch is made on that: G1 membership (rollout_composite),
the flip reward/potential and G_NOM termination (flip_env2d), and G1
calibration (calibrate_quad2d_g1.py).

DECISION: the stored trajectory `theta` column (and the final-state theta in
eval_states.txt / roa_labels.txt, and the handoff state in
handoff_states.txt) is TRUE attitude on [-pi, pi]. It is the physically
meaningful quantity, the flip/composite/regenerated-baseline datasets are
new, and the folded column is actively misleading for any downstream
consumer -- a trajectory that rotates through inversion reads, in folded
coordinates, as a bounce off +-pi/2, which is not a trajectory of any
continuous system.

CONSEQUENCE: the regenerated baseline's theta columns no longer match the
archived quadrotor2D_rl file's, whose final-state theta is folded (measured:
its column 8 spans exactly [-1.570796, 1.570796] = [-pi/2, pi/2], while its
INIT theta column spans -3.141593..2.908407 -- init states were sampled, not
observed, so they were already true). Only the theta columns differ, and
only where |true theta| > pi/2; every other column and every label is
unaffected. Initial states are read from the archived file and passed
through unchanged, so pairing between the datasets is untouched.
`test_rollout_core_matches_the_reference_implementation` therefore compares
the reference script's folded theta against `fold_pitch(ours)` (defined in
that test module) rather than against ours directly.

CONTROLLER 2 IS NOT AFFECTED: it keeps receiving the env's native folded
observation, exactly as it was trained and exactly as the shipped baseline
was generated. Changing its input would redefine its region of attraction
and invalidate every comparison against that baseline.
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


def true_theta_from_quat(quat):
    '''TRUE pitch (rotation about the body y axis), in [-pi, pi], from a
    body-orientation quaternion.

    THE PROBLEM (Finding C1).  The env's observed `theta` is
    `p.getEulerFromQuaternion(quat)[1]`, and PyBullet returns the euler branch
    whose PITCH IS CLAMPED TO [-pi/2, pi/2], pushing the remainder of the
    rotation into roll/yaw ~ +-pi.  A nearly-inverted drone therefore reads as
    nearly upright.  Measured directly on this machine:

        set pitch 2.00 -> getEulerFromQuaternion -> [3.1416, 1.1416, 3.1416]
        set pitch 3.00 -> getEulerFromQuaternion -> [3.1416, 0.1416, 3.1416]
        through the real env: init theta 3.0000 -> observed theta 0.1416

    `normalize_angle` cannot undo this: the folded value is already inside
    [-pi, pi], so normalization is a no-op and the fold is invisible.

    THE FIX.  The rotation matrix carries the full rotation with no branch cut.
    For a rotation about y alone,

        R = [[ cos t, 0, sin t],
             [     0, 1,     0],
             [-sin t, 0, cos t]]

    so `atan2(R[0, 2], R[0, 0]) == t` for every t in [-pi, pi], both signs, with
    no dependence on which euler branch PyBullet happened to pick (verified
    against a real-env sweep over |theta| up to pi -- see
    tests/test_quad_composition/test_rollout2d.py's real-env attitude tests,
    max error 0.0).  The 2D quadrotor stays in the x-z plane (measured
    |R[1, 0]|, |R[1, 2]| ~ 1e-19 after stepping), so this is exact here rather
    than an approximation of a general 3D attitude.

    USE THIS FOR EVERY SUPERVISORY DECISION -- G1 membership, the flip
    reward/potential, G_NOM termination, G1 calibration.  NEVER feed it to
    controller 2: controller 2 was trained on, and generated the shipped
    baseline dataset from, the env's own FOLDED observation, and substituting
    a different attitude convention would redefine its region of attraction
    and destroy comparability with that baseline.
    '''
    rot = np.asarray(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
    return float(math.atan2(rot[0, 2], rot[0, 0]))


def true_theta(env):
    '''TRUE pitch of the drone in `env`, in [-pi, pi] (see
    `true_theta_from_quat` for why the observation's own theta cannot be used).

    Reads `env.quat`, the orientation `BaseAviary._update_and_store_kinematic_information`
    caches on every reset and every step -- the exact same quaternion the
    observation's folded theta was derived from, so the two can never drift
    apart within a tick, and no extra PyBullet round trip is needed.
    '''
    return true_theta_from_quat(env.quat)


def state_from_obs(obs):
    '''env order [x, x_dot, z, z_dot, theta, theta_dot] -> dataset order.

    theta here is the env's own GIMBAL-FOLDED pitch (Finding C1).  This is the
    right function for anything that must reproduce the env's native view --
    and the wrong one for any attitude decision.  Use `state_from_env` for
    those.
    '''
    x, x_dot, z, z_dot, theta, theta_dot = obs[:6]
    return [float(x), float(z), float(normalize_angle(theta)),
            float(x_dot), float(z_dot), float(theta_dot)]


def state_from_env(env, obs):
    '''Dataset-order state with TRUE attitude (Finding C1).

    Identical to `state_from_obs` except that theta is recovered from the
    drone's rotation matrix instead of read out of the folded observation.
    This is the state every supervisory decision on this branch is made on,
    and the state written to the stored trajectories.
    '''
    state = state_from_obs(obs)
    state[2] = true_theta(env)
    return state


def _normalized_dataset_row(state):
    '''dataset order [x, z, theta, x_dot, z_dot, theta_dot] with theta
    normalized to [-pi, pi], matching what state_from_env produces for every
    later row (and what generate_quadrotor_2d_trajectories_rl.py's
    run_trajectory does for its own seed row, line ~494).

    The seed row is built from the REQUESTED init state, not read back off the
    env, and needs no unfolding: an init theta is already TRUE attitude on
    +-pi (the archived quadrotor2D_rl init column spans -3.141593..2.908407,
    i.e. it was never passed through the env's folded observation). So row 0
    and every later row are the same quantity -- see rollout_composite's
    ATTITUDE CONVENTION note.
    '''
    x, z, theta, x_dot, z_dot, theta_dot = state
    return [float(x), float(z), float(normalize_angle(theta)),
            float(x_dot), float(z_dot), float(theta_dot)]


def make_env(seed=None):
    '''Build the 2D quadrotor env alone, with TERMINATION bounds applied and
    no controller loaded.

    Factored out of `make_env_and_ctrl2` so callers that need an env but not
    controller 2 -- e.g. G1 calibration, which must not load controller 2 at
    all (spec D1) -- do not have to load and immediately discard it just to
    get an env instance.
    '''
    env = make('quadrotor', seed=seed, **ENV_CONFIG)
    for idx, (lo, hi) in TERMINATION.items():
        env.state_space.low[idx] = lo
        env.state_space.high[idx] = hi
    return env


def make_env_and_ctrl2(model_path, output_dir):
    '''Build the env and load controller 2 (safe_explorer_ppo) unmodified.'''
    env_func = partial(make, 'quadrotor', **ENV_CONFIG)
    ctrl2 = make('safe_explorer_ppo', env_func, **ALGO_CONFIG, output_dir=output_dir)
    ctrl2.load(model_path)
    ctrl2.obs_normalizer.set_read_only()
    env = make_env()
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

    ATTITUDE CONVENTION (Findings C1/C2). Every g1 test below -- step 0 and
    step >= 1 alike -- is made on TRUE attitude read off the env's rotation
    matrix via `state_from_env`/`true_theta`, never on the env's own
    gimbal-folded observation theta. Before the fix, step 0 tested the raw
    (true, +-pi) `init_state[2]` while every later step tested the FOLDED obs
    theta, so the same physical state was classified two different ways
    depending on when it was seen -- and a nearly-inverted drone (true theta
    3.0 -> folded 0.1416) triggered a spurious handoff on step 1.
    Controller 2 is unaffected: `_act` still hands it the untouched `obs`.
    '''
    obs, info = set_initial_state(env, init_state)
    trajectory = [_normalized_dataset_row(init_state)]

    if ctrl1 is None:
        handoff_index, latched = -1, True
    else:
        # Same quantity, same code path as the loop below (Finding C2).
        # `set_initial_state` has just placed the drone at `init_state`, so
        # this is `normalize_angle(init_state[2])` -- read back off the env
        # rather than recomputed, so the two can never diverge.
        state = state_from_env(env, obs)
        latched = bool(g1.contains(abs(state[2]), abs(state[5])))
        handoff_index = 0 if latched else -1

    ctrl2_success = False
    for step in range(max_steps):
        action = _act(ctrl2 if latched else ctrl1, obs, info)
        obs, _, done, info = env.step(action)
        state = state_from_env(env, obs)
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
