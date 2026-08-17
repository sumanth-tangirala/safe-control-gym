'''The hand-coded controller 1 must flip the drone, stay inside the actuator,
stay deterministic, and drop into `rollout_composite` untouched.

These are deliberately few and fast.  They guard exactly the claims
`quad_composition/geometric_flip3d.py` is being promoted on:

  1. it reduces tilt from FULL INVERSION (the whole reason controller 1
     exists, and the case a `sin(tilt)` error stalls on);
  2. every action it emits lies inside `env.physical_action_bounds` (the 3D
     env acts on the PHYSICAL actuator -- rollout3d.py docstring, item 4);
  3. it is deterministic and stateless -- same state, same action, always;
  4. it reaches `G_NOM_3D` from 180 deg in the isolated attitude-only setting;
  5. `rollout3d.rollout_composite` accepts it as controller 1 with NO change
     to `rollout3d.py`.

Statistical success RATES are not tested here -- they are measured by
`evaluate_geometric_flip3d.py`, which needs hundreds of rollouts and does not
belong in a test suite.
'''
import os
import sys

import numpy as np
import pybullet as p
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from quad_composition.flip_env3d import G_NOM_3D, random_rotation_quat  # noqa: E402
from quad_composition.geometric_flip3d import (GeometricFlipController3D,  # noqa: E402
                                               IdentityNormalizer, make_ctrl1, mixer)
from quad_composition.rollout3d import (QUAT_SLICE, canonical_quat_wxyz,  # noqa: E402
                                        ctrl1_observation, make_env, make_env_and_ctrl2,
                                        omega_norm, quat_wxyz_to_pybullet, rollout_composite,
                                        set_initial_state, state_from_env, tilt,
                                        tilt_from_quat_wxyz)

RATE_BOUND = 24.0


@pytest.fixture(scope='module')
def env():
    '''One env for the whole module: building a PyBullet env costs more than
    every rollout in this file put together, and nothing here mutates it in a
    way the next test can see (`set_initial_state` resets it).
    '''
    built = make_env(seed=0)
    yield built
    built.close()


@pytest.fixture(scope='module')
def ctrl(env):
    return GeometricFlipController3D(env)


def _state(pos=(0.0, 0.0, 1.0), euler=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0),
           rates=(0.0, 0.0, 0.0)):
    '''A dataset-order 13-dim row, from an Euler attitude that is convenient to
    WRITE DOWN.  Nothing under test ever reads an Euler angle back out.
    '''
    quat = canonical_quat_wxyz(p.getQuaternionFromEuler(list(euler)))
    return np.array(list(pos) + quat + list(vel) + list(rates), dtype=float)


def _obs18(rot, rates=(0.0, 0.0, 0.0), pos_vel=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)):
    '''Controller 1's 18-dim observation, assembled by hand in
    `ctrl1_observation`'s documented layout: [x, x_dot, y, y_dot, z, z_dot,
    R00..R22 row-major, p, q, r].
    '''
    return np.concatenate([np.asarray(pos_vel, dtype=float),
                           np.asarray(rot, dtype=float).reshape(9),
                           np.asarray(rates, dtype=float)])


def _rollout(env, ctrl, init, steps):
    '''Controller 1 alone, through the same call `rollout3d._act_ctrl1` makes.
    Returns the (tilt, |omega|) trace and whether the env terminated.
    '''
    obs, info = set_initial_state(env, init)
    trace, done = [], False
    for _ in range(steps):
        action = ctrl.select_action(ctrl.obs_normalizer(ctrl1_observation(env, obs)), info)
        obs, _, done, info = env.step(action)
        state = np.asarray(state_from_env(env, obs), dtype=float)
        trace.append((tilt_from_quat_wxyz(state[QUAT_SLICE]), omega_norm(state)))
        if done:
            break
    return trace, done


# ---------------------------------------------------------------------------
# 1. It flips.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('euler', [
    (np.pi, 0.0, 0.0),   # inverted by roll
    (0.0, np.pi, 0.0),   # inverted by pitch -- the attitude Euler angles fold
])
def test_reduces_tilt_from_full_inversion(env, ctrl, euler):
    '''THE CORE CLAIM.  From exactly pi of tilt with zero body rates, tilt
    falls.

    Exact inversion is the hard case, not a soft one: `cross(b3, z)` is
    exactly zero there, so a purely reactive law has no error to act on and
    sits on the (unstable) equilibrium forever.  Both inversions below start
    with `|e| == 0`.
    '''
    init = _state(euler=euler)
    obs, _ = set_initial_state(env, init)
    assert tilt(env) == pytest.approx(np.pi, abs=1e-6), 'the test must start inverted'

    trace, _ = _rollout(env, ctrl, init, 60)
    tilts = [t for t, _ in trace]
    assert tilts[-1] < np.pi / 2, f'tilt only fell from 180 to {np.degrees(tilts[-1]):.1f} deg'
    assert min(tilts) < np.radians(30.0)


def test_inverted_at_rest_commands_a_real_torque(env, ctrl):
    '''The singular case, at the level of a single action.

    At exactly pi the tilt error vanishes and an unguarded law returns four
    equal motor thrusts -- zero torque, a permanent stall.  The action must
    instead be asymmetric across the roll pair, and the pair being asked to
    give way must be at the actuator FLOOR: the roll command is far larger
    than four bounded motors can realise around a `thrust_down` collective, so
    saturation resolves it in favour of torque.  That is the design (module
    docstring), and it is why the collective is a request, not a guarantee.
    '''
    low, _ = env.physical_action_bounds
    rot = np.diag([1.0, -1.0, -1.0])            # 180 deg about body x
    action = ctrl.select_action(_obs18(rot))
    assert action[0] > action[2] and action[1] > action[3], 'no roll torque when inverted'
    assert action[2:] == pytest.approx(low[2:], rel=1e-9)


def test_upright_at_rest_commands_no_torque(ctrl):
    '''The other zero of the error: upright must be a fixed point of the
    attitude law, not a place the singular tie-break fires.
    '''
    action = ctrl.select_action(_obs18(np.eye(3)))
    assert action == pytest.approx(np.full(4, action[0]), rel=1e-9)
    assert float(np.sum(action)) == pytest.approx(ctrl.thrust_up, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. It stays inside the actuator.
# ---------------------------------------------------------------------------

def test_actions_are_within_physical_action_bounds(env, ctrl):
    '''Over random attitudes on SO(3) crossed with body rates at the env's own
    termination bound -- the widest input the closed state space can present.
    '''
    low, high = env.physical_action_bounds
    rng = np.random.default_rng(0)
    for _ in range(200):
        quat = quat_wxyz_to_pybullet(random_rotation_quat(rng))
        rot = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        rates = rng.uniform(-RATE_BOUND, RATE_BOUND, size=3)
        action = ctrl.select_action(_obs18(rot, rates))
        assert action.shape == (4,)
        assert np.all(action >= low - 1e-12) and np.all(action <= high + 1e-12)


def test_actions_stay_in_bounds_along_a_rollout(env, ctrl):
    '''The same guarantee on the states the controller actually drives itself
    into, which is not the same distribution as the sampler above.
    '''
    low, high = env.physical_action_bounds
    obs, info = set_initial_state(env, _state(euler=(2.5, 0.4, 1.0), rates=(5.0, -7.0, 3.0)))
    for _ in range(80):
        action = ctrl.select_action(ctrl.obs_normalizer(ctrl1_observation(env, obs)), info)
        assert np.all(action >= low - 1e-12) and np.all(action <= high + 1e-12)
        obs, _, done, info = env.step(action)
        if done:
            break


# ---------------------------------------------------------------------------
# 3. It is deterministic and stateless.
# ---------------------------------------------------------------------------

def test_same_state_gives_the_identical_action(env, ctrl):
    '''Bitwise identical, twice in a row AND after the controller has been
    driven through an unrelated state -- there is no phase latch, so history
    cannot change the answer.
    '''
    rng = np.random.default_rng(3)
    quat = quat_wxyz_to_pybullet(random_rotation_quat(rng))
    rot = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    obs = _obs18(rot, rates=(1.5, -2.0, 0.5))

    first = ctrl.select_action(obs)
    assert np.array_equal(ctrl.select_action(obs), first)

    ctrl.select_action(_obs18(np.diag([1.0, -1.0, -1.0]), rates=(9.0, 9.0, 9.0)))
    assert np.array_equal(ctrl.select_action(obs), first)

    # And a controller that has never been called at all agrees.
    assert np.array_equal(GeometricFlipController3D(env).select_action(obs), first)


def test_action_ignores_position_and_translational_velocity(ctrl):
    '''Controller 1 is an ATTITUDE controller (spec D1/D2): its action must be
    a function of the rotation matrix and body rates alone.  If position ever
    leaks in, G1 starts being pulled toward RoA2.
    '''
    rot = np.asarray(p.getMatrixFromQuaternion(
        p.getQuaternionFromEuler([0.6, -1.1, 2.0]))).reshape(3, 3)
    here = ctrl.select_action(_obs18(rot, rates=(1.0, 2.0, -3.0),
                                     pos_vel=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)))
    elsewhere = ctrl.select_action(_obs18(rot, rates=(1.0, 2.0, -3.0),
                                          pos_vel=(1.7, -2.9, -1.7, 2.9, 0.2, -2.9)))
    assert np.array_equal(here, elsewhere)


def test_rejects_the_wrong_observation_width(ctrl):
    '''The 12-dim native observation carries Euler angles at 6:8; silently
    reading them as rotation-matrix entries would be the 3D restatement of
    Finding C1.  Fail loudly instead.
    '''
    with pytest.raises(ValueError):
        ctrl.select_action(np.zeros(12))


# ---------------------------------------------------------------------------
# 4. It reaches G_nom from 180 deg (attitude-only).
# ---------------------------------------------------------------------------

def test_attitude_only_reaches_g_nom_from_180_degrees(env, ctrl):
    '''The isolated attitude-only sanity check: position at the stabilization
    goal, zero translational velocity, zero body rates, tilt exactly pi.
    `G_NOM_3D` is the region controller 1 is built against (10 deg, 4 rad/s).
    '''
    trace, _ = _rollout(env, ctrl, _state(pos=(0.0, 0.0, 1.0), euler=(np.pi, 0.0, 0.0)), 200)
    assert any(G_NOM_3D.contains(t, w) for t, w in trace), (
        f'never entered G_nom; best tilt {np.degrees(min(t for t, _ in trace)):.1f} deg')


# ---------------------------------------------------------------------------
# 5. It drops into the composition unchanged.
# ---------------------------------------------------------------------------

def test_rollout_composite_accepts_it_as_controller_1():
    '''`rollout3d.rollout_composite` is written against `select_action` and
    `obs_normalizer` only -- never against SAC -- so this object is controller
    1 with NO edit to `rollout3d.py`.  Builds its own env because it needs
    controller 2 (LQR) alongside.
    '''
    env, ctrl2 = make_env_and_ctrl2()
    try:
        ctrl1 = make_ctrl1(env)
        assert isinstance(ctrl1.obs_normalizer, IdentityNormalizer)
        result = rollout_composite(env, ctrl1, ctrl2, G_NOM_3D,
                                   _state(euler=(0.0, np.pi, 0.0)), max_steps=300)
        assert result.flip_success, 'controller 1 did not reach G1 from inversion'
        assert result.handoff_index > 0
        # After the latch, controller 2 flies -- the handoff is not the last
        # state, i.e. the composition really did continue.
        assert len(result.trajectory) > result.handoff_index + 1
    finally:
        ctrl2.close()
        env.close()


# ---------------------------------------------------------------------------
# 6. The mixer's signs, against the real dynamics.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('axis, tau, rate_index', [
    (0, 0.002, 9),    # +tau_x -> +p
    (1, 0.002, 10),   # +tau_y -> +q
    (2, 0.0005, 11),  # +tau_z -> +r
])
def test_mixer_signs_match_the_real_dynamics(env, axis, tau, rate_index):
    '''A positive commanded torque must produce a positive body rate on ITS
    OWN axis and (near) nothing on the others.  The whole control law is built
    on this convention; a flipped sign here turns the PD into positive
    feedback, so it is measured against PyBullet rather than argued on paper.
    '''
    mixer_inv = np.linalg.inv(mixer(env))
    low, high = env.physical_action_bounds
    command = np.zeros(4)
    command[0] = env.MASS * env.GRAVITY_ACC
    command[1 + axis] = tau

    set_initial_state(env, _state())
    obs, _, _, _ = env.step(np.clip(mixer_inv @ command, low, high))
    rates = np.array([obs[9], obs[10], obs[11]])
    assert rates[axis] > 0.05
    assert np.all(np.abs(np.delete(rates, axis)) < 1e-3)
