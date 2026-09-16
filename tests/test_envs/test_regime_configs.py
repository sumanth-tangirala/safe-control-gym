'''The regime files must describe one dynamical system, in sync with the record.

configs/system/<name>.yaml holds the plant; the regime files extend it and add
kill boxes. Two invariants are load-bearing and cheap to assert:

- reference_run in the collection regime is documentation nobody consumes, and
  the values that must reach the env are duplicated into the system file's
  task_config_overrides "kept in sync by hand". Hand-sync is the failure mode
  this repo has already paid for once -- a policy trained at 10 N was scored
  against a 2000 N reference -- so the sync is asserted here instead.
- The collection and physical regimes must resolve to the SAME plant through
  the extends chain, differing only in where episodes die.
'''
import os
from xml.etree import ElementTree

import yaml

from safe_control_gym.experiments.train_sb3 import load_collection_bounds

REPO = os.path.join(os.path.dirname(__file__), '..', '..')
SYSTEM = os.path.join(REPO, 'configs/system/cartpole.yaml')
COLLECTION = os.path.join(REPO, 'configs/collection/cartpole.yaml')
PHYSICAL = os.path.join(REPO, 'configs/physical/cartpole.yaml')

PENDULUM_SYSTEM = os.path.join(REPO, 'configs/system/inverted_pendulum.yaml')
PENDULUM_COLLECTION = os.path.join(REPO, 'configs/collection/inverted_pendulum.yaml')
PENDULUM_PHYSICAL = os.path.join(REPO, 'configs/physical/inverted_pendulum.yaml')

QUAD3D_SYSTEM = os.path.join(REPO, 'configs/system/quadrotor3d.yaml')
QUAD3D_COLLECTION = os.path.join(REPO, 'configs/collection/quadrotor3d.yaml')
QUAD3D_PHYSICAL = os.path.join(REPO, 'configs/physical/quadrotor3d.yaml')

QUAD2D_SYSTEM = os.path.join(REPO, 'configs/system/quadrotor2d.yaml')
QUAD2D_COLLECTION = os.path.join(REPO, 'configs/collection/quadrotor2d.yaml')
QUAD2D_PHYSICAL = os.path.join(REPO, 'configs/physical/quadrotor2d.yaml')


def raw(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def test_reference_run_matches_the_system_plant():
    '''The documented generation config and the consumed plant cannot drift.'''
    reference = raw(COLLECTION)['reference_run']
    plant = raw(SYSTEM)['task_config_overrides']
    assert reference['control_bound'] == plant['action_scale']
    assert reference['ctrl_freq'] == plant['ctrl_freq']
    assert reference['pyb_freq'] == plant['pyb_freq']


def test_regimes_share_one_plant():
    '''Collection and physical differ ONLY in the termination box.'''
    collection = load_collection_bounds(COLLECTION)
    physical = load_collection_bounds(PHYSICAL)
    for key in ('task_config_overrides', 'init_state_randomization_info',
                'state_layout', 'angle_observation', 'normalize_observation',
                'normalized_rl_action_space'):
        assert collection[key] == physical[key], f'{key} differs between regimes'
    assert collection['env_attributes'] != physical['env_attributes']


def test_extends_resolves_and_child_wins():
    '''The regime file's own keys survive the merge; inherited keys arrive.'''
    resolved = load_collection_bounds(COLLECTION)
    assert 'extends' not in resolved
    # Inherited from the system file.
    assert resolved['task_config_overrides']['action_scale'] == 2000.0
    # The child's own keys, absent from the system file.
    assert resolved['reference_task'] == 'stabilization'
    assert resolved['env_attributes']['x_dot_threshold'] == 5.0


def test_pendulum_reference_run_matches_the_system_plant():
    '''Same hand-sync hazard, and here it bites on the integrator rate.

    The pendulum's shipped dataset ran three explicit-Euler substeps per control
    step; the env default is one, and one reproduces 299/300 sampled labels
    instead of 300/300. Nothing consumes reference_run, so only this assertion
    stops the documented run and the consumed plant from drifting apart.
    '''
    reference = raw(PENDULUM_COLLECTION)['reference_run']
    plant = raw(PENDULUM_SYSTEM)['task_config_overrides']
    assert reference['control_bound'] == plant['u_sat']
    assert reference['ctrl_freq'] == plant['ctrl_freq']
    assert reference['pyb_freq'] == plant['pyb_freq']
    assert reference['theta_dot_max'] == plant['theta_dot_max']
    assert reference['goal_threshold'] == plant['goal_threshold']


def test_pendulum_regimes_share_one_plant():
    '''Collection and physical differ ONLY in the reference block.

    Unlike the cartpole there is no termination box to differ IN:
    inverted_pendulum._get_done has no out-of-bounds test, theta is wrapped and
    theta_dot is clipped, so neither regime carries env_attributes and the two
    resolve to identical env behaviour. Asserted rather than assumed, because a
    stray env_attributes block here would silently invent a kill boundary the
    reference dataset never had.
    '''
    collection = load_collection_bounds(PENDULUM_COLLECTION)
    physical = load_collection_bounds(PENDULUM_PHYSICAL)
    assert 'env_attributes' not in collection
    assert 'env_attributes' not in physical
    reference_only = {'reference_success', 'reference_dataset', 'reference_task',
                      'reference_run'}
    assert set(collection) - set(physical) == reference_only
    for key in set(physical):
        assert collection[key] == physical[key], f'{key} differs between regimes'


def test_pendulum_extends_resolves_and_child_wins():
    '''The regime file's own keys survive the merge; inherited keys arrive.'''
    resolved = load_collection_bounds(PENDULUM_COLLECTION)
    assert 'extends' not in resolved
    # Inherited from the system file.
    assert resolved['task_config_overrides']['pyb_freq'] == 300
    assert resolved['task_config_overrides']['u_sat'] == 0.6371781908344007
    assert resolved['state_layout'] == ['theta', 'theta_dot']
    # The child's own keys, absent from the system file.
    assert resolved['reference_task'] == 'reach'
    assert resolved['reference_dataset'] == 'deterministic/pendulum_lqr_50k'

    # The physical regime is nothing but the extends line.
    assert set(raw(PENDULUM_PHYSICAL)) == {'extends'}


def test_quad2d_reference_run_matches_the_system_plant():
    '''Same hand-sync hazard; here it bites on the integrator rate and the goal.

    The env defaults are 60 Hz control over 240 Hz physics and a 0.05 m goal
    tolerance. The shipped dataset ran 100/5000 with a 0.2 m tolerance, and
    under terminate_on_goal the tolerance is what decides where the episode ends
    and therefore what the terminal state -- and so the label -- is.
    '''
    reference = raw(QUAD2D_COLLECTION)['reference_run']
    plant = raw(QUAD2D_SYSTEM)['task_config_overrides']
    assert reference['ctrl_freq'] == plant['ctrl_freq']
    assert reference['pyb_freq'] == plant['pyb_freq']
    assert reference['episode_len_sec'] == plant['episode_len_sec']
    assert reference['success_threshold'] == \
        plant['task_info']['stabilization_goal_tolerance']
    # The goal is written [x, z, ...] in the file order the success rule uses,
    # and [x, z] in the env's stabilization_goal.
    assert reference['success_goal'][:2] == \
        [float(v) for v in plant['task_info']['stabilization_goal']]
    assert reference['normalized_rl_action_space'] is True


def test_quad2d_reference_run_records_the_damping_bug():
    '''The one plant fact no constructor kwarg can carry.

    Every shipped quadrotor dataset ran at PyBullet's DEFAULT damping because
    base_aviary's changeDynamics call targeted client 0; the library is fixed,
    so an env built from these configs runs at zero damping and is a different
    plant. Measured over three shipped trajectories, one open-loop step from
    each recorded state: 1.1e-6 deviation at damping 0.04, 1.2e-2 at 0.

    Asserted so that the fact survives in a file somebody runs, not only in a
    comment -- and so that deleting it from reference_run fails loudly.
    '''
    reference = raw(QUAD2D_COLLECTION)['reference_run']
    assert reference['linear_damping'] == 0.04
    assert reference['angular_damping'] == 0.04


def test_quad2d_regimes_share_one_plant():
    '''Collection and physical differ ONLY in the termination box.

    The quadrotors terminate on state_space rather than on the threshold
    attributes cartpole uses, so the box that differs is state_space_bounds and
    neither regime carries env_attributes.
    '''
    collection = load_collection_bounds(QUAD2D_COLLECTION)
    physical = load_collection_bounds(QUAD2D_PHYSICAL)
    for key in ('task_config_overrides', 'init_state_randomization_info',
                'state_layout', 'angle_observation', 'normalize_observation',
                'normalized_rl_action_space', 'curriculum'):
        assert collection[key] == physical[key], f'{key} differs between regimes'
    assert 'env_attributes' not in collection
    assert 'env_attributes' not in physical
    assert collection['state_space_bounds'] != physical['state_space_bounds']


def test_quad2d_physical_relaxes_the_collection_box():
    '''The physical kill box must be a strict superset, or the ROA comparison lies.'''
    collection = load_collection_bounds(QUAD2D_COLLECTION)['state_space_bounds']
    physical = load_collection_bounds(QUAD2D_PHYSICAL)['state_space_bounds']
    for channel, (low, high) in physical.items():
        c_low, c_high = collection[channel]
        assert low <= c_low, f'{channel} lower bound is tighter than collection'
        assert high >= c_high, f'{channel} upper bound is tighter than collection'
    # theta is deliberately absent from the physical box: _get_done masks the
    # angle out of the out-of-bounds test, so it is a normalisation scale only.
    assert 'theta' not in physical
    assert 'theta' in collection


def test_quad2d_extends_resolves_and_child_wins():
    '''The regime file's own keys survive the merge; inherited keys arrive.'''
    resolved = load_collection_bounds(QUAD2D_COLLECTION)
    assert 'extends' not in resolved
    # Inherited from the system file.
    assert resolved['task_config_overrides']['pyb_freq'] == 5000
    assert resolved['task_config_overrides']['ctrl_freq'] == 100
    assert resolved['state_layout'] == ['x', 'x_dot', 'z', 'z_dot',
                                        'theta', 'theta_dot']
    # The child's own keys, absent from the system file.
    assert resolved['reference_task'] == 'reach'
    assert resolved['reference_dataset'] == 'deterministic/quadrotor2D_rl'
    assert resolved['state_space_bounds']['theta_dot'] == [-8.0, 8.0]

    # The physical regime is the extends line plus its kill box.
    assert set(raw(QUAD2D_PHYSICAL)) == {'extends', 'state_space_bounds'}


def test_quad2d_urdf_carries_the_documented_physical_parameters():
    '''Mass and inertia are NOT pinned in task_config_overrides, on purpose.

    `inertial_prop` assigns self.MASS after loadURDF has already built the body,
    so pinning it there desynchronises the simulated body from the model the
    goal and the symbolic dynamics are computed from. The values the dataset
    description records therefore have to be guarded against cf2x.urdf instead.
    '''
    urdf = os.path.join(REPO, 'safe_control_gym/envs/gym_pybullet_drones/'
                              'assets/cf2x.urdf')
    tree = ElementTree.parse(urdf).getroot()
    inertial = tree.find('.//link[@name="base_link"]/inertial')
    assert float(inertial.find('mass').get('value')) == 0.027
    assert float(inertial.find('inertia').get('iyy')) == 1.4e-5
    assert float(tree.find('properties').get('arm')) == 0.0397


def test_quad3d_reference_run_matches_the_system_plant():
    """Same hand-sync hazard; here it bites on the integrator rate.

    The env defaults are 60 Hz control over 240 Hz physics. The shipped dataset
    ran 100/5000, and its regime files pinned no plant at all until 2026-07-31,
    so training ran at the defaults against a reference collected at 100/5000.
    """
    reference = raw(QUAD3D_COLLECTION)['reference_run']
    plant = raw(QUAD3D_SYSTEM)['task_config_overrides']
    assert reference['ctrl_freq'] == plant['ctrl_freq']
    assert reference['pyb_freq'] == plant['pyb_freq']
    assert reference['success_threshold'] == \
        plant['task_info']['stabilization_goal_tolerance']
    # The goal is [x, y, z] in the env's stabilization_goal; success_goal is the
    # full 12-D env state, whose z sits at index 4.
    assert reference['success_goal'][4] == plant['task_info']['stabilization_goal'][2]
    assert reference['normalized_rl_action_space'] is False


def test_quad3d_horizon_is_shortened_but_never_truncating():
    """The system file deliberately does NOT match the dataset's horizon.

    The collector ran episode_len_sec 1000 (100,000 control steps, 5 million
    physics steps per episode), which never came close to binding: the longest
    shipped trajectory is 636 states. The system file pins 10 s = 1000 steps so
    a training episode is finite. That is only safe while it still clears the
    longest trajectory, so the ordering is asserted rather than trusted.
    """
    reference = raw(QUAD3D_COLLECTION)['reference_run']
    plant = raw(QUAD3D_SYSTEM)['task_config_overrides']
    plant_steps = plant['episode_len_sec'] * plant['ctrl_freq']
    reference_steps = reference['episode_len_sec'] * reference['ctrl_freq']
    assert reference['longest_shipped_trajectory'] < plant_steps <= reference_steps


def test_quad3d_reference_run_records_the_damping_bug():
    """The one plant fact no constructor kwarg can carry.

    On this system the changeDynamics call was not merely misdirected but a
    no-op: the LQR takes PyBullet client 0 and never resets it, so client 0 held
    zero bodies and the call addressed a body that did not exist. Measured
    replaying shipped trajectories, step-1 error 1.5e-5 flat at damping 0.04
    against 1.0e-1 and growing at 0.
    """
    reference = raw(QUAD3D_COLLECTION)['reference_run']
    assert reference['linear_damping'] == 0.04
    assert reference['angular_damping'] == 0.04


def test_quad3d_regimes_share_one_plant():
    """Collection and physical differ ONLY in the termination box."""
    collection = load_collection_bounds(QUAD3D_COLLECTION)
    physical = load_collection_bounds(QUAD3D_PHYSICAL)
    for key in ('task_config_overrides', 'init_state_randomization_info',
                'state_layout', 'angle_observation', 'rotation_observation',
                'normalize_observation', 'normalized_rl_action_space',
                'curriculum'):
        assert collection[key] == physical[key], f'{key} differs between regimes'
    assert 'env_attributes' not in collection
    assert 'env_attributes' not in physical
    assert collection['state_space_bounds'] != physical['state_space_bounds']


def test_quad3d_physical_relaxes_the_collection_box():
    """The physical kill box must be a strict superset, or the ROA comparison lies."""
    collection = load_collection_bounds(QUAD3D_COLLECTION)['state_space_bounds']
    physical = load_collection_bounds(QUAD3D_PHYSICAL)['state_space_bounds']
    for channel, (low, high) in physical.items():
        c_low, c_high = collection[channel]
        assert low <= c_low, f'{channel} lower bound is tighter than collection'
        assert high >= c_high, f'{channel} upper bound is tighter than collection'
    # The three Euler angles are deliberately absent from the physical box:
    # _get_done masks them out of the out-of-bounds test, so they are a
    # normalisation scale only.
    for angle in ('phi', 'theta', 'psi'):
        assert angle not in physical
        assert angle in collection


def test_quad3d_extends_resolves_and_child_wins():
    """The regime file's own keys survive the merge; inherited keys arrive."""
    resolved = load_collection_bounds(QUAD3D_COLLECTION)
    assert 'extends' not in resolved
    # Inherited from the system file.
    assert resolved['task_config_overrides']['pyb_freq'] == 5000
    assert resolved['task_config_overrides']['ctrl_freq'] == 100
    assert resolved['state_layout'] == ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                                        'phi', 'theta', 'psi', 'p', 'q', 'r']
    # The child's own keys, absent from the system file.
    assert resolved['reference_task'] == 'reach'
    assert resolved['reference_dataset'] == 'deterministic/quadrotor3D_lqr'
    assert resolved['state_space_bounds']['p'] == [-24.0, 24.0]

    # The physical regime is the extends line plus its kill box.
    assert set(raw(QUAD3D_PHYSICAL)) == {'extends', 'state_space_bounds'}


def test_quad3d_urdf_carries_the_documented_physical_parameters():
    """Mass and inertia are NOT pinned in task_config_overrides, on purpose.

    Same reasoning as quadrotor2d: `inertial_prop` assigns self.MASS after
    loadURDF has already built the body. quadrotor3d additionally depends on
    izz, which the 2D system never uses.
    """
    urdf = os.path.join(REPO, 'safe_control_gym/envs/gym_pybullet_drones/'
                              'assets/cf2x.urdf')
    tree = ElementTree.parse(urdf).getroot()
    inertial = tree.find('.//link[@name="base_link"]/inertial')
    assert float(inertial.find('mass').get('value')) == 0.027
    assert float(inertial.find('inertia').get('ixx')) == 1.4e-5
    assert float(inertial.find('inertia').get('izz')) == 2.17e-5
    assert float(tree.find('properties').get('arm')) == 0.0397
