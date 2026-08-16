'''Tests for the attitude-only handoff region G1.

Spec: docs/superpowers/specs/2026-08-16-quad-composition-g1-design.md (D1)
'''
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from quad_composition.g1 import G1Region, attitude_2d


def test_membership_is_exclusive_at_the_boundary():
    g1 = G1Region(tilt_c=0.2, w_c=1.5)
    assert g1.contains(np.array([0.19]), np.array([1.4]))[0]
    assert not g1.contains(np.array([0.20]), np.array([1.4]))[0]
    assert not g1.contains(np.array([0.19]), np.array([1.5]))[0]


def test_membership_uses_magnitude_so_sign_does_not_matter():
    g1 = G1Region(tilt_c=0.2, w_c=1.5)
    assert g1.contains(np.array([-0.19]), np.array([-1.4]))[0]


def test_attitude_2d_reads_theta_and_theta_dot_not_position():
    # dataset order [x, z, theta, x_dot, z_dot, theta_dot]
    states = np.array([[0.5, 1.2, -0.3, 0.4, -0.6, 2.5]])
    tilt, omega = attitude_2d(states)
    assert tilt[0] == 0.3
    assert omega[0] == 2.5


def test_round_trips_through_a_dict():
    g1 = G1Region(tilt_c=0.2, w_c=1.5)
    assert G1Region.from_dict(g1.to_dict()) == g1
    assert math.isclose(g1.to_dict()['tilt_c_deg'], math.degrees(0.2))
