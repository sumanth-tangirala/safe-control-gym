'''Tests for the split train/eval collection scheme.

Spec: docs/superpowers/specs/2026-07-25-noisy-pendulum-collection-design.md
'''

import math
import os
import sys

import numpy as np
import pytest

SHIPPED_LQR_LABELS = ('/common/users/shared/pracsys/genMoPlan/data_trajectories/'
                      'deterministic/pendulum/lqr/roa_labels.txt')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# --- grid -------------------------------------------------------------------

def test_grid_is_half_open_and_stays_within_the_domain():
    '''theta is periodic, so [-pi, pi) must not include both endpoints.'''
    from generate_inverted_pendulum_trajectories import sample_initial_states
    grid = sample_initial_states(0, False, 0, 2 * math.pi, 0.04)
    theta = np.unique(grid[:, 0])
    theta_dot = np.unique(grid[:, 1])
    assert theta.max() < math.pi, 'theta must not reach or exceed +pi'
    assert theta_dot.max() < 2 * math.pi, 'theta_dot must not reach or exceed +2pi'
    assert len(theta) == 158
    assert len(theta_dot) == 315
    assert len(grid) == 49770


def test_grid_reproduces_the_shipped_deterministic_dataset():
    '''The 49,770 grid points must be the ones the shipped datasets were built on.

    Compared as sorted point sets: roa_labels.txt rows are in trajectory-index
    order, not grid order. The residual is ~3e-6 because the external repo
    started from -3.14159 (pi truncated to 5 dp) rather than -pi.
    '''
    if not os.path.exists(SHIPPED_LQR_LABELS):
        pytest.skip('shipped dataset not mounted')
    from generate_inverted_pendulum_trajectories import sample_initial_states
    reference = np.loadtxt(SHIPPED_LQR_LABELS, delimiter=',')[:, :2]
    grid = sample_initial_states(0, False, 0, 2 * math.pi, 0.04)
    assert grid.shape == reference.shape
    order = np.lexsort  # sort both by (theta_dot, theta) so rows correspond
    grid = grid[order((grid[:, 1], grid[:, 0]))]
    reference = reference[order((reference[:, 1], reference[:, 0]))]
    assert np.abs(grid - reference).max() < 1e-5
