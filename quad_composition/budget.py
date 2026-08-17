'''Analytic lower bound on the vertical velocity a rotation must cost.

To rotate from tilt t0 down to t*, the drone crosses an arc where vertical
acceleration is negative whatever it commands, because thrust points downward
when inverted.  The thrust-optimal schedule is max thrust while cos(theta) > 0
and min thrust while cos(theta) < 0; traversing at the maximum permitted body
rate minimises time spent there.  Hence

    dzd_min(t0) = (1/w_max) * integral from t* to t0 of zdd*(th) dth

which is a strict lower bound on the cost of any rotation, for any controller.
'''

import numpy as np

G = 9.81

# norm_act_scale=0.1 (spec D6): total thrust 0.23838..0.29136 N over m=0.027 kg.
A_MIN_RESTRICTED = 8.829
A_MAX_RESTRICTED = 10.791

# Physical Crazyflie, for comparison only.
A_MIN_PHYSICAL = 4.172
A_MAX_PHYSICAL = 21.976


def _zdd_star(theta, a_min, a_max):
    return np.where(theta < np.pi / 2, a_max * np.cos(theta) - G,
                    a_min * np.cos(theta) - G)


def min_delta_zdot(tilt0, tilt_target, w_max, a_min, a_max, n=4001):
    '''Least possible change in vertical velocity rotating tilt0 -> tilt_target.'''
    tilt0 = float(abs(tilt0))
    if tilt0 <= tilt_target:
        return 0.0
    grid = np.linspace(tilt_target, tilt0, n)
    return float(np.trapz(_zdd_star(grid, a_min, a_max), grid) / w_max)


def budget_feasible(states, tilt_target, w_max, zd_bound, a_min, a_max):
    '''Can each dataset-order state reach tilt_target without breaking |zdot|?'''
    s = np.atleast_2d(np.asarray(states, dtype=float))
    tilts, zd = np.abs(s[:, 2]), s[:, 4]
    losses = np.array([min_delta_zdot(t, tilt_target, w_max, a_min, a_max)
                       for t in tilts])
    return (zd + losses) >= -zd_bound
