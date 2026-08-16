'''The handoff region G1.

G1's FORM is attitude-only and fixed a priori (spec D1): a recovery controller
has authority over attitude and nothing else, so an attitude-only goal region is
the non-contrived choice.  Position and translational velocity must never enter
this definition -- if they did, G1 would be pulled toward RoA2 and the whole
experiment would be circular.
'''

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class G1Region:
    '''G1 = {|theta| < tilt_c, |theta_dot| < w_c}.'''

    tilt_c: float   # radians
    w_c: float      # rad/s

    def contains(self, tilt, omega):
        '''Elementwise membership.  Half-open: the boundary is outside.'''
        tilt = np.abs(np.asarray(tilt, dtype=float))
        omega = np.abs(np.asarray(omega, dtype=float))
        return (tilt < self.tilt_c) & (omega < self.w_c)

    def to_dict(self):
        return {
            'form': 'attitude_only',
            'tilt_c_rad': float(self.tilt_c),
            'tilt_c_deg': float(np.degrees(self.tilt_c)),
            'w_c_rad_s': float(self.w_c),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(tilt_c=float(d['tilt_c_rad']), w_c=float(d['w_c_rad_s']))


def attitude_2d(states):
    '''(|theta|, |theta_dot|) from dataset-order [x, z, theta, x_dot, z_dot, theta_dot].'''
    s = np.atleast_2d(np.asarray(states, dtype=float))
    return np.abs(s[:, 2]), np.abs(s[:, 5])
