'''The supervisory guard g(x): does controller 2 (LQR) alone succeed from a
dataset-order 13-dim initial state x?

THE PIECE OF THE ORIGINAL DESIGN THAT WAS NEVER BUILT: the composition
(`rollout3d.rollout_composite`) always ran controller 1 (the flip policy)
first, even from states already inside RoA2 -- and over the 40000-state
paired baseline/composite comparison, 1051 of those states were ones LQR
ALONE would have solved, that the unguarded composition broke. The guard
lets `rollout_composite` decline to run controller 1 in exactly the states
where it is not needed.

FITTING (see `fit_quad3d_guard.py`, run once, offline): a logistic regression
over four physically-motivated, rotation/translation-invariant scalars,
fitted on `quadrotor3D_lqr_regenerated/eval_states.txt` (40000 rows: init(13),
final(13), ctrl2_success) with an 80/20 train/test split (seed 42). Held-out
test accuracy 0.839, precision 0.721, recall 0.412 (n=8000; see
`fit_quad3d_guard.py`'s docstring and the guard3d report for the full
candidate comparison -- a majority-class floor, a single threshold on
distance-from-goal, and a depth-3 decision tree were all tried and are less
accurate or less simple).

The coefficients below are the FROZEN result of that fit -- this module has
no sklearn dependency at runtime; `predict` is five multiplications, a sum,
and a sigmoid, evaluated directly on `rollout_composite`'s hot path.

TILT COMES FROM THE ROTATION MATRIX, NEVER FROM EULER ANGLES -- see
`rollout3d.tilt_from_quat_wxyz` and rollout3d.py's module docstring, item 1.
`guard_features` calls it directly; there is no other admissible way to
extract attitude from a dataset-order state on this branch.
'''

import math
from dataclasses import dataclass

import numpy as np

from quad_composition.rollout3d import GOAL_STATE, QUAT_SLICE, RATE_SLICE, tilt_from_quat_wxyz

FEATURE_NAMES = ('tilt_rad', 'omega_norm', 'speed', 'dist_from_goal')


def guard_features(state):
    '''[tilt, |omega|, speed, dist_from_goal] from one dataset-order 13-dim
    state [x, y, z, qw, qx, qy, qz, x_dot, y_dot, z_dot, p, q, r].

    All four are scalars an LQR practitioner would reach for first: LQR is a
    local linear controller, so its region of attraction is expected to
    shrink as any of these grows. Rotation/translation-invariant by
    construction (norms and a goal-relative distance, not raw per-axis
    components), so the guard generalises across the DIRECTION of an offset,
    not just its magnitude along particular axes.
    '''
    s = np.asarray(state, dtype=float)
    tilt = tilt_from_quat_wxyz(s[QUAT_SLICE])
    omega = float(np.linalg.norm(s[RATE_SLICE]))
    speed = float(np.linalg.norm(s[7:10]))
    dist = float(np.linalg.norm(s[0:3] - np.asarray(GOAL_STATE[0:3])))
    return np.array([tilt, omega, speed, dist], dtype=float)


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


@dataclass(frozen=True)
class LogisticGuard:
    '''g(x) = sigmoid(coef . (features(x) - mean) / std + intercept) >= threshold.

    `mean`/`std` are the TRAIN-split standardization used when fitting
    `coef`/`intercept` -- see `fit_quad3d_guard.py::fit_logreg`. `threshold`
    defaults to the standard 0.5 decision boundary; it is a plain field
    (not baked into `coef`/`intercept`) so a caller can build a
    higher-precision variant from the SAME fitted model by passing a
    different threshold, without refitting.
    '''

    mean: tuple
    std: tuple
    coef: tuple
    intercept: float
    threshold: float = 0.5

    def predict_proba(self, state):
        features = guard_features(state)
        mean = np.asarray(self.mean, dtype=float)
        std = np.asarray(self.std, dtype=float)
        coef = np.asarray(self.coef, dtype=float)
        z = float(np.dot(coef, (features - mean) / std)) + self.intercept
        return _sigmoid(z)

    def predict(self, state):
        '''True: predicts controller 2 (LQR) alone succeeds from `state` --
        `rollout_composite`'s guard hook should run LQR alone. False:
        predicts LQR alone fails -- run the existing (unguarded) composition.
        '''
        return self.predict_proba(state) >= self.threshold

    def __call__(self, state):
        return self.predict(state)


# ---------------------------------------------------------------------------
# The frozen fit. Reproduce with:
#   python3 fit_quad3d_guard.py
# mean/std/coef/intercept are copied verbatim from that run's `logreg`
# candidate (seed=42, test_frac=0.2) -- see the guard3d report for the full
# metrics table this was chosen from.
# ---------------------------------------------------------------------------
FITTED_GUARD = LogisticGuard(
    mean=(1.267272154302145, 14.199645179315207, 2.4967555379521533, 1.641848055866723),
    std=(0.6803459990245617, 7.846489530460009, 0.8297072949190303, 0.5154809911266641),
    coef=(-0.8213880425202105, -0.4271514855081449, -0.7023659305928698, -0.3470623101342991),
    intercept=-1.8356290780949776,
    threshold=0.5,
)


def lqr_success_guard(state):
    '''Default guard callable: `rollout3d.rollout_composite(..., guard=lqr_success_guard)`.'''
    return FITTED_GUARD.predict(state)
