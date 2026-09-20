'''Shared quad3d setup for the twin-curtain stochastic collection.

Everything about the plant is transcribed from generate_quadrotor_3d_noisy.py
(env construction, LQR controller, the body/world rate injection asymmetry).
The only difference is the disturbance: a pair of vertical Gaussian curtains
standing at x = +X_C and x = -X_C, unbounded in y and z, force along +y.

Design: .superpowers/sdd/quad3d-corridor-design-memo.md

TWIN CURTAIN [user, 2026-08-19 -- revises the memo's single-curtain choice]
----------------------------------------------------------------------------
The memo's §B2 recommendation was one curtain at x = +0.9. The adopted design
is TWO: x = +0.9 AND x = -0.9, both forcing +y, each an independent
'altitude_gated_sine' disturbance entry. DisturbanceList seeds every entry in
its list with its own child RNG stream (disturbances.py's DisturbanceList.seed,
spawned from the env's seed sequence in list order), so the two curtains draw
independent (A, phi) -- amplitude and phase -- every rollout. This is two
independent gusts standing back-to-back, not one symmetric field with a single
random draw.

Consequence for the goal-calm budget (memo §A): the two curtains' skirts both
reach x = 0, so the mean load at the goal is the SUM of both contributions.
`sigma(0, 1.0)` (this module) is the combined envelope and it is ~2x the
single-curtain memo value:

  - gate(0) doubles: 1.53e-3 -> ~3.06e-3 of peak.
  - quarter-ball F_max ceiling roughly halves: 0.743 N -> ~0.372 N
    (F_max <= 2 * 5.70e-4 / gate(0), memo §A's rule, gate(0) now ~3.06e-3).
  - cost at the goal at F_max = 0.20 roughly doubles: 6.7% -> ~13.4% of the
    stabilization ball (‖x_ss‖ = 21.925 * sigma(0, F_max) / 2, memo §A).

The sweep's proposed ladder top of 0.50 N (see q3_corridor_sweep.py) is above
this twin ceiling on purpose -- a deliberate over-the-line probe, not an
oversight.
'''
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from functools import partial  # noqa: E402

from generate_quadrotor_3d_noisy import (DET, STATE_BOUNDS, TASK_INFO, eval_starts,  # noqa: E402,F401
                                         inject_sampler, inject_stored, rollout_seed, to_row13)
from safe_control_gym.utils.registration import make  # noqa: E402

# --- geometry ----------------------------------------------------------
PROFILE = 'gaussian'          # ALTITUDE_PROFILES entry
X_C = 0.9                     # curtain centre, m; the mirror stands at -X_C
WIDTH = 0.25                  # gaussian width parameter, m
GATE_COORD_INDEX = 0          # env state order [x, x_dot, y, y_dot, z, ...] -- x is index 0
TWIN = True                   # two curtains at +-X_C, not one

# Altitudes (here: x) where a single curtain's profile is above 1% of ITS OWN
# peak. Everything outside both intervals is calm enough that a crossing is
# effectively undisturbed by that curtain.
_K = WIDTH * math.sqrt(2 * math.log(100))
BAND = ((X_C - _K, X_C + _K), (-X_C - _K, -X_C + _K))   # (+curtain, -curtain)

SINE_PERIOD = 2.0   # seconds; re-anchored on the curtain's measured 1.45s crossing (memo §B2 item 3)
PERIOD = SINE_PERIOD

HORIZON = 2000       # control steps at ctrl_freq=100 -> 20s; the deadline decision (memo §D)


def sigma(x, f_max):
    '''Combined draw-bound envelope at x, in newtons -- the sum of the two
    independent curtains' peaks at +X_C and -X_C.

    Each curtain draws its own noise independently (see the module docstring),
    but their MEANS add: mean force at x is sigma(x, f_max) / 2, because the
    mean of a sum of independent draws is the sum of their means. This is the
    quantity the goal-calm budget (memo §A) is checked against.
    '''
    def _profile(centre):
        return math.exp(-0.5 * ((x - centre) / WIDTH) ** 2)
    return f_max * (_profile(X_C) + _profile(-X_C))


# A noise model names the full disturbance stack for a collection campaign.
# 'corridor' scales with f_max; 'ambient' is a fixed zero-mean Gaussian side
# force (std in newtons) that costs no standing tilt and blurs the success
# boundary in both directions. Swapping models is changing one name; every
# dataset description records the resolved stack. Mirrors q2_corridor_common's
# NOISE_MODELS shape exactly (memo §C.2); the entries mean different things
# (an x-gated plane, not a z-gated altitude band) so this is a twin registry,
# not a shared one.
NOISE_MODELS = {
    'sine': {'corridor': 'altitude_gated_sine', 'ambient': 0.0},
    'sine+ambient': {'corridor': 'altitude_gated_sine', 'ambient': None},  # ambient required
    'ambient': {'corridor': None, 'ambient': None},                   # ambient required
    'uniform': {'corridor': 'altitude_gated', 'ambient': 0.0},        # falsified for quad2d; kept for reproduction
}


def resolve_noise_model(model, f_max, ambient=None):
    '''Resolve a NOISE_MODELS entry plus (f_max, ambient) into the concrete
    disturbance stack build() installs -- also what a dataset description
    embeds, so a reader can reproduce the stack without re-deriving it from
    the model name.

    Returns {'model', 'f_max', 'ambient', 'corridor'} where 'corridor' is
    either None (no corridor term) or a list of TWO dicts -- the +X_C curtain
    first, then the -X_C curtain, a fixed and documented order -- each handed
    to `disturbances` as one dynamics-channel entry.
    '''
    if model not in NOISE_MODELS:
        raise ValueError(f'[ERROR] q3_corridor_common.resolve_noise_model(): '
                         f'unknown model {model!r}; choose from {sorted(NOISE_MODELS)}.')
    entry = NOISE_MODELS[model]

    if entry['ambient'] is None:
        if ambient is None:
            raise ValueError(f'[ERROR] q3_corridor_common.resolve_noise_model(): '
                             f'model {model!r} requires an ambient std (got None).')
        resolved_ambient = float(ambient)
    else:
        if ambient is not None:
            raise ValueError(f'[ERROR] q3_corridor_common.resolve_noise_model(): '
                             f'model {model!r} has a fixed ambient of '
                             f"{entry['ambient']}; do not pass ambient= (got {ambient!r}).")
        resolved_ambient = float(entry['ambient'])

    corridor_func = entry['corridor']
    if corridor_func is None:
        if f_max != 0:
            raise ValueError(f'[ERROR] q3_corridor_common.resolve_noise_model(): '
                             f'model {model!r} has no corridor term, so f_max must '
                             f'be exactly 0 (got {f_max!r}).')
    elif f_max < 0:
        raise ValueError(f'[ERROR] q3_corridor_common.resolve_noise_model(): '
                         f'f_max must be >= 0 (got {f_max!r}).')

    corridor = None
    if corridor_func is not None and f_max > 0:
        base = {'disturbance_func': corridor_func, 'f_max': float(f_max),
                'profile': PROFILE, 'width': WIDTH,
                # mask multiplies the disturb_force 3-vector elementwise, and
                # quadrotor.py hands that vector to pybullet unchanged for a
                # THREE_D quad. So [0, 1, 0] puts the force on Y while
                # GATE_COORD_INDEX gates on X: a crosswind sheet that shoves the
                # drone sideways as it crosses, not a headwind it flies into.
                # Probed 2026-08-19 at the curtain peak: force = [0, 0.068, 0].
                'state_index': GATE_COORD_INDEX, 'mask': [0, 1, 0]}
        if corridor_func == 'altitude_gated_sine':
            base['period'] = SINE_PERIOD
        # Fixed order: +X_C curtain first, then -X_C. DisturbanceList spawns
        # child RNG streams in list order, so this order is also what fixes
        # which stream each curtain draws from -- reproducibility depends on
        # never reordering this.
        corridor = [dict(base, centre=X_C), dict(base, centre=-X_C)]

    return {'model': model, 'f_max': float(f_max), 'ambient': resolved_ambient,
            'corridor': corridor}


def build(f_max, model='sine+ambient', ambient=None):
    '''model selects a NOISE_MODELS entry -- the full disturbance stack for a
    campaign. Clones generate_quadrotor_3d_noisy.build()'s env kwargs, with
    two changes: episode_len_sec=20 (HORIZON=2000 at ctrl_freq=100, the
    deadline decision, memo §D) instead of the collector's own huge allowance,
    and the twin-curtain disturbance stack in place of the collector's
    zero-mean 3-axis uniform bound.

    Twin curtain: for any corridor-bearing model this installs TWO independent
    AltitudeGatedSineNoise (or AltitudeGatedNoise, for model='uniform')
    instances -- one centred at x=+X_C, one at x=-X_C -- followed by the
    ambient white-noise term if the model has one. See the module docstring
    for why the two curtains draw independent gusts.
    '''
    stack = resolve_noise_model(model, f_max, ambient)

    kw = dict(quad_type=3, task='stabilization', task_info=TASK_INFO,
              ctrl_freq=100, pyb_freq=5000, gui=False, randomized_init=False,
              episode_len_sec=20, cost='quadratic', done_on_out_of_bound=True)
    dynamics = []
    if stack['corridor'] is not None:
        dynamics.extend(dict(d) for d in stack['corridor'])   # +X_C then -X_C
    if stack['ambient'] > 0:
        # Zero-mean per-step wobble via upstream WhiteNoise, +y only. Costs no
        # standing tilt (no mean), so it may act at the goal.
        dynamics.append({'disturbance_func': 'white_noise',
                        'std': stack['ambient'], 'mask': [0, 1, 0]})
    if dynamics:
        kw['disturbances'] = {'dynamics': dynamics}
    env_func = partial(make, 'quadrotor', **kw)
    env = env_func()
    for i, (lo, hi) in STATE_BOUNDS.items():
        env.state_space.low[i], env.state_space.high[i] = lo, hi
    ctrl = make('lqr', env_func, q_lqr=[1] * 12, r_lqr=[0.1] * 4,
                discrete_dynamics=True)
    return env, ctrl


def roll(env, ctrl, row13, seed, keep=False, track_band=False):
    '''One eval rollout from a shipped 13-D grouped/quaternion start state
    (eval_states.txt columns 0:13). Uses inject_stored -- the eval path's
    world = R @ body conversion (see generate_quadrotor_3d_noisy's module
    docstring and its inject_stored). Do NOT use this with a sampler-drawn
    12-D start; that path needs inject_sampler's body-as-world conflation
    instead (see generate_quadrotor_3d_noisy.inject_sampler and this module's
    re-export of it).

    Entry-cut success: the rollout stops the instant info['goal_reached'] is
    True, matching generate_quadrotor_3d_noisy.run() -- the entry state IS
    the last stored state, which keeps "terminal state in the goal set" and
    "label 1" the same statement.

    `track_band` additionally reports whether the trajectory ever entered
    either curtain's 1%-of-peak band in x (gate >= 1% of that curtain's own
    peak), which the eval split can use to skip states neither curtain can
    reach.
    '''
    env.reset(seed=int(seed))
    ctrl.reset()
    obs = inject_stored(env, row13)
    info = {'current_step': 0}
    traj = [to_row13(env.state)] if keep else None

    def _gate_x():
        x = float(np.asarray(env.state).ravel()[GATE_COORD_INDEX])
        return min(abs(x - X_C), abs(x + X_C)) <= _K

    entered = _gate_x()
    success = False
    steps = 0
    for steps in range(1, HORIZON + 1):
        obs, _, terminated, truncated, info = env.step(ctrl.select_action(obs, info))
        if keep:
            traj.append(to_row13(env.state))
        if track_band and not entered:
            entered = _gate_x()
        if info.get('goal_reached', False):
            success = True
            break
        if terminated or truncated:
            break
    if track_band:
        return success, steps, traj, entered
    return success, steps, traj
