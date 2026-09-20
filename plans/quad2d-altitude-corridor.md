# quad2d Altitude Corridor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect a quad2d stochastic dataset under a one-sided +x disturbance whose
magnitude is gated by altitude, forming a corridor of disturbed air the vehicle must
cross to reach the goal.

**Architecture:** One new `Disturbance` subclass reads altitude off the env and uses it
as the upper bound of a one-sided uniform draw. The altitude envelope is a *strategy*:
resolved by name from an `ALTITUDE_PROFILES` registry (only `'gaussian'` — profile D —
exists now), and the draw law is isolated in a `_draw(bound)` method so a future law is
a subclass override. Everything else reuses the existing `dynamics` disturbance channel
and the five-script q2 collection pipeline (common, validate, sweep, collect, reduce).
Two measurement scripts run before collection and their results decide the level ladder.

**Tech Stack:** Python 3.10, numpy, PyBullet, gymnasium, pytest, `safe_control_gym`
registry. Design spec: `docs/superpowers/specs/2026-08-17-quad2d-altitude-corridor-design.md`.

## Global Constraints

- Noise model: `sigma(z) = f_max * profile(z)`, `F_x ~ U(0, sigma(z))`, redrawn every control step, `F_z = 0`.
- The profile is a named strategy from `ALTITUDE_PROFILES`. The only entry is `'gaussian'`: `exp(-0.5 * ((z - centre) / width) ** 2)` with centre `0.55`, width `0.12`. Do not change these values without revising the spec; new profiles are new registry entries, never edits to `'gaussian'`.
- The draw law lives in `AltitudeGatedNoise._draw(bound)` and nowhere else. An alternative law (the spec's banked sine) is a subclass overriding `_draw`, not an edit.
- Disturbance mode is `dynamics`; for `TWO_D` the 2-vector maps to `[Fx, 0, Fz]`, so the mask is `[1, 0]`.
- Plant settings transcribed from `q2_common.py` and never re-derived: `ctrl_freq=100`, `pyb_freq=5000`, `episode_len_sec=1000`, `cost='quadratic'`, `done_on_out_of_bound=True`, `normalized_rl_action_space=True`, `constraints=SAFE_EXPLORER_CONSTRAINTS`, `done_on_violation=False`.
- `HORIZON = 1200`, `TOL = 0.2` (entry-cut), goal `[0, 1]`.
- State bounds applied after construction: `x +/-1.0`, `x_dot +/-1.0`, `z [0.1, 1.5]`, `z_dot +/-1.0`, `theta_dot +/-8.0`; theta unbounded.
- Controller is `safe_explorer_ppo` with the shipped model and `obs_normalizer.set_read_only()`.
- Env state order is `[x, x_dot, z, z_dot, theta, theta_dot]`, so altitude is index `2`.
- Level bracket for the sweep: `f_max` in `0.002`-`0.020` N. Do not seed the sweep from the existing family's `0.070`-`0.200` N.
- Style is enforced by `pre-commit` only: single quotes, `'''` docstrings. A hook denies `--no-verify`.
- Datasets are outputs. Never write under `DATA_ROOT`; a hook denies it.
- Collection runs are hours long. Never launch one in a turn's foreground.

---

### Task 1: `AltitudeGatedNoise` disturbance class

**Files:**
- Modify: `safe_control_gym/envs/disturbances.py` (add class after `SignalDependentNoise`, ~line 317)
- Test: `tests/test_envs/test_altitude_gated_noise.py`

**Interfaces:**
- Consumes: `Disturbance` base class — `__init__(self, env, dim, mask=None, **kwargs)`, `seed(self, env, stream=None)` which sets `self.np_random`, and `self.mask` as a float32 array or `None`.
- Produces: `ALTITUDE_PROFILES` dict mapping name -> `fn(z, params) -> float in [0, 1]`; `AltitudeGatedNoise(env, dim, mask=None, f_max=0.0, profile='gaussian', state_index=2, **profile_params)` with `sigma(z) -> float`, `_draw(bound) -> ndarray` (the only place variates are drawn), and `apply(target, env) -> ndarray`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_envs/test_altitude_gated_noise.py`:

```python
'''AltitudeGatedNoise: a one-sided draw whose bound is a function of altitude.

The load-bearing properties are that the bound tracks the vehicle's altitude,
that the draw is one-sided so the family has a non-zero mean, and that the
number of variates consumed does not depend on f_max -- which is what keeps
two levels paired under the same seed.
'''
import numpy as np
import pytest

from safe_control_gym.envs.disturbances import ALTITUDE_PROFILES, AltitudeGatedNoise


class _FakeEnv:
    '''Minimal stand-in: the class only ever reads `state` and `np_random`.'''

    def __init__(self, z):
        self.state = np.array([0.0, 0.0, z, 0.0, 0.0, 0.0])
        self.np_random = np.random.default_rng(0)


def _make(z, **kwargs):
    env = _FakeEnv(z)
    cfg = dict(f_max=0.05, centre=0.55, width=0.12)
    cfg.update(kwargs)
    dist = AltitudeGatedNoise(env, dim=2, mask=[1, 0], **cfg)
    dist.seed(env)
    return dist, env


def test_sigma_peaks_at_the_centre():
    dist, _ = _make(0.55)
    assert dist.sigma(0.55) == pytest.approx(0.05)


def test_sigma_falls_off_as_a_gaussian():
    dist, _ = _make(0.55)
    # one width out is exp(-0.5) of the peak
    assert dist.sigma(0.67) == pytest.approx(0.05 * np.exp(-0.5), rel=1e-6)


def test_sigma_is_effectively_zero_at_the_goal():
    '''The whole point of placing the corridor below the goal.'''
    dist, _ = _make(0.55)
    assert dist.sigma(1.0) < 1e-4


def test_draw_is_one_sided_and_within_the_bound():
    dist, env = _make(0.55)
    for _ in range(200):
        out = dist.apply(np.zeros(2), env)
        assert 0.0 <= out[0] <= dist.sigma(0.55)


def test_mask_kills_the_z_component():
    dist, env = _make(0.55)
    for _ in range(50):
        out = dist.apply(np.zeros(2), env)
        assert out[1] == 0.0


def test_bound_tracks_the_env_altitude():
    '''Same disturbance object, two altitudes, different support.'''
    dist, env = _make(0.55)
    env.state[2] = 0.55
    hot = max(dist.apply(np.zeros(2), env)[0] for _ in range(400))
    env.state[2] = 0.95
    cold = max(dist.apply(np.zeros(2), env)[0] for _ in range(400))
    assert cold < hot / 10


def test_variate_count_is_independent_of_f_max():
    '''Levels stay paired: the same seed consumes the same stream at any f_max.'''
    outs = []
    for f_max in (0.01, 0.05):
        env = _FakeEnv(0.55)
        dist = AltitudeGatedNoise(env, dim=2, mask=[1, 0], f_max=f_max,
                                  centre=0.55, width=0.12)
        dist.seed(env, np.random.default_rng(7))
        outs.append([dist.apply(np.zeros(2), env)[0] for _ in range(20)])
    ratio = np.array(outs[1]) / np.array(outs[0])
    assert np.allclose(ratio, 5.0)


def test_rejects_a_negative_f_max():
    env = _FakeEnv(0.55)
    with pytest.raises(ValueError, match='f_max'):
        AltitudeGatedNoise(env, dim=2, f_max=-1.0)


def test_rejects_a_non_positive_width():
    env = _FakeEnv(0.55)
    with pytest.raises(ValueError, match='width'):
        AltitudeGatedNoise(env, dim=2, f_max=0.05, width=0.0)


def test_rejects_an_unknown_profile():
    env = _FakeEnv(0.55)
    with pytest.raises(ValueError, match='profile'):
        AltitudeGatedNoise(env, dim=2, f_max=0.05, profile='tanh_wall')


def test_a_registered_profile_is_swappable():
    '''The strategy seam: a new envelope is a registry entry, nothing else.'''
    ALTITUDE_PROFILES['flat_test'] = lambda z, params: 1.0
    try:
        dist, _ = _make(0.55, profile='flat_test')
        assert dist.sigma(0.1) == pytest.approx(0.05)
        assert dist.sigma(1.4) == pytest.approx(0.05)
    finally:
        del ALTITUDE_PROFILES['flat_test']


def test_draw_law_lives_only_in_draw():
    '''A subclass overriding _draw changes the law without touching apply().'''

    class _Frozen(AltitudeGatedNoise):
        def _draw(self, bound):
            return np.full(self.dim, bound)

    env = _FakeEnv(0.55)
    dist = _Frozen(env, dim=2, mask=[1, 0], f_max=0.05, centre=0.55, width=0.12)
    dist.seed(env)
    out = dist.apply(np.zeros(2), env)
    assert out[0] == pytest.approx(dist.sigma(0.55))
    assert out[1] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_envs/test_altitude_gated_noise.py -v`
Expected: FAIL — `ImportError: cannot import name 'AltitudeGatedNoise'`

- [ ] **Step 3: Write the implementation**

In `safe_control_gym/envs/disturbances.py`, after `SignalDependentNoise`:

```python
def _gaussian_profile(z, params):
    '''Profile D from the corridor spec: a Gaussian bump, in [0, 1].'''
    if params['width'] <= 0:
        raise ValueError('[ERROR] AltitudeGatedNoise: width must be positive.')
    return np.exp(-0.5 * ((z - params['centre']) / params['width']) ** 2)


# The altitude envelope is a strategy. A new shape is a new entry here plus its
# parameters in the disturbance config -- nothing else changes. Entries map
# name -> fn(z, params) returning a value in [0, 1]; f_max supplies the units.
ALTITUDE_PROFILES = {'gaussian': _gaussian_profile}


class AltitudeGatedNoise(Disturbance):
    '''One-sided uniform noise whose bound is a function of altitude.

        F ~ U(0, sigma(z)),   sigma(z) = f_max * profile(z)

    ``sigma`` is the BOUND of the draw, not a scale on a fixed distribution --
    the support itself moves with altitude, so mean (sigma/2) and standard
    deviation (sigma/(2 sqrt(3))) are locked at a ratio of 1/sqrt(3).

    One-sided on purpose. Every other family here is symmetric about zero; this
    one has a non-zero mean, so the vehicle must hold a standing tilt against it
    rather than merely reject jitter.

    ``profile`` names an entry in ``ALTITUDE_PROFILES``; the remaining kwargs
    are handed to it untouched. The draw law lives in ``_draw`` and nowhere
    else, so an alternative law (the spec's banked per-rollout sinusoid) is a
    subclass overriding ``_draw``, not an edit here.

    The altitude is read from ``env.state``, which ``_get_observation`` refreshes
    at the end of each step, so at ``before_step`` time it is the state entering
    this step. It is the TRUE state -- the observation disturbance is applied to
    a copy afterwards -- which is what we want: the corridor is a property of the
    airspace, not of what the vehicle believes.

    The draw is normalised then scaled rather than drawn directly on
    ``[0, sigma]``, so the number of variates consumed does not depend on
    ``f_max``. Two levels sharing a seed then see the same underlying stream,
    which is the pairing property ``rollout_seed`` exists to provide.
    '''

    def __init__(self,
                 env,
                 dim,
                 mask=None,
                 f_max=0.0,
                 profile='gaussian',
                 state_index=2,
                 **profile_params
                 ):
        super().__init__(env, dim, mask)
        if f_max < 0:
            raise ValueError('[ERROR] AltitudeGatedNoise.__init__(): f_max must be '
                             'non-negative; it is the upper bound of a one-sided draw.')
        if profile not in ALTITUDE_PROFILES:
            raise ValueError(f'[ERROR] AltitudeGatedNoise.__init__(): unknown profile '
                             f'{profile!r}; registered: {sorted(ALTITUDE_PROFILES)}.')
        self.f_max = float(f_max)
        self.profile = profile
        self.profile_params = {k: float(v) for k, v in profile_params.items()}
        self.state_index = int(state_index)
        self.sigma(0.0)   # fail at construction, not mid-rollout, on bad params

    def sigma(self, z):
        '''The draw bound at altitude z, in newtons.'''
        return self.f_max * ALTITUDE_PROFILES[self.profile](z, self.profile_params)

    def _draw(self, bound):
        '''The draw law -- the ONLY place variates are consumed.'''
        return self.np_random.uniform(0.0, 1.0, size=self.dim) * bound

    def apply(self,
              target,
              env
              ):
        z = float(np.asarray(env.state).ravel()[self.state_index])
        noise = self._draw(self.sigma(z))
        if self.mask is not None:
            noise *= self.mask
        disturbed = target + noise
        return disturbed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_envs/test_altitude_gated_noise.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add safe_control_gym/envs/disturbances.py tests/test_envs/test_altitude_gated_noise.py
git commit -m "Add AltitudeGatedNoise with a pluggable altitude-profile registry"
```

---

### Task 2: Register the type and verify it reaches the plant

**Files:**
- Modify: `safe_control_gym/envs/disturbances.py:370-376` (the `DISTURBANCE_TYPES` dict)
- Test: `tests/test_envs/test_altitude_corridor_env.py`

**Interfaces:**
- Consumes: `AltitudeGatedNoise` from Task 1; `make('quadrotor', ...)` from `safe_control_gym.utils.registration`.
- Produces: the config key `'altitude_gated'`, usable as
  `disturbances={'dynamics': [{'disturbance_func': 'altitude_gated', 'f_max': F, 'profile': 'gaussian', 'centre': 0.55, 'width': 0.12, 'mask': [1, 0]}]}`.
  Profile params pass through `create_disturbance_list` untouched — it already forwards every key except `disturbance_func` as kwargs.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_envs/test_altitude_corridor_env.py`:

```python
'''The corridor disturbance as the env actually applies it.

Two things the unit tests cannot see: that the config key resolves, and that a
2-D dynamics disturbance masked to [1, 0] moves the vehicle in +x only.
'''
import numpy as np
import pytest

from safe_control_gym.utils.registration import make

TASK_INFO = {'stabilization_goal': [0, 1], 'stabilization_goal_tolerance': 0.2}


def _env(f_max, centre=0.55, width=0.12):
    return make('quadrotor', quad_type=2, task='stabilization', task_info=TASK_INFO,
                ctrl_freq=100, pyb_freq=5000, episode_len_sec=10, cost='quadratic',
                done_on_out_of_bound=False, gui=False, randomized_init=False,
                disturbances={'dynamics': [{'disturbance_func': 'altitude_gated',
                                            'f_max': f_max, 'profile': 'gaussian',
                                            'centre': centre, 'width': width,
                                            'mask': [1, 0]}]})


def test_config_key_resolves_to_the_class():
    from safe_control_gym.envs.disturbances import AltitudeGatedNoise
    env = _env(0.05)
    try:
        dist = env.disturbances['dynamics'].disturbances[0]
        assert isinstance(dist, AltitudeGatedNoise)
        assert dist.dim == 2
    finally:
        env.close()


def test_hover_inside_the_band_drifts_positive_x():
    '''A one-sided force has a mean, so the drift has a sign.'''
    env = _env(0.05)
    try:
        env.reset(seed=0)
        import pybullet as pb
        pb.resetBasePositionAndOrientation(
            env.DRONE_ID, [0, 0, 0.55], pb.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=env.PYB_CLIENT)
        env._update_and_store_kinematic_information()
        env._get_observation()   # refresh env.state after the teleport
        hover = env.U_GOAL.copy()   # per-pair thrust mg/2, physical units
        for _ in range(100):
            env.step(hover)
        assert env.state[0] > 0.01      # x has moved +
        assert env.state[1] > 0.0       # and is still moving +
    finally:
        env.close()


def test_hover_at_the_goal_altitude_does_not_drift():
    '''sigma(1.0) is ~0.09% of peak, so the goal is calm by construction.'''
    env = _env(0.05)
    try:
        env.reset(seed=0)
        import pybullet as pb
        pb.resetBasePositionAndOrientation(
            env.DRONE_ID, [0, 0, 1.0], pb.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=env.PYB_CLIENT)
        env._update_and_store_kinematic_information()
        env._get_observation()   # refresh env.state after the teleport
        hover = env.U_GOAL.copy()   # per-pair thrust mg/2, physical units
        for _ in range(100):
            env.step(hover)
        assert abs(env.state[0]) < 1e-3
    finally:
        env.close()


def test_f_max_zero_is_bit_identical_to_no_disturbance():
    '''The baseline level must be the same code path, not a different env.'''
    import pybullet as pb
    finals = []
    for dist in (True, False):
        env = _env(0.0) if dist else make(
            'quadrotor', quad_type=2, task='stabilization', task_info=TASK_INFO,
            ctrl_freq=100, pyb_freq=5000, episode_len_sec=10, cost='quadratic',
            done_on_out_of_bound=False, gui=False, randomized_init=False)
        try:
            env.reset(seed=3)
            pb.resetBasePositionAndOrientation(
                env.DRONE_ID, [0, 0, 0.55], pb.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=env.PYB_CLIENT)
            env._update_and_store_kinematic_information()
            env._get_observation()   # refresh env.state after the teleport
            hover = env.U_GOAL.copy()   # per-pair thrust mg/2, physical units
            for _ in range(50):
                env.step(hover)
            finals.append(env.state.copy())
        finally:
            env.close()
    assert np.allclose(finals[0], finals[1], atol=0, rtol=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_envs/test_altitude_corridor_env.py -v`
Expected: FAIL — `AssertionError: [ERROR] in BenchmarkEnv._setup_disturbances(), disturbance type not available.`

- [ ] **Step 3: Add the registry entry**

In `safe_control_gym/envs/disturbances.py`, extend `DISTURBANCE_TYPES`:

```python
DISTURBANCE_TYPES = {'impulse': ImpulseDisturbance,
                     'step': StepDisturbance,
                     'uniform': UniformNoise,
                     'white_noise': WhiteNoise,
                     'periodic': PeriodicNoise,
                     'signal_dependent': SignalDependentNoise,
                     'altitude_gated': AltitudeGatedNoise,
                     }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_envs/test_altitude_corridor_env.py -v`
Expected: 4 passed

If `test_f_max_zero_is_bit_identical_to_no_disturbance` fails, the cause is the
disturbance consuming variates from a stream the no-disturbance env does not
have. That is expected and acceptable only if the drift assertion still holds;
record the divergence in the spec rather than loosening the test silently.

- [ ] **Step 5: Commit**

```bash
git add safe_control_gym/envs/disturbances.py tests/test_envs/test_altitude_corridor_env.py
git commit -m "Register altitude_gated and pin that it drives +x inside the band only"
```

---

### Task 3: `q2_corridor_common.py` — shared build and roll

**Files:**
- Create: `q2_corridor_common.py`
- Test: `tests/test_envs/test_q2_corridor_common.py`

**Interfaces:**
- Consumes: `AltitudeGatedNoise` via the `'altitude_gated'` config key (Task 2); `q2_common.py` for the transcribed plant settings.
- Produces: `DET`, `MODEL`, `HORIZON=1200`, `TOL=0.2`, `CENTRE=0.55`, `WIDTH=0.12`, `BAND=(0.186, 0.914)`, `sigma(z, f_max)`, `build(f_max)` returning `(env, ctrl)`, `roll(env, ctrl, state6, seed, keep=False)` returning `(success, steps, traj)`, `rollout_seed(base, split_id, index, trial)`, `grid_states(lo, hi)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_envs/test_q2_corridor_common.py`:

```python
'''The corridor collector's shared setup.

Pinned here because these constants are transcribed from the shipped
deterministic set and a silent drift in any of them invalidates the whole
comparison.
'''
import importlib.util
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    'q2_corridor_common', os.path.join(ROOT, 'q2_corridor_common.py'))
q2c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q2c)


def test_plant_constants_match_the_shipped_set():
    assert q2c.HORIZON == 1200
    assert q2c.TOL == 0.2
    assert q2c.CENTRE == 0.55
    assert q2c.WIDTH == 0.12


def test_sigma_is_zero_at_the_goal_to_four_decimals():
    assert round(q2c.sigma(1.0, 0.05), 4) == 0.0


def test_band_bounds_are_the_one_percent_contour():
    lo, hi = q2c.BAND
    assert abs(q2c.sigma(lo, 1.0) - 0.01) < 1e-9
    assert abs(q2c.sigma(hi, 1.0) - 0.01) < 1e-9


def test_rollout_seed_excludes_the_level():
    '''Levels must be paired: same coordinates, same stream, any f_max.'''
    import inspect
    src = inspect.getsource(q2c.rollout_seed)
    assert 'f_max' not in src and 'level' not in src


def test_build_wires_the_corridor_to_the_dynamics_channel():
    from safe_control_gym.envs.disturbances import AltitudeGatedNoise
    env, ctrl = q2c.build(0.01)
    try:
        assert 'dynamics' in env.disturbances
        assert 'action' not in env.disturbances
        dist = env.disturbances['dynamics'].disturbances[0]
        assert isinstance(dist, AltitudeGatedNoise)
        assert dist.profile == 'gaussian'
        assert list(dist.mask) == [1.0, 0.0]
    finally:
        env.close()


def test_build_at_zero_installs_no_disturbance():
    env, ctrl = q2c.build(0.0)
    try:
        assert env.disturbances == {}
    finally:
        env.close()


def test_state_bounds_are_applied_after_construction():
    env, _ = q2c.build(0.0)
    try:
        assert env.state_space.low[2] == 0.1
        assert env.state_space.high[2] == 1.5
        assert env.state_space.high[5] == 8.0
        # index 4 (theta) is untouched by the corridor build: it keeps the
        # env's own finite default (85 deg), not an override.
        assert env.state_space.high[4] == env.theta_threshold_radians
    finally:
        env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_envs/test_q2_corridor_common.py -v`
Expected: FAIL — `FileNotFoundError` on `q2_corridor_common.py`

- [ ] **Step 3: Write the implementation**

Create `q2_corridor_common.py`:

```python
'''Shared quad2d setup for the altitude-corridor stochastic collection.

Everything about the plant is transcribed from q2_common.py, which in turn came
from deterministic/quadrotor2D_rl. The only difference is the disturbance: a
one-sided +x force whose bound is gated by altitude, forming a corridor of
disturbed air below the goal.

Spec: docs/superpowers/specs/2026-08-17-quad2d-altitude-corridor-design.md
'''
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.environ.get('SCG_REPO', '.'))
from functools import partial  # noqa: E402

from generate_quadrotor_2d_trajectories_rl import (ALGO_CONFIGS,  # noqa: E402
                                                   SAFE_EXPLORER_CONSTRAINTS,
                                                   normalize_angle)
from safe_control_gym.utils.registration import make  # noqa: E402

DET = os.environ.get(
    'Q2_DET_DIR',
    '/common/users/shared/pracsys/genMoPlan/data_trajectories/'
    'deterministic/quadrotor2D_rl')
MODEL = os.environ.get(
    'Q2_MODEL',
    'examples/rl/models/safe_explorer_ppo/safe_explorer_ppo_model_quadrotor_2D_stab.pt')

HORIZON = 1200
TOL = 0.2
GOAL = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])   # file order [x,z,theta,xd,zd,td]

PROFILE = 'gaussian'               # ALTITUDE_PROFILES entry -- profile D
CENTRE = 0.55                      # corridor centre, m
WIDTH = 0.12                       # gaussian width parameter, m
# Altitudes where sigma is above 1% of peak. Everything outside is calm enough
# that a trajectory crossing it is effectively undisturbed.
_K = WIDTH * math.sqrt(2 * math.log(100))
BAND = (CENTRE - _K, CENTRE + _K)   # (0.186, 0.914)

# env state order is [x, x_dot, z, z_dot, theta, theta_dot]
STATE_BOUNDS = {0: (-1.0, 1.0), 1: (-1.0, 1.0), 2: (0.1, 1.5),
                3: (-1.0, 1.0), 5: (-8.0, 8.0)}
TASK_INFO = {'stabilization_goal': [0, 1], 'stabilization_goal_tolerance': TOL}


def sigma(z, f_max):
    '''The draw bound at altitude z, in newtons.'''
    return f_max * math.exp(-0.5 * ((z - CENTRE) / WIDTH) ** 2)


def rollout_seed(base, split_id, index, trial):
    '''Pure function of the coordinates -- the noise scale is deliberately absent.

    A resumed shard draws exactly what an uninterrupted run would have drawn,
    and every rung of the ladder sees the same stream per (start, trial), so
    rungs are paired and rung-to-rung differences carry far less variance than
    the individual estimates.
    '''
    return int((base + split_id * 1_000_003 + index * 7919 + trial * 104_729)
               % (2 ** 31 - 1))


def build(f_max):
    kw = dict(quad_type=2, task='stabilization', task_info=TASK_INFO,
              ctrl_freq=100, pyb_freq=5000, gui=False, randomized_init=False,
              episode_len_sec=1000, cost='quadratic', done_on_out_of_bound=True,
              normalized_rl_action_space=True,
              constraints=SAFE_EXPLORER_CONSTRAINTS, done_on_violation=False)
    if f_max > 0:
        kw['disturbances'] = {'dynamics': [{'disturbance_func': 'altitude_gated',
                                            'f_max': f_max, 'profile': PROFILE,
                                            'centre': CENTRE, 'width': WIDTH,
                                            'mask': [1, 0]}]}
    env_func = partial(make, 'quadrotor', **kw)
    cfg = ALGO_CONFIGS['safe_explorer_ppo'].copy()
    tmp = tempfile.mkdtemp(prefix='q2corr-')
    ctrl = make('safe_explorer_ppo', env_func, **cfg, output_dir=tmp)
    ctrl.load(MODEL)
    ctrl.obs_normalizer.set_read_only()
    env = env_func()
    for i, (lo, hi) in STATE_BOUNDS.items():
        env.state_space.low[i], env.state_space.high[i] = lo, hi
    return env, ctrl


def roll(env, ctrl, state6, seed, keep=False, track_band=False):
    '''state6 is FILE order [x, z, theta, x_dot, z_dot, theta_dot].

    `track_band` additionally reports whether the trajectory ever entered the
    corridor, which the eval split uses to skip states the disturbance cannot
    reach.
    '''
    import pybullet as pb
    x, z, theta, x_dot, z_dot, theta_dot = state6
    obs, info = env.reset(seed=int(seed))
    pb.resetBasePositionAndOrientation(
        env.DRONE_ID, [x, 0, z], pb.getQuaternionFromEuler([0, theta, 0]),
        physicsClientId=env.PYB_CLIENT)
    pb.resetBaseVelocity(env.DRONE_ID, [x_dot, 0, z_dot], [0, theta_dot, 0],
                         physicsClientId=env.PYB_CLIENT)
    env._update_and_store_kinematic_information()
    obs = env._get_observation()
    if getattr(env, 'constraints', None) is not None:
        info['constraint_values'] = env.constraints.get_values(env, only_state=True)
    traj = [[x, z, normalize_angle(theta), x_dot, z_dot, theta_dot]] if keep else None
    entered = BAND[0] <= z <= BAND[1]
    success = False
    steps = 0
    for steps in range(1, HORIZON + 1):
        action = ctrl.select_action(ctrl.obs_normalizer(obs), info)
        obs, _, terminated, truncated, info = env.step(action)
        if keep:
            xx, xd, zz, zd, th, td = obs[:6]
            traj.append([xx, zz, normalize_angle(th), xd, zd, td])
        if track_band and not entered:
            entered = BAND[0] <= float(env.state[2]) <= BAND[1]
        if terminated or truncated:
            success = bool(info.get('goal_reached', False))
            break
    if track_band:
        return success, steps, traj, entered
    return success, steps, traj


def grid_states(lo, hi):
    '''Rows [lo:hi) of roa_labels.txt: 6 state columns plus the shipped label.'''
    rows = np.loadtxt(DET + '/roa_labels.txt', delimiter=',',
                      skiprows=lo, max_rows=hi - lo, ndmin=2)
    return rows[:, 0:6], rows[:, 6].astype(int)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_envs/test_q2_corridor_common.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add q2_corridor_common.py tests/test_envs/test_q2_corridor_common.py
git commit -m "Add the corridor collector's shared setup, transcribed from q2_common"
```

---

### Task 4: `q2_corridor_validate.py` — the level-0 gate

**Files:**
- Create: `q2_corridor_validate.py`

**Interfaces:**
- Consumes: `build`, `roll`, `grid_states`, `rollout_seed` from `q2_corridor_common`.
- Produces: a script printing `agreement N/M` and exiting non-zero below threshold. No importable API.

This gate must pass before any noisy level runs. `q2_validate.py` is the direct
precedent — read it first and match its shape.

- [ ] **Step 1: Write the script**

Create `q2_corridor_validate.py`:

```python
'''Level-0 gate: at f_max = 0 the corridor collector must reproduce the shipped
deterministic quad2d labels. Anything less means the plant transcription is
wrong and every noisy level built on it would be wrong the same way.

Usage: python q2_corridor_validate.py [--n 300] [--min_agreement 0.98]
'''
import argparse

import numpy as np

from q2_corridor_common import DET, build, roll, rollout_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--min_agreement', type=float, default=0.98)
    ap.add_argument('--base_seed', type=int, default=20260817)
    args = ap.parse_args()

    # Balanced sample, as in q2_validate.py: the grid is ordered, so a raw
    # prefix is all one corner where everything fails and passes trivially at
    # 100% without ever exercising the success branch.
    import os
    rows = np.loadtxt(os.path.join(DET, 'roa_labels.txt'), delimiter=',')
    lab = rows[:, 6].astype(int)
    rng = np.random.default_rng(0)
    pick = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(lab == 1), args.n // 2, replace=False),
        rng.choice(np.flatnonzero(lab == 0), args.n // 2, replace=False)]))
    starts, det_labels = rows[pick, 0:6], lab[pick]

    env, ctrl = build(0.0)
    try:
        got = np.zeros(len(starts), dtype=int)
        for i, s in enumerate(starts):
            ok, _, _ = roll(env, ctrl, s, rollout_seed(args.base_seed, 1, int(pick[i]), 0))
            got[i] = int(ok)
    finally:
        env.close()

    agree = int((got == det_labels).sum())
    frac = agree / len(starts)
    on1 = float((got[det_labels == 1] == 1).mean())
    on0 = float((got[det_labels == 0] == 0).mean())
    print(f'agreement {agree}/{len(starts)} = {frac:.4f}  '
          f'(success rows {on1:.3f}, failure rows {on0:.3f})')
    if frac < args.min_agreement:
        raise SystemExit(f'FAIL: below {args.min_agreement}')
    print('PASS')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the gate**

Run: `python q2_corridor_validate.py --n 300`
Expected: `agreement 29x/300 = 0.98xx` then `PASS`.

The shipped quad2d family reproduced at 0.9949, so anything below ~0.98 is a
transcription bug, not chaos. Do not proceed past this task until it passes.

- [ ] **Step 3: Commit**

```bash
git add q2_corridor_validate.py
git commit -m "Add the corridor level-0 gate against the shipped deterministic labels"
```

---

### Task 5: `q2_corridor_entry.py` — measure the entry rate

**Files:**
- Create: `q2_corridor_entry.py`

**Interfaces:**
- Consumes: `build`, `roll` (with `track_band=True`), `grid_states`, `BAND` from `q2_corridor_common`.
- Produces: `entry_rate.npz` with keys `entered` (uint8, per state), `start_z` (float64), `band` (2,). Consumed by Task 6 and Task 7.

Spec item 3. 42.86% of eval starts sit above the band; this measures how many
of them actually descend into it, which sets both the sweep's interpretation and
Task 7's skip margin.

- [ ] **Step 1: Write the script**

Create `q2_corridor_entry.py`:

```python
'''What fraction of trajectories actually enter the corridor.

Run at f_max = 0: the question is whether the UNDISTURBED trajectory passes
through the band, which is what decides whether the disturbance can reach that
start state at all. Noise can pull a marginal trajectory in, which is what the
margin in q2_corridor_collect.py is for.

Rows are sampled at random from the full grid -- a prefix of the ordered grid
is one corner and would measure that corner, not the set (the same trap the
level-0 gate hit). Workers follow q2_collect.py's Pool pattern: each builds
its own env and rolls a contiguous slice of the picked rows.

Usage: python q2_corridor_entry.py --n 20000 --procs 24 --out entry_rate.npz
'''
import argparse
import os
from multiprocessing import Pool

import numpy as np

from q2_corridor_common import BAND, DET, build, roll, rollout_seed

ARGS = None
S_PICK = None
IDX_PICK = None


def _init(a, s_pick, idx_pick):
    global ARGS, S_PICK, IDX_PICK
    ARGS, S_PICK, IDX_PICK = a, s_pick, idx_pick


def _range(rng_pair):
    lo, hi = rng_pair
    env, ctrl = build(0.0)
    entered = np.zeros(hi - lo, dtype=np.uint8)
    try:
        for i in range(lo, hi):
            _, _, _, hit = roll(env, ctrl, S_PICK[i],
                                rollout_seed(ARGS.base_seed, 1, int(IDX_PICK[i]), 0),
                                track_band=True)
            entered[i - lo] = int(hit)
    finally:
        env.close()
    return lo, hi, entered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20_000)
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--base_seed', type=int, default=20260817)
    ap.add_argument('--out', default='entry_rate.npz')
    args = ap.parse_args()

    rows = np.loadtxt(os.path.join(DET, 'roa_labels.txt'), delimiter=',')
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(rows), args.n, replace=False))
    picked = rows[idx, 0:6]

    edges = np.linspace(0, args.n, args.procs + 1).astype(int)
    ranges = [(int(edges[k]), int(edges[k + 1])) for k in range(args.procs)]
    entered = np.zeros(args.n, dtype=np.uint8)
    with Pool(args.procs, initializer=_init,
              initargs=(args, picked, idx)) as pool:
        for lo, hi, e in pool.imap_unordered(_range, ranges):
            entered[lo:hi] = e

    z = picked[:, 1]
    np.savez(args.out, entered=entered, start_z=z, row_index=idx,
             band=np.asarray(BAND))
    for lo, hi, label in [(0.0, BAND[0], 'below band'),
                          (BAND[0], BAND[1], 'inside band'),
                          (BAND[1], 2.0, 'above band')]:
        sel = (z >= lo) & (z < hi)
        if sel.sum():
            print(f'{label:<12} n={sel.sum():>6}  entered={entered[sel].mean():.4f}')
    print(f'overall entered = {entered.mean():.4f}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it in the background**

Run: `nohup python3 q2_corridor_entry.py --n 20000 --procs 24 --out entry_rate.npz > entry.log 2>&1 &`

Poll with `tail entry.log`. Do not block a turn on it.

- [ ] **Step 3: Record the result in the spec**

Append the three per-region entry rates to the "Corridor coverage" section of
`docs/superpowers/specs/2026-08-17-quad2d-altitude-corridor-design.md`, replacing
the note that the entry rate is unmeasured.

- [ ] **Step 4: Commit**

```bash
git add q2_corridor_entry.py docs/superpowers/specs/2026-08-17-quad2d-altitude-corridor-design.md
git commit -m "Measure how many quad2d trajectories actually enter the corridor"
```

---

### Task 6: `q2_corridor_sweep.py` — choose the level ladder

**Files:**
- Create: `q2_corridor_sweep.py`

**Interfaces:**
- Consumes: `build`, `roll`, `grid_states`, `rollout_seed` from `q2_corridor_common`.
- Produces: `sweep.npz` with keys `levels` (float64), `p_success` (float64, per level), `fraction_interior` (float64, per level), `hit_horizon` (int64, per level). The chosen four levels feed Task 7.

`q2_sweep.py` is the precedent — same shape, different level bracket. The
bracket is `0.002`-`0.020` N and must not be seeded from the existing family's
`0.070`-`0.200` N, which are zero-mean and un-gated.

- [ ] **Step 1: Write the script**

Create `q2_corridor_sweep.py`:

```python
'''Choose the corridor level ladder, and measure what the spec says to measure.

Reports three things per level: mean p_success, fraction_interior (the share of
states with 0 < p < 1, i.e. the transition shell), and the horizon-hit count.
fraction_interior is the load-bearing one. Per-step draws average along a
crossing -- estimated at a 4-9% spread on the delivered impulse -- so if the
shell is too thin the family is not usable and the spec's frozen-field fallback
should be revisited rather than the levels pushed higher.

States are sampled at random from the full grid (a prefix is one corner) and
each level runs the same picked states, seeded by original row index so levels
stay paired. Workers follow q2_collect.py's Pool pattern.

Usage: python q2_corridor_sweep.py --n 400 --trials 30 --procs 24 --out sweep.npz
'''
import argparse
import os
from multiprocessing import Pool

import numpy as np

from q2_corridor_common import DET, HORIZON, build, roll, rollout_seed

# The 0.002-0.020 bracket came from an open-loop impulse estimate and measured
# far too weak: retention 0.97 at 0.016, fraction_interior 0.0025. The entry
# data shows 90.75% of successes cross the band, so there is no structural
# ceiling -- the ladder extends upward until retention actually falls.
LEVELS = [0.0, 0.002, 0.004, 0.006, 0.009, 0.012, 0.016, 0.020]
EXT_LEVELS = [0.03, 0.05, 0.08, 0.13, 0.20, 0.30]

ARGS = None
S_PICK = None
IDX_PICK = None
LEVEL = None


def _init(a, s_pick, idx_pick, level):
    global ARGS, S_PICK, IDX_PICK, LEVEL
    ARGS, S_PICK, IDX_PICK, LEVEL = a, s_pick, idx_pick, level


def _range(rng_pair):
    lo, hi = rng_pair
    trials = 1 if LEVEL == 0 else ARGS.trials
    env, ctrl = build(LEVEL)
    p = np.zeros(hi - lo)
    hits = 0
    try:
        for i in range(lo, hi):
            ok_count = 0
            for k in range(trials):
                ok, steps, _ = roll(env, ctrl, S_PICK[i],
                                    rollout_seed(ARGS.base_seed, 1,
                                                 int(IDX_PICK[i]), k))
                ok_count += int(ok)
                hits += int(steps >= HORIZON)
            p[i - lo] = ok_count / trials
    finally:
        env.close()
    return lo, hi, p, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--base_seed', type=int, default=20260817)
    ap.add_argument('--out', default='sweep.npz')
    ap.add_argument('--levels', type=float, nargs='+', default=LEVELS)
    args = ap.parse_args()

    rows = np.loadtxt(os.path.join(DET, 'roa_labels.txt'), delimiter=',')
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(rows), args.n, replace=False))
    picked = rows[idx, 0:6]

    edges = np.linspace(0, args.n, args.procs + 1).astype(int)
    ranges = [(int(edges[k]), int(edges[k + 1])) for k in range(args.procs)]

    p_all, interior_all, horizon_all = [], [], []
    for level in args.levels:
        p = np.zeros(args.n)
        hits = 0
        with Pool(args.procs, initializer=_init,
                  initargs=(args, picked, idx, level)) as pool:
            for lo, hi, pv, h in pool.imap_unordered(_range, ranges):
                p[lo:hi] = pv
                hits += h
        interior = float(np.mean((p > 0) & (p < 1)))
        p_all.append(float(p.mean()))
        interior_all.append(interior)
        horizon_all.append(hits)
        print(f'f_max={level:.3f}  p_success={p.mean():.4f}  '
              f'fraction_interior={interior:.4f}  hit_horizon={hits}', flush=True)

    np.savez(args.out, levels=np.asarray(args.levels),
             p_success=np.asarray(p_all),
             fraction_interior=np.asarray(interior_all),
             hit_horizon=np.asarray(horizon_all),
             row_index=idx)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it in the background**

Run: `nohup python3 q2_corridor_sweep.py --n 400 --trials 30 --procs 24 --out sweep.npz > sweep.log 2>&1 &`

- [ ] **Step 3: Choose four levels and record them**

Pick four noisy levels spanning retention roughly 1.0 down to ~0.3 of the
`f_max = 0` baseline, matching the shape of the shipped families (quad2d ran
1.00 / 0.93 / 0.84 / 0.57 / 0.29). Append the table and the chosen levels to the
spec's "To be measured before collection" section, and change its status line
from `sweep not yet run`.

**Stop condition:** if `fraction_interior` stays below ~0.05 at every level, the
transition shell is too thin. Do not raise the levels to compensate — that
trades a measurement problem for a physics one. Revisit the frozen-field
alternative in the spec's Rejected alternatives.

- [ ] **Step 4: Commit**

```bash
git add q2_corridor_sweep.py docs/superpowers/specs/2026-08-17-quad2d-altitude-corridor-design.md
git commit -m "Sweep the corridor level bracket and record the chosen ladder"
```

---

### Task 7: `q2_corridor_collect.py` — the sharded collector

**Files:**
- Create: `q2_corridor_collect.py`

**Interfaces:**
- Consumes: `build`, `roll`, `grid_states`, `rollout_seed`, `BAND`, `HORIZON` from `q2_corridor_common`; the levels chosen in Task 6.
- Produces: per-shard `.npz`. Train keys: `states` (float32), `offsets` (int64), `starts` (float64), `labels` (uint8), `seeds` (int64), `lo`, `hi`, `f_max`. Eval keys: `starts`, `hits` (int32), `trials_used` (int32, per state), `det_labels`, `lo`, `hi`, `f_max`. Consumed by Task 8.

`q2_collect.py` is the precedent. Two differences: the level parameter is
`f_max`, and the eval split implements the spec's skip mitigation.

- [ ] **Step 1: Write the script**

Create `q2_corridor_collect.py`:

```python
'''quad2d stochastic collection under the altitude corridor.

Eval implements the spec's mitigation: a state whose undisturbed trajectory
never comes within MARGIN of the corridor cannot be reached by the disturbance,
so it is rolled once instead of `trials` times and its p_success is 0 or 1. The
margin covers trajectories that noise could pull into the band. `trials_used` is
stored per state so the reducer can report exactly how many were shortcut rather
than leaving it implicit.

Usage:
  python q2_corridor_collect.py --split train --level 0.009 \\
      --shard 0 --nshards 40 --out train_0.npz
  python q2_corridor_collect.py --split eval --level 0.009 --trials 50 \\
      --shard 0 --nshards 56 --out eval_0.npz
'''
import argparse
import os

import numpy as np

from q2_corridor_common import BAND, build, grid_states, roll, rollout_seed

N_STATES = 489_789
N_TRAIN = 500_000
TRAIN_SPLIT_ID, EVAL_SPLIT_ID = 0, 1
MARGIN = 0.10          # metres of altitude slack around the band


def sample_starts(n, seed):
    '''Random starts within the shipped sampling bounds, off the eval lattice.'''
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(-1.0, 1.0, n),      # x
        rng.uniform(0.1, 1.5, n),       # z
        rng.uniform(-np.pi, np.pi, n),  # theta
        rng.uniform(-1.0, 1.0, n),      # x_dot
        rng.uniform(-1.0, 1.0, n),      # z_dot
        rng.uniform(-8.0, 8.0, n),      # theta_dot
    ])


def shard_train(args, lo, hi):
    starts = sample_starts(N_TRAIN, args.base_seed)[lo:hi]
    env, ctrl = build(args.level, draw=args.draw)
    states, offsets, labels, seeds = [], [0], [], []
    try:
        for i in range(len(starts)):
            seed = rollout_seed(args.base_seed, TRAIN_SPLIT_ID, lo + i, 0)
            ok, _, traj = roll(env, ctrl, starts[i], seed, keep=True)
            states.append(np.asarray(traj, dtype=np.float32))
            offsets.append(offsets[-1] + len(traj))
            labels.append(ok)
            seeds.append(seed)
    finally:
        env.close()
    np.savez(args.out,
             states=np.concatenate(states),
             offsets=np.asarray(offsets, np.int64),
             starts=starts.astype(np.float64),
             labels=np.asarray(labels, np.uint8),
             seeds=np.asarray(seeds, np.int64),
             lo=lo, hi=hi, f_max=args.level, draw=args.draw)
    return int(np.sum(labels)), len(labels)


def shard_eval(args, lo, hi):
    starts, det_labels = grid_states(lo, hi)
    env, ctrl = build(args.level, draw=args.draw)
    hits = np.zeros(len(starts), dtype=np.int32)
    used = np.zeros(len(starts), dtype=np.int32)
    try:
        for i in range(len(starts)):
            # First roll doubles as the reachability probe.
            ok, _, _, entered = roll(env, ctrl, starts[i],
                                     rollout_seed(args.base_seed, EVAL_SPLIT_ID, lo + i, 0),
                                     track_band=True)
            hits[i] = int(ok)
            used[i] = 1
            reachable = entered or (
                BAND[0] - MARGIN <= starts[i][1] <= BAND[1] + MARGIN)
            if args.level == 0 or not reachable:
                continue
            for k in range(1, args.trials):
                ok, _, _ = roll(env, ctrl, starts[i],
                                rollout_seed(args.base_seed, EVAL_SPLIT_ID, lo + i, k))
                hits[i] += int(ok)
                used[i] += 1
    finally:
        env.close()
    np.savez(args.out, starts=starts, hits=hits, trials_used=used,
             det_labels=det_labels, lo=lo, hi=hi, f_max=args.level,
             draw=args.draw)
    return int(hits.sum()), int(used.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'eval'], required=True)
    ap.add_argument('--level', type=float, required=True)
    ap.add_argument('--draw', choices=['uniform', 'sine'], default='sine')
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--nshards', type=int, required=True)
    ap.add_argument('--base_seed', type=int, default=20260817)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # Idempotent: a completed shard is never redone, so resubmitting a partly
    # failed array costs only the missing work.
    if os.path.exists(args.out):
        print(f'{args.out} exists, skipping')
        return

    total = N_TRAIN if args.split == 'train' else N_STATES
    edges = np.linspace(0, total, args.nshards + 1).astype(int)
    lo, hi = int(edges[args.shard]), int(edges[args.shard + 1])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    fn = shard_train if args.split == 'train' else shard_eval
    got, n = fn(args, lo, hi)
    print(f'{args.split} f_max={args.level} shard {args.shard}/{args.nshards} '
          f'[{lo}:{hi}] -> {got}/{n}', flush=True)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke-test one tiny shard of each split**

Run:
```bash
python q2_corridor_collect.py --split eval --level 0.009 --trials 5 \
    --shard 0 --nshards 4898 --out /tmp/smoke_eval.npz
python -c "
import numpy as np
d = np.load('/tmp/smoke_eval.npz')
print('states', d['starts'].shape, 'trials_used min/max',
      d['trials_used'].min(), d['trials_used'].max())
assert d['trials_used'].min() >= 1
assert d['hits'].max() <= d['trials_used'].max()
print('OK')
"
```
Expected: `trials_used` between 1 and 5, `OK`.

- [ ] **Step 3: Commit**

```bash
git add q2_corridor_collect.py
git commit -m "Add the corridor collector, skipping trials on states the band cannot reach"
```

---

### Task 8: `q2_corridor_reduce.py` — dataset layout and description

**Files:**
- Create: `q2_corridor_reduce.py`

**Interfaces:**
- Consumes: shard `.npz` files from Task 7.
- Produces: a directory per level containing `train.npz`, `eval_success_prob.npz`, `eval_states.txt`, `roa_labels.txt`, `dataset_description.json`. No importable API.

`q2_reduce.py` is the precedent — read it and match its output contract exactly,
so the corridor family is readable by whatever already reads the shipped quad2d
set. The one addition is the geometry block in the description.

- [ ] **Step 1: Write the script**

Create `q2_corridor_reduce.py`. Follow `q2_reduce.py` for the file layout and key
names, and extend the description with a `corridor` block — without it the level
names are not comparable with any other family:

```python
    description['generation_parameters']['corridor'] = {
        'mechanism': 'altitude_gated',
        'channel': 'dynamics',
        'one_sided': True,
        'direction': '+x',
        'profile': 'gaussian',
        'formula': 'F_x ~ U(0, f_max * exp(-0.5*((z-centre)/width)**2))',
        'f_max': float(f_max),
        'centre': 0.55,
        'width': 0.12,
        'sigma_at_goal': float(sigma(1.0, f_max)),
        'band_1pct': [round(BAND[0], 4), round(BAND[1], 4)],
        'draw_law': 'sine',
        'period_s': 2.0,
        'redraw': 'per-rollout phase and amplitude; deterministic within a rollout',
        'skip_margin_m': 0.10,
    }
    description['eval_statistics']['trials_shortcut'] = int((used == 1).sum())
    description['eval_statistics']['trials_full'] = int((used > 1).sum())
```

- [ ] **Step 2: Verify the description round-trips**

Run:
```bash
python -c "
import json
d = json.load(open('<level_dir>/dataset_description.json'))
c = d['generation_parameters']['corridor']
assert c['f_max'] > 0 and c['sigma_at_goal'] < 1e-4
assert c['band_1pct'] == [0.186, 0.914]
print('description OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add q2_corridor_reduce.py
git commit -m "Add the corridor reducer, recording band geometry beside the level name"
```

---

## After the plan

Collection itself is not a task here. It is hours of cluster time and belongs in
an sbatch array, submitted per the `compute-resources` skill, after Tasks 4-6
have all passed. Order is load-bearing: the level-0 gate proves the transcription,
the entry measurement sets the interpretation, and the sweep sets the ladder.

Once the data lands, ingest into the wiki: `datasets.md` gains the family,
`architecture.md` gains the seventh disturbance type, `glossary.md` gains
"corridor", and `log.md` gets an entry. Then run `python3 .claude/wiki_lint.py`.
