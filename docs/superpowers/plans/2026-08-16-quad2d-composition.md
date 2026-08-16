# Quadrotor-2D Controller Composition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a flip controller for quadrotor-2D, compose it with the existing `safe_explorer_ppo` controller by latching on first entry into an attitude-only handoff region `G1`, and measure what fraction of `G1` falls outside that controller's region of attraction.

**Architecture:** A shared rollout core drives all three datasets so the baseline is reproduced by the same code path that produces the composition. `G1`'s *form* is fixed a priori (attitude only); its *parameters* are calibrated from controller 1's own exit distribution without ever consulting `RoA2`, then frozen before any composition rollout runs. Controller 1 is a SAC policy trained against an attitude-only reward on exactly controller 2's action space.

**Tech Stack:** Python 3.10, PyBullet, safe-control-gym `make()` registry, SAC from `safe_control_gym/controllers/sac/`, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-16-quad-composition-g1-design.md`

## Global Constraints

Copied verbatim from the spec. Every task's requirements implicitly include these.

- System limits are **unchanged** from `quadrotor2D_rl`: `x ±1.0`, `z ∈ [0.1, 1.5]`, `|ẋ| ≤ 1.0`, `|ż| ≤ 1.0`, `|θ̇| ≤ 8.0`, `theta` has no termination bound.
- Success radius `0.2`; `ctrl_freq=100`; `pyb_freq=5000`; `max_steps=1200`; `cost='quadratic'`; `task_info={'stabilization_goal': [0, 1], 'stabilization_goal_tolerance': 0.2}`.
- **Action space for BOTH controllers**: `normalized_rl_action_space=True`, `norm_act_scale=0.1` (the default — do not override). TWR max 1.100, α_max 53.1 rad/s².
- Dataset state order is `[x, z, theta, x_dot, z_dot, theta_dot]`; env obs order is `[x, x_dot, z, z_dot, theta, theta_dot]`. These differ — convert explicitly, never by slicing.
- `theta` is stored normalised to `[-pi, pi]`.
- `G1` is attitude-only: `{|theta| < tilt_c, |theta_dot| < w_c}`. No position or velocity term may ever be added to it.
- Controller 2 is `examples/rl/models/safe_explorer_ppo/safe_explorer_ppo_model_quadrotor_2D_stab.pt`, loaded unmodified.
- Shipped baseline for comparison: `/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/quadrotor2D_rl/`.

---

### Task 1: `G1Region` — the handoff region

**Files:**
- Create: `quad_composition/__init__.py`
- Create: `quad_composition/g1.py`
- Test: `tests/test_quad_composition/test_g1.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `G1Region(tilt_c: float, w_c: float)` with `.contains(tilt, omega) -> np.ndarray[bool]`, `.to_dict() -> dict`, `G1Region.from_dict(d) -> G1Region`; module function `attitude_2d(states) -> (abs_theta, abs_theta_dot)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quad_composition/test_g1.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quad_composition/test_g1.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quad_composition'`

- [ ] **Step 3: Write minimal implementation**

```python
# quad_composition/__init__.py
'''Controller composition experiments for the quadrotor systems.'''
```

```python
# quad_composition/g1.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_quad_composition/test_g1.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add quad_composition/__init__.py quad_composition/g1.py tests/test_quad_composition/test_g1.py
git commit -m "Add attitude-only G1 handoff region"
```

---

### Task 2: Rollout core, and prove it reproduces the shipped baseline

This is the highest-value task in the plan. If this rollout does not reproduce
`quadrotor2D_rl`, every later number is measured against a different system than
the published baseline and the comparison is void.

**Files:**
- Create: `quad_composition/rollout2d.py`
- Test: `tests/test_quad_composition/test_rollout2d.py`

**Interfaces:**
- Consumes: `G1Region` from Task 1.
- Produces:
  - `ENV_CONFIG: dict` — the frozen env kwargs.
  - `make_env_and_ctrl2(model_path, output_dir) -> (env, ctrl2)`
  - `set_initial_state(env, init_state) -> (obs, info)` where `init_state` is dataset order.
  - `state_from_obs(obs) -> list[float]` in dataset order.
  - `rollout_composite(env, ctrl1, ctrl2, g1, init_state, max_steps=1200) -> RolloutResult`
  - `RolloutResult(trajectory: list, handoff_index: int, flip_success: bool, ctrl2_success: bool)` — `handoff_index` is `-1` when no handoff fired.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quad_composition/test_rollout2d.py
'''The rollout core must reproduce the shipped quadrotor2D_rl dataset.

Spec: docs/superpowers/specs/2026-08-16-quad-composition-g1-design.md (D3, D4, D5)
'''
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SHIPPED = ('/common/users/shared/pracsys/genMoPlan/data_trajectories/'
           'deterministic/quadrotor2D_rl/eval_states.txt')
MODEL = os.path.join(REPO_ROOT, 'examples/rl/models/safe_explorer_ppo/'
                                'safe_explorer_ppo_model_quadrotor_2D_stab.pt')


def test_state_from_obs_reorders_env_obs_into_dataset_order():
    from quad_composition.rollout2d import state_from_obs
    # env order [x, x_dot, z, z_dot, theta, theta_dot]
    obs = np.array([0.1, 0.2, 1.3, 0.4, 0.5, 0.6])
    # dataset order [x, z, theta, x_dot, z_dot, theta_dot]
    assert state_from_obs(obs) == pytest.approx([0.1, 1.3, 0.5, 0.2, 0.4, 0.6])


def test_env_uses_controller_2s_restricted_action_space():
    '''Spec D6: TWR 1.10, alpha 53.1 -- not the physical actuator.'''
    from quad_composition.rollout2d import ENV_CONFIG
    assert ENV_CONFIG['normalized_rl_action_space'] is True
    assert 'norm_act_scale' not in ENV_CONFIG, 'must inherit the 0.1 default'


@pytest.mark.slow
def test_baseline_rollout_reproduces_the_shipped_labels():
    '''ctrl1=None must reproduce quadrotor2D_rl on its own initial states.'''
    if not os.path.exists(SHIPPED):
        pytest.skip('shipped dataset not mounted')
    import tempfile
    from quad_composition.rollout2d import (make_env_and_ctrl2, rollout_composite)

    rows = np.loadtxt(SHIPPED, delimiter=',', max_rows=40)
    inits, finals, labels = rows[:, :6], rows[:, 6:12], rows[:, 12].astype(int)

    with tempfile.TemporaryDirectory() as tmp:
        env, ctrl2 = make_env_and_ctrl2(MODEL, tmp)
        for init, final, label in zip(inits, finals, labels):
            res = rollout_composite(env, None, ctrl2, None, init)
            assert res.ctrl2_success == bool(label), f'label mismatch from {init}'
            assert np.allclose(res.trajectory[-1], final, atol=1e-4), \
                f'final state mismatch from {init}'
            assert res.handoff_index == -1, 'no handoff without ctrl1'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quad_composition/test_rollout2d.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quad_composition.rollout2d'`

- [ ] **Step 3: Write minimal implementation**

```python
# quad_composition/rollout2d.py
'''Shared rollout core for the quadrotor-2D composition experiments.

All three datasets (baseline, flip-only, composite) go through
`rollout_composite` so the baseline is reproduced by the same code path that
produces the composition.  The env configuration below is copied from
generate_quadrotor_2d_trajectories_rl.py and must not drift from it.
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

GOAL_STATE = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # dataset order


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def state_from_obs(obs):
    '''env order [x, x_dot, z, z_dot, theta, theta_dot] -> dataset order.'''
    x, x_dot, z, z_dot, theta, theta_dot = obs[:6]
    return [float(x), float(z), float(normalize_angle(theta)),
            float(x_dot), float(z_dot), float(theta_dot)]


def make_env_and_ctrl2(model_path, output_dir):
    '''Build the env and load controller 2 unmodified.'''
    env_func = partial(make, 'quadrotor', **ENV_CONFIG)
    ctrl2 = make('safe_explorer_ppo', env_func, **ALGO_CONFIG, output_dir=output_dir)
    ctrl2.load(model_path)
    ctrl2.obs_normalizer.set_read_only()
    env = env_func()
    for idx, (lo, hi) in TERMINATION.items():
        env.state_space.low[idx] = lo
        env.state_space.high[idx] = hi
    return env, ctrl2


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
    trajectory: list
    handoff_index: int      # -1 when no handoff fired
    flip_success: bool      # reached G1 (True by definition when ctrl1 is None)
    ctrl2_success: bool     # composite reached the goal ball


def _act(ctrl, obs, info):
    return ctrl.select_action(ctrl.obs_normalizer(obs), info)


def rollout_composite(env, ctrl1, ctrl2, g1, init_state, max_steps=MAX_STEPS):
    '''Run ctrl1 until the first state inside g1, then latch to ctrl2 forever.

    ctrl1=None runs ctrl2 from the start, reproducing the baseline.  The latch is
    permanent: once handed off, g1 is never consulted again (spec D3).
    '''
    obs, info = set_initial_state(env, init_state)
    trajectory = [list(map(float, init_state))]

    if ctrl1 is None:
        handoff_index, latched = 0, True
    else:
        tilt, omega = abs(normalize_angle(init_state[2])), abs(init_state[5])
        latched = bool(g1.contains(tilt, omega))
        handoff_index = 0 if latched else -1

    ctrl2_success = False
    for step in range(max_steps):
        action = _act(ctrl2 if latched else ctrl1, obs, info)
        obs, _, done, info = env.step(action)
        state = state_from_obs(obs)
        trajectory.append(state)

        if not latched:
            if bool(g1.contains(abs(state[2]), abs(state[5]))):
                latched = True
                handoff_index = len(trajectory) - 1
        elif done:
            ctrl2_success = bool(info.get('goal_reached', False))
            break

        if done and not latched:
            break   # out of bounds before ever reaching G1

    return RolloutResult(trajectory=trajectory,
                         handoff_index=handoff_index,
                         flip_success=handoff_index >= 0,
                         ctrl2_success=ctrl2_success)
```

- [ ] **Step 4: Run the fast tests**

Run: `python3 -m pytest tests/test_quad_composition/test_rollout2d.py -v -m "not slow"`
Expected: 2 passed

- [ ] **Step 5: Run the equivalence test against the shipped dataset**

Run: `python3 -m pytest tests/test_quad_composition/test_rollout2d.py -v -m slow`
Expected: PASS.

If it fails, do **not** proceed. Diff `ENV_CONFIG` and `ALGO_CONFIG` against
`generate_quadrotor_2d_trajectories_rl.py` lines 123–195 and 570–620, and check
`goal_tolerance` is 0.2 (the CLI default is 0.05, but the shipped dataset used 0.2).

- [ ] **Step 6: Register the `slow` marker**

```ini
# add to pytest.ini or setup.cfg under [tool:pytest]
markers =
    slow: tests that boot PyBullet and load checkpoints
```

- [ ] **Step 7: Commit**

```bash
git add quad_composition/rollout2d.py tests/test_quad_composition/test_rollout2d.py
git commit -m "Add rollout core and prove it reproduces quadrotor2D_rl"
```

---

### Task 3: Attitude-only training environment

**Files:**
- Create: `quad_composition/flip_env2d.py`
- Test: `tests/test_quad_composition/test_flip_env2d.py`

**Interfaces:**
- Consumes: `ENV_CONFIG`, `TERMINATION`, `state_from_obs`, `normalize_angle` from Task 2; `G1Region` from Task 1.
- Produces: `FlipEnv2D(g_nom: G1Region, seed: int)` — a `gym.Wrapper` exposing standard `reset()/step()`; module constants `G_NOM = G1Region(tilt_c=0.175, w_c=1.0)`, `SHAPING_GAMMA = 0.99`, `BONUS = 100.0`, `OOB_PENALTY = -100.0`; function `sample_uniform_state(rng) -> np.ndarray`.

The reward is **attitude only** (spec D2). Adding a position or velocity term
would contrive `G1` back toward `RoA2` and invalidate the experiment.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quad_composition/test_flip_env2d.py
'''The flip controller's objective must be attitude-only.

Spec: docs/superpowers/specs/2026-08-16-quad-composition-g1-design.md (D2)
'''
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from quad_composition.flip_env2d import (BONUS, G_NOM, SHAPING_GAMMA,
                                         potential, sample_uniform_state,
                                         shaped_reward)


def test_potential_depends_only_on_attitude():
    '''Two states differing only in position and velocity share a potential.'''
    a = np.array([0.0, 1.0, 0.4, 0.0, 0.0, 2.0])
    b = np.array([-0.9, 0.2, 0.4, 0.9, -0.8, 2.0])
    assert potential(a) == pytest.approx(potential(b))


def test_potential_rises_as_attitude_improves():
    upright = np.array([0.0, 1.0, 0.05, 0.0, 0.0, 0.1])
    tilted = np.array([0.0, 1.0, 2.50, 0.0, 0.0, 5.0])
    assert potential(upright) > potential(tilted)


def test_shaping_is_potential_based():
    '''r = gamma * Phi(s') - Phi(s), so cycles accumulate no reward.'''
    s = np.array([0.0, 1.0, 2.0, 0.0, 0.0, 4.0])
    s2 = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 2.0])
    r = shaped_reward(s, s2, in_g_nom=False, out_of_bounds=False)
    assert r == pytest.approx(SHAPING_GAMMA * potential(s2) - potential(s))


def test_entering_g_nom_pays_the_bonus():
    s = np.array([0.0, 1.0, 0.30, 0.0, 0.0, 1.5])
    s2 = np.array([0.0, 1.0, 0.05, 0.0, 0.0, 0.2])
    assert G_NOM.contains(abs(s2[2]), abs(s2[5]))
    r = shaped_reward(s, s2, in_g_nom=True, out_of_bounds=False)
    assert r > BONUS


def test_uniform_sampler_respects_the_closed_state_space():
    rng = np.random.default_rng(0)
    states = np.array([sample_uniform_state(rng) for _ in range(4000)])
    assert np.abs(states[:, 0]).max() < 1.0        # x
    assert states[:, 1].min() > 0.1 and states[:, 1].max() < 1.5   # z
    assert np.abs(states[:, 2]).max() <= np.pi     # theta
    assert np.abs(states[:, 3]).max() < 1.0        # x_dot
    assert np.abs(states[:, 4]).max() < 1.0        # z_dot
    assert np.abs(states[:, 5]).max() < 8.0        # theta_dot
    # full attitude coverage, not just near-upright
    assert np.abs(states[:, 2]).max() > 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quad_composition/test_flip_env2d.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quad_composition.flip_env2d'`

- [ ] **Step 3: Write minimal implementation**

```python
# quad_composition/flip_env2d.py
'''Training environment for controller 1 -- the flip controller.

The reward is ATTITUDE ONLY (spec D2).  Controller 1 has authority over attitude
and essentially none over position or translational velocity, so rewarding it for
those would (a) ask for something it cannot deliver and (b) pull G1 toward RoA2,
which is precisely the contrivance this experiment exists to avoid.
'''

import numpy as np

from quad_composition.g1 import G1Region
from quad_composition.rollout2d import normalize_angle

# Nominal training target.  This is NOT the G1 that triggers handoff -- that is
# calibrated later from measured exits (spec D1 step 2) and may be looser.
G_NOM = G1Region(tilt_c=0.175, w_c=1.0)     # 10 deg, 1 rad/s

SHAPING_GAMMA = 0.99
BONUS = 100.0
OOB_PENALTY = -100.0

# Normalisers for the potential: full tilt range and the theta_dot bound.
TILT_SCALE = np.pi
RATE_SCALE = 8.0

# Closed state space, dataset order [x, z, theta, x_dot, z_dot, theta_dot].
STATE_LOW = np.array([-1.0, 0.1, -np.pi, -1.0, -1.0, -8.0])
STATE_HIGH = np.array([1.0, 1.5, np.pi, 1.0, 1.0, 8.0])


def potential(state):
    '''Phi(s), attitude only.  Higher is closer to upright and still.'''
    theta = normalize_angle(float(state[2]))
    theta_dot = float(state[5])
    return -(abs(theta) / TILT_SCALE + abs(theta_dot) / RATE_SCALE)


def shaped_reward(state, next_state, in_g_nom, out_of_bounds):
    '''Potential-based shaping plus terminal terms.

    Potential-based shaping leaves the optimal policy unchanged, so the bonus is
    what the policy actually optimises and the shaping only speeds it up.
    '''
    reward = SHAPING_GAMMA * potential(next_state) - potential(state)
    if in_g_nom:
        reward += BONUS
    if out_of_bounds:
        reward += OOB_PENALTY
    return reward


def sample_uniform_state(rng):
    '''Uniform over the closed state space (spec: training distribution).'''
    return rng.uniform(STATE_LOW, STATE_HIGH)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_quad_composition/test_flip_env2d.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add quad_composition/flip_env2d.py tests/test_quad_composition/test_flip_env2d.py
git commit -m "Add attitude-only reward and uniform state sampler for controller 1"
```

---

### Task 4: Training script for controller 1

**Files:**
- Create: `train_quadrotor_2d_flip.py`
- Test: `tests/test_quad_composition/test_train_smoke.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces: a checkpoint at `--output_dir/flip_model.pt` loadable by `make('sac', ...).load()`; CLI `--output_dir`, `--max_env_steps`, `--seed`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quad_composition/test_train_smoke.py
'''Controller 1 training must run end-to-end and emit a loadable checkpoint.'''
import os
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.slow
def test_training_emits_a_loadable_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, 'train_quadrotor_2d_flip.py'),
             '--output_dir', tmp, '--max_env_steps', '2000', '--seed', '0'],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=1800)
        assert result.returncode == 0, result.stderr[-3000:]
        assert os.path.exists(os.path.join(tmp, 'flip_model.pt'))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quad_composition/test_train_smoke.py -v -m slow`
Expected: FAIL — `train_quadrotor_2d_flip.py` does not exist, non-zero returncode.

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
'''Train controller 1 (the flip controller) for quadrotor-2D.

Attitude-only objective, on exactly controller 2's action space (spec D2, D6).
'''

import argparse
import os
from functools import partial

import numpy as np

from quad_composition.flip_env2d import G_NOM, sample_uniform_state
from quad_composition.rollout2d import ALGO_CONFIG, ENV_CONFIG, TERMINATION
from safe_control_gym.utils.registration import make

SAC_CONFIG = {
    'hidden_dim': 128, 'norm_obs': False, 'norm_reward': False,
    'clip_obs': 10.0, 'clip_reward': 10.0,
    'gamma': 0.99, 'tau': 0.005, 'init_temperature': 0.2,
    'actor_lr': 0.001, 'critic_lr': 0.001, 'entropy_lr': 0.001,
    'train_interval': 100, 'train_batch_size': 256,
    'max_env_steps': 1000000, 'warm_up_steps': 1000,
    'rollout_batch_size': 4, 'num_workers': 1,
    'max_buffer_size': 1000000, 'deque_size': 10, 'eval_batch_size': 10,
    'log_interval': 10000, 'save_interval': 0, 'num_checkpoints': 0,
    'eval_interval': 0, 'eval_save_best': False, 'tensorboard': False,
    'training': True,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_env_steps', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    def env_func():
        from quad_composition.flip_env2d import FlipTrainingEnv
        env = make('quadrotor', **ENV_CONFIG)
        for idx, (lo, hi) in TERMINATION.items():
            env.state_space.low[idx] = lo
            env.state_space.high[idx] = hi
        return FlipTrainingEnv(env, G_NOM, seed=args.seed)

    config = dict(SAC_CONFIG, max_env_steps=args.max_env_steps, seed=args.seed)
    ctrl = make('sac', env_func, **config, output_dir=args.output_dir)
    ctrl.learn()
    ctrl.save(os.path.join(args.output_dir, 'flip_model.pt'))


if __name__ == '__main__':
    main()
```

Add `FlipTrainingEnv` to `quad_composition/flip_env2d.py`:

```python
import gym

from quad_composition.rollout2d import set_initial_state, state_from_obs


class FlipTrainingEnv(gym.Wrapper):
    '''Attitude-only objective over the 2D quadrotor.

    reset() places the drone at a uniform sample of the closed state space;
    step() replaces the env reward with the attitude-only shaped reward and
    terminates on G_nom entry or out-of-bounds.
    '''

    def __init__(self, env, g_nom, seed=0):
        super().__init__(env)
        self.g_nom = g_nom
        self.rng = np.random.default_rng(seed)
        self._state = None

    def reset(self, **kwargs):
        init = sample_uniform_state(self.rng)
        obs, info = set_initial_state(self.env, init)
        self._state = np.asarray(state_from_obs(obs), dtype=float)
        return obs, info

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        next_state = np.asarray(state_from_obs(obs), dtype=float)
        in_g_nom = bool(self.g_nom.contains(abs(next_state[2]), abs(next_state[5])))
        out_of_bounds = bool(done and not in_g_nom)
        reward = shaped_reward(self._state, next_state, in_g_nom, out_of_bounds)
        self._state = next_state
        return obs, reward, bool(done or in_g_nom), info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_quad_composition/test_train_smoke.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Launch the real training run**

```bash
python3 train_quadrotor_2d_flip.py \
    --output_dir models/quad2d_flip --max_env_steps 1000000 --seed 0
```

- [ ] **Step 6: Commit**

```bash
git add train_quadrotor_2d_flip.py quad_composition/flip_env2d.py tests/test_quad_composition/test_train_smoke.py
git commit -m "Add SAC training for the quadrotor-2D flip controller"
```

---

### Task 5: Calibrate `G1` from controller 1's exits, then freeze it

`RoA2` must not be consulted anywhere in this task. That ordering is what makes
the final measurement a discovery rather than a construction (spec D1).

**Files:**
- Create: `calibrate_quad2d_g1.py`
- Modify: `quad_composition/g1.py` (add `fit_from_exits`)
- Test: `tests/test_quad_composition/test_g1_calibration.py`

**Interfaces:**
- Consumes: `G1Region` from Task 1, `rollout_composite` from Task 2, the checkpoint from Task 4.
- Produces: `fit_from_exits(tilts, omegas, quantile=0.9) -> G1Region`; a JSON file `models/quad2d_flip/g1.json` holding `G1Region.to_dict()` plus the calibration sample.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quad_composition/test_g1_calibration.py
'''G1's parameters come from controller 1's exits, never from RoA2 (spec D1).'''
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from quad_composition.g1 import G1Region, fit_from_exits


def test_fit_covers_the_requested_quantile_of_exits():
    rng = np.random.default_rng(0)
    tilts = np.abs(rng.normal(0.0, 0.1, size=5000))
    omegas = np.abs(rng.normal(0.0, 0.8, size=5000))
    g1 = fit_from_exits(tilts, omegas, quantile=0.9)
    assert g1.contains(tilts, omegas).mean() >= 0.80
    assert g1.tilt_c == np.quantile(tilts, 0.9)
    assert g1.w_c == np.quantile(omegas, 0.9)


def test_fit_is_monotone_in_the_quantile():
    rng = np.random.default_rng(1)
    tilts = np.abs(rng.normal(0.0, 0.1, size=2000))
    omegas = np.abs(rng.normal(0.0, 0.8, size=2000))
    tight = fit_from_exits(tilts, omegas, quantile=0.5)
    loose = fit_from_exits(tilts, omegas, quantile=0.95)
    assert tight.tilt_c < loose.tilt_c and tight.w_c < loose.w_c


def test_fit_rejects_an_empty_sample():
    import pytest
    with pytest.raises(ValueError, match='no exits'):
        fit_from_exits(np.array([]), np.array([]), quantile=0.9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quad_composition/test_g1_calibration.py -v`
Expected: FAIL with `ImportError: cannot import name 'fit_from_exits'`

- [ ] **Step 3: Write minimal implementation**

Append to `quad_composition/g1.py`:

```python
def fit_from_exits(tilts, omegas, quantile=0.9):
    '''Fit G1 to controller 1's measured exit attitudes (spec D1 step 2).

    (tilt_c, w_c) are quantiles of what controller 1 actually delivers -- the
    tightest attitude region it hits reliably.  RoA2 is deliberately not an
    argument to this function: if it were, G1 would be fitted to the answer.
    '''
    tilts = np.abs(np.asarray(tilts, dtype=float))
    omegas = np.abs(np.asarray(omegas, dtype=float))
    if tilts.size == 0 or omegas.size == 0:
        raise ValueError('no exits to calibrate from')
    return G1Region(tilt_c=float(np.quantile(tilts, quantile)),
                    w_c=float(np.quantile(omegas, quantile)))
```

Create `calibrate_quad2d_g1.py`:

```python
#!/usr/bin/env python3
'''Calibrate G1 from controller 1's exit attitudes, then freeze it.

RoA2 is not loaded, imported, or referenced anywhere in this script.  Run this
BEFORE generating any composition dataset.
'''

import argparse
import json
import os
import tempfile

import numpy as np

from quad_composition.flip_env2d import G_NOM, sample_uniform_state
from quad_composition.g1 import fit_from_exits
from quad_composition.rollout2d import (ALGO_CONFIG, ENV_CONFIG, TERMINATION,
                                        make_env_and_ctrl2, set_initial_state,
                                        state_from_obs)
from safe_control_gym.utils.registration import make


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flip_model', required=True)
    parser.add_argument('--output', default='models/quad2d_flip/g1.json')
    parser.add_argument('--num_rollouts', type=int, default=5000)
    parser.add_argument('--quantile', type=float, default=0.9)
    parser.add_argument('--settle_steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    tilts, omegas = [], []

    with tempfile.TemporaryDirectory() as tmp:
        env, _ = make_env_and_ctrl2(
            os.path.join('examples/rl/models/safe_explorer_ppo',
                         'safe_explorer_ppo_model_quadrotor_2D_stab.pt'), tmp)
        ctrl1 = make('sac', lambda: env, **dict(ALGO_CONFIG, training=False),
                     output_dir=tmp)
        ctrl1.load(args.flip_model)
        ctrl1.obs_normalizer.set_read_only()

        for _ in range(args.num_rollouts):
            obs, info = set_initial_state(env, sample_uniform_state(rng))
            best = None
            for _ in range(args.settle_steps):
                action = ctrl1.select_action(ctrl1.obs_normalizer(obs), info)
                obs, _, done, info = env.step(action)
                s = state_from_obs(obs)
                score = abs(s[2]) / np.pi + abs(s[5]) / 8.0
                if best is None or score < best[0]:
                    best = (score, abs(s[2]), abs(s[5]))
                if done:
                    break
            if best is not None:
                tilts.append(best[1])
                omegas.append(best[2])

    g1 = fit_from_exits(tilts, omegas, quantile=args.quantile)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump({'g1': g1.to_dict(),
                   'calibration': {'num_rollouts': args.num_rollouts,
                                   'quantile': args.quantile,
                                   'seed': args.seed,
                                   'exit_tilt_quantiles': {
                                       str(q): float(np.quantile(tilts, q))
                                       for q in (0.5, 0.75, 0.9, 0.95)},
                                   'exit_rate_quantiles': {
                                       str(q): float(np.quantile(omegas, q))
                                       for q in (0.5, 0.75, 0.9, 0.95)}},
                   'roa2_consulted': False}, fh, indent=2)
    print(f'G1 = {g1.to_dict()}  -> {args.output}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_quad_composition/test_g1_calibration.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the calibration and freeze `G1`**

```bash
python3 calibrate_quad2d_g1.py --flip_model models/quad2d_flip/flip_model.pt
git add models/quad2d_flip/g1.json
git commit -m "Freeze G1 from controller 1 calibration"
```

The commit is the freeze. Any later change to `g1.json` invalidates every
composition dataset generated from it.

- [ ] **Step 6: Commit the code**

```bash
git add calibrate_quad2d_g1.py quad_composition/g1.py tests/test_quad_composition/test_g1_calibration.py
git commit -m "Add G1 calibration from controller 1 exit attitudes"
```

---

### Task 6: Generate the flip-only and composition datasets

**Files:**
- Create: `generate_quadrotor_2d_composition.py`
- Test: `tests/test_quad_composition/test_composition_datasets.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 5.
- Produces: two output directories, `--mode flip` writing `quadrotor2D_flip/` and `--mode composite` writing `quadrotor2D_flip_to_rl/`, each with `trajectories/`, `eval_states.txt`, `roa_labels.txt`, `handoff_states.txt`, `dataset_description.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quad_composition/test_composition_datasets.py
'''Dataset invariants for the composition (spec D3, D7, D8).'''
import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_impossible_label_combination_is_rejected():
    from generate_quadrotor_2d_composition import validate_labels
    # (flip_success=0, ctrl2_success=1) cannot occur: no handoff, no controller 2
    with pytest.raises(ValueError, match='impossible label'):
        validate_labels(np.array([0]), np.array([1]))
    validate_labels(np.array([1, 1, 0]), np.array([1, 0, 0]))   # all legal


def test_handoff_index_minus_one_means_flip_failed():
    from generate_quadrotor_2d_composition import labels_from_result
    from quad_composition.rollout2d import RolloutResult
    res = RolloutResult(trajectory=[[0] * 6], handoff_index=-1,
                        flip_success=False, ctrl2_success=False)
    assert labels_from_result(res) == (0, 0)


def test_eval_states_row_is_init_final_and_two_labels():
    from generate_quadrotor_2d_composition import eval_states_row
    from quad_composition.rollout2d import RolloutResult
    init = [0.1, 1.2, 0.3, 0.4, 0.5, 0.6]
    final = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    res = RolloutResult(trajectory=[init, final], handoff_index=1,
                        flip_success=True, ctrl2_success=True)
    row = eval_states_row(init, res)
    assert len(row) == 14           # 6 + 6 + 2
    assert row[:6] == pytest.approx(init)
    assert row[6:12] == pytest.approx(final)
    assert row[12:] == [1, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quad_composition/test_composition_datasets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generate_quadrotor_2d_composition'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
'''Generate the flip-only and composite datasets for quadrotor-2D.

--mode flip       controller 1 alone, truncated at first G1 entry
--mode composite  controller 1 then controller 2, latching on first G1 entry

Both share the initial states of the shipped quadrotor2D_rl dataset so all
comparisons are paired (spec D7).
'''

import argparse
import json
import os
import tempfile

import numpy as np

from quad_composition.g1 import G1Region
from quad_composition.rollout2d import (ALGO_CONFIG, GOAL_TOLERANCE, MAX_STEPS,
                                        make_env_and_ctrl2, rollout_composite)
from safe_control_gym.utils.registration import make

CTRL2_MODEL = ('examples/rl/models/safe_explorer_ppo/'
               'safe_explorer_ppo_model_quadrotor_2D_stab.pt')


def validate_labels(flip_success, ctrl2_success):
    '''(flip_success=0, ctrl2_success=1) cannot occur -- no handoff, no ctrl 2.'''
    bad = (np.asarray(flip_success) == 0) & (np.asarray(ctrl2_success) == 1)
    if bad.any():
        raise ValueError(f'impossible label combination in {int(bad.sum())} rows')


def labels_from_result(result):
    return int(result.flip_success), int(result.ctrl2_success)


def eval_states_row(init, result):
    flip, ctrl2 = labels_from_result(result)
    return list(map(float, init)) + list(map(float, result.trajectory[-1])) + [flip, ctrl2]


def load_initial_states(baseline_dir):
    path = os.path.join(baseline_dir, 'eval_states.txt')
    return np.loadtxt(path, delimiter=',')[:, :6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['flip', 'composite'], required=True)
    parser.add_argument('--flip_model', required=True)
    parser.add_argument('--g1', default='models/quad2d_flip/g1.json')
    parser.add_argument('--baseline_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    with open(args.g1) as fh:
        g1 = G1Region.from_dict(json.load(fh)['g1'])

    inits = load_initial_states(args.baseline_dir)
    if args.limit:
        inits = inits[:args.limit]

    os.makedirs(os.path.join(args.output_dir, 'trajectories'), exist_ok=True)
    rows, handoffs = [], []

    with tempfile.TemporaryDirectory() as tmp:
        env, ctrl2 = make_env_and_ctrl2(CTRL2_MODEL, tmp)
        ctrl1 = make('sac', lambda: env, **dict(ALGO_CONFIG, training=False),
                     output_dir=tmp)
        ctrl1.load(args.flip_model)
        ctrl1.obs_normalizer.set_read_only()

        for idx, init in enumerate(inits):
            res = rollout_composite(env, ctrl1, ctrl2, g1, init, max_steps=MAX_STEPS)
            if args.mode == 'flip' and res.handoff_index >= 0:
                res.trajectory = res.trajectory[:res.handoff_index + 1]
            np.savetxt(os.path.join(args.output_dir, 'trajectories',
                                    f'sequence_{idx}.txt'),
                       np.array(res.trajectory), delimiter=',', fmt='%.6f')
            rows.append(eval_states_row(init, res))
            handoff = (res.trajectory[res.handoff_index]
                       if res.handoff_index >= 0 else [-1.0] * 6)
            handoffs.append(list(map(float, init)) + list(map(float, handoff)))

    rows = np.array(rows)
    validate_labels(rows[:, 12], rows[:, 13])
    np.savetxt(os.path.join(args.output_dir, 'eval_states.txt'), rows,
               delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(args.output_dir, 'roa_labels.txt'),
               np.column_stack([rows[:, :6], rows[:, 12:]]),
               delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(args.output_dir, 'handoff_states.txt'),
               np.array(handoffs), delimiter=',', fmt='%.6f')

    with open(os.path.join(args.output_dir, 'dataset_description.json'), 'w') as fh:
        json.dump({
            'dataset_name': f'Quadrotor-2D {args.mode} trajectories',
            'purpose': 'EVALUATION ONLY' if args.mode == 'composite' else 'controller 1 alone',
            'g1': g1.to_dict(),
            'controller_1': {'type': 'sac', 'model': args.flip_model,
                             'objective': 'attitude-only'},
            'controller_2': {'type': 'safe_explorer_ppo', 'model': CTRL2_MODEL},
            'handoff': {'operator': 'sequential latch on first entry into G1'},
            'action_space': {'normalized_rl_action_space': True,
                             'norm_act_scale': 0.1,
                             'twr_max': 1.100, 'alpha_max_rad_s2': 53.1},
            'labels': {
                'flip_success': '1 if controller 1 reached G1',
                'ctrl2_success': '1 if the composite reached the goal ball',
                'note': '(flip_success=0, ctrl2_success=1) cannot occur'},
            'success_criteria': {'type': 'radius', 'threshold': GOAL_TOLERANCE},
            'statistics': {
                'total': int(len(rows)),
                'flip_success': int(rows[:, 12].sum()),
                'ctrl2_success': int(rows[:, 13].sum())},
        }, fh, indent=2)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_quad_composition/test_composition_datasets.py -v`
Expected: 3 passed

- [ ] **Step 5: Smoke-run both modes on 200 initial states**

```bash
BASE=/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/quadrotor2D_rl
python3 generate_quadrotor_2d_composition.py --mode flip --limit 200 \
    --flip_model models/quad2d_flip/flip_model.pt --baseline_dir $BASE \
    --output_dir /tmp/q2d_flip_smoke
python3 generate_quadrotor_2d_composition.py --mode composite --limit 200 \
    --flip_model models/quad2d_flip/flip_model.pt --baseline_dir $BASE \
    --output_dir /tmp/q2d_comp_smoke
```

Verify the prefix invariant by hand:

```bash
python3 -c "
import numpy as np
f=np.loadtxt('/tmp/q2d_flip_smoke/trajectories/sequence_0.txt',delimiter=',',ndmin=2)
c=np.loadtxt('/tmp/q2d_comp_smoke/trajectories/sequence_0.txt',delimiter=',',ndmin=2)
assert np.allclose(f, c[:len(f)]), 'flip trajectory must prefix the composite'
print('prefix invariant OK')"
```

- [ ] **Step 6: Commit**

```bash
git add generate_quadrotor_2d_composition.py tests/test_quad_composition/test_composition_datasets.py
git commit -m "Add flip-only and composite dataset generators"
```

---

### Task 7: Measure non-subsumption — the primary result

**Files:**
- Create: `analyze_quad2d_composition.py`
- Test: `tests/test_quad_composition/test_metrics.py`

**Interfaces:**
- Consumes: the datasets from Task 6.
- Produces: `non_subsumption(flip_success, ctrl2_success) -> (point, lo, hi)` using a bootstrap CI; `composed_gain(baseline_labels, composite_labels) -> dict`; a printed report and `results/quad2d_composition.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quad_composition/test_metrics.py
'''The primary metric: what fraction of G1 falls outside RoA2.'''
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analyze_quad2d_composition import composed_gain, non_subsumption


def test_non_subsumption_is_measured_only_over_actual_handoffs():
    '''Rows where the flip never reached G1 say nothing about RoA2.'''
    flip = np.array([1, 1, 1, 1, 0, 0])
    ctrl2 = np.array([1, 1, 0, 0, 0, 0])
    point, lo, hi = non_subsumption(flip, ctrl2)
    assert point == pytest.approx(0.5)      # 2 of 4 handoffs failed
    assert lo <= point <= hi


def test_non_subsumption_is_zero_when_g1_is_subsumed():
    flip = np.array([1, 1, 1])
    ctrl2 = np.array([1, 1, 1])
    point, _, _ = non_subsumption(flip, ctrl2)
    assert point == pytest.approx(0.0)


def test_non_subsumption_needs_at_least_one_handoff():
    with pytest.raises(ValueError, match='no handoffs'):
        non_subsumption(np.array([0, 0]), np.array([0, 0]))


def test_composed_gain_is_paired_over_shared_initial_states():
    baseline = np.array([0, 0, 1, 0])
    composite = np.array([1, 0, 1, 1])
    gain = composed_gain(baseline, composite)
    assert gain['baseline_rate'] == pytest.approx(0.25)
    assert gain['composed_rate'] == pytest.approx(0.75)
    assert gain['won'] == 2      # states the composition rescued
    assert gain['lost'] == 0     # states the composition broke
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quad_composition/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analyze_quad2d_composition'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
'''Measure the primary result: is G1 subsumed by RoA2?

non_subsumption = 1 - P(ctrl2_success = 1 | flip_success = 1), measured over
REAL handoffs.  The claim under test is that it is bounded away from both 0
(G1 subsumed, composition trivially sound) and 1 (G1 disjoint, composition
useless).
'''

import argparse
import json
import os

import numpy as np


def non_subsumption(flip_success, ctrl2_success, n_boot=10000, seed=0):
    '''(point estimate, lo, hi) of 1 - P(ctrl2 succeeds | handoff fired).'''
    flip = np.asarray(flip_success).astype(bool)
    ctrl2 = np.asarray(ctrl2_success).astype(bool)
    handed = ctrl2[flip]
    if handed.size == 0:
        raise ValueError('no handoffs: cannot measure non-subsumption')
    point = 1.0 - handed.mean()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, handed.size, size=(n_boot, handed.size))
    boot = 1.0 - handed[draws].mean(axis=1)
    return float(point), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def composed_gain(baseline_labels, composite_labels):
    '''Paired comparison over shared initial states.'''
    base = np.asarray(baseline_labels).astype(bool)
    comp = np.asarray(composite_labels).astype(bool)
    if base.shape != comp.shape:
        raise ValueError('paired comparison needs identical initial states')
    return {
        'baseline_rate': float(base.mean()),
        'composed_rate': float(comp.mean()),
        'won': int((comp & ~base).sum()),
        'lost': int((~comp & base).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--composite_dir', required=True)
    parser.add_argument('--baseline_dir', required=True)
    parser.add_argument('--output', default='results/quad2d_composition.json')
    args = parser.parse_args()

    comp = np.loadtxt(os.path.join(args.composite_dir, 'eval_states.txt'), delimiter=',')
    base = np.loadtxt(os.path.join(args.baseline_dir, 'eval_states.txt'), delimiter=',')
    base = base[:len(comp)]

    point, lo, hi = non_subsumption(comp[:, 12], comp[:, 13])
    gain = composed_gain(base[:, 12], comp[:, 13])

    result = {
        'non_subsumption': {'point': point, 'ci95': [lo, hi],
                            'n_handoffs': int(comp[:, 12].sum())},
        'composed_gain': gain,
        'handoff_rate': float(comp[:, 12].mean()),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump(result, fh, indent=2)

    print(f"non-subsumption : {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]"
          f"  over {int(comp[:, 12].sum())} handoffs")
    print(f"baseline        : {gain['baseline_rate']:.4f}")
    print(f"composed        : {gain['composed_rate']:.4f}"
          f"  (+{gain['won']} won, -{gain['lost']} lost)")
    if point < 0.02:
        print('WARNING: G1 is effectively subsumed by RoA2 -- the primary claim fails.')
    if point > 0.98:
        print('WARNING: G1 barely intersects RoA2 -- handoffs almost never succeed.')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_quad_composition/test_metrics.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the full generation and the analysis**

```bash
BASE=/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic/quadrotor2D_rl
OUT=/common/users/shared/pracsys/genMoPlan/data_trajectories/deterministic
python3 generate_quadrotor_2d_composition.py --mode flip \
    --flip_model models/quad2d_flip/flip_model.pt --baseline_dir $BASE \
    --output_dir $OUT/quadrotor2D_flip
python3 generate_quadrotor_2d_composition.py --mode composite \
    --flip_model models/quad2d_flip/flip_model.pt --baseline_dir $BASE \
    --output_dir $OUT/quadrotor2D_flip_to_rl
python3 analyze_quad2d_composition.py \
    --composite_dir $OUT/quadrotor2D_flip_to_rl --baseline_dir $BASE
```

- [ ] **Step 6: Commit**

```bash
git add analyze_quad2d_composition.py tests/test_quad_composition/test_metrics.py results/quad2d_composition.json
git commit -m "Measure non-subsumption of G1 by RoA2 on quadrotor-2D"
```

---

### Task 8: Validate against the analytic budget

Distinguishes "controller 1 is undertrained" from "that region is physically
unreachable" — spec risk 4. Without this, a policy that has learned everything
learnable looks like a failure.

**Files:**
- Create: `quad_composition/budget.py`
- Test: `tests/test_quad_composition/test_budget.py`

**Interfaces:**
- Consumes: `quadrotor2D_flip/` from Task 6.
- Produces: `min_delta_zdot(tilt0, tilt_target, w_max, a_min, a_max) -> float`; `budget_feasible(states, tilt_target, ...) -> np.ndarray[bool]`; constants `A_MIN_RESTRICTED = 8.829`, `A_MAX_RESTRICTED = 10.791`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quad_composition/test_budget.py
'''The analytic vertical-velocity budget (spec: flip feasibility).'''
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from quad_composition.budget import (A_MAX_RESTRICTED, A_MIN_RESTRICTED,
                                     budget_feasible, min_delta_zdot)


def test_no_rotation_costs_nothing():
    assert min_delta_zdot(0.1, 0.2, 8.0, A_MIN_RESTRICTED, A_MAX_RESTRICTED) == 0.0


def test_larger_rotations_cost_more():
    small = min_delta_zdot(np.radians(60), np.radians(10), 8.0,
                           A_MIN_RESTRICTED, A_MAX_RESTRICTED)
    large = min_delta_zdot(np.radians(150), np.radians(10), 8.0,
                           A_MIN_RESTRICTED, A_MAX_RESTRICTED)
    assert large < small < 0


def test_faster_rotation_costs_less():
    slow = min_delta_zdot(np.radians(150), np.radians(10), 8.0,
                          A_MIN_RESTRICTED, A_MAX_RESTRICTED)
    fast = min_delta_zdot(np.radians(150), np.radians(10), 24.0,
                          A_MIN_RESTRICTED, A_MAX_RESTRICTED)
    assert fast > slow


def test_matches_the_spec_recoverable_tilt_at_zero_zdot():
    '''Spec: 107 deg at zdot=0 under the restricted actuator.'''
    for tilt_deg, expected in ((105, True), (110, False)):
        state = np.array([[0.0, 1.0, np.radians(tilt_deg), 0.0, 0.0, 0.0]])
        got = budget_feasible(state, np.radians(10), 8.0, 1.0,
                              A_MIN_RESTRICTED, A_MAX_RESTRICTED)[0]
        assert got == expected, f'{tilt_deg} deg should be feasible={expected}'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_quad_composition/test_budget.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quad_composition.budget'`

- [ ] **Step 3: Write minimal implementation**

```python
# quad_composition/budget.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_quad_composition/test_budget.py -v`
Expected: 4 passed

- [ ] **Step 5: Compare controller 1 against the bound**

```bash
python3 -c "
import numpy as np, os
from quad_composition.budget import *
from quad_composition.g1 import G1Region
import json
D=os.environ['OUT']+'/quadrotor2D_flip'
rows=np.loadtxt(D+'/eval_states.txt',delimiter=',')
g1=G1Region.from_dict(json.load(open('models/quad2d_flip/g1.json'))['g1'])
feas=budget_feasible(rows[:,:6], g1.tilt_c, 8.0, 1.0, A_MIN_RESTRICTED, A_MAX_RESTRICTED)
flip=rows[:,12].astype(bool)
print('budget-feasible      : %.4f' % feas.mean())
print('controller 1 reached : %.4f' % flip.mean())
print('capture of feasible  : %.4f' % flip[feas].mean())
print('succeeded though infeasible (bound violated - investigate): %d' % (flip & ~feas).sum())
"
```

`capture of feasible` well below 1.0 means controller 1 is undertrained.
Any nonzero count in the last line means the bound or the actuator constants are
wrong — the analytic bound must never be beaten.

- [ ] **Step 6: Commit**

```bash
git add quad_composition/budget.py tests/test_quad_composition/test_budget.py
git commit -m "Add analytic flip budget to separate undertraining from infeasibility"
```

---

## Self-Review

**Spec coverage:**

| Spec item | Task |
|---|---|
| D1 `G1` form fixed a priori | 1 |
| D1 parameters from calibration, `RoA2` not consulted | 5 |
| D2 attitude-only reward | 3 |
| D3 latch on first `G1` entry | 2 |
| D4 limits unchanged | 2 (`ENV_CONFIG`, `TERMINATION`) |
| D5 controller 2 is the existing checkpoint | 2 |
| D6 restricted action space | 2 (test asserts `norm_act_scale` absent), 4 |
| D7 dataset layout | 6 |
| D8 formats incl. `handoff_states.txt` | 6 |
| Primary metric + CI | 7 |
| Secondary: composed gain, paired | 7 |
| Validation: baseline reproduced | 2 |
| Validation: `(0,1)` count zero | 6 |
| Validation: flip prefixes composite | 6 |
| Validation: `G1` frozen before rollouts | 5 (committed `g1.json`) |
| Validation: measured vs analytic budget | 8 |
| Risk 4: undertraining vs infeasibility | 8 |

**Gap found and accepted:** the spec's secondary metric `|Flip⁻¹(G1) \ RoA2|` is
computed as `gain['won']` in Task 7 rather than as an explicit set — the paired
count is the same quantity and needs no extra machinery.

**Gap found and accepted:** spec risk 2 (the 2D non-subsumption floor is
unconfirmed below 9.08° tilt) is *resolved* by this plan rather than tested
separately — Task 5 calibrates `G1` from continuous rollout states, which are not
grid-quantised, and Task 7 measures over those.

**Not covered — deliberately deferred:** quadrotor-3D. It is a separate plan; the
`quad_composition/` modules are written per-system (`rollout2d`, `flip_env2d`) so
3D adds siblings rather than editing these.

---

## Execution Order and Compute

Tasks 1–3 and 8 are pure-Python and fast. Task 4's real run is the long pole
(1M env steps at 100 Hz). Tasks 6–7 are ~490k rollouts × 2 modes; parallelise with
the same CPU-affinity pattern as `generate_quadrotor_2d_trajectories_rl.py`
(`get_available_cpus`) before launching the full run — the smoke run in Task 6
step 5 is capped at 200 states precisely so this is discovered early.

**Hard ordering constraint:** Task 5's `git commit` of `g1.json` must land before
any Task 6 full run. That commit is the freeze that makes the Task 7 measurement a
discovery rather than a fit.
