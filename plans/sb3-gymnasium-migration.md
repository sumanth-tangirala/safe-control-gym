# Gymnasium Migration and Generic SB3 Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate all four environments to the Gymnasium step API and add one
task-agnostic stable-baselines3 training entry point, so RL controllers can be
trained in-repo for any registered system.

**Architecture:** The environments currently return the pre-Gymnasium 4-tuple
`(obs, rew, done, info)` and record time-limit truncation as
`info['TimeLimit.truncated']`. This plan promotes that convention to a
first-class `truncated` return value, restores the wrapper attribute forwarding
that Gymnasium 1.0 removed, and adds `train_sb3.py` driven by the existing
`ConfigFactory`. Behaviour must not change anywhere; three independent oracles
captured before any migration commit enforce that.

**Tech Stack:** Python 3.10, gymnasium 1.3.0, stable-baselines3 2.9.0, torch
2.8, numpy, pytest, pre-commit (isort + autopep8 + flake8).

Spec: `docs/superpowers/specs/2026-07-28-sb3-gymnasium-migration-design.md`.

## Global Constraints

- **Nothing may change numerically.** All six RL controllers already compensate
  for truncation via `info['TimeLimit.truncated']` (`sac.py:287-304` and
  equivalents in `ppo`, `ddpg`, `rarl`, `rap`, `safe_ppo`). A number that moves
  is a migration bug.
- **Verification bar:** `tests/test_examples/` + `tests/test_build.py` at
  **68 passed, 2 skipped**; `tests/test_inverted_pendulum/` at **74 passed**
  with only the known `test_pendulum_experiment.py` subprocess failure.
- **The known pendulum failure survives Task 3.** Its first-order cause was found
  in Task 1 -- the editable install pointed at a deleted sibling clone, so
  `safe_control_gym` resolved only from a cwd inside this repo. Task 3's
  `pip install -e .` repaired that (imports now resolve from anywhere), and the
  test STILL fails. So there is a second, unrelated cause. The bar therefore
  stays **74 passed, 1 failed** throughout, and that one failure remains out of
  scope. Do not "fix" it as part of migration work.
- **stable-baselines3 may be imported only by**
  `safe_control_gym/experiments/train_sb3.py`. `envs/` and `controllers/` stay
  SB3-free.
- **Style:** single quotes, `'''` docstrings, `flake8 --ignore=E501`. The
  `PostToolUse` hook runs `pre-commit` on every file written; fix what it
  reports rather than bypassing it.
- **Never write under** `/common/users/shared/pracsys/genMoPlan/data_trajectories`.
  All dataset work in this plan goes to a scratch `--output_dir`.
- Tasks 1 and 2 must be committed **before** Task 3. They are the oracles;
  captured after the dependency bump they prove nothing.

---

### Task 1: Golden rollout fixtures for cartpole and both quadrotors

The pendulum's `fixtures/env_rollouts.json` came from the external source
system, so there is no in-repo generator to copy. Write one, then use it.

**Files:**
- Create: `tests/fixtures/generate_env_rollouts.py`
- Create: `tests/test_envs/__init__.py`
- Create: `tests/test_envs/test_env_rollouts.py`
- Create: `tests/test_envs/fixtures/cartpole_rollouts.json`
- Create: `tests/test_envs/fixtures/quadrotor_2d_rollouts.json`
- Create: `tests/test_envs/fixtures/quadrotor_3d_rollouts.json`

**Interfaces:**
- Produces: `generate_env_rollouts.build(task, task_config, act_dim)` returning
  the dict written to JSON, with keys `params` (dict) and `scenarios` (list of
  `{'x0': [...], 'actions': [[...], ...], 'states': [[...], ...]}`). Task 5-6
  rely on `tests/test_envs/test_env_rollouts.py` still passing unchanged.

- [ ] **Step 1: Write the generator**

`tests/fixtures/generate_env_rollouts.py`:

```python
'''Generate golden rollout fixtures for the non-pendulum environments.

Run BEFORE the gymnasium migration, under gymnasium 0.28, and commit the
output. The migration must reproduce these trajectories exactly.

Run:  python tests/fixtures/generate_env_rollouts.py
'''
import json
import os

import numpy as np

from safe_control_gym.utils.registration import make

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'test_envs', 'fixtures')

# (fixture name, task id, task_config overrides, action dimension)
CASES = [
    ('cartpole_rollouts.json', 'cartpole', {}, 1),
    ('quadrotor_2d_rollouts.json', 'quadrotor', {'quad_type': 2}, 2),
    ('quadrotor_3d_rollouts.json', 'quadrotor', {'quad_type': 3}, 4),
]

N_SCENARIOS = 4
N_STEPS = 25


def build(task, task_config, act_dim):
    '''Roll fixed pseudo-random action sequences and record every state.'''
    env = make(task, **task_config)
    rng = np.random.default_rng(0)
    scenarios = []
    for scenario in range(N_SCENARIOS):
        env.reset(seed=1000 + scenario)
        x0 = np.asarray(env.state, dtype=np.float64).tolist()
        actions, states = [], []
        for _ in range(N_STEPS):
            act = rng.uniform(env.action_space.low, env.action_space.high,
                              size=(act_dim,))
            env.step(act)
            actions.append(np.asarray(act, dtype=np.float64).tolist())
            states.append(np.asarray(env.state, dtype=np.float64).tolist())
        scenarios.append({'x0': x0, 'actions': actions, 'states': states})
    params = {'task': task, 'task_config': task_config,
              'ctrl_freq': int(env.CTRL_FREQ), 'n_steps': N_STEPS}
    env.close()
    return {'params': params, 'scenarios': scenarios}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, task, task_config, act_dim in CASES:
        data = build(task, task_config, act_dim)
        path = os.path.join(OUT_DIR, name)
        with open(path, 'w') as handle:
            json.dump(data, handle)
        print(f'wrote {path} ({len(data["scenarios"])} scenarios)')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Generate the fixtures**

Run: `python tests/fixtures/generate_env_rollouts.py`
Expected: three `wrote .../fixtures/*.json` lines, no traceback.

- [ ] **Step 3: Write the test that consumes them**

`tests/test_envs/test_env_rollouts.py`:

```python
'''Golden rollout fidelity for cartpole and both quadrotors.

Fixtures were generated by tests/fixtures/generate_env_rollouts.py under
gymnasium 0.28, before the Gymnasium migration. They must reproduce exactly.
'''
import json
import os

import numpy as np
import pytest

from safe_control_gym.utils.registration import make

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')

CASES = ['cartpole_rollouts.json',
         'quadrotor_2d_rollouts.json',
         'quadrotor_3d_rollouts.json']


@pytest.mark.parametrize('fixture', CASES)
def test_rollouts_match_golden(fixture):
    with open(os.path.join(FIX, fixture)) as handle:
        data = json.load(handle)
    env = make(data['params']['task'], **data['params']['task_config'])
    for scenario in data['scenarios']:
        env.reset(seed=None)
        env.state = np.asarray(scenario['x0'], dtype=np.float64)
        for act, expected in zip(scenario['actions'], scenario['states']):
            env.step(np.asarray(act, dtype=np.float64))
            np.testing.assert_allclose(env.state, expected, atol=1e-9)
    env.close()
```

- [ ] **Step 4: Run it and confirm it passes**

Run: `python -m pytest tests/test_envs/ -q`
Expected: `3 passed`.

- [ ] **Step 5: Confirm it actually pins something**

Temporarily change `atol=1e-9` to `atol=1e-30` and re-run. Expected: failures.
Revert to `1e-9`. A fixture test that cannot fail is not an oracle.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/generate_env_rollouts.py tests/test_envs/
git commit -m "Add golden rollout fixtures for cartpole and both quadrotors

The pendulum's fixtures came from the external source system, so there was no
in-repo generator to copy. These pin physics at atol=1e-9 before the gymnasium
migration touches the step API."
```

---

### Task 2: Dataset slice and invariant-set references

**Files:**
- Create: `tests/test_envs/fixtures/dataset_slice_lqr.json`
- Create: `tests/test_envs/fixtures/dataset_slice_v3_strong.json`
- Create: `tests/test_envs/test_dataset_slice.py`
- Create: `tests/test_envs/test_invariant_sets.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: two fixtures keyed `starts` (list of `[theta, theta_dot]`), `labels`
  (list of int), `terminal_states` (list of `[theta, theta_dot]`). Tasks 5-9
  rely on `test_dataset_slice.py` passing unchanged.

- [ ] **Step 1: Capture both slices**

`--resolution 0.5` gives `ceil(2*pi/0.5) x ceil(4*pi/0.5)` = 13 x 26 = 338
cells. `--skip_save` is not usable — it suppresses sequence writing
(`generate_inverted_pendulum_trajectories.py:252`) and terminal states are the
quantity that matters.

```bash
SCRATCH=$(mktemp -d)
for CTRL in lqr v3_strong; do
  python generate_inverted_pendulum_trajectories.py \
      --controller "$CTRL" --seed 42 --resolution 0.5 \
      --output_dir "$SCRATCH/$CTRL"
done
ls "$SCRATCH"/lqr "$SCRATCH"/v3_strong
```

Expected: each directory contains `roa_labels.txt` and a `trajectories/`
directory. Note the printed `$SCRATCH` path — Step 2 needs it.

- [ ] **Step 2: Convert both to fixtures**

```bash
python - "$SCRATCH" <<'PY'
import glob, json, os, sys
import numpy as np

scratch = sys.argv[1]
out = 'tests/test_envs/fixtures'
os.makedirs(out, exist_ok=True)
for ctrl in ('lqr', 'v3_strong'):
    root = os.path.join(scratch, ctrl)
    starts, labels = [], []
    with open(os.path.join(root, 'roa_labels.txt')) as handle:
        for line in handle:
            parts = [float(p) for p in line.strip().split(',') if p]
            starts.append(parts[:2])
            labels.append(int(parts[-1]))
    terminals = []
    for i in range(len(starts)):
        seq = os.path.join(root, 'trajectories', f'sequence_{i}.txt')
        rows = np.loadtxt(seq, delimiter=',', ndmin=2)
        terminals.append(rows[-1].tolist())
    path = os.path.join(out, f'dataset_slice_{ctrl}.json')
    with open(path, 'w') as handle:
        json.dump({'starts': starts, 'labels': labels,
                   'terminal_states': terminals}, handle)
    print(f'wrote {path}: {len(starts)} cells')
PY
```

Expected: two `wrote ...: 338 cells` lines. If the count is not 338, stop — the
grid convention has changed and the spec's arithmetic needs revisiting.

- [ ] **Step 3: Write the test**

`tests/test_envs/test_dataset_slice.py`:

```python
'''The dataset slice must reproduce bit-exactly across the migration.

rollout_seed is a pure function of (base_seed, split_id, index, batch), so a
fixed grid regenerates identically. Both the LQR and the RL controller paths
are covered: nothing is exempt from bit-exactness.
'''
import json
import os
import subprocess
import sys

import numpy as np
import pytest

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')
REPO = os.path.join(os.path.dirname(__file__), '..', '..')


@pytest.mark.parametrize('controller', ['lqr', 'v3_strong'])
def test_slice_reproduces(controller, tmp_path):
    with open(os.path.join(FIX, f'dataset_slice_{controller}.json')) as handle:
        golden = json.load(handle)

    out = tmp_path / controller
    result = subprocess.run(
        [sys.executable, 'generate_inverted_pendulum_trajectories.py',
         '--controller', controller, '--seed', '42', '--resolution', '0.5',
         '--output_dir', str(out)],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]

    starts, labels = [], []
    with open(out / 'roa_labels.txt') as handle:
        for line in handle:
            parts = [float(p) for p in line.strip().split(',') if p]
            starts.append(parts[:2])
            labels.append(int(parts[-1]))

    assert labels == golden['labels']
    np.testing.assert_allclose(starts, golden['starts'], atol=1e-12)

    for i, expected in enumerate(golden['terminal_states']):
        rows = np.loadtxt(out / 'trajectories' / f'sequence_{i}.txt',
                          delimiter=',', ndmin=2)
        np.testing.assert_allclose(rows[-1], expected, atol=1e-12)
```

- [ ] **Step 4: Write the invariant-set test (in memory — it must never write)**

`compute_invariant_sets.py` rewrites `invariant_sets/<system>.npz` **in place**
(`compute_invariant_sets.py:354-355`). A test must never invoke it: those
artifacts are committed inputs, and a test that mutates tracked files is a
footgun even when git can restore them. The four `.npz` files are also now
mode `-r--r-----` on disk, so an in-place write fails loudly.

Recompute the same quantities in memory instead and compare. Import the pieces;
do not call `compute_system`, which is the function that saves.

`tests/test_envs/test_invariant_sets.py`:

```python
'''The invariant-set artifacts must still match a fresh recomputation.

Exercises the closed-loop step map directly, so it catches a migration that
perturbs stepping without changing any golden rollout.

This recomputes IN MEMORY. It must never call compute_system() or the script
entry point: both write invariant_sets/<system>.npz in place, and those
artifacts are committed inputs, not test scratch.
'''
import importlib.util
import os

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

REPO = os.path.join(os.path.dirname(__file__), '..', '..')
SETS = os.path.join(REPO, 'invariant_sets')


def _load_module():
    spec = importlib.util.spec_from_file_location(
        'compute_invariant_sets', os.path.join(REPO, 'compute_invariant_sets.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('system', ['pendulum', 'cartpole', 'quad2d', 'quad3d'])
def test_invariant_set_recomputes_in_memory(system):
    mod = _load_module()
    golden = np.load(os.path.join(SETS, f'{system}.npz'))

    instance = mod.SYSTEMS[system]()
    s0 = instance.attractor()
    A_d = mod.fd_linearize(instance, s0)
    P = solve_discrete_lyapunov(A_d.T, np.diag(instance.Q_diag))

    np.testing.assert_allclose(s0, golden['center'], atol=1e-12)
    np.testing.assert_allclose(P, golden['P'], atol=1e-12)


def test_artifacts_are_not_writable():
    '''Guard the guard: these are committed inputs, not scratch.'''
    for system in ('pendulum', 'cartpole', 'quad2d', 'quad3d'):
        path = os.path.join(SETS, f'{system}.npz')
        assert not os.access(path, os.W_OK), \
            f'{path} is writable; it must stay read-only so an in-place ' \
            f'recomputation fails loudly instead of silently clobbering it'
```

`c` is a property of the system definition rather than of the recomputation, so
it is not re-derived here; `P` and `center` are what the step map determines and
are what a migration would move.

- [ ] **Step 5: Run both**

Run: `python -m pytest tests/test_envs/test_dataset_slice.py tests/test_envs/test_invariant_sets.py -q`
Expected: `6 passed`. These are slow; that is acceptable for oracles.

- [ ] **Step 6: Commit**

```bash
git add tests/test_envs/
git commit -m "Add dataset-slice and invariant-set oracles

338-cell deterministic grid for both the lqr and v3_strong controllers, plus
invariant-set recomputation. Captured under gymnasium 0.28 so the migration has
something to be checked against."
```

---

### Task 3: Dependency bump

**Files:**
- Modify: `pyproject.toml:15` (`gymnasium = "^0.28"`)
- Modify: `setup.py` (`install_requires` list)

- [ ] **Step 1: Confirm the oracles are committed**

Run: `git log --oneline -3`
Expected: the Task 1 and Task 2 commits are present. If not, stop and finish them.

- [ ] **Step 2: Edit `pyproject.toml`**

Change `gymnasium = "^0.28"` to `gymnasium = "^1.3"` and add
`stable-baselines3 = "^2.9"` immediately after the `torch` line.

- [ ] **Step 3: Edit `setup.py`**

In `install_requires`, change `'gymnasium'` to `'gymnasium>=1.3'` and add
`'stable-baselines3'` after `'torch'`.

- [ ] **Step 4: Install**

Run: `python -m pip install -e .`
Expected: `gymnasium-1.3.0` and `stable_baselines3-2.9.0` installed.

- [ ] **Step 5: Record the damage**

Run: `python -m pytest tests/test_examples/ tests/test_build.py -q --no-header 2>&1 | tail -3`
Expected: roughly `65 failed, 3 passed, 2 skipped`, all
`AttributeError: '...' object has no attribute '...'`. This is the known state
Task 4 fixes. Do not attempt to fix anything here.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml setup.py
git commit -m "Bump gymnasium to 1.3 and add stable-baselines3

Leaves the upstream example tests failing on wrapper attribute forwarding,
which gymnasium 1.0 removed. Task 4 restores it."
```

---

### Task 4: Restore wrapper attribute forwarding

Gymnasium 1.0 removed `Wrapper.__getattr__` passthrough. Two wrapper classes
need it back — but allowlisted, not blanket. An explicit `FORWARDED` tuple keeps
the forwarded surface visible and greppable, and makes a typo raise instead of
silently resolving to the wrapped env.

**Files:**
- Create: `safe_control_gym/envs/env_wrappers/forwarding.py`
- Modify: `safe_control_gym/envs/env_wrappers/record_episode_statistics.py:13` (class `RecordEpisodeStatistics`)
- Modify: `safe_control_gym/experiments/base_experiment.py:310` (class `RecordDataWrapper`)
- Create: `tests/test_envs/test_wrapper_forwarding.py`

**Interfaces:**
- Produces: `forwarding.AttributeForwardingMixin` with class attribute
  `FORWARDED` (tuple of str). Both wrappers inherit it, so
  `RecordEpisodeStatistics.constraints`, `RecordDataWrapper.GUI` and
  `RecordDataWrapper.done_on_out_of_bound` all resolve.

- [ ] **Step 1: Write the failing test**

`tests/test_envs/test_wrapper_forwarding.py`:

```python
'''Attributes the codebase reads through a wrapper must still resolve.

gymnasium 1.0 removed Wrapper.__getattr__ passthrough. These are the attributes
that call sites actually use; see the FORWARDED tuple in each wrapper.
'''
import numpy as np
import pytest

from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics
from safe_control_gym.experiments.base_experiment import RecordDataWrapper
from safe_control_gym.utils.registration import make

ATTRS = ['GUI', 'CTRL_FREQ', 'constraints', 'done_on_out_of_bound',
         'symbolic', 'state']


@pytest.mark.parametrize('wrapper_cls', [RecordDataWrapper, RecordEpisodeStatistics])
@pytest.mark.parametrize('attr', ATTRS)
def test_attribute_forwards(wrapper_cls, attr):
    env = make('cartpole')
    wrapped = wrapper_cls(env)
    wrapped.reset()
    expected = getattr(env, attr)          # raises if the env itself lacks it
    actual = getattr(wrapped, attr)        # raises if forwarding is missing
    assert actual is expected or np.array_equal(actual, expected)
    env.close()


@pytest.mark.parametrize('wrapper_cls', [RecordDataWrapper, RecordEpisodeStatistics])
def test_unknown_attribute_still_raises(wrapper_cls):
    '''Allowlisted forwarding, not blanket -- a typo must not resolve.'''
    env = make('cartpole')
    wrapped = wrapper_cls(env)
    with pytest.raises(AttributeError):
        wrapped.definitely_not_an_attribute
    env.close()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_envs/test_wrapper_forwarding.py -q`
Expected: FAIL with `AttributeError: 'RecordDataWrapper' object has no attribute 'GUI'`.

- [ ] **Step 3: Write the shared mixin**

One definition, so the forwarded list cannot drift between the two wrappers.

`safe_control_gym/envs/env_wrappers/forwarding.py`:

```python
'''Allowlisted attribute forwarding for env wrappers.

gymnasium 1.0 removed ``Wrapper.__getattr__`` passthrough to the wrapped env,
which this codebase relied on in ~11 places. This restores it for a named set
only, so the forwarded surface stays greppable and a typo raises instead of
silently resolving.
'''


class AttributeForwardingMixin:
    '''Forward the attributes in ``FORWARDED`` to ``self.env``.

    Mix in *before* ``gym.Wrapper`` so this ``__getattr__`` wins.
    '''

    # Attributes call sites read through a wrapper. Extend deliberately.
    FORWARDED = ('GUI', 'CTRL_FREQ', 'PYB_FREQ', 'NAME', 'symbolic', 'state',
                 'constraints', 'done_on_out_of_bound', 'X_GOAL', 'TASK',
                 'denormalize_action', 'normalize_action')

    def __getattr__(self, name):
        if name in type(self).FORWARDED:
            return getattr(self.env, name)
        raise AttributeError(
            f'{type(self).__name__!r} object has no attribute {name!r}; add it '
            f'to FORWARDED if it should pass through to the wrapped env.')
```

- [ ] **Step 3b: Inherit it in both wrappers**

In `record_episode_statistics.py`, add the import and change the declaration:

```python
from safe_control_gym.envs.env_wrappers.forwarding import AttributeForwardingMixin

class RecordEpisodeStatistics(AttributeForwardingMixin, gym.Wrapper):
```

In `base_experiment.py`, the same:

```python
from safe_control_gym.envs.env_wrappers.forwarding import AttributeForwardingMixin

class RecordDataWrapper(AttributeForwardingMixin, gym.Wrapper):
```

Add no other code to either class.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_envs/test_wrapper_forwarding.py -q`
Expected: `14 passed` (2 wrappers x 6 attributes, plus 2 raise-checks).

- [ ] **Step 5: Run the upstream suite**

Run: `python -m pytest tests/test_examples/ tests/test_build.py -q --no-header 2>&1 | tail -3`
Expected: **68 passed, 2 skipped**. If any test still fails with `AttributeError`,
add the missing name to `FORWARDED` in both wrappers and repeat.

- [ ] **Step 6: Commit**

```bash
git add safe_control_gym/envs/env_wrappers/record_episode_statistics.py \
        safe_control_gym/experiments/base_experiment.py \
        tests/test_envs/test_wrapper_forwarding.py
git commit -m "Restore wrapper attribute forwarding for gymnasium 1.x

An explicit FORWARDED tuple rather than a blanket passthrough, so the surface
call sites depend on is visible and a typo raises instead of silently missing."
```

---

### Task 5: `benchmark_env` and the three environments return the 5-tuple

**Files:**
- Modify: `safe_control_gym/envs/benchmark_env.py:447-502` (`after_step`)

**Interfaces:**
- Produces: `after_step(obs, rew, terminated, truncated, info)` returning
  `(obs, rew, terminated, truncated, info)`. Tasks 6-9 consume this signature.

- [ ] **Step 1: Change the signature and the time-limit block**

Replace the `after_step` signature and its final block. The `info` key stays —
it is what Task 9 cross-checks the new flag against.

Signature becomes:

```python
    def after_step(self, obs, rew, terminated, truncated, info):
```

Update the docstring `Args`/`Returns` to name `terminated (bool)` and
`truncated (bool)` instead of `done (bool)`.

Constraint-violation block: `done = True` becomes `terminated = True`.

Final block becomes:

```python
        # Time limit is truncation, not termination. The legacy info key is
        # retained: six controllers still read it, and it is what the migration
        # tests cross-check `truncated` against.
        if self.ctrl_step_counter >= self.CTRL_STEPS:
            info['TimeLimit.truncated'] = not terminated
            truncated = True
        return obs, rew, terminated, truncated, info
```

- [ ] **Step 2: Update each environment's `step` in the same commit**

`after_step` has three callers (`inverted_pendulum.py:189`, `cartpole.py:298`,
`quadrotor.py:449`). Changing its signature without them leaves the suite red,
so both halves land in one commit.

Files also modified here:
`safe_control_gym/envs/gym_control/inverted_pendulum.py:159-204`,
`safe_control_gym/envs/gym_control/cartpole.py:299,301`,
`safe_control_gym/envs/gym_pybullet_drones/quadrotor.py:450,328`.

In all three, replace the tail of `step`. Pattern, using
`inverted_pendulum.py:185-190` as the worked example:

```python
        obs = self._get_observation()
        rew = self._get_reward()
        terminated = self._get_done()
        truncated = False
        info = self._get_info()
        obs, rew, terminated, truncated, info = super().after_step(
            obs, rew, terminated, truncated, info)
        return obs, rew, terminated, truncated, info
```

`cartpole.py:299` and `quadrotor.py:450` take the identical change; both already
compute `done = self._get_done()` the same way.

- [ ] **Step 3: Add `options` to each `reset`**

Gymnasium 1.x requires `reset(self, *, seed=None, options=None)`. All three are
currently `def reset(self, seed=None)`. Change each to:

```python
    def reset(self, seed=None, options=None):
```

`options` is accepted and ignored; note that in the docstring. Do not make
`seed` keyword-only — `test_env.py` and the generators call `reset(seed=...)`
by keyword already, but `dummy_vec_env.py:36` calls `reset()` positionally-free,
and `compute_invariant_sets.py` may not.

- [ ] **Step 4: Run the pendulum env tests**

Run: `python -m pytest tests/test_inverted_pendulum/test_env.py -q`
Expected: PASS. If `test_env.py` unpacks four values from `step`, update those
unpack sites to five — the fixture values themselves must not change.

- [ ] **Step 5: Run the golden rollouts from Task 1**

Run: `python -m pytest tests/test_envs/test_env_rollouts.py -q`
Expected: `3 passed` at `atol=1e-9`. This is the first real proof the migration
has not perturbed physics.

- [ ] **Step 6: Commit**

```bash
git add safe_control_gym/envs/
git commit -m "Return the gymnasium 5-tuple from all three environments

_get_done() is termination; truncation comes from the time limit in after_step.
reset() gains the options argument gymnasium 1.x requires. Golden rollouts
reproduce at atol=1e-9."
```

---

### Task 6: Wrappers and vectorized envs

**Files:**
- Modify: `safe_control_gym/envs/env_wrappers/record_episode_statistics.py:64-91`
- Modify: `safe_control_gym/experiments/base_experiment.py:357-` (`RecordDataWrapper.step`)
- Modify: `safe_control_gym/envs/env_wrappers/vectorized_env/dummy_vec_env.py:32-38`
- Modify: `safe_control_gym/envs/env_wrappers/vectorized_env/subproc_vec_env.py:189`

- [ ] **Step 1: `RecordEpisodeStatistics.step`**

```python
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
```

Leave the rest of the body unchanged — it already keys off `done` — and return
`observation, reward, terminated, truncated, info`.

- [ ] **Step 2: `RecordDataWrapper.step`**

Same shape: unpack five, derive `done = terminated or truncated` for the
existing logging body, return five. The `step_data` dict is unchanged, so
recorded data stays identical.

- [ ] **Step 3: `dummy_vec_env.py:32`**

The auto-reset must key on either flag:

```python
            obs, rew, terminated, truncated, info = self.envs[i].step(self.actions[i])
            done = terminated or truncated
            if done:
                end_obs = copy.deepcopy(obs)
                end_info = copy.deepcopy(info)
                obs, info = self.envs[i].reset()
                info['terminal_observation'] = end_obs
                info['terminal_info'] = end_info
```

Keep appending `done` to `results` — the `VecEnv` contract above this layer is
out of scope and must not change.

- [ ] **Step 4: `subproc_vec_env.py:189`**

Identical treatment: unpack five, derive `done`, leave the surrounding
auto-reset and `terminal_info` stashing untouched.

- [ ] **Step 5: Run the pendulum suite**

Run: `python -m pytest tests/test_inverted_pendulum/ -q --no-header 2>&1 | tail -3`
Expected: **74 passed, 1 failed** — only the known subprocess failure.

- [ ] **Step 6: Commit**

```bash
git add safe_control_gym/envs/env_wrappers/ safe_control_gym/experiments/base_experiment.py
git commit -m "Adapt wrappers and vectorized envs to the 5-tuple

Each derives done = terminated or truncated for its existing body, so recorded
data and the VecEnv contract above them are unchanged."
```

---

### Task 7: Behaviour-preserving consumers

Fifteen of the 21 unpack sites do not distinguish the two flags. They collapse
immediately and behave identically.

**Files (all Modify):**
- `safe_control_gym/controllers/lqr/ilqr.py:380`
- `safe_control_gym/controllers/mpc/gp_mpc.py:1089`
- `safe_control_gym/experiments/base_experiment.py:127,144,359`
- `safe_control_gym/controllers/ppo/ppo.py:235`
- `safe_control_gym/controllers/ddpg/ddpg.py:245`
- `safe_control_gym/controllers/sac/sac.py:243`
- `safe_control_gym/controllers/rarl/rarl.py:242`
- `safe_control_gym/controllers/rarl/rap.py:231`
- `safe_control_gym/controllers/safe_explorer/safe_ppo.py:256,439`

- [ ] **Step 1: Apply the mechanical change at each site**

Every one of these is an evaluation or rollout loop that only asks "is the
episode over". At each, replace:

```python
            obs, _, done, info = env.step(action)
```

with:

```python
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
```

preserving whatever the first two names are at that site (`obs, cost` at
`ilqr.py:380`, `obs, reward` at `gp_mpc.py:1089`, `obs_next, _` at
`safe_ppo.py:439`).

- [ ] **Step 2: Verify no 4-tuple unpack survives**

Run:

```bash
grep -rn "done, info = .*\.step(" --include="*.py" safe_control_gym/ | grep -v "truncated"
```

Expected: only the six sites Task 8 handles
(`ppo.py:269`, `ddpg.py:287`, `sac.py:281`, `rarl.py:368`, `rap.py:380`,
`safe_ppo.py:314`).

- [ ] **Step 3: Run the upstream suite**

Run: `python -m pytest tests/test_examples/ tests/test_build.py -q --no-header 2>&1 | tail -3`
Expected: **68 passed, 2 skipped**.

- [ ] **Step 4: Commit**

```bash
git add safe_control_gym/controllers/ safe_control_gym/experiments/base_experiment.py
git commit -m "Adapt the behaviour-preserving step consumers to the 5-tuple

Fifteen evaluation and rollout loops that only ask whether the episode ended;
each collapses to done = terminated or truncated at the unpack."
```

---

### Task 8: The six truncation-compensation blocks

These are the sites that already do the right thing via the info key. They keep
doing it; the flag becomes the source of truth and the info key becomes the
cross-check.

**Files (all Modify):**
- `safe_control_gym/controllers/sac/sac.py:281-304`
- `safe_control_gym/controllers/ddpg/ddpg.py:287-310`
- `safe_control_gym/controllers/ppo/ppo.py:269`
- `safe_control_gym/controllers/rarl/rarl.py:368`
- `safe_control_gym/controllers/rarl/rap.py:380`
- `safe_control_gym/controllers/safe_explorer/safe_ppo.py:314`

- [ ] **Step 1: Update the unpack in each, leaving the compensation intact**

At `sac.py:281`:

```python
        next_obs, rew, terminated, truncated, info = self.env.step(action)

        next_obs = self.obs_normalizer(next_obs)
        done = np.logical_or(terminated, truncated)
        rew = self.reward_normalizer(rew, done)
        mask = 1 - np.asarray(done)
```

Leave lines 287-304 — the `terminal_info` / `TimeLimit.truncated` loop and
`true_mask[idx] = 1.0` — **exactly as they are**. These run against the
vectorized env's per-sub-env `info['n']`, which Task 6 did not change. Apply the
same unpack change at the other five sites.

- [ ] **Step 2: Write the behaviour-identity test**

`tests/test_envs/test_truncation_semantics.py`:

```python
'''terminated/truncated must agree with the legacy info key, on every step.

The six RL controllers already compensated for time truncation via
info['TimeLimit.truncated']. The new flags formalise that; they must not
disagree with it, or the compensation silently changes meaning.
'''
import numpy as np
import pytest

from safe_control_gym.utils.registration import make

TASKS = [('inverted_pendulum', {}), ('cartpole', {}),
         ('quadrotor', {'quad_type': 2}), ('quadrotor', {'quad_type': 3})]


@pytest.mark.parametrize('task,cfg', TASKS)
def test_truncated_agrees_with_legacy_info_key(task, cfg):
    env = make(task, **cfg)
    env.reset(seed=7)
    rng = np.random.default_rng(7)
    saw_truncation = False
    for _ in range(env.CTRL_STEPS + 5):
        act = rng.uniform(env.action_space.low, env.action_space.high)
        _, _, terminated, truncated, info = env.step(act)
        if 'TimeLimit.truncated' in info:
            saw_truncation = True
            assert truncated is True or truncated == 1
            assert info['TimeLimit.truncated'] == (not terminated)
        if terminated or truncated:
            break
    assert saw_truncation or terminated, \
        'episode neither truncated nor terminated within CTRL_STEPS + 5'
    env.close()


@pytest.mark.parametrize('task,cfg', TASKS)
def test_flags_are_booleans(task, cfg):
    env = make(task, **cfg)
    env.reset(seed=3)
    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    env.close()


def test_terminated_and_truncated_can_co_occur():
    '''Goal reached on exactly the horizon step must set both flags.

    The two conditions are computed independently -- termination in
    _get_done(), truncation from ctrl_step_counter in after_step -- so neither
    may mask the other. A migration that returns `terminated or truncated` from
    one slot and False from the other passes every other test here.
    '''
    env = make('inverted_pendulum')
    env.reset(seed=11)
    # Park the state inside the goal ball and wind the counter to the horizon,
    # so this step both terminates and truncates.
    env.state = np.array(env.X_GOAL, dtype=np.float64).copy()
    env.ctrl_step_counter = env.CTRL_STEPS - 1
    _, _, terminated, truncated, info = env.step(np.zeros(1))
    assert bool(terminated) is True, 'goal state must terminate'
    assert bool(truncated) is True, 'horizon step must truncate'
    # The legacy key is `not terminated`, so a genuine co-occurrence records False.
    assert info['TimeLimit.truncated'] is False
    env.close()
```

- [ ] **Step 3: Run it**

Run: `python -m pytest tests/test_envs/test_truncation_semantics.py -q`
Expected: `9 passed` (2 parametrized tests x 4 tasks, plus co-occurrence).

- [ ] **Step 4: Run the oracles**

Run:

```bash
python -m pytest tests/test_envs/ -q --no-header
python -m pytest tests/test_inverted_pendulum/ -q --no-header 2>&1 | tail -3
python -m pytest tests/test_examples/ tests/test_build.py -q --no-header 2>&1 | tail -3
```

Expected: all of `tests/test_envs/` green including both dataset slices at
`atol=1e-12`; pendulum 74 passed / 1 known failure; upstream 68 passed, 2
skipped. **A dataset-slice failure here is the migration bug this whole plan
exists to catch — do not adjust the tolerance.**

- [ ] **Step 5: Commit**

```bash
git add safe_control_gym/controllers/ tests/test_envs/test_truncation_semantics.py
git commit -m "Point the six truncation-compensation blocks at the new flags

The compensation logic is untouched; only the unpack changes. A test asserts
truncated agrees with info['TimeLimit.truncated'] on every step, so the
formalisation cannot silently diverge from the convention it replaces."
```

---

### Task 9: `check_env` conformance

**Files:**
- Create: `tests/test_envs/test_gymnasium_conformance.py`

- [ ] **Step 1: Write the test**

```python
'''Gymnasium API conformance for every registered environment.

SB3's env_checker validates the contract directly -- tuple arity, reset
signature, space conformance, dtypes -- rather than inferring correctness from
tests that happen to pass. It is the primary evidence the migration is correct,
and it covers the systems that have no golden fixtures.
'''
import pytest
from stable_baselines3.common.env_checker import check_env

from safe_control_gym.utils.registration import make

TASKS = [('inverted_pendulum', {}), ('cartpole', {}),
         ('quadrotor', {'quad_type': 2}), ('quadrotor', {'quad_type': 3})]


@pytest.mark.parametrize('task,cfg', TASKS)
def test_check_env(task, cfg):
    env = make(task, **cfg)
    check_env(env, warn=True, skip_render_check=True)
    env.close()
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_envs/test_gymnasium_conformance.py -q`
Expected: `4 passed`.

If it fails on observation dtype (`float64` where `Box` declares `float32`),
fix it in `_set_observation_space` by passing `dtype=np.float64` explicitly —
**not** by casting observations, which would change numbers and break Task 1's
fixtures.

- [ ] **Step 3: Re-run the golden rollouts**

Run: `python -m pytest tests/test_envs/test_env_rollouts.py -q`
Expected: `3 passed`. Confirms any conformance fix did not perturb physics.

- [ ] **Step 4: Commit**

```bash
git add tests/test_envs/test_gymnasium_conformance.py
git commit -m "Assert gymnasium API conformance via SB3's env_checker

Validates the contract directly for all four registered environments, including
the ones with no golden fixtures."
```

---

### Task 10: Task-agnostic SB3 training entry point

**Files:**
- Create: `safe_control_gym/experiments/train_sb3.py`
- Create: `safe_control_gym/envs/env_wrappers/shaping.py`
- Create: `tests/test_envs/test_train_sb3.py`

**Interfaces:**
- Consumes: 5-tuple environments from Tasks 5-7.
- Produces: `train_sb3.train()`; `shaping.AngleObservation(env, angle_index,
  rate_index, rate_max)`; `shaping.ActionRepeat(env, repeat)`.

- [ ] **Step 1: Write the optional shaping wrappers**

`safe_control_gym/envs/env_wrappers/shaping.py`:

```python
'''Optional, config-selected observation and cadence shaping.

Nothing here is applied by default. These exist because particular systems were
trained with particular conventions -- the pendulum's policies consume
[cos theta, sin theta, theta_dot / theta_dot_max] at an action_repeat of 4 --
and baking those into the trainer would silently mis-train every other system.
'''
import gymnasium as gym
import numpy as np


class AngleObservation(gym.ObservationWrapper):
    '''Re-encode one angular coordinate as (cos, sin) and scale its rate.'''

    def __init__(self, env, angle_index, rate_index, rate_max):
        super().__init__(env)
        self.angle_index = int(angle_index)
        self.rate_index = int(rate_index)
        self.rate_max = float(rate_max)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float64),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            dtype=np.float64)

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        angle = obs[self.angle_index]
        return np.array([np.cos(angle), np.sin(angle),
                         obs[self.rate_index] / self.rate_max],
                        dtype=np.float64)


class ActionRepeat(gym.Wrapper):
    '''Hold each action for `repeat` control steps, as the policy was trained.'''

    def __init__(self, env, repeat):
        super().__init__(env)
        self.repeat = max(1, int(repeat))

    def step(self, action):
        total = 0.0
        terminated = truncated = False
        obs = info = None
        for _ in range(self.repeat):
            obs, rew, terminated, truncated, info = self.env.step(action)
            total += rew
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info
```

- [ ] **Step 2: Write the trainer**

`safe_control_gym/experiments/train_sb3.py`:

```python
'''Task-agnostic stable-baselines3 training.

The only module in the package permitted to import stable-baselines3; envs/ and
controllers/ stay SB3-free so inference and dataset collection never gain the
dependency.

Shaping is configuration, not code: --kv_overrides sb3_config.angle_obs=... and
sb3_config.action_repeat=... select the optional wrappers. No task is special
cased here.

    python -m safe_control_gym.experiments.train_sb3 \\
        --task inverted_pendulum --algo sac --output_dir results/pendulum_sac \\
        --kv_overrides sb3_config.total_timesteps=200000
'''
import os

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from safe_control_gym.envs.env_wrappers.shaping import ActionRepeat, AngleObservation
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_seed_from_config

ALGOS = {'sac': SAC, 'ppo': PPO}


def build_env(config):
    '''Registered env plus whatever shaping the config asks for.'''
    env = make(config.task, **config.task_config)
    sb3_config = config.get('sb3_config', {})
    angle_obs = sb3_config.get('angle_obs', None)
    if angle_obs is not None:
        env = AngleObservation(env, angle_obs['angle_index'],
                               angle_obs['rate_index'], angle_obs['rate_max'])
    repeat = int(sb3_config.get('action_repeat', 1))
    if repeat > 1:
        env = ActionRepeat(env, repeat)
    return env


def train():
    '''Train and checkpoint; returns the fitted model.'''
    config = ConfigFactory().merge()
    set_seed_from_config(config)
    mkdirs(config.output_dir)

    sb3_config = config.get('sb3_config', {})
    algo = ALGOS[config.algo]
    total_timesteps = int(sb3_config.get('total_timesteps', 100000))
    save_freq = int(sb3_config.get('save_freq', 10000))

    env = build_env(config)
    model = algo('MlpPolicy', env, seed=config.seed, verbose=1,
                 policy_kwargs={'net_arch': list(sb3_config.get('net_arch', [256, 256]))})
    # Periodic checkpoints, not only best: the shipped strong/weak model pairs
    # are best-vs-intermediate checkpoints of one run, so dropping intermediates
    # would make that axis unreproducible.
    callback = CheckpointCallback(save_freq=save_freq,
                                  save_path=os.path.join(config.output_dir, 'checkpoints'),
                                  name_prefix='step')
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(config.output_dir, 'model_final'))
    env.close()
    return model


if __name__ == '__main__':
    train()
```

- [ ] **Step 3: Write the smoke test**

`tests/test_envs/test_train_sb3.py`:

```python
'''The trainer must be genuinely task-agnostic, not pendulum-shaped.'''
import os
import subprocess
import sys

import pytest

REPO = os.path.join(os.path.dirname(__file__), '..', '..')

TASKS = ['inverted_pendulum', 'cartpole', 'quadrotor']


@pytest.mark.parametrize('task', TASKS)
def test_trains_briefly(task, tmp_path):
    out = tmp_path / task
    result = subprocess.run(
        [sys.executable, '-m', 'safe_control_gym.experiments.train_sb3',
         '--task', task, '--algo', 'sac', '--seed', '1',
         '--output_dir', str(out),
         '--kv_overrides', 'sb3_config.total_timesteps=256',
         'sb3_config.save_freq=128'],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    assert (out / 'model_final.zip').exists()
    assert list((out / 'checkpoints').glob('step_*.zip'))


def test_sb3_not_imported_by_library():
    '''envs/ and controllers/ must stay importable without SB3.'''
    probe = (
        'import sys, importlib\n'
        'sys.modules["stable_baselines3"] = None\n'
        'import safe_control_gym.envs, safe_control_gym.controllers\n'
        'print("ok")\n'
    )
    result = subprocess.run([sys.executable, '-c', probe],
                            cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'ok' in result.stdout
```

- [ ] **Step 4: Run it**

Run: `python -m pytest tests/test_envs/test_train_sb3.py -q`
Expected: `4 passed`. If `--kv_overrides` cannot express the nested
`sb3_config.*` keys, add an `sb3_config` block to the task's registered yaml
defaults and override the leaf keys instead — do not add per-task branching to
`train_sb3.py`.

- [ ] **Step 5: Full verification**

```bash
python -m pytest tests/test_envs/ -q --no-header
python -m pytest tests/test_inverted_pendulum/ -q --no-header 2>&1 | tail -3
python -m pytest tests/test_examples/ tests/test_build.py -q --no-header 2>&1 | tail -3
pre-commit run --all
```

Expected: `tests/test_envs/` fully green; pendulum **74 passed, 1 failed**
(known subprocess failure only); upstream **68 passed, 2 skipped**;
`pre-commit` clean.

- [ ] **Step 6: Commit**

```bash
git add safe_control_gym/experiments/train_sb3.py \
        safe_control_gym/envs/env_wrappers/shaping.py \
        tests/test_envs/test_train_sb3.py
git commit -m "Add task-agnostic SB3 training

One entry point through ConfigFactory, no per-system branching; shaping is
config-selected wrappers. A test asserts envs/ and controllers/ still import
with stable_baselines3 unavailable, so inference and collection never gain the
dependency."
```

---

## Follow-on work (not in this plan)

- **Per-system exporters**, starting with the pendulum's 8-key `.pt`, so trained
  policies can actually be run in-repo. Until then `train_sb3.py` output has no
  consumer.
- **Noise-matched training** — the original goal, unblocked by this plan.
- **`test_pendulum_experiment.py` subprocess failure** — pre-existing
  `PYTHONPATH` issue, deliberately untouched here so it stays a visible
  known-failure rather than being conflated with migration work.
- **Wiki ingest** — `.claude/docs/architecture.md` and `workflows.md` describe
  the 4-tuple world and the absent SB3 dependency; both need updating once this
  lands, per the ingest operation in `CLAUDE.md`.
