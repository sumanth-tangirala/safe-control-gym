# Spec: Collection Oracles for Every System

Date: 2026-07-29
Scope: `tests/test_envs/` only. No production code changes.

## Goal

Give every dataset family a fixture that fails if collection output changes.

Concretely: capture `(starts, labels, terminal_states)` from each collector into
a committed JSON fixture, and add a test that re-runs the collector and compares
at `atol=1e-12`.

## Motivation

Five collectors totalling 5,821 lines produce six shipped dataset families. Only
the inverted pendulum has an oracle
(`tests/test_envs/fixtures/dataset_slice_{lqr,v3_strong}.json`).

The plan is to unify those five collectors into one. That refactor is
unverifiable without oracles: dataset semantics can shift with nothing failing,
and the shift surfaces weeks later as bad model behaviour rather than as a red
test.

This is not hypothetical. During the Gymnasium migration a one-line change to
`state_space`'s declared dtype moved out-of-bounds thresholds by up to 1.9e-7
and flipped an `out_of_bounds` decision. Every suite stayed green. It was caught
by review, not by tests, because no oracle covered that path. The pendulum's
338-cell slice reproducing at `atol=1e-12` is the only reason the rest of that
migration could be called verified.

Oracles first is therefore a sequencing constraint, not a preference: they must
exist before any collector is touched, or they are measuring the new behaviour
rather than the old.

## Prior state (measured)

| collector | lines | args | controller | noise | splits | sampling |
| --- | --- | --- | --- | --- | --- | --- |
| `generate_inverted_pendulum_trajectories.py` | 895 | 18 | RL + LQR | yes | yes | grid + random |
| `generate_cartpole_trajectories.py` | 801 | 18 | LQR only | no | no | **grid only** |
| `generate_quadrotor_2d_trajectories.py` | 1000 | 39 | LQR only | no | no | grid + random |
| `generate_quadrotor_2d_trajectories_rl.py` | 1624 | 39 | RL | no | no | grid + random |
| `generate_quadrotor_3d_trajectories.py` | 1501 | 39 | LQR only | no | no | grid + random |

All five write the same shapes: `roa_labels.txt` (`init_state..., label`) and
`trajectories/sequence_<i>.txt` (one state per line).

Existing dataset families under `DATA_ROOT/deterministic/`: `cartpole_pybullet`,
`pendulum`, `quadrotor2D_rl`, `quadrotor3D_lqr`, plus `humanoid_get_up_medium`
and a `pendulum_lqr_50k` symlink.

## Decisions

### Capture strategy differs per system, because dimensionality does

`cartpole` has no `--random_init`; it is grid-only over 4 dimensions. The other
three accept `--random_init --num_trajs N --seed S`.

A grid does not survive dimensionality: quad3d has 12 state dimensions, so even
two points per axis is 4,096 rollouts at ~250 steps/s. Random sampling with a
fixed seed is equally deterministic and does not explode.

| system | mode | why |
| --- | --- | --- |
| cartpole | coarse grid | only mode available; 4 dims is tractable |
| quad2d | `--random_init --num_trajs --seed` | 6 dims |
| quad2d_rl | `--random_init --num_trajs --seed` | 6 dims, plus a loaded policy |
| quad3d | `--random_init --num_trajs --seed` | 12 dims; a grid is impossible |

The pendulum keeps its existing 338-cell grid fixtures unchanged.

### Every fixture must contain both labels

A fixture where every rollout succeeds — or every one fails — pins almost
nothing: a regression that flips the success rule would still reproduce. Each
capture must be tuned (via bounds or trajectory count) until both labels appear,
and the test asserts that both are present. If a system cannot be made to
produce both cheaply, that is a finding to report, not a fixture to accept.

### Runtime budget: under 3 minutes per system

These run in the normal suite. The pendulum's two slices take ~130s combined.
Anything slower gets skipped in practice, which makes it worthless. Tune
`--num_trajs` and bounds to fit, favouring fewer rollouts over shorter horizons
— truncating the horizon changes which labels appear.

### Pin collector output only

`quad2d`'s collector loads a `safe_explorer_ppo` policy, and its invariant-set
artifact does not reproduce (a ReLU crease inside `fd_linearize`'s stencil; see
`.claude/docs/datasets.md`). The oracle therefore pins what the collector
*wrote* — starts, labels, terminal states — and must not assert anything derived
from `compute_invariant_sets.py`. Do not pass `--invariant_terminal_sets`.

### Scratch output only

Captures write to a temp directory, never under
`/common/users/shared/pracsys/genMoPlan/data_trajectories`. A `PreToolUse` hook
denies writes there, and the data is shared.

## Formats

One fixture per system, `tests/test_envs/fixtures/dataset_slice_<system>.json`:

| key | type | notes |
| --- | --- | --- |
| `starts` | list of list of float | initial states, in collector order |
| `labels` | list of int | 1 = success, 0 = failure |
| `terminal_states` | list of list of float | last row of each `sequence_<i>.txt` |
| `command` | list of str | the exact argv that produced it |
| `n` | int | rollout count, for a fast shape assertion |

`command` is what makes a fixture regenerable a year from now, and follows RL
Zoo's convention of storing `command.txt` beside every run rather than inferring
the run from its directory name.

## Order of work

The four captures are independent and should run in parallel. The shared test
file is written once, up front, so no capture task contends on it.

1. Write `tests/test_envs/test_dataset_slices.py`, parameterised over whichever
   fixtures exist, so it passes with zero fixtures and strengthens as each lands.
2. In parallel, one task per system: tune the invocation, capture the fixture,
   confirm both labels appear and it runs under 3 minutes.
3. Verify the whole suite, then commit.

## Verification

- Each fixture reproduces at `atol=1e-12` on a re-run.
- Each contains both labels.
- Each capture completes in under 3 minutes.
- `tests/test_envs/` and `tests/test_inverted_pendulum/` stay green (121 passed
  as of `249ed89c`).
- `invariant_sets/*.npz` remain byte-identical to `HEAD`.
- **Each fixture is proven falsifiable**: perturb one recorded value, confirm
  the test fails, restore. A fixture that cannot fail is not an oracle — this
  check exists because the pendulum's `atol` mutation passed trivially and the
  gap was only found by corrupting a value instead.

## Out of scope

- **Unifying the collectors.** The follow-on this exists to make safe.
- **The generic runtime, processors and exporters.** Separate spec.
- **Noise support in the collectors.** Only the pendulum has it; adding it
  elsewhere belongs with the unification.
- **Trajectory tracking.** No terminal-state success label exists for it.
