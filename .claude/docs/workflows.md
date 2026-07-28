# Workflows

Load when running tests, lint, examples, or a collection job.

## Environment

Python 3.10. Assume the project env is already active — `python3`, `pytest` and
`pre-commit` should resolve from it. Which env that is on a given machine
belongs in `CLAUDE.local.md`, which is not committed.

Fresh setup:

```bash
conda create -n safe python=3.10 && conda activate safe
python -m pip install --upgrade pip
python -m pip install -e .
```

`pycddlib` needs `gmp` (`conda install -c anaconda gmp` or
`sudo apt-get install libgmp-dev`). MPC via acados needs a separate acados
build; nothing else depends on it.

## Tests

```bash
python3 -m pytest ./tests/                         # everything
python3 -m pytest ./tests/test_inverted_pendulum/  # this fork's tests
python3 -m pytest ./tests/test_examples/           # upstream example smoke tests
```

`tests/test_inverted_pendulum/` is the fork's own and the one to keep green:
`test_env.py`, `test_registration.py`, `test_pendulum_lqr.py`,
`test_pendulum_rl.py`, `test_pendulum_noise.py`, `test_pendulum_experiment.py`,
`test_generate.py`, `test_collection_splits.py`.

`tests/test_hpo/` needs a MySQL-backed Optuna store and does not run
standalone. Do not treat its failures as a regression you introduced.

The root-level `test_*.py` files are ad-hoc verification scripts from earlier
work, not on any pytest path. Run them directly if you need them.

## Lint

`pre-commit` is the only style gate. A `PostToolUse` hook already runs it on
every file Claude writes, so most of the time this is automatic.

```bash
pre-commit install       # once per clone — the session hook warns if missing
pre-commit run --all     # whole repo
pre-commit run --files path/to/file.py
```

A hook denies `git commit --no-verify`. If pre-commit is failing, the fix is the
code, not the flag.

## Running an example

Every example follows the same `ConfigFactory` shape — `--algo`, `--task`,
`--overrides` (files, applied in order), `--kv_overrides` (dotted scalars):

```bash
python examples/inverted_pendulum/pendulum_experiment.py \
    --algo pendulum_lqr --task inverted_pendulum \
    --overrides examples/inverted_pendulum/config_overrides/noise/control_proportional_high.yaml \
    --kv_overrides n_episodes=10
```

```bash
cd examples/
python3 lqr/lqr_experiment.py --algo lqr --task cartpole \
    --overrides ./lqr/config_overrides/cartpole/cartpole_stab.yaml \
                ./lqr/config_overrides/cartpole/lqr_cartpole_stab.yaml
```

Output goes to `results/` (gitignored).

## Collection runs

**These take hours. Never run one in the foreground of a turn.** Background it
and poll, or hand the user the command.

```bash
# train split — 300k rollouts
python generate_inverted_pendulum_trajectories.py \
    --controller lqr --split train --parallel --num_workers 32

# eval split — grid batches until mean_se < 0.01
python generate_inverted_pendulum_trajectories.py \
    --controller lqr --split eval --parallel --num_workers 32

# noisy variant
python generate_inverted_pendulum_trajectories.py \
    --controller v3_strong --split train --noise control_proportional_med

# invariant-terminal-set scheme (off by default)
python generate_inverted_pendulum_trajectories.py \
    --controller lqr --invariant_terminal_sets
```

Flags worth knowing before launching:

- `--output_dir` — otherwise derived from controller and noise level.
- `--skip_save` — compute labels and stats without writing sequences. The cheap
  way to sanity-check a change.
- `--overwrite` — regenerate even where sequence files already exist.
- `--seed` (default 42) — feeds `rollout_seed`; changing it changes every rollout.
- `--parallel --num_workers N` — `get_available_cpus()` respects affinity/taskset.

An eval run can be killed at any time: the published dataset is the checkpoint,
and `dataset_description.json` records `converged: false`. A train run is not
checkpointed the same way — check before killing one.

Where to run these: `.claude/docs/compute.md`.

## Invariant sets

```bash
python compute_invariant_sets.py --systems pendulum cartpole quad2d quad3d
```

Rewrites `invariant_sets/*.npz`, which are committed and loaded by the
generators. `--skip_validation` skips the boundary-sampling check — do not use
it for an artifact you intend to commit.

---

Related: [datasets.md](datasets.md) for what a collection run produces, [compute.md](compute.md) for which machine, [conventions.md](conventions.md) for what pre-commit will demand.
