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
python3 -m pytest ./tests/test_envs/               # gymnasium/SB3 migration oracles
python3 -m pytest ./tests/test_examples/           # upstream example smoke tests
```

`tests/test_inverted_pendulum/` is the fork's own and the one to keep green:
`test_env.py`, `test_registration.py`, `test_pendulum_lqr.py`,
`test_pendulum_rl.py`, `test_pendulum_noise.py`, `test_pendulum_experiment.py`,
`test_generate.py`, `test_collection_splits.py`. **75 passed**, 0 known
failures (verified directly). The bar used to be 74 passed plus one accepted
`test_pendulum_experiment.py` subprocess failure; that failure was a broken
editable install — `safe_control_gym.pth` pointed at a deleted sibling clone,
so the package only resolved from a cwd inside this repo, and the test shells
out from a script under `examples/` where it doesn't. `pip install -e .`
repaired it. If a run reports 74/1 again, that is a stale environment, not a
regression — re-run `pip install -e .` before treating it as one.

`tests/test_envs/` is a new directory, added by the Gymnasium migration, and
holds the oracles that protect it:

- `test_env_rollouts.py` — golden rollouts for cartpole and both quadrotors
  (the pendulum's own fixtures already lived under `test_inverted_pendulum/`),
  captured under gymnasium 0.28 before any migration commit. `atol=1e-9`.
- `test_dataset_slice.py` — a 338-cell `lqr` and `v3_strong` grid
  (`--resolution 0.5`) over `generate_inverted_pendulum_trajectories.py`,
  captured the same way. `atol=1e-12`. This is the oracle that most directly
  protects the datasets: a failure here is the migration bug the whole plan
  exists to catch, and the tolerance must not be loosened to make one pass.
- `test_truncation_semantics.py` — `terminated`/`truncated` agree with the
  legacy `info['TimeLimit.truncated']` on every step, for all four
  environments, including the case where both flags are true on the same step.
- `test_wrapper_forwarding.py` — `AttributeForwardingMixin`'s `FORWARDED`
  allowlist resolves the attributes call sites actually read, and an
  unlisted attribute still raises.
- `test_gymnasium_conformance.py` — `stable_baselines3.common.env_checker.check_env`
  passes on every registered environment; the primary evidence the migration
  is correct, because it validates the API contract directly rather than
  inferring correctness from tests that happen to pass.
- `test_episode_flags_initialised.py` — `goal_reached`/`out_of_bounds` exist
  immediately after `reset()`, for all four environments (a CartPole gap this
  migration's review surfaced and fixed).
- `test_train_sb3.py` — a short `train_sb3` run per env id, base and composite
  alike, completes and writes a loadable model; the run directory carries
  `config.yml`, `args.yml` and `command.txt`; a second run takes `_2` rather
  than clobbering; `--env_id` aliases `--task` and passing both is an error;
  four systems launched concurrently all get a directory; and `envs/` and
  `controllers/` still import with `stable_baselines3` blanked out of
  `sys.modules`.
- `test_composite_env_ids.py` — each composite `(system, task)` id passes
  `check_env`, and building it produces the same environment as building its
  base id with the same config, asserted observation-by-observation. That
  equivalence is what keeps runs under the two ids comparable, so a composite
  yaml drifting from its base is a test failure rather than a silent change.
- `test_concurrent_pybullet_envs.py` — two quadrotor envs alive in one process
  must reproduce a single-env rollout exactly. Guards the `physicsClientId`
  fix; reachable from ordinary training because `EvalCallback` holds a second
  env open. See [datasets.md](datasets.md) for what it was hiding.
- `test_eval_policy.py` — evaluation is deterministic under a fixed seed, the
  report always carries absolute numbers for both controllers, `--skip_baseline`
  yields `NO_BASELINE` rather than a default `PASS`, and the pendulum resolves
  to `pendulum_lqr`.

There is deliberately no `test_invariant_sets.py` in this directory: `quad2d`
and `quad3d` do not reproduce the committed `.npz` artifacts bit-exactly (see
`.claude/docs/datasets.md`), so a test gating on that would fail for a reason
nobody intends to fix. `invariant_sets/*.npz` stay read-only on disk
(`-r--r-----`) for the same reason `compute_invariant_sets.py` rewrites them in
place is documented there.

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

## Training with stable-baselines3

`safe_control_gym/experiments/train_sb3.py` is task-agnostic: it works for any
registered `--task` through the same `ConfigFactory` shape, with no
per-system branching in the trainer itself.

```bash
python -m safe_control_gym.experiments.train_sb3 \
    --env_id cartpole_stabilization --algo sac --output_dir logs \
    --overrides configs/sb3/cartpole_stabilization_sac.yaml --use_gpu
```

`--env_id` is an alias for `--task`; they are the same registry lookup, and
passing both is an error rather than a silent precedence rule. `--task` is not
renamed because `configuration.py` defines it once for every entry point in the
repo.

Runs land in `<output_dir>/<algo>/<env_id>_<run>/`, following RL Baselines3 Zoo:
`best_model.zip`, `checkpoints/`, `model_final.zip`, plus `config.yml`,
`args.yml` and `command.txt`. `<run>` auto-increments, so re-running never
clobbers. `config.yml` is load-bearing — `eval_policy` rebuilds the environment
and its wrappers from it, so a run without one cannot be evaluated.

Per-system hyperparameters live in `configs/sb3/<env_id>_<algo>.yaml`, passed
through `--overrides`, and never in the env yaml: a composite id's yaml has to
stay a faithful copy of its base. The quadrotor configs override
`task_config.randomized_init: True` there, because the env default is `False`
and SAC would otherwise see one initial state for the whole run. All four are
hand-written starting points; no hyperparameter search has been run.

## Evaluating a trained policy

```bash
python -m safe_control_gym.experiments.eval_policy \
    --run logs/sac/cartpole_stabilization_1 --n_episodes 100 --seed 0
```

Rolls out the policy and the system's LQR (`pendulum_lqr` for the pendulum,
`lqr` elsewhere) from the *same* seeded initial states — asserted, not assumed —
and writes `eval.json` beside the weights. `PASS` when the policy's success rate
is within `--margin` (default 0.05) of the baseline's.

Success is computed from the state, not read from `info['goal_reached']`:
`_get_info` gates that key on `COST == Cost.QUADRATIC`, and training uses
`rl_reward`, so for cartpole and both quadrotors it is absent entirely. It is
evaluated at the **terminal** state, because under `rl_reward` an episode does
not stop on entering the goal ball.

The bar is vacuous where LQR itself is weak. The mitigation is disclosure, not a
cleverer rule: absolute numbers for both controllers are always printed and
stored, so never quote a verdict without them.

Task-specific shaping (the pendulum's `[cos, sin, thdot/max]` observation
re-encoding, its `action_repeat` of 4) is not hardcoded — it is an optional
wrapper from `envs/env_wrappers/shaping.py`, selected via
`sb3_config.angle_obs` / `sb3_config.action_repeat` in the config, so other
systems train on their own native observation with no wrapper at all.

`--use_gpu` selects SB3's device (defaults to CPU, like every other entry
point here). Pass it if a GPU is free: measured on an idle ilab2 (64 cores,
RTX A4500), SAC on the pendulum, `net_arch: [256, 256]`, both devices back to
back with OMP/MKL threads pinned to 8 — **cpu 65.6 steps/s vs cuda 111.0
steps/s, GPU 1.69x faster**. An earlier docstring asserted the opposite from
the general principle that small MLP policies favour CPU; that was wrong,
measured on a since-corrected run, and the corrected number above is what to
trust. The environment itself is never the bottleneck either way — the
pendulum alone steps at ~12,500/s.

`train_sb3.py` is the only module in the package allowed to import
`stable_baselines3`; see `.claude/docs/architecture.md` for why, and for the
current gap (no exporter, so a trained model has no in-repo consumer yet).

## Exporting a trained policy

SB3 writes a `.zip` (policy + critics + optimizers). `pendulum_rl` loads an
8-key `.pt` (actor weights plus run constants). Convert with:

```bash
python scripts/export_sb3_pendulum.py <model.zip> <out.pt> --action_repeat 4
```

It extends `controllers/pendulum_rl/models/manifest.json` with the git SHA,
source zip, and SB3/torch versions. `tests/test_envs/test_export_sb3_pendulum.py`
verifies the round trip against SB3's own `predict(deterministic=True)` over 200
random states at `1e-6`; the original external port achieved `~3e-7`, so that
bar is established rather than invented.

Pendulum only. Cartpole and the quadrotors have no native actor controller to
export into, so their trained policies cannot yet be run or collected with.

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

# the three stochastic pendulum families, each a different MECHANISM
--torque_noise TAU                        # uniform, pre-saturation
--noise_alpha A --noise_beta B            # gaussian sigma = A + B|u|, pre-saturation
--noise_alpha A --noise_beta B --external_noise   # ... POST-saturation
```

The three are mutually exclusive and each writes to its own family directory
(`noisy_torque/`, `signal_dependent/`, `external_torque/`). `--external_noise`
needs an alpha/beta; `--noise_alpha` and `--noise_beta` must be given together,
since defaulting the missing one to zero would silently collect a level nobody
asked for and the directory name would not say so.

Verification for the signal-dependent path, both cheap and both worth running
after any change to `disturbances.py` or `_preprocess_control`:

```bash
python pend_sig_validate.py sigma      # empirical std == alpha + beta|u|
python pend_sig_validate.py gate       # alpha = beta = 0 reproduces tau_0.00
python pend_sig_sweep.py 2000 20       # level sweep; SIG_EXTERNAL=1 for sat(u)+w
```

Cartpole has the same family, without the placement switch (its LQR never
saturates, so there is nothing for a placement to change):

```bash
python cp_collect.py --split {train,eval} --alpha A --beta B --trials 100 \
    --shard S --nshards N --out <dir>/shard.npz
python cp_gauss_sweep.py --config I --shard S --nshards N --out shard.npz
python cp_gauss_sweep.py --merge --out-dir <dir>
```

`--alpha/--beta` and the uniform `--level` are mutually exclusive and the
collector says so. On a cluster that cannot see the iLab filesystem, both
cartpole entry points need `CP_DET_DIR` (holding `eval_states.txt`) and
`CP_SIGMA0` (the deterministic labels) — see the runbook in
`docs/superpowers/specs/2026-08-17-cartpole-gaussian-signal-collection.md`.

Amarel batch scripts for the whole pipeline live in `scripts/`:
`sbatch_pendulum_external_pair.sh` (any pendulum level, ALPHA/BETA from env),
`sbatch_cartpole_gauss_sweep.sh` and `sbatch_cartpole_gauss_collect.sh`,
each taking `MODE=collect` then `MODE=finalize`.

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

Related: [architecture.md](architecture.md) for the two-stack split `train_sb3.py` lives inside, [datasets.md](datasets.md) for what a collection run produces, [compute.md](compute.md) for which machine, [conventions.md](conventions.md) for what pre-commit will demand.
