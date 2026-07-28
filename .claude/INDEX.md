# Index

Catalog of the wiki. Read this first, then open only what it points at.
Updated on every ingest. Schema and workflows: `CLAUDE.md`. History:
`.claude/log.md`.

## Pages

| Page | Load when | Contents |
| --- | --- | --- |
| [architecture.md](docs/architecture.md) | touching `safe_control_gym/`, adding an env/controller/filter | The library-vs-scripts split and why collection policy stays out of the library. The string registry (`register`/`make`) and every registered id. How `ConfigFactory.merge()` layers defaults, `--overrides` and `--kv_overrides`. Package map. The pendulum noise families and their preset naming. |
| [datasets.md](docs/datasets.md) | generating, reading, or reasoning about a dataset | `DATA_ROOT` layout and the per-controller/per-noise-level directory scheme. All five generators. The two success-labelling regimes — invariant terminal sets vs entry-cut — and the non-normality result that forces the first. Train/eval split semantics, the mean-SE stopping rule, `train.npz` and `eval_success_prob.npz` key layouts, the float32 decision, `rollout_seed`, atomic incremental publication. |
| [workflows.md](docs/workflows.md) | running tests, lint, examples, or a collection job | Which test directories are meaningful and which (`test_hpo`) cannot run standalone. `pre-commit` invocations. The `ConfigFactory` command shape for examples. Collection commands with the flags worth knowing before launching, and which splits are safe to kill. |
| [conventions.md](docs/conventions.md) | writing code or a commit that lands here | Style as `.pre-commit-config.yaml` enforces it: single quotes, `'''` docstrings, line length deliberately unenforced, the `tests/` and `transformations.py` exemptions. Comment and docstring form. Commit message form with examples from the log. The spec-then-plan workflow and the house rule of recording rejected alternatives with the measurement that rejected them. |
| [compute.md](docs/compute.md) | deciding *where* a job runs | What the jobs need, naming no machines: collection is CPU-bound and honours affinity, only RL *training* wants a GPU, dataset sizes on shared storage, which splits suit a preemptible allocation. Machine selection is site-specific and belongs in `CLAUDE.local.md`. |
| [glossary.md](docs/glossary.md) | a term in a spec is unfamiliar | ROA, invariant terminal set, non-normal closed loop, entry-cut, noise floor, noise preset, split, batch, mean-SE stopping rule, drift, half-open grid, horizon, `U_SAT`, terminal-state model. |

## Wiki files

| File | Role |
| --- | --- |
| `CLAUDE.md` | The schema. Three layers, three operations, wiki conventions, resident invariants. |
| `CLAUDE.local.md` | Machine- and account-specific facts for one checkout: which env, which cluster, which remote is writable. Gitignored, so it may not exist here. Nothing in the committed wiki may depend on it. |
| `.claude/INDEX.md` | This catalog. |
| `.claude/log.md` | Append-only history of ingests, filed queries, and lint passes. Newest first. |
| `.claude/wiki_lint.py` | Mechanical half of lint: constants-vs-source, index coverage, orphans, log parseability. |

## Hooks (`.claude/hooks/`)

| File | Event | Behaviour |
| --- | --- | --- |
| `session_start.sh` | `SessionStart` | Emits branch, uncommitted-path count, HEAD, newest `plans/` and spec file, and a warning if `pre-commit install` has not been run. |
| `guard_write.py` | `PreToolUse(Write\|Edit)` | Denies writes under `DATA_ROOT` and to `invariant_sets/*.npz`, with the regeneration command in the reason. |
| `guard_bash.py` | `PreToolUse(Bash)` | Denies `git commit/push --no-verify`, bare `git push --force`/`-f` (`--force-with-lease` passes), `rm -r` under `DATA_ROOT`, and `rm` of `invariant_sets/`. |
| `format_python.sh` | `PostToolUse(Write\|Edit)` | Runs `pre-commit run --files <file>` on `.py`/`.yaml`/`.toml`. Clean file → one pass, silent. Dirty file → fixers apply, second pass reports residual flake8 errors with exit 2. |

Config: `.claude/settings.json`. Personal overrides go in
`.claude/settings.local.json` (gitignored).

## Source layer

What the wiki is derived from. Read these when the wiki is wrong or silent.

| Path | What it is |
| --- | --- |
| `safe_control_gym/` | The library. Upstream code plus the pendulum additions. |
| `generate_*_trajectories*.py` | Dataset collectors, one per system. Root scripts, not library code. |
| `compute_invariant_sets.py` | Produces `invariant_sets/{pendulum,cartpole,quad2d,quad3d}.npz`. |
| `calibrate_*_bounds.py`, `visualize_*.py`, `viz_*.py`, `compare_trajectories.py` | Analysis and figure scripts. Not on any test path. |
| `examples/` | Upstream-style runnable demos, one directory per algorithm. |
| `tests/` | pytest. `tests/test_inverted_pendulum/` is this fork's; the rest is upstream. |
| `docs/superpowers/specs/` | Dated design specs. Decisions with their measured justification. |
| `plans/` | Implementation plans derived from specs. |
| `scripts/` | One-off conversion utilities for the pendulum RL policies. |
| `invariant_sets/` | Committed `.npz` ellipsoid artifacts. |
