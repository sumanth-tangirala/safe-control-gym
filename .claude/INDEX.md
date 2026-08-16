# Index

Catalog of the wiki. Read this first, then open only what it points at.
Updated on every ingest. Schema and workflows: `CLAUDE.md`. History:
`.claude/log.md`.

## Pages

| Page | Load when | Contents |
| --- | --- | --- |
| [architecture.md](docs/architecture.md) | touching `safe_control_gym/`, adding an env/controller/filter | The library-vs-scripts split and why collection policy stays out of the library. All four envs' Gymnasium 5-tuple step API and the two `terminated` sources vs the single `truncated` one. The two coexisting RL stacks — native controllers vs training-only stable-baselines3, confined to `train_sb3.py` — and the no-exporter-yet gap. The string registry (`register`/`make`), every registered id, and the composite `(system, task)` ids that disambiguate the two axes `--task` used to carry at once. How `ConfigFactory.merge()` layers defaults, `--overrides` and `--kv_overrides`. Package map. The two unrelated noise mechanisms: the pendulum's `pendulum_noise.py` presets, and the upstream `disturbances` mechanism every other env has — its six usable types including the fork's `signal_dependent`, and the `external_action_disturbance` switch that decides whether an action disturbance sits inside or outside the actuator saturation, cartpole's three modes as different physics (matched `action`, unmatched `dynamics`, POMDP-only `observation`), and the chosen cartpole axis with its seeding and magnitude traps. The `dynamics` mode as actually used by the quadrotor collections: per-substep re-application (PyBullet clears external forces each step), COM application meaning no torque, and the two disturbance classes that exist but are unregistered. |
| [datasets.md](docs/datasets.md) | generating, reading, or reasoning about a dataset | `DATA_ROOT` layout and the per-controller/per-noise-level directory scheme. All five generators. The two success-labelling regimes — invariant terminal sets vs entry-cut — and the non-normality result that forces the first. Train/eval split semantics, the mean-SE stopping rule, `train.npz` and `eval_success_prob.npz` key layouts, the float32 decision, `rollout_seed`, atomic incremental publication. Why `quad2d`'s invariant-set ellipsoid does not reproduce bit-exactly and why that is expected. Why the five collectors share no output contract, which files were backfilled by scripts outside this repo, and that `cal_set.txt`/`test_set.txt` have no producer anywhere. The `physicsClientId` damping bug that means every shipped quadrotor dataset ran at PyBullet's default damping rather than zero. The six stochastic families and their three mechanisms — matched `action` (pendulum, cartpole), unmatched `dynamics` (both quadrotors), and the pendulum's signal-dependent and external-torque families. Why the external one is the only family whose ROA is not a subset of the deterministic one, and the level-naming trap that created. The cartpole re-collection and the six defects it corrects, including a 20x control-bound error. Why the deterministic cartpole description states its own success rule wrongly, and why labels cannot detect that — only final states can. Noise levels are coupled to the success rule and the horizon and do not transfer across either. The two rate-injection directions for quad3d and what each costs if reversed. |
| [workflows.md](docs/workflows.md) | running tests, lint, examples, or a collection job | Which test directories are meaningful and which (`test_hpo`) cannot run standalone: the fork's `tests/test_inverted_pendulum/` (75 passed, 0 known failures) and the new `tests/test_envs/` oracle suite (golden rollouts, dataset-slice bit-exactness, truncation semantics, wrapper forwarding, `check_env` conformance, SB3 training smoke). `pre-commit` invocations. The `ConfigFactory` command shape for examples, for `train_sb3.py` — including the RL-Zoo run layout, where per-system configs live, and the measured GPU-vs-CPU number — and for `eval_policy.py`'s LQR-relative acceptance bar. Collection commands with the flags worth knowing before launching, and which splits are safe to kill. |
| [conventions.md](docs/conventions.md) | writing code or a commit that lands here | Style as `.pre-commit-config.yaml` enforces it: single quotes, `'''` docstrings, line length deliberately unenforced, the `tests/` and `transformations.py` exemptions. Comment and docstring form. Commit message form with examples from the log. The spec-then-plan workflow and the house rule of recording rejected alternatives with the measurement that rejected them. |
| [compute.md](docs/compute.md) | deciding *where* a job runs | What the jobs need, naming no machines: collection is CPU-bound and honours affinity, RL *training* (native stack or SB3) wants a GPU — measured 1.69x faster for SB3 SAC on an idle host — dataset sizes on shared storage, which splits suit a preemptible allocation. Machine selection is site-specific and belongs in `CLAUDE.local.md`. Three ways a scheduler reports success and produces nothing — nodes that cannot write, preemption without `--requeue`, stdout that never lands — and why only an expected-vs-actual shard check catches any of them. A fourth: a job running stale code because `git pull | tail` hid an aborted merge. The submit-count and CPU caps that shape array sizing. The cartpole's per-reset URDF write and the 15-45x NFS penalty it causes. |
| [glossary.md](docs/glossary.md) | a term in a spec is unfamiliar | ROA, invariant terminal set, non-normal closed loop, entry-cut, noise floor, noise preset, matched/unmatched uncertainty, internal vs external uncertainty, stochastic ROA, split, batch, mean-SE stopping rule, drift, half-open grid, horizon, `U_SAT`, terminal-state model, `terminated`/`truncated`, `check_env`. Labels cannot validate a success rule; bounded-time reach probability; interior fraction and its dependence on K. Saturation placement and why it, not matchedness, decides whether noise can rescue a failing state; the zero-gain result and its two measured causes; `alpha`/`beta` and why the alpha turn never arrives under an entry-cut rule. |
| [dm-control-cartpole.md](docs/dm-control-cartpole.md) | comparing our cartpole RL to dm_control's swingup, or tuning `Cost.SHAPED` | Verified upstream constants (gear 10, rail ±1.8, 1000 steps, no termination ever). The dense tolerance-product reward with its per-factor floors, the sparse hold-the-region variant, and the swingup hanging start. Point-by-point mapping to the physical regime: same force authority and obs encoding, opposite choices on termination, start distribution and success metric — and why the dataset factory forces ours. `Cost.SHAPED` provenance and where it diverges (span-normalised errors, no floors, goal/oob bonuses). |

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
