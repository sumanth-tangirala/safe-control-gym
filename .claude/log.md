# Log

Append-only, newest first. One entry per ingest, per filed query answer, per
lint pass. Keep the `## [YYYY-MM-DD] <op> | <subject>` prefix exactly — it is
what makes the log greppable:

```bash
grep '^## \[' .claude/log.md | head -5      # what happened recently
grep '^## \[' .claude/log.md | grep ingest  # what has been read into the wiki
```

`<op>` is one of `ingest`, `query`, `lint`.

---

## [2026-07-29] ingest | gymnasium/SB3 migration lands in architecture, workflows, glossary, compute

The prior entry below described the migration's code but only actually filed
the quad2d ellipsoid finding into `docs/datasets.md`; `architecture.md`,
`workflows.md` and `glossary.md` still described the pre-migration 4-tuple
world with zero mentions of gymnasium, SB3 or `terminated`/`truncated` (the
plan's own "Follow-on work" section says as much). This entry does that
ingest, from `docs/superpowers/specs/2026-07-28-sb3-gymnasium-migration-design.md`,
`plans/sb3-gymnasium-migration.md`, `git log --oneline main..HEAD` (31 commits),
and the landed code.

`architecture.md`: added the Gymnasium 5-tuple / `terminated`+`truncated`
section (two termination sources, one truncation source, `info['TimeLimit.truncated']`
kept for six controllers), a "Two RL stacks coexist" section (native
controllers unchanged; SB3 confined to `train_sb3.py`; no exporter yet), and
package-map rows for `train_sb3.py`, `forwarding.py`, `shaping.py`. Also fixed
a claim that predates this migration and that the new addition would have
made worse to leave: "import-only; no CLI entry points" was already false —
`train_rl_controller.py` has had a `__main__` block all along.

`workflows.md`: corrected the pendulum test bar to **75 passed, 0 known
failures** (verified directly) — the old "74 + 1 known failure" was a broken
editable install (`safe_control_gym.pth` pointed at a deleted sibling clone),
fixed by `pip install -e .`, not a real failure. Documented the new
`tests/test_envs/` oracle directory (golden rollouts, dataset-slice at
`atol=1e-12`, truncation semantics, wrapper forwarding, `check_env`
conformance, episode-flag initialisation, SB3 training smoke) and why
`test_invariant_sets.py` is deliberately absent from it. Added the
`train_sb3.py` command with the corrected GPU measurement.

`glossary.md`: added `terminated`/`truncated` and `check_env`.

`compute.md`: replaced the unmeasured "RL training wants a GPU" framing with
the measured number — SB3 SAC on the pendulum, `net_arch [256, 256]`, idle
ilab2, threads pinned: **cpu 65.6 steps/s vs cuda 111.0 steps/s, GPU 1.69x
faster** — and the correction of the earlier wrong claim (asserted from
general principle, first re-check ran on a loaded host and measured
contention, not devices).

`datasets.md` was checked against the sources and found already consistent
(the quad2d finding and the read-only `invariant_sets/*.npz` note match the
current code and the absence of `tests/test_envs/test_invariant_sets.py`); no
edit needed there this time.

Found in passing, not part of this migration: an untracked
`scripts/export_sb3_pendulum.py` and `tests/test_envs/test_export_sb3_pendulum.py`
exist in the working tree but are not committed to any branch. The spec and
plan are explicit that no exporter is in scope and a trained SB3 policy has no
in-repo consumer yet; those uncommitted files look like separate, unlanded
work on that follow-on and were not treated as source for this ingest.

`INDEX.md` summaries updated for architecture.md, workflows.md, compute.md and
glossary.md to match.

## [2026-07-29] ingest | quad2d ellipsoid finding

Migrated all four envs to the Gymnasium 5-tuple and added task-agnostic SB3
training (`plans/sb3-gymnasium-migration.md`). Filed the quad2d invariant-set
finding into `docs/datasets.md`: its ellipsoid is not reproducible because a
ReLU policy puts a crease inside the finite-difference stencil, and that is
expected rather than a defect, because `validate()` is what establishes
invariance. Human ruled no investigation needed; recorded so it is not
re-derived.

## [2026-07-28] lint | split machine-specific facts out of the wiki

Moved the conda env path and the cluster/machine guidance to CLAUDE.local.md
(gitignored) ahead of opening a PR against the shared repo. Added two checks to
wiki_lint.py: committed pages may name no absolute path except DATA_ROOT, and
every page must carry a title, a "Load when" line and a Related line -- the
second added after a bad test edit truncated a page and every existing check
still passed. Fixed conventions.md and glossary.md, which had never matched the
"Load when" convention.

## [2026-07-28] lint | bootstrap pass

First run of `.claude/wiki_lint.py`. 6 pages, 8 constants verified against
`generate_inverted_pendulum_trajectories.py`. Clean.

## [2026-07-28] ingest | karpathy/llm-wiki gist

Read `gist.github.com/karpathy/442a6bf555914893e9891c11519de94f`. Restructured
the wiki to the pattern it describes: `CLAUDE.md` became the schema (three
layers, three operations), `INDEX.md` became a content catalog rather than a
static repo map, added this log and `wiki_lint.py`.

Deviations recorded in `CLAUDE.md` under "Deviations from the pattern": raw
sources here are the repo itself rather than a curated document collection, and
the index is hand-ordered by task rather than by entity.

## [2026-07-28] ingest | plans/invariant-terminal-sets-recollection.md

Non-normality excursion gains (pendulum 2.6, quad2D 3.2, quad3D 4.9, cartpole
5.4), the discrete-Lyapunov ellipsoid construction, and the fixed-horizon
rationale filed into `docs/datasets.md` and `docs/glossary.md`.

## [2026-07-28] ingest | docs/superpowers/specs/2026-07-25-noisy-pendulum-collection-design.md

Train/eval split semantics, the mean-SE stopping rule, the entry-cut success
rule, the float32-vs-int16 measurement, the half-open grid fix, and the
atomic-publication scheme filed into `docs/datasets.md`. Terms filed into
`docs/glossary.md`.

## [2026-07-28] ingest | repository bootstrap

Initial pass over `safe_control_gym/`, the root generators,
`.pre-commit-config.yaml`, `tests/`, and the git log. Produced
`docs/architecture.md`, `docs/workflows.md`, `docs/conventions.md`,
`docs/compute.md`.
