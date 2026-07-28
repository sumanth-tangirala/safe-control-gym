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
