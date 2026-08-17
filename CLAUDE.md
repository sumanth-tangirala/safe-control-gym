# CLAUDE.md — wiki schema

This repo keeps an LLM-maintained wiki under `.claude/`, following
[karpathy/llm-wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
This file is the schema: it says how the wiki is structured and what to do when
ingesting, querying, or linting it. You write and maintain the wiki; the human
curates sources and asks the questions.

Keep this file small. If you want to add a paragraph here, it almost certainly
belongs on a wiki page with one line added to `.claude/INDEX.md`.

## What this repo is

A fork of `safe-control-gym` (learnsyslab, UofT DSL) used as a **trajectory
dataset factory**. Upstream provides the environments, controllers and safety
filters; this fork adds an inverted pendulum system and the `generate_*.py`
collectors at the repo root, which roll out closed loops over a state grid and
label each start state by whether it reaches the goal.

The consumer is a downstream flow-matching model that predicts **terminal
states**. That single fact drives most design decisions here — a success label
must be a function of the terminal state, which is why success is defined by
membership in an invariant ellipsoid rather than by a goal ball.
See `.claude/docs/datasets.md`.

## Three layers

**Raw sources** — immutable, read but never rewritten to make the wiki tidy:
the code under `safe_control_gym/` and the root scripts, the dated specs in
`docs/superpowers/specs/`, the plans in `plans/`, and the git log. The specs in
particular are historical records of decisions with their measured
justification. When the wiki and a source disagree, the source wins and the
wiki is wrong.

**The wiki** — `.claude/docs/*.md`, plus `.claude/INDEX.md` and
`.claude/log.md`. Entirely yours. Derived knowledge: what the code means, why a
constant has the value it does, what to do and in what order. It exists so that
the next session does not re-derive from scratch what this one already worked
out.

**The schema** — this file.

`CLAUDE.local.md` sits outside all three: machine- and account-specific facts
for one checkout, gitignored. Anything true only of one machine — which conda
env, which cluster, which remote is writable — goes there, never onto a wiki
page. Wiki pages must stay correct on a checkout where that file is absent.

## Operations

### Ingest

Triggered when a new spec or plan lands, a meaningful commit series lands, or
the human explains something that is not written down anywhere.

1. Read the source.
2. Say what you took from it and what it changes, before writing.
3. Update every affected page — a single spec here typically touches
   `datasets.md` and `glossary.md`, sometimes `workflows.md`. Do not append to
   one page and leave the contradiction on another.
4. Where a source overturns an existing claim, replace the claim; do not stack
   a newer paragraph on top of a stale one.
5. Update `INDEX.md` if the page set or a page's one-line summary changed.
6. Append an entry to `log.md`.
7. Run `python3 .claude/wiki_lint.py`.

Carry the *measurement* across, not just the conclusion. "float32, because
int16's 9.6e-5 step quantises the slowest 1% of motion into noise" survives
review; "we use float32" does not.

### Query

Answering any question about this repo.

1. Read `INDEX.md` first, then open only the pages it points at. Do not preload
   the doc set — the index exists so that you don't.
2. Verify anything load-bearing against the source layer before asserting it.
   Wiki pages are derived and can be stale; `wiki_lint.py` catches drifted
   constants but not drifted prose.
3. If the answer took real work and will be wanted again — a comparison, a
   trace through the collection path, a benchmark — file it as a new page under
   `.claude/docs/`, catalog it in `INDEX.md`, and log it. Otherwise it dies in
   the transcript.

### Lint

Ask for this periodically, and do it unprompted after a large change.

Mechanical half — `python3 .claude/wiki_lint.py` — checks that every constant
the wiki quotes still matches the source, that `INDEX.md` catalogs exactly the
pages that exist, that no page is an orphan, and that `log.md` stays parseable.

Judgement half, which no script can do:

- Claims that are still literally true but no longer the way things are done.
- Two pages that disagree.
- A concept referenced repeatedly with no page of its own.
- Cross-references that should exist and don't.
- Gaps worth filling — something the code does that no page explains.

Report what you find before fixing it.

## Wiki conventions

- One page per subject, named for the subject. Six exist; prefer growing one
  over adding a seventh unless the subject is genuinely separate.
- Every page opens with a one-line "load when…" so the index entry and the page
  agree.
- Every page ends with a `Related` line linking its neighbours. Orphans are a
  lint failure.
- Link with repo-relative paths (`.claude/docs/datasets.md`) — clickable in the
  terminal and in Obsidian.
- `INDEX.md` is content-oriented: every page, its link, one line on what is in
  it. Updated on every ingest.
- `log.md` is chronological, newest first, `## [YYYY-MM-DD] <op> | <subject>`.
  The prefix is load-bearing; `grep '^## \[' .claude/log.md | head -5` is how
  you find out what happened recently.

## Resident invariants

Hold regardless of the task.

1. **Datasets are outputs, never inputs to edit.** Everything under
   `/common/users/shared/pracsys/genMoPlan/data_trajectories` is shared and
   costs hours of compute. Change the generator and re-run it. A hook denies
   writes there.
2. **`invariant_sets/*.npz` are computed artifacts.** Regenerate with
   `python compute_invariant_sets.py --systems <name>`; never hand-edit.
3. **Reproducibility is load-bearing.** Every rollout's noise derives from
   `rollout_seed(base_seed, split_id, index, batch)`, a pure function of its
   coordinates. Do not introduce a stateful RNG into a rollout path — a resumed
   run must draw exactly what an uninterrupted run would have drawn.
4. **`pre-commit` is the only style gate.** It runs on every file you write.
   Fix what it reports; a hook denies `--no-verify`.
5. **Collection runs are hours long.** Never launch one in the foreground of a
   turn. Background it and poll, or hand the human the command.
6. **Design before code for anything non-trivial.** Specs go in
   `docs/superpowers/specs/`, plans in `plans/`, and both are ingested here
   afterwards.

## Syscalls

Hooks in `.claude/settings.json`, scripts in `.claude/hooks/`:

- `SessionStart` → `session_start.sh`: branch, dirty count, newest plan/spec.
- `PreToolUse(Write|Edit)` → `guard_write.py`: denies dataset and artifact writes.
- `PreToolUse(Bash)` → `guard_bash.py`: denies `--no-verify`, bare force-push,
  recursive deletes under the data root.
- `PostToolUse(Write|Edit)` → `format_python.sh`: runs `pre-commit` on the file,
  feeds back only what the fixers could not resolve.

## Deviations from the pattern

Recorded so they read as choices rather than omissions.

- **Raw sources are the repo, not a curated document collection.** There is no
  `raw/` directory and nothing gets clipped into one. Ingest is driven by specs,
  plans and commits landing.
- **The index is ordered by task, not by entity.** A code repo is queried as
  "I am about to do X", not "tell me about entity Y".
- **No search tooling.** Six pages; `INDEX.md` plus grep is sufficient. Revisit
  past roughly twenty.
- **`wiki_lint.py` exists** because a code wiki has a failure mode a prose wiki
  does not: a page asserting a constant the code no longer has. That check is
  mechanical, so it is a script rather than a judgement call.
