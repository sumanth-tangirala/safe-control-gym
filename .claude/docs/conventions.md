# Conventions

Load when writing code or a commit that lands in this repo.

## Style, as enforced

`.pre-commit-config.yaml` is the specification; this section is the summary.
A `PostToolUse` hook runs it on every file written, so violations come back
immediately rather than at commit time.

- **Single quotes.** `double-quote-string-fixer` rewrites `"x"` to `'x'`.
  Docstrings use `'''`, not `"""` — the whole codebase does.
- **Line length is not enforced.** `flake8` runs with `--ignore=E501` and
  `autopep8` with `--max-line-length=1000`. Wrap for readability, not for a
  linter. `isort` uses `--line-length=110`.
- **autopep8 in place**, with `tests/` and
  `safe_control_gym/math_and_models/transformations.py` on a looser rule set
  (`E501,E201,E241,E127` also ignored) because of their matrix literals.
- **`safe_control_gym/__init__.py`** is exempt from flake8 (registration imports
  that look unused).
- Also enforced: valid AST, yaml/toml parse, no merge-conflict markers, no
  `pdb`/`breakpoint` left behind, no added file over 10 MB, trailing whitespace,
  final newline, docstring first.

The 10 MB cap is deliberate — datasets belong under `DATA_ROOT`, not in git.
The one committed binary family is `invariant_sets/*.npz`, which are small.

## Docstrings

Module docstring first, `'''` delimited, one-line summary then a blank line then
detail. Upstream uses a Google-ish `Args:` block for functions with non-obvious
signatures; the fork's newer code mostly relies on a single explanatory sentence
plus the *reason*, e.g. `rollout_seed`'s "Purity is what lets a resumed run draw
exactly the noise an uninterrupted run would have drawn." Prefer the reason over
restating the signature.

## Comments

Write the why, not the what. The existing high-value comments are all
justifications of a non-obvious constant or choice:

```python
GRID_RESOLUTION = 0.04  # 158 x 315 = 49,770 states, matching the shipped datasets
```

If a number was measured, say what it was measured against.

## Commits

From the existing log: imperative mood, capitalised, no trailing period, no
`feat:`/`fix:` prefix. The subject states the effect, not the file touched.

```
Name split descriptions per split so they cannot clobber each other
Fix the pendulum grid overshooting its domain
Add the eval collection split: per-cell success probabilities
Gate invariant-terminal-set collection behind --invariant_terminal_sets (default off)
```

Group related file changes into one commit — `Commit the eval dataset as a
group, npz last` is the pattern for ordering within a group when readers may
observe a partial state.

## Design docs

Non-trivial work is specced before it is written.

- `docs/superpowers/specs/YYYY-MM-DD-<topic>.md` — the design. Sections used
  throughout: Goal, Motivation, Prior state, Decisions, Formats. Decisions carry
  their measured justification (comparison tables, rejected alternatives with the
  number that killed them).
- `plans/<topic>.md` — the implementation plan derived from a spec.

Read the newest of each before proposing changes to collection. They record
findings — the non-normality excursion gains, the float32-vs-int16 comparison,
the grid off-by-one — that are expensive to rediscover and easy to contradict by
accident.

When you make a comparable decision, record the alternative you rejected and the
measurement that rejected it. That is the house style for these documents.

## Scope discipline

Root scripts own collection policy; `safe_control_gym/` owns systems and
controllers. Adding a grid, split, stopping rule, or label definition to the
library is the wrong layer — see `.claude/docs/architecture.md`.

---

Related: [architecture.md](architecture.md) for the layering this page defends, [workflows.md](workflows.md) for running the lint it describes.
