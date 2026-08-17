#!/usr/bin/env bash
# SessionStart: page in the small amount of state that CLAUDE.md cannot know statically —
# where the working tree is right now, and which design docs are still open.
set -uo pipefail

root=${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null)}
[[ -n $root && -d $root/.git ]] || exit 0
cd "$root" || exit 0

branch=$(git branch --show-current 2>/dev/null || echo '?')
dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
head=$(git log -1 --format='%h %s' 2>/dev/null)

lines=("branch ${branch}, ${dirty} uncommitted path(s)"
       "HEAD: ${head}")

if [[ ! -x .git/hooks/pre-commit ]]; then
    lines+=("pre-commit is NOT installed as a git hook here — run 'pre-commit install' before relying on commit-time linting.")
fi

plan=$(ls -t plans/*.md 2>/dev/null | head -1)
[[ -n $plan ]] && lines+=("newest plan: ${plan}")
spec=$(ls -t docs/superpowers/specs/*.md 2>/dev/null | head -1)
[[ -n $spec ]] && lines+=("newest spec: ${spec}")

# Wiki drift. Advisory: whether a source change needs a wiki edit is a
# judgement call, so this surfaces it rather than blocking on it. Correctness
# failures (a page quoting a constant the code no longer has) are separate and
# do fail the lint.
if [[ -x .claude/wiki_lint.py ]]; then
    while IFS= read -r line; do
        [[ -n $line ]] && lines+=("${line}")
    done < <(python3 .claude/wiki_lint.py 2>/dev/null | grep -E '^(wrong|wiki may be stale|    )')
fi

context=$(printf 'Repo state:\n'; printf -- '- %s\n' "${lines[@]}")

python3 -c '
import json, sys
ctx = sys.stdin.read()
json.dump({"hookSpecificOutput": {"hookEventName": "SessionStart",
                                  "additionalContext": ctx}}, sys.stdout)
' <<<"$context"
