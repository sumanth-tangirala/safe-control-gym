#!/usr/bin/env bash
# PostToolUse(Write|Edit): run the repo's own pre-commit hooks on the file just written.
#
# Pass 1 lets the fixers (isort, autopep8, trailing-whitespace, double-quote-string-fixer)
# rewrite the file in place. Pass 2 reports only what the fixers could not resolve —
# real flake8 errors — back to the model with exit 2.
set -uo pipefail

payload=$(cat)

file=$(printf '%s' "$payload" | python3 -c '
import json, sys
d = json.load(sys.stdin)
resp = d.get("tool_response") or {}
inp = d.get("tool_input") or {}
print(resp.get("filePath") or inp.get("file_path") or "")
' 2>/dev/null) || exit 0

[[ -n $file && -f $file ]] || exit 0
case $file in
    *.py | *.yaml | *.yml | *.toml) ;;
    *) exit 0 ;;
esac

command -v pre-commit >/dev/null 2>&1 || exit 0

root=${CLAUDE_PROJECT_DIR:-$(git -C "$(dirname "$file")" rev-parse --show-toplevel 2>/dev/null)}
[[ -n $root && -d $root/.git ]] || exit 0
cd "$root" || exit 0

# Only files tracked by (or inside) this repo — pre-commit cannot see anything else.
[[ $file == "$root"/* ]] || exit 0

# Already clean is the common case — one pass, no second invocation.
pre-commit run --files "$file" >/dev/null 2>&1 && exit 0
out=$(pre-commit run --files "$file" 2>&1) && exit 0

{
    echo "pre-commit still fails on ${file#"$root"/} after auto-fix:"
    printf '%s\n' "$out" | grep -Ev '^\[INFO\]|^- hook id:|^- exit code:' | tail -40
} >&2
exit 2
