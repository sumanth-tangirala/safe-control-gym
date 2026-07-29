#!/usr/bin/env python3
'''PreToolUse(Bash): block the shell commands that quietly destroy work here.

Lint bypasses defeat the repo's only style gate; force-pushes and recursive
deletes under the shared data root are unrecoverable for everyone else too.
'''
import json
import re
import sys

DATA_ROOT = '/common/users/shared/pracsys/genMoPlan/data_trajectories'

# `[^|;&\n]*` deliberately excludes newlines: a real bypass is one command on one
# line, whereas a heredoc or multi-line script can mention `git commit` and a
# flag many lines apart and is not a bypass. Crossing newlines produced false
# denials on ordinary scripts.
RULES = [
    (re.compile(r'\bgit\s+(commit|push)\b[^|;&\n]*\s(--no-verify|-n)\b'),
     'pre-commit is this repo\'s only lint gate (isort/autopep8/flake8, see '
     '.pre-commit-config.yaml). Fix the reported failure instead of bypassing it.'),

    # (?![\w-]) so --force-with-lease, the safe form, is not caught by the --force prefix.
    (re.compile(r'\bgit\s+push\b[^|;&\n]*\s(--force|-f)(?![\w-])'),
     'Plain force-push discards commits with no recovery for anyone who already fetched. '
     'Use --force-with-lease, or say explicitly that you want the history overwritten.'),

    (re.compile(r'\brm\s+(-[a-zA-Z]*[rR][a-zA-Z]*\s+)+[^|;&\n]*' + re.escape(DATA_ROOT)),
     f'Recursive delete under the shared dataset root ({DATA_ROOT}). These datasets are '
     'hours of compute and are read by other people. Delete them yourself if you mean it.'),

    (re.compile(r'\brm\s+[^|;&]*\binvariant_sets/'),
     'invariant_sets/*.npz are the success-ellipsoid artifacts every generator loads at '
     'startup. Regenerate with compute_invariant_sets.py rather than deleting them.'),
]


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    command = (payload.get('tool_input') or {}).get('command') or ''
    for pattern, reason in RULES:
        if pattern.search(command):
            json.dump({'hookSpecificOutput': {'hookEventName': 'PreToolUse',
                                              'permissionDecision': 'deny',
                                              'permissionDecisionReason': reason}}, sys.stdout)
            return 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
