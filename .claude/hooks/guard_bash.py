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

    (re.compile(r'--output[_-]dir[= ]\s*[\'"]?' + re.escape(DATA_ROOT)),
     f'That writes directly into the shared dataset root ({DATA_ROOT}). Those datasets '
     'are hours of compute and are read by other people. Collect to a scratch directory '
     'and have someone move it deliberately.'),
]

# Six generator/visualisation scripts DEFAULT --output_dir to the shared data root
# (e.g. generate_quadrotor_2d_trajectories_rl.py -> .../quadrotor2D_rl). Omitting the
# flag therefore overwrites shipped datasets, and guard_write.py cannot see it because
# a script invoked through Bash never touches the Write/Edit tools.
# Must be an actual invocation -- `python3 generate_x.py ...` -- not merely a mention.
# Matching the bare filename also fired on `grep generate_x.py` and `sed -n ... file`,
# which read the script rather than run it.
GENERATOR = re.compile(r'\bpython[\d.]*\s+(?:-\w+\s+)*[\w./-]*generate_\w*trajector\w*\.py\b')
OUTPUT_DIR = re.compile(r'--output[_-]dir\b')


def check_generator_output(command):
    '''Require an explicit --output_dir on any trajectory generator.'''
    if not GENERATOR.search(command) or OUTPUT_DIR.search(command):
        return None
    return ('This runs a trajectory generator without --output_dir. Several of them '
            f'default to the shared dataset root ({DATA_ROOT}), so omitting it can '
            'overwrite shipped datasets. Pass an explicit --output_dir under a scratch '
            'directory.')


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    command = (payload.get('tool_input') or {}).get('command') or ''

    reasons = [reason for pattern, reason in RULES if pattern.search(command)]
    missing_output_dir = check_generator_output(command)
    if missing_output_dir:
        reasons.append(missing_output_dir)

    if reasons:
        json.dump({'hookSpecificOutput': {'hookEventName': 'PreToolUse',
                                          'permissionDecision': 'deny',
                                          'permissionDecisionReason': reasons[0]}}, sys.stdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
