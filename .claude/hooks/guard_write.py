#!/usr/bin/env python3
'''PreToolUse(Write|Edit): keep edits out of shared datasets and computed artifacts.

Both are *produced* by scripts in this repo and consumed by other people's runs.
Hand-editing them silently desynchronises the data from the code that made it.
'''
import json
import os
import sys

DATA_ROOT = '/common/users/shared/pracsys/genMoPlan/data_trajectories'

DENY_PREFIX = [
    (DATA_ROOT,
     'Shared dataset root. Datasets are produced by generate_*_trajectories.py and read by '
     'other people. Change the generator and re-run it; never hand-edit a dataset.'),
]

DENY_MATCH = [
    ('invariant_sets', '.npz',
     'Computed artifact. Regenerate with `python compute_invariant_sets.py --systems <name>`.'),
]


def decide(path):
    path = os.path.abspath(path)
    for prefix, reason in DENY_PREFIX:
        if path == prefix or path.startswith(prefix + os.sep):
            return reason
    parent = os.path.basename(os.path.dirname(path))
    for directory, suffix, reason in DENY_MATCH:
        if parent == directory and path.endswith(suffix):
            return reason
    return None


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    path = (payload.get('tool_input') or {}).get('file_path')
    if not path:
        return 0

    reason = decide(path)
    if reason is None:
        return 0

    json.dump({'hookSpecificOutput': {'hookEventName': 'PreToolUse',
                                      'permissionDecision': 'deny',
                                      'permissionDecisionReason': reason}}, sys.stdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
