'''The dataset slice must reproduce bit-exactly across the migration.

rollout_seed is a pure function of (base_seed, split_id, index, batch), so a
fixed grid regenerates identically. Both the LQR and the RL controller paths
are covered: nothing is exempt from bit-exactness.
'''
import json
import os
import subprocess
import sys

import numpy as np
import pytest

FIX = os.path.join(os.path.dirname(__file__), 'fixtures')
REPO = os.path.join(os.path.dirname(__file__), '..', '..')


@pytest.mark.parametrize('controller', ['lqr', 'v3_strong'])
def test_slice_reproduces(controller, tmp_path):
    with open(os.path.join(FIX, f'dataset_slice_{controller}.json')) as handle:
        golden = json.load(handle)

    out = tmp_path / controller
    result = subprocess.run(
        [sys.executable, 'generate_inverted_pendulum_trajectories.py',
         '--controller', controller, '--seed', '42', '--resolution', '0.5',
         '--output_dir', str(out)],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]

    starts, labels = [], []
    with open(out / 'roa_labels.txt') as handle:
        for line in handle:
            parts = [float(p) for p in line.strip().split(',') if p]
            starts.append(parts[:2])
            labels.append(int(parts[-1]))

    assert labels == golden['labels']
    np.testing.assert_allclose(starts, golden['starts'], atol=1e-12)

    for i, expected in enumerate(golden['terminal_states']):
        rows = np.loadtxt(out / 'trajectories' / f'sequence_{i}.txt',
                          delimiter=',', ndmin=2)
        np.testing.assert_allclose(rows[-1], expected, atol=1e-12)
