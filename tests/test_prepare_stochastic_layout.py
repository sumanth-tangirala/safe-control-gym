import json

import numpy as np

from prepare_stochastic_layout import prepare, verify


def test_cartpole_layout_uses_declared_canonical_state_order(tmp_path):
    state_order = ['x', 'theta', 'x_dot', 'theta_dot']
    states = np.array(
        [[1.0, 2.0, 30.0, 4.0], [-1.0, -3.0, -40.0, -5.0]],
        dtype=np.float32,
    )
    np.savez(
        tmp_path / 'train.npz',
        states=states,
        offsets=np.array([0, 1, 2], dtype=np.int64),
        starts=states.astype(np.float64),
        labels=np.array([1, 0], dtype=np.uint8),
        seeds=np.array([1, 2], dtype=np.int64),
    )
    np.savez(
        tmp_path / 'eval_success_prob.npz',
        starts=states.astype(np.float64),
        successes=np.array([1, 0], dtype=np.int32),
        trials=np.array([1, 1], dtype=np.int32),
        p_success=np.array([1.0, 0.0]),
        n_batches=np.array(1, dtype=np.int64),
    )
    (tmp_path / 'train_description.json').write_text(
        json.dumps({'dataset_name': 'test', 'state_order': state_order})
    )
    (tmp_path / 'eval_description.json').write_text(
        json.dumps({'state_order': state_order})
    )

    assert prepare(str(tmp_path), n_cal=1) == (2, 2)
    assert verify(str(tmp_path), n_cal=1) == (0.5, 2, 2)

    description = json.loads((tmp_path / 'dataset_description.json').read_text())
    assert description['state_space']['state_order'] == state_order
    assert description['achieved_bounds']['theta']['min'] == -3.0
    assert description['achieved_bounds']['theta']['max'] == 2.0
    assert description['achieved_bounds']['x_dot']['min'] == -40.0
    assert description['achieved_bounds']['x_dot']['max'] == 30.0
