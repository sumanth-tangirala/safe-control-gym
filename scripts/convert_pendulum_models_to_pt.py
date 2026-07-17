"""Convert the extracted pendulum policy weights (.npz) to native torch .pt.

Run in the *safe-control-gym* (``scg``) conda env, so the ``.pt`` is written by
the same torch that the controller loads it with (maximally portable). The
``.npz`` files are the version-agnostic handoff produced by
``scripts/extract_pendulum_rl_policies.py`` in the source system's env; this
step turns each into a native ``PendulumActor.state_dict()`` checkpoint:

    {
        'actor_state_dict': <PendulumActor.state_dict()>,
        'obs_dim', 'act_dim', 'hidden_dims', 'activation',
        'u_sat', 'theta_dot_max', 'action_repeat',
    }

Each conversion is validated: the reloaded ``.pt`` actor must match the raw
NumPy forward on random observations.

Run:  python scripts/convert_pendulum_models_to_pt.py
"""

import glob
import math
import os

import numpy as np
import torch

from safe_control_gym.controllers.pendulum_rl.pendulum_rl import PendulumActor

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'safe_control_gym/controllers/pendulum_rl/models')


def _numpy_forward(data, obs):
    h = np.asarray(obs, dtype=np.float32)
    for i in range(int(data['n_hidden'])):
        h = np.maximum(0.0, data[f'hidden_{i}_weight'].astype(np.float32) @ h
                       + data[f'hidden_{i}_bias'].astype(np.float32))
    mean = data['mu_weight'].astype(np.float32) @ h + data['mu_bias'].astype(np.float32)
    return float(data['u_sat']) * math.tanh(float(mean.reshape(-1)[0]))


def convert_one(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    n_hidden = int(data['n_hidden'])
    hidden_dims = [int(data[f'hidden_{i}_weight'].shape[0]) for i in range(n_hidden)]
    obs_dim = int(data['hidden_0_weight'].shape[1])
    act_dim = int(data['mu_weight'].shape[0])
    u_sat = float(data['u_sat'])

    actor = PendulumActor(obs_dim, act_dim, hidden_dims, u_sat)
    with torch.no_grad():
        for i in range(n_hidden):
            actor.net.fcs[i].weight.copy_(torch.as_tensor(data[f'hidden_{i}_weight'], dtype=torch.float32))
            actor.net.fcs[i].bias.copy_(torch.as_tensor(data[f'hidden_{i}_bias'], dtype=torch.float32))
        actor.mu_layer.weight.copy_(torch.as_tensor(data['mu_weight'], dtype=torch.float32))
        actor.mu_layer.bias.copy_(torch.as_tensor(data['mu_bias'], dtype=torch.float32))
    actor.eval()

    ckpt = {
        'actor_state_dict': actor.state_dict(),
        'obs_dim': obs_dim, 'act_dim': act_dim, 'hidden_dims': hidden_dims,
        'activation': str(data['activation']),
        'u_sat': u_sat, 'theta_dot_max': float(data['theta_dot_max']),
        'action_repeat': int(data['action_repeat']),
    }
    pt_path = npz_path[:-len('.npz')] + '.pt'
    torch.save(ckpt, pt_path)

    # Validate: reload and compare the actor forward to the raw NumPy forward.
    reloaded = PendulumActor(obs_dim, act_dim, hidden_dims, u_sat)
    reloaded.load_state_dict(torch.load(pt_path, map_location='cpu')['actor_state_dict'])
    reloaded.eval()
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        obs = rng.uniform(-1.0, 1.0, size=obs_dim).astype(np.float32)
        with torch.no_grad():
            got = float(reloaded(torch.as_tensor(obs, dtype=torch.float32)).item())
        max_err = max(max_err, abs(got - _numpy_forward(data, obs)))
    assert max_err <= 1e-5, f'{os.path.basename(pt_path)}: forward mismatch {max_err:.2e}'
    return pt_path, max_err


def main():
    npzs = sorted(glob.glob(os.path.join(MODELS_DIR, '*.npz')))
    if not npzs:
        raise SystemExit(f'no .npz models found in {MODELS_DIR} '
                         '(run scripts/extract_pendulum_rl_policies.py first)')
    for npz in npzs:
        pt, err = convert_one(npz)
        print(f'[ok] {os.path.basename(pt)}  (forward err {err:.2e})')
    print(f'\nWrote {len(npzs)} .pt files -> {MODELS_DIR}')


if __name__ == '__main__':
    main()
