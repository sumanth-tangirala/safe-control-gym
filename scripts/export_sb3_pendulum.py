'''Export a stable_baselines3 SAC ``.zip`` to the native ``pendulum_rl`` ``.pt``.

This is the bridge ``safe_control_gym/experiments/train_sb3.py`` leaves open: SB3
saves a training artifact (actor + critics + optimizers); ``pendulum_rl`` loads an
inference artifact (actor weights + run constants). This script converts one to
the other, following the same shape as ``scripts/convert_pendulum_models_to_pt.py``
(the ``.npz``-handoff precedent), but reading directly from an SB3 ``.zip``:

    {
        'actor_state_dict': <PendulumActor.state_dict()>,
        'obs_dim', 'act_dim', 'hidden_dims', 'activation',
        'u_sat', 'theta_dot_max', 'action_repeat',
    }

Like ``train_sb3.py``, this script is permitted to import stable_baselines3;
``envs/`` and ``controllers/`` (including ``pendulum_rl``) stay SB3-free.

``theta_dot_max`` and ``action_repeat`` are not stored in the SB3 zip -- they are
properties of the ``AngleObservation``/``ActionRepeat`` wrappers the policy was
*trained* under (see ``safe_control_gym/envs/env_wrappers/shaping.py``), so they
must be supplied (or left at the shipped convention's defaults: ``theta_dot_max
= 2*pi``, ``action_repeat = 4``). ``u_sat`` is read off the SB3 model's action
space bounds unless overridden, which is correct as long as training used
``normalized_rl_action_space: False`` (the repo default), so the action space
bounds *are* the physical torque bounds and SB3's squashed output already equals
``u_sat * tanh(mu(...))``.

Every export is validated against the source model before being trusted: the
reloaded ``.pt`` actor must reproduce ``model.predict(obs, deterministic=True)``
to within ``1e-5`` (the tolerance the original ``.npz`` port established, see
``scripts/extract_pendulum_rl_policies.py``). A mismatch means the exported
weights do not reproduce the trained policy and the export is refused.

Provenance is written to a ``manifest.json`` alongside the output ``.pt``
(created if absent, extended -- keyed by variant name -- if present): git SHA,
the source zip's absolute path, the checkpoint step (inferred from the SB3
``CheckpointCallback`` filename convention ``..._<N>_steps.zip``, else from the
loaded model's ``num_timesteps``), and the SB3/torch versions. This replaces the
old external ``source_zip`` scheme (an absolute path on the source system that
produced the shipped ``v1..v4`` models) now that the ``.zip`` is produced by
this repo's own ``train_sb3.py``: the SHA of the exporting commit is the
provenance that actually resolves on this checkout.

Run:
    python scripts/export_sb3_pendulum.py <model.zip> <out.pt> \\
        [--action_repeat 4] [--theta_dot_max 6.283185307179586] [--u_sat U]

Refuses to target a shipped model name (``v1..v4_{strong,weak}.pt`` under
``safe_control_gym/controllers/pendulum_rl/models/``) -- those are committed
artifacts from the original external training; export to a new name or a
scratch path instead.
'''

import argparse
import datetime
import json
import math
import os
import re
import subprocess

import numpy as np
import torch

from safe_control_gym.controllers.pendulum_rl.pendulum_rl import PendulumActor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_MODELS_DIR = os.path.join(
    REPO_ROOT, 'safe_control_gym/controllers/pendulum_rl/models')
SHIPPED_NAME_RE = re.compile(r'^v[1-4]_(strong|weak)\.pt$')

DEFAULT_THETA_DOT_MAX = 2 * math.pi
DEFAULT_ACTION_REPEAT = 4
FWD_TOL = 1e-5


def _git_sha():
    '''HEAD SHA of the exporting checkout, or ``None`` outside a git repo.'''
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _infer_checkpoint_step(zip_path, model):
    '''``CheckpointCallback`` names checkpoints ``..._<N>_steps.zip``; fall back
    to the loaded model's own step counter (e.g. for ``model_final.zip``, which
    has no step in its name but still knows how long it trained).'''
    m = re.search(r'_(\d+)_steps(?:\.zip)?$', os.path.basename(zip_path))
    if m:
        return int(m.group(1))
    step = getattr(model, 'num_timesteps', None)
    return int(step) if step else None


def _extract_actor_weights(model):
    '''Pull the deterministic SAC actor's ``latent_pi``/``mu`` Linear layers.'''
    sd = model.policy.actor.state_dict()
    layer_idx = sorted({int(k.split('.')[1]) for k in sd if k.startswith('latent_pi.')})
    if not layer_idx:
        raise ValueError('no latent_pi.* layers found; is this an SAC actor?')
    weights, hidden_dims, obs_dim = [], [], None
    for i, li in enumerate(layer_idx):
        w, b = sd[f'latent_pi.{li}.weight'], sd[f'latent_pi.{li}.bias']
        weights.append((w, b))
        hidden_dims.append(int(w.shape[0]))
        if i == 0:
            obs_dim = int(w.shape[1])
    mu_w, mu_b = sd['mu.weight'], sd['mu.bias']
    act_dim = int(mu_w.shape[0])
    return weights, (mu_w, mu_b), obs_dim, act_dim, hidden_dims


def _action_space_u_sat(model):
    '''Infer ``u_sat`` from the SB3 action space, requiring symmetric bounds.'''
    high = np.asarray(model.action_space.high, dtype=np.float64).reshape(-1)
    low = np.asarray(model.action_space.low, dtype=np.float64).reshape(-1)
    if high.size != 1:
        raise ValueError(f'expected a scalar action, got shape {high.shape}')
    if not np.allclose(high, -low):
        raise ValueError(
            f'action space is not symmetric (low={low}, high={high}); '
            'PendulumActor assumes u_sat * tanh(...) with symmetric bounds. '
            'Pass --u_sat explicitly if this is expected (e.g. a normalized '
            'action space), or retrain with normalized_rl_action_space: False.')
    return float(high[0])


def _refuse_shipped_target(out_path):
    resolved = os.path.abspath(out_path)
    in_shipped_dir = os.path.dirname(resolved) == os.path.abspath(SHIPPED_MODELS_DIR)
    if in_shipped_dir and SHIPPED_NAME_RE.match(os.path.basename(resolved)):
        raise SystemExit(
            f'[ERROR] refusing to overwrite shipped model {resolved!r}; it is a '
            'committed artifact from the original external training. Export to a '
            'new name or a scratch path instead.')


def export(zip_path, out_path, action_repeat=DEFAULT_ACTION_REPEAT,
           theta_dot_max=DEFAULT_THETA_DOT_MAX, u_sat=None):
    '''Convert one SB3 ``.zip`` to a native ``pendulum_rl`` ``.pt``.

    Returns ``(out_path, forward_max_err)``. Raises if the reloaded actor does
    not reproduce ``model.predict(deterministic=True)`` within ``FWD_TOL``.
    '''
    _refuse_shipped_target(out_path)
    from stable_baselines3 import SAC
    from stable_baselines3 import __version__ as sb3_version

    model = SAC.load(zip_path, device='cpu')
    weights, (mu_w, mu_b), obs_dim, act_dim, hidden_dims = _extract_actor_weights(model)
    if u_sat is None:
        u_sat = _action_space_u_sat(model)

    actor = PendulumActor(obs_dim, act_dim, hidden_dims, u_sat)
    with torch.no_grad():
        for i, (w, b) in enumerate(weights):
            actor.net.fcs[i].weight.copy_(w.to(torch.float32))
            actor.net.fcs[i].bias.copy_(b.to(torch.float32))
        actor.mu_layer.weight.copy_(mu_w.to(torch.float32))
        actor.mu_layer.bias.copy_(mu_b.to(torch.float32))
    actor.eval()

    ckpt = {
        'actor_state_dict': actor.state_dict(),
        'obs_dim': obs_dim, 'act_dim': act_dim, 'hidden_dims': hidden_dims,
        'activation': 'relu',
        'u_sat': u_sat, 'theta_dot_max': float(theta_dot_max),
        'action_repeat': int(action_repeat),
    }
    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(ckpt, out_path)

    # Validate: reload and compare the actor forward to SB3's own predict().
    reloaded = PendulumActor(obs_dim, act_dim, hidden_dims, u_sat)
    reloaded.load_state_dict(torch.load(out_path, map_location='cpu')['actor_state_dict'])
    reloaded.eval()
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        obs = rng.uniform(-1.0, 1.0, size=obs_dim).astype(np.float32)
        sb3_action, _ = model.predict(obs, deterministic=True)
        with torch.no_grad():
            got = float(reloaded(torch.as_tensor(obs, dtype=torch.float32)).item())
        max_err = max(max_err, abs(got - float(np.asarray(sb3_action).reshape(-1)[0])))
    if max_err > FWD_TOL:
        raise AssertionError(
            f'{os.path.basename(out_path)}: forward mismatch {max_err:.2e} exceeds '
            f'{FWD_TOL:.0e} -- the exported weights do not reproduce the trained '
            'policy. Not writing manifest provenance for a broken export.')

    info = {
        'variant': os.path.splitext(os.path.basename(out_path))[0],
        'source': 'export_sb3_pendulum.py',
        'git_sha': _git_sha(),
        'source_zip': os.path.abspath(zip_path),
        'checkpoint_step': _infer_checkpoint_step(zip_path, model),
        'sb3_version': sb3_version,
        'torch_version': torch.__version__,
        'net_arch': hidden_dims,
        'obs_dim': obs_dim, 'act_dim': act_dim,
        'u_sat': u_sat, 'theta_dot_max': float(theta_dot_max),
        'action_repeat': int(action_repeat),
        'forward_max_err': max_err,
        'exported_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _update_manifest(out_dir or '.', info)
    return out_path, max_err


def _update_manifest(models_dir, info):
    '''Create or extend ``manifest.json`` alongside the exported ``.pt``.'''
    manifest_path = os.path.join(models_dir, 'manifest.json')
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            'u_sat': info['u_sat'], 'theta_dot_max': info['theta_dot_max'],
            'action_repeat': info['action_repeat'],
            'obs_transform': '[cos th, sin th, thdot/thdot_max]',
            'models': [],
        }
    manifest.setdefault('models', [])
    # Keyed by variant: re-exporting the same output path replaces its entry
    # rather than accumulating stale duplicates.
    manifest['models'] = [m for m in manifest['models'] if m.get('variant') != info['variant']]
    manifest['models'].append(info)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(
        description='Export a stable_baselines3 SAC .zip to a native pendulum_rl .pt.')
    parser.add_argument('zip_path', help='Source SB3 .zip (e.g. model_final.zip or a checkpoint).')
    parser.add_argument('out_path', help='Output .pt path. Must not be a shipped model name.')
    parser.add_argument('--action_repeat', type=int, default=DEFAULT_ACTION_REPEAT,
                        help=f'Control cadence baked into the .pt (default {DEFAULT_ACTION_REPEAT}).')
    parser.add_argument('--theta_dot_max', type=float, default=DEFAULT_THETA_DOT_MAX,
                        help='AngleObservation rate_max used at train time '
                        f'(default {DEFAULT_THETA_DOT_MAX!r}, i.e. 2*pi).')
    parser.add_argument('--u_sat', type=float, default=None,
                        help='Control saturation. Default: inferred from the SB3 '
                        'action space bounds (requires normalized_rl_action_space: False).')
    args = parser.parse_args()

    out_path, max_err = export(args.zip_path, args.out_path,
                               action_repeat=args.action_repeat,
                               theta_dot_max=args.theta_dot_max,
                               u_sat=args.u_sat)
    print(f'[ok] wrote {out_path}  (forward err vs SB3 predict: {max_err:.2e})')


if __name__ == '__main__':
    main()
