'''Inverted-pendulum SAC controller, in the repo's native PyTorch style.

Like safe-control-gym's own ``sac``/``ppo`` controllers, the policy is a
hand-written ``torch.nn.Module`` (built on the repo's ``MLP`` block) and
inference runs under ``torch.no_grad()`` -- no stable-baselines3 dependency. The
actor reproduces the trained SB3 SAC actor's *deterministic* forward:

    net_out = relu(W2 @ relu(W1 @ [cos t, sin t, tdot/tdot_max] + b1) + b2)
    action  = u_sat * tanh(mu(net_out))          # squash + symmetric unscale

Note the actor differs from the repo's own ``MLPActor`` by one activation: SB3's
``latent_pi`` applies ``relu`` *after* its last shared layer (before ``mu``),
which we reproduce via ``MLP(..., output_act='relu')``. Weights are loaded from
the version-agnostic ``.npz`` produced by
``scripts/extract_pendulum_rl_policies.py``. The policy is re-queried every
``action_repeat`` calls (the trained control cadence). These are the *standalone*
swing-up controllers (no LQR handoff).
'''

import math
import os

import numpy as np
import torch
import torch.nn as nn

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.math_and_models.neural_networks import MLP
from safe_control_gym.math_and_models.normalization import BaseNormalizer

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def _resolve_model_path(model_path):
    '''Resolve a full path or a bundled short name (e.g. ``v1_strong``).'''
    if model_path is None:
        raise ValueError('[ERROR] PendulumRL requires a model_path (path or bundled name, e.g. "v1_strong").')
    if os.path.isfile(model_path):
        return model_path
    bundled = os.path.join(MODELS_DIR, f'{model_path}.pt')
    if os.path.isfile(bundled):
        return bundled
    raise FileNotFoundError(f'[ERROR] PendulumRL model not found: {model_path!r} '
                            f'(also tried {bundled!r}).')


class PendulumActor(nn.Module):
    '''Deterministic SAC actor: obs -> ``u_sat * tanh(mu(latent_pi(obs)))``.

    Uses the repo's ``MLP`` for the shared body, with ``output_act`` set so a
    ``relu`` follows the last shared layer (matching the trained SB3 policy).
    '''

    def __init__(self, obs_dim, act_dim, hidden_dims, u_sat, activation='relu'):
        super().__init__()
        self.net = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1],
                       act=activation, output_act=activation)
        self.mu_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.register_buffer('u_sat', torch.tensor(float(u_sat)))

    def forward(self, obs):
        return self.u_sat * torch.tanh(self.mu_layer(self.net(obs)))


class PendulumRL(BaseController):
    '''Standalone trained SAC swing-up policy as a state-feedback controller.'''

    def __init__(self, env_func, model_path=None, action_repeat=None, **kwargs):
        '''Creates the task env and loads the policy weights.

        Args:
            env_func (Callable): Function to instantiate the inverted pendulum env.
            model_path (str): Bundled name (``v1_strong`` ... ``v4_weak``) or a
                path to an extracted ``.npz`` policy.
            action_repeat (int, optional): Overrides the policy's stored
                action-repeat (default: use the value baked into the ``.npz``).
        '''
        super().__init__(env_func, **kwargs)
        self.env = env_func()
        # Identity normalizer -- the obs transform lives in select_action; this
        # only satisfies the trajectory scripts' ``obs_normalizer`` contract.
        self.obs_normalizer = BaseNormalizer()
        self.actor = None
        self._action_repeat_override = action_repeat
        self._count = 0
        self._held = None
        if model_path is not None:
            self.load(model_path)

    def load(self, path):
        '''Build the torch actor and load its state-dict + metadata from ``.pt``.'''
        ckpt = torch.load(_resolve_model_path(path), map_location=self.device)
        self._u_sat = float(ckpt['u_sat'])
        self._theta_dot_max = float(ckpt['theta_dot_max'])

        self.actor = PendulumActor(int(ckpt['obs_dim']), int(ckpt['act_dim']),
                                   list(ckpt['hidden_dims']), self._u_sat,
                                   activation=ckpt.get('activation', 'relu'))
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.actor.to(self.device).eval()

        self._action_repeat = max(1, int(self._action_repeat_override
                                          if self._action_repeat_override is not None
                                          else ckpt['action_repeat']))
        self.reset()

    def _policy_action(self, obs):
        '''Deterministic torch forward: physical [theta, theta_dot] -> torque.'''
        theta, thetadot = float(obs[0]), float(obs[1])
        feat = torch.as_tensor(
            [math.cos(theta), math.sin(theta), thetadot / self._theta_dot_max],
            dtype=torch.float32, device=self.device)
        with torch.no_grad():
            u = float(self.actor(feat).item())
        return float(np.clip(u, -self._u_sat, self._u_sat))

    def reset(self):
        '''Clear the action-repeat latch (call between episodes).'''
        self._count = 0
        self._held = None

    def close(self):
        '''Cleans up resources.'''
        self.env.close()

    def select_action(self, obs, info=None):
        '''Return the (repeat-held) policy torque for the current observation.'''
        if self.actor is None:
            raise RuntimeError('[ERROR] PendulumRL has no policy loaded; pass model_path or call load().')
        if self._count % self._action_repeat == 0:
            self._held = self._policy_action(np.asarray(obs, dtype=np.float64))
        self._count += 1
        return np.array([self._held], dtype=np.float64)
