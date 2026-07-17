'''SB3-free inverted-pendulum SAC controller.

Wraps a trained SAC swing-up policy behind the safe-control-gym controller
interface without depending on stable-baselines3 or torch 2.x. The actor MLP
weights are loaded from a version-agnostic ``.npz`` (produced by
``scripts/extract_pendulum_rl_policies.py``) and the deterministic policy is
reproduced with a pure-NumPy forward pass::

    h = [cos theta, sin theta, theta_dot / theta_dot_max]
    for (W, b) in hidden layers: h = relu(W @ h + b)
    mean = W_mu @ h + b_mu
    action = clip(u_sat * tanh(mean), [-u_sat, u_sat])

The policy is re-queried every ``action_repeat`` calls and the action held in
between, matching the control cadence the policy was trained under. These are
the *standalone* swing-up controllers (no LQR handoff).
'''

import math
import os

import numpy as np

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.math_and_models.normalization import BaseNormalizer

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def _resolve_model_path(model_path):
    '''Resolve a full path or a bundled short name (e.g. ``v1_strong``).'''
    if model_path is None:
        raise ValueError('[ERROR] PendulumRL requires a model_path (path or bundled name, e.g. "v1_strong").')
    if os.path.isfile(model_path):
        return model_path
    bundled = os.path.join(MODELS_DIR, f'{model_path}.npz')
    if os.path.isfile(bundled):
        return bundled
    raise FileNotFoundError(f'[ERROR] PendulumRL model not found: {model_path!r} '
                            f'(also tried {bundled!r}).')


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
        self._action_repeat_override = action_repeat
        self._layers = None
        self._count = 0
        self._held = None
        if model_path is not None:
            self.load(model_path)

    def load(self, path):
        '''Load actor MLP weights + metadata from an extracted ``.npz``.'''
        data = np.load(_resolve_model_path(path), allow_pickle=False)
        n_hidden = int(data['n_hidden'])
        self._layers = [
            (data[f'hidden_{i}_weight'].astype(np.float32),
             data[f'hidden_{i}_bias'].astype(np.float32))
            for i in range(n_hidden)
        ]
        self._mu_w = data['mu_weight'].astype(np.float32)
        self._mu_b = data['mu_bias'].astype(np.float32)
        self._u_sat = float(data['u_sat'])
        self._theta_dot_max = float(data['theta_dot_max'])
        stored_repeat = int(data['action_repeat'])
        self._action_repeat = max(1, int(self._action_repeat_override
                                          if self._action_repeat_override is not None
                                          else stored_repeat))
        self.reset()

    def _policy_action(self, obs):
        '''Deterministic SAC forward: physical [theta, theta_dot] -> torque.'''
        theta, thetadot = float(obs[0]), float(obs[1])
        h = np.array([math.cos(theta), math.sin(theta), thetadot / self._theta_dot_max],
                     dtype=np.float32)
        for w, b in self._layers:
            h = np.maximum(0.0, w @ h + b)
        mean = self._mu_w @ h + self._mu_b
        u = self._u_sat * math.tanh(float(mean.reshape(-1)[0]))
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
        if self._layers is None:
            raise RuntimeError('[ERROR] PendulumRL has no policy loaded; pass model_path or call load().')
        if self._count % self._action_repeat == 0:
            self._held = self._policy_action(np.asarray(obs, dtype=np.float64))
        self._count += 1
        return np.array([self._held], dtype=np.float64)
