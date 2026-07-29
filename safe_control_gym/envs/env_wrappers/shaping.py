'''Optional, config-selected observation and cadence shaping.

Nothing here is applied by default. These exist because particular systems were
trained with particular conventions -- the pendulum's policies consume
[cos theta, sin theta, theta_dot / theta_dot_max] at an action_repeat of 4 --
and baking those into the trainer would silently mis-train every other system.
'''
import gymnasium as gym
import numpy as np


class AngleObservation(gym.ObservationWrapper):
    '''Re-encode one angular coordinate as (cos, sin) and scale its rate.'''

    def __init__(self, env, angle_index, rate_index, rate_max):
        super().__init__(env)
        self.angle_index = int(angle_index)
        self.rate_index = int(rate_index)
        self.rate_max = float(rate_max)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float64),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            dtype=np.float64)

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        angle = obs[self.angle_index]
        return np.array([np.cos(angle), np.sin(angle),
                         obs[self.rate_index] / self.rate_max],
                        dtype=np.float64)


class ActionRepeat(gym.Wrapper):
    '''Hold each action for `repeat` control steps, as the policy was trained.'''

    def __init__(self, env, repeat):
        super().__init__(env)
        self.repeat = max(1, int(repeat))

    def step(self, action):
        total = 0.0
        terminated = truncated = False
        obs = info = None
        for _ in range(self.repeat):
            obs, rew, terminated, truncated, info = self.env.step(action)
            total += rew
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info
