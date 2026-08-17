'''Make an env's info dict survive the trip to a subprocess.

SubprocVecEnv runs each env in its own process and ships every reset/step
result back over a pipe, which means pickling it. These envs put a CasADi
symbolic model into `info` at reset:

    info keys at reset: current_step, physical_parameters, symbolic_model,
                        u_reference, x_reference

and CasADi MX objects refuse to pickle outside a casadi context:

    Exception: Cannot pickle MX objects without a casadi context.

So every worker died on startup, surfacing as EOFError in the parent and then
BrokenPipeError from the close() in the finally block -- neither of which names
the actual cause.

Dropping the key is safe for training and evaluation. The symbolic model is
consumed through the `env.symbolic` ATTRIBUTE -- that is what lqr, ilqr, mpc and
gp_mpc read -- and nothing reads it out of `info`. The attribute is untouched;
only the copy in the info dict goes.
'''
import gymnasium as gym

# Keys whose values cannot cross a process boundary. Narrow on purpose: dropping
# anything picklable would hide information for no reason, so this names the one
# offender rather than filtering by trial pickling.
UNPICKLABLE_INFO_KEYS = ('symbolic_model',)


class PicklableInfo(gym.Wrapper):
    '''Strip info entries that cannot be pickled, leaving everything else.'''

    def _clean(self, info):
        if not any(key in info for key in UNPICKLABLE_INFO_KEYS):
            return info
        return {k: v for k, v in info.items() if k not in UNPICKLABLE_INFO_KEYS}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._clean(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._clean(info)
