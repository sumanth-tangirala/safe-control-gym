'''The success rule, defined once.

Training-time validation and the acceptance bar must measure the same thing. If
a callback logs one notion of success while `eval_policy` accepts on another,
the curve you watch is not the curve you are judged on -- and nothing would ever
fail to warn you.

So both import from here. `eval_policy` re-exports these names for callers that
already reach for them there.

Why the rule is not `info['goal_reached']`: `_get_info` in cartpole.py and
quadrotor.py gates that key on ``COST == Cost.QUADRATIC``, and RL training uses
``rl_reward``, so for three of the four systems the key is simply absent. This
applies the envs' own test -- ``||state - X_GOAL|| < tolerance`` -- to the state
directly, which holds whatever the cost is.

Note it is a norm over the FULL state, velocities and rates included, not over
position. For the 3D quadrotor that is twelve dimensions inside a single 0.05
ball, which is a demanding bar; expect low absolute numbers there and read the
comparison against LQR rather than the value alone.
'''
import numpy as np


def goal_tolerance(env):
    '''Radius of the goal ball, by whichever name this env gives it.

    Resolved on `.unwrapped`. `AttributeForwardingMixin` forwards an allowlist
    containing neither `goal_threshold` nor `TASK_INFO`, so on the pendulum --
    the one system whose training config wraps its env -- reading either from
    the wrapper raises AttributeError.

    The pendulum names it `goal_threshold` (0.075, from its yaml); cartpole and
    the quadrotors use `TASK_INFO['stabilization_goal_tolerance']` (0.05).
    Checking the pendulum's name first matters: its TASK_INFO happens to hold
    0.075 too, so reading TASK_INFO would agree by coincidence today and
    diverge silently the moment the yaml changed.
    '''
    base = env.unwrapped
    threshold = getattr(base, 'goal_threshold', None)
    if threshold is not None:
        return float(threshold)
    return float(base.TASK_INFO['stabilization_goal_tolerance'])


def at_goal(env, tolerance):
    '''The envs' own stabilization test, applied to the current state.'''
    base = env.unwrapped
    return bool(np.linalg.norm(np.asarray(base.state) - np.asarray(base.X_GOAL)) < tolerance)
