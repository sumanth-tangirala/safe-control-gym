'''Widen the initial-state distribution as the policy earns it.

Why this exists, measured rather than assumed. Under a random policy, over 300
episodes per system, the number of steps landing inside the goal ball:

    inverted_pendulum      4 hits in 99,049 steps   (~1 in 25,000)
    cartpole               0 hits in 16,712 steps
    quadrotor2d            0 hits in  2,775 steps
    quadrotor3d            0 hits in  4,980 steps

With a sparse reward, three of four systems therefore see no positive reward
ever. The replay buffer contains only the step cost and the out-of-bounds
penalty, so the critic learns that every state is equally bad and no gradient
points at the goal. Those runs cannot learn, however long they are given.

Relaxing the success criterion does not rescue quadrotor3d either: measured on
configuration coordinates alone, ignoring velocity, its closest random approach
is 0.631 and it never enters a ball of radius 0.5. Its median distance to the
goal is 14.8, dominated by body rates running to +/-24.

So the fix has to be on the initial states. Start each episode near the goal,
where even a poor policy succeeds sometimes, and widen toward the full
collection region as success is demonstrated. The policy always trains on the
hardest distribution it can currently handle.

Driven by measured success rate, not by a step schedule. A schedule has to be
guessed per system and is wrong twice: it holds back a system that learns
quickly and pushes one that has not learned past what it can do. Success rate is
already computed for the acceptance bar, so the curriculum and the metric agree
by construction.

The end state is the full collection region -- the regime the datasets were
collected under. A curriculum that stopped short would leave a policy that has
never seen the states it will be asked to handle.
'''
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from safe_control_gym.experiments.success import at_goal, goal_tolerance


def interpolate_ranges(full, fraction, goal_state=None, layout=None):
    '''Shrink each initial-state range toward the goal by `fraction`.

    fraction 0 gives a point distribution at the goal, 1 gives the full
    collection region. Each channel is contracted about the goal value rather
    than about the midpoint of its range: contracting about the midpoint would
    start quadrotor2d's altitude at 0.8 m when its goal is 1.0 m, so the easiest
    curriculum stage would not actually be easy.
    '''
    out = {}
    for name, spec in full.items():
        low, high = float(spec['low']), float(spec['high'])
        centre = 0.0
        if goal_state is not None and layout is not None:
            channel = name[len('init_'):] if name.startswith('init_') else name
            if channel in layout:
                centre = float(goal_state[layout.index(channel)])
        centre = min(max(centre, low), high)
        out[name] = {'distrib': spec.get('distrib', 'uniform'),
                     'low': centre + (low - centre) * fraction,
                     'high': centre + (high - centre) * fraction}
    return out


def set_goal_tolerance(env, tolerance):
    """Set the goal-ball radius, by whichever name this env gives it.

    The pendulum keeps `goal_threshold`; cartpole and the quadrotors keep
    `TASK_INFO['stabilization_goal_tolerance']`. Both are read by _get_done, so
    changing them moves what counts as success -- which is the point, and why the
    curriculum must always end at the true value.
    """
    base = env.unwrapped
    if getattr(base, 'goal_threshold', None) is not None:
        base.goal_threshold = float(tolerance)
    else:
        base.TASK_INFO = dict(base.TASK_INFO)
        base.TASK_INFO['stabilization_goal_tolerance'] = float(tolerance)


class InitStateCurriculum(BaseCallback):
    '''Widen the initial-state range when the policy clears a success threshold.

    Evaluates on its own env every `eval_freq` steps. Clearing `threshold`
    advances the fraction by `step`; the run finishes at fraction 1.0, the full
    collection region.

    Its env must not be the training env -- stepping that would advance the
    state the algorithm is learning from. The training envs are updated through
    `env_setter`, which the caller supplies because a VecEnv needs env_method
    while a plain env is assigned directly.

    `eval/curriculum_fraction` is logged so a run's difficulty is visible
    alongside its success rate; a rising success rate at a fixed fraction and a
    rising fraction mean different things.
    '''

    def __init__(self, eval_env, env_setter, tolerance_setter, full_ranges, layout,
                 start=0.1, step=0.15, threshold=0.5, n_episodes=10,
                 tolerance_start=None, tolerance_final=None,
                 eval_freq=10000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.env_setter = env_setter
        self.full_ranges = full_ranges
        self.layout = layout
        self.fraction = float(start)
        self.step_size = float(step)
        self.threshold = float(threshold)
        self.n_episodes = int(n_episodes)
        self.eval_freq = int(eval_freq)
        self.goal_state = np.asarray(eval_env.unwrapped.X_GOAL, dtype=float)
        self.tolerance_setter = tolerance_setter
        # A second axis, needed only where no initial state is easy enough.
        # quadrotor3d scores 0% under a random policy even when started exactly
        # at the goal -- one random thrust leaves a 0.05-radius ball in 12
        # dimensions before _get_done is evaluated -- so widening the start
        # distribution cannot help it. At a tolerance of 0.5 the same setup
        # scores 88%. The others need no tolerance curriculum: cartpole reaches
        # 47% and quadrotor2d 31% at their true tolerance once started close.
        self.tolerance_final = float(goal_tolerance(eval_env)
                                     if tolerance_final is None else tolerance_final)
        self.tolerance = float(self.tolerance_final if tolerance_start is None
                               else tolerance_start)
        self._apply()

    def _ranges(self):
        return interpolate_ranges(self.full_ranges, self.fraction,
                                  self.goal_state, self.layout)

    def _apply(self):
        ranges = self._ranges()
        self.env_setter(ranges)
        # The evaluation env moves with the training envs: the threshold has to
        # measure the distribution being trained on, not a different one.
        self.eval_env.unwrapped.INIT_STATE_RAND_INFO = ranges
        if self.tolerance_setter is not None:
            self.tolerance_setter(self.tolerance)
            set_goal_tolerance(self.eval_env, self.tolerance)

    def _success_rate(self):
        wins = 0
        for i in range(self.n_episodes):
            obs, _ = self.eval_env.reset(seed=self.n_calls + i)
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
            wins += at_goal(self.eval_env, self.tolerance)
        return wins / self.n_episodes

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        rate = self._success_rate()
        self.logger.record('eval/curriculum_fraction', self.fraction)
        self.logger.record('eval/curriculum_tolerance', self.tolerance)
        self.logger.record('eval/curriculum_success_rate', rate)
        if rate >= self.threshold and not self._finished():
            # Tighten the tolerance first, then widen the initial states. Doing
            # it the other way round would leave a policy that only ever
            # succeeded against a loose tolerance being asked to handle the full
            # region at the true one -- the two hardest changes at once.
            if self.tolerance > self.tolerance_final:
                self.tolerance = max(self.tolerance_final,
                                     self.tolerance * (1.0 - self.step_size))
            elif self.fraction < 1.0:
                self.fraction = min(1.0, self.fraction + self.step_size)
            self._apply()
            if self.verbose:
                print(f'curriculum: success {rate:.2f} -> fraction '
                      f'{self.fraction:.3f}, tolerance {self.tolerance:.4f}')
        return True

    def _finished(self):
        return self.fraction >= 1.0 and self.tolerance <= self.tolerance_final
