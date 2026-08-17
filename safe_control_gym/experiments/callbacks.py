'''Training-time callbacks that log what the acceptance bar actually measures.

SB3's own `EvalCallback` records mean episode reward and length. Neither is the
criterion a policy is accepted on here, which is terminal-state membership of
the goal ball -- so a run could be watched to completion without ever showing
the number that decides it.

`SuccessRateCallback` closes that: same rule as `eval_policy`, imported from
`success.py` rather than restated, logged every `eval_freq` steps.
'''
from stable_baselines3.common.callbacks import BaseCallback

from safe_control_gym.experiments.success import at_goal, goal_tolerance


class SuccessRateCallback(BaseCallback):
    '''Log terminal-state success rate on a held-out env during training.

    Runs `n_episodes` deterministic episodes every `eval_freq` calls and emits
    `eval/success_rate`, `eval/reached_goal_any_step_rate` and
    `eval/out_of_bounds_rate` through SB3's logger, so they reach TensorBoard
    (and wandb, when it is syncing TensorBoard) alongside the built-in curves.

    `success_rate` is measured at the TERMINAL state; under `rl_reward` an
    episode does not stop on entering the goal ball, so a policy can arrive and
    then drift out. `reached_goal_any_step_rate` is logged next to it precisely
    so that gap is visible rather than hidden -- a large spread between the two
    means the policy can reach the goal but not hold it, which is a different
    problem from never reaching it.

    Its env must not be the training env. Stepping that env would advance the
    state the algorithm is learning from; a separate one is required, and for
    the quadrotors that only became safe once base_aviary.py's changeDynamics
    call was given its physicsClientId.
    '''

    def __init__(self, eval_env, n_episodes=10, eval_freq=10000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_episodes = n_episodes
        self.eval_freq = eval_freq
        self.tolerance = goal_tolerance(eval_env)

    def _run_episode(self, seed):
        obs, _ = self.eval_env.reset(seed=seed)
        reached_any = at_goal(self.eval_env, self.tolerance)
        terminated = truncated = False
        out_of_bounds = False
        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = self.eval_env.step(action)
            reached_any = reached_any or at_goal(self.eval_env, self.tolerance)
            out_of_bounds = out_of_bounds or bool(info.get('out_of_bounds', False))
        return at_goal(self.eval_env, self.tolerance), reached_any, out_of_bounds

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        # Seeded off n_calls so each evaluation draws the same initial states
        # for every run of a given config -- otherwise a rising curve could be
        # easier starts rather than a better policy.
        results = [self._run_episode(self.n_calls + i) for i in range(self.n_episodes)]
        success, reached, oob = zip(*results)
        self.logger.record('eval/success_rate', float(sum(success)) / len(success))
        self.logger.record('eval/reached_goal_any_step_rate', float(sum(reached)) / len(reached))
        self.logger.record('eval/out_of_bounds_rate', float(sum(oob)) / len(oob))
        return True
