'''Score a trained policy against its system's LQR baseline.

    python -m safe_control_gym.experiments.eval_policy \\
        --run logs/sac/cartpole_stabilization_1 --n_episodes 100 --seed 0

The environment is rebuilt from the run's own config.yml, shaping wrappers
included, so evaluation cannot silently disagree with training about what the
task was. Policy and baseline are then rolled out from the *same* seeded initial
states -- asserted, not assumed -- and both sets of numbers are written to
eval.json beside the weights.

Why a relative bar. Every system here has an LQR (`pendulum_lqr` for the
inverted pendulum, `lqr` elsewhere), and all five collectors already use one, so
comparing against it needs no per-system constant invented in advance. A fixed
threshold would have to be guessed before anyone knows what is achievable, and a
wrong guess either passes bad policies or blocks good ones.

Its failure mode is a weak baseline: where LQR itself performs badly the bar is
vacuous and `PASS` means little. That is handled by disclosure rather than by a
cleverer rule -- absolute numbers for both controllers are always reported next
to the verdict.

Success is computed here, not read from info['goal_reached'].
`_get_info` in cartpole.py and quadrotor.py gates that key on
`COST == Cost.QUADRATIC`, and RL training uses `cost: rl_reward`, so for three of
the four systems the key is simply absent. This applies the envs' own rule --
``||state - X_GOAL|| < tolerance`` -- directly to the state, which holds whatever
the cost is.

Success is evaluated at the TERMINAL state, not at any point in the episode.
Under `rl_reward` an episode does not stop when the goal ball is entered, so
"reached it at some point" counts trajectories that arrived and then left. The
terminal state is also what the downstream flow-matching model predicts, which
makes it the number worth reporting. `reached_goal_any_step` is recorded
alongside it, so the gap between the two is visible rather than hidden.
'''
import argparse
import json
import os
from functools import partial

import munch
import numpy as np
import yaml

from safe_control_gym.experiments.success import at_goal, goal_tolerance
from safe_control_gym.experiments.train_sb3 import (ALGOS, apply_collection_bounds, build_env,
                                                    load_collection_bounds, with_collection_init)
from safe_control_gym.utils.registration import get_config, make

# Re-exported: callers reach for these here, and success.py is where they are
# defined so the training-time callback cannot drift from the acceptance bar.
__all__ = ['at_goal', 'baseline_id', 'episode_seeds', 'evaluate', 'goal_tolerance']

# Systems whose LQR is not the generic one, keyed by SYSTEM rather than by env
# id. Keyed by id, adding the reach task silently dropped the pendulum back to
# the generic `lqr` -- the smoke run scored inverted_pendulum_reach against the
# wrong controller while inverted_pendulum_stabilization used the right one.
BASELINE_OVERRIDES = {'inverted_pendulum': 'pendulum_lqr'}

DEFAULT_MARGIN = 0.05


def baseline_id(env_id):
    '''Which LQR controller stabilises this system, for any task variant.'''
    for system, controller in BASELINE_OVERRIDES.items():
        if env_id == system or env_id.startswith(system + '_'):
            return controller
    return 'lqr'


def episode_seeds(seed, n_episodes):
    '''Per-episode seeds. Shared by policy and baseline so inits coincide.'''
    rng = np.random.default_rng(seed)
    return [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=n_episodes)]


def run_episode(env, act, seed, tolerance):
    '''One episode; returns its record.'''
    obs, info = env.reset(seed=seed)
    initial_state = np.asarray(env.unwrapped.state, dtype=float).copy()
    total_reward, steps, reached_any = 0.0, 0, at_goal(env, tolerance)
    terminated = truncated = False
    out_of_bounds = violated = False

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(act(obs, info))
        total_reward += float(reward)
        steps += 1
        reached_any = reached_any or at_goal(env, tolerance)
        out_of_bounds = out_of_bounds or bool(info.get('out_of_bounds', False))
        violated = violated or bool(np.any(info.get('constraint_violation', False)))

    return {
        'initial_state': initial_state.tolist(),
        'terminal_state': np.asarray(env.unwrapped.state, dtype=float).tolist(),
        'success': at_goal(env, tolerance),
        'reached_goal_any_step': reached_any,
        'return': total_reward,
        'length': steps,
        'terminated': bool(terminated),
        'truncated': bool(truncated),
        'out_of_bounds': out_of_bounds,
        'constraint_violation': violated,
    }


def summarise(episodes):
    '''Per-controller metrics block.'''
    def frac(key):
        return float(np.mean([e[key] for e in episodes]))
    return {
        'success_rate': frac('success'),
        'reached_goal_any_step_rate': frac('reached_goal_any_step'),
        'mean_return': float(np.mean([e['return'] for e in episodes])),
        'mean_episode_length': float(np.mean([e['length'] for e in episodes])),
        'out_of_bounds_rate': frac('out_of_bounds'),
        'constraint_violation_rate': frac('constraint_violation'),
        'terminated_frac': frac('terminated'),
        'truncated_frac': frac('truncated'),
    }


def load_run_config(run_dir):
    with open(os.path.join(run_dir, 'config.yml')) as handle:
        return yaml.safe_load(handle)


def resolve_weights(run_dir, model_name):
    '''best_model if EvalCallback wrote one, else the final weights.'''
    if model_name:
        path = os.path.join(run_dir, model_name)
        return path if path.endswith('.zip') else path + '.zip'
    for candidate in ('best_model.zip', 'model_final.zip'):
        path = os.path.join(run_dir, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'No best_model.zip or model_final.zip in {run_dir}')


def evaluate(run_dir, n_episodes, seed, margin, model_name=None, skip_baseline=False,
             eval_bounds_path=None):
    '''Roll out policy and baseline from identical states; return the report.'''
    config = load_run_config(run_dir)
    env_id, algo = config['task'], config['algo']

    # Which regime to score in. An explicit --eval_bounds wins; otherwise the
    # run's own collection_bounds is reused, so a policy trained in the
    # collection regime is scored in it by default rather than by remembering a
    # flag. Falling back to neither means scoring on the training distribution,
    # which is a materially easier question -- recorded in the report so a
    # number can never be quoted without saying which region produced it.
    trained_under = config.get('sb3_config', {}).get('collection_bounds')
    bounds_path = eval_bounds_path or trained_under
    bounds = load_collection_bounds(bounds_path)

    # Applied to the config BEFORE either env is built, so policy and baseline
    # face the identical distribution -- the whole comparison rests on that.
    config = dict(config)
    config['task_config'] = with_collection_init(dict(config['task_config']), bounds)
    # Point build_env at the regime being scored, rather than clearing the key.
    #
    # Clearing it was wrong once the observation encoding moved into this same
    # file: build_env then applied no wrappers, so a pendulum policy expecting
    # (cos, sin, rate) was handed the raw 2-channel state and SB3 rejected the
    # shape. Setting it makes one file the authority for both the regime and the
    # encoding, which is the point of having one file.
    # sb3_config keeps the run's OWN collection_bounds, which is where the
    # observation encoding lives -- the model's input shape was fixed at
    # training time and must not change here. The evaluation region is passed
    # separately as `regime`.
    config['sb3_config'] = dict(config.get('sb3_config') or {})
    task_config = config['task_config']
    seeds = episode_seeds(seed, n_episodes)

    # Policy: the wrapped env it was trained on. Shaping changes the observation,
    # so a policy evaluated on the bare env would be fed the wrong vector.
    policy_env = build_env(munch.munchify(config), regime=bounds)
    apply_collection_bounds(policy_env, env_id, bounds)
    tolerance = goal_tolerance(policy_env)
    weights = resolve_weights(run_dir, model_name)
    model = ALGOS[algo].load(weights, device='cpu')
    try:
        policy_episodes = [
            run_episode(policy_env, lambda obs, info: model.predict(obs, deterministic=True)[0],
                        s, tolerance)
            for s in seeds]
    finally:
        policy_env.close()

    report = {
        'env_id': env_id, 'algo': algo, 'n_episodes': n_episodes, 'seed': seed,
        'weights': os.path.basename(weights),
        'goal_tolerance': tolerance,
        'eval_bounds': bounds_path,
        'reference_success': bounds.get('reference_success'),
        'reference_dataset': bounds.get('reference_dataset'),
        # Which task the reference was measured under, against which this run
        # ran. Every shipped dataset predates terminate_on_goal, so its numbers
        # are reach numbers; scoring a stabilization policy against them
        # compares two different problems. Recorded rather than silently
        # compared, and reflected in beats_reference below.
        'reference_task': bounds.get('reference_task'),
        'terminate_on_goal': bool(policy_env.unwrapped.terminate_on_goal),
        'policy': summarise(policy_episodes),
    }

    if skip_baseline:
        report.update({'baseline_id': None, 'margin': margin, 'verdict': 'NO_BASELINE'})
        return report

    # Baseline: the bare env. LQR is a state-feedback law and does not consume
    # the shaped observation.
    base_id = baseline_id(env_id)
    # The baseline runs on the RAW action space, never the normalised one.
    # normalized_rl_action_space exists to give the POLICY a symmetric [-1, 1]
    # box; the env then denormalises. LQR is a state-feedback law emitting
    # physical units, so handing it a normalised env silently rescales its
    # output -- measured on cartpole, a 5 N command was applied as 50 N.
    baseline_config = {k: v for k, v in task_config.items()
                       if k != 'normalized_rl_action_space'}
    baseline_config['normalized_rl_action_space'] = False
    env_func = partial(make, env_id, **baseline_config)
    # LQR's q_lqr/r_lqr default to None and are then passed straight to
    # get_cost_weight_matrix, which does len() on them. The registered yaml is
    # where the real defaults live, so it is loaded rather than reinvented here.
    controller = make(base_id, env_func, **get_config(base_id))
    baseline_env = env_func()
    apply_collection_bounds(baseline_env, env_id, bounds)
    try:
        def act(obs, info):
            return controller.select_action(np.asarray(baseline_env.unwrapped.state), info)
        baseline_episodes = [run_episode(baseline_env, act, s, tolerance) for s in seeds]
    finally:
        baseline_env.close()
        controller.close()

    # The whole point of sharing seeds. Asserted rather than trusted: a wrapper
    # or a reset-order change could desynchronise them, and then the comparison
    # is between two different problems.
    for i, (a, b) in enumerate(zip(policy_episodes, baseline_episodes)):
        np.testing.assert_allclose(
            a['initial_state'], b['initial_state'], atol=0,
            err_msg=f'episode {i}: policy and baseline started from different states')

    baseline = summarise(baseline_episodes)
    policy_success = report['policy']['success_rate']
    reference = report.get('reference_success')

    # terminate_on_goal is the reach/stabilization distinction, so it is what
    # decides whether the reference number describes the same problem.
    ran_task = 'reach' if report['terminate_on_goal'] else 'stabilization'
    reference_task = report.get('reference_task')
    comparable = reference_task is None or reference_task == ran_task

    # A baseline pinned at 0 or 1 cannot rank anything: at 1.0 every adequate
    # policy passes, at 0.0 every useless one does. Say so rather than emitting
    # a PASS that means nothing -- this is the saturation the collection-box
    # eval regions exist to avoid, and it must be visible when it happens.
    # Saturated at either end, or so weak that the margin swallows it. The
    # second case is not hypothetical: a 20-episode quad3d smoke run put LQR at
    # 0.050 against a margin of 0.05, so a policy scoring exactly 0.000 cleared
    # the bar and reported PASS. A bar a dead policy can clear ranks nothing.
    if baseline['success_rate'] in (0.0, 1.0) or baseline['success_rate'] <= margin:
        verdict = 'NON_DISCRIMINATING'
    elif policy_success >= baseline['success_rate'] - margin:
        verdict = 'PASS'
    else:
        verdict = 'FAIL'

    report.update({
        'baseline_id': base_id,
        'baseline': baseline,
        'margin': margin,
        'verdict': verdict,
        # Against the controller that actually produced the shipped dataset,
        # measured over this same region. The bar for "worth re-collecting with".
        #
        # Withheld when the tasks differ: the references are reach numbers, and
        # a stabilization policy holding position is solving a harder problem,
        # so the comparison would flatter or damn it for the wrong reason. None
        # says "not comparable", which is the honest answer.
        'beats_reference': (None if reference is None or not comparable
                            else bool(policy_success >= reference)),
    })
    return report


def render(report):
    '''Absolute numbers for both controllers, so a weak baseline is visible.'''
    lines = [f"{report['env_id']}  {report['algo']}  weights={report['weights']}",
             f"{report['n_episodes']} episodes, seed {report['seed']}, "
             f"goal tolerance {report['goal_tolerance']}"]
    blocks = [('policy', report['policy'])]
    if report.get('baseline'):
        blocks.append((report['baseline_id'], report['baseline']))
    keys = ['success_rate', 'mean_return', 'mean_episode_length',
            'out_of_bounds_rate', 'terminated_frac']
    lines.append(f"{'':22s}" + ''.join(f'{k:>22s}' for k in keys))
    for name, block in blocks:
        lines.append(f'{name:22s}' + ''.join(f'{block[k]:>22.4f}' for k in keys))
    if report.get('reference_success') is not None:
        ran = 'reach' if report.get('terminate_on_goal') else 'stabilization'
        reference_task = report.get('reference_task')
        lines.append(f"reference   {report['reference_success']:.4f}  "
                     f"({report.get('reference_dataset')}, task={reference_task}) -- "
                     f"beats_reference={report.get('beats_reference')}")
        if reference_task is not None and reference_task != ran:
            lines.append(f'  this run is {ran}, the reference is {reference_task} -- '
                         f'not the same problem, so no comparison is made.')
    if report.get('eval_bounds'):
        lines.append(f"eval region: {report['eval_bounds']}")
    lines.append(f"verdict: {report['verdict']}  (margin {report['margin']})")
    if report['verdict'] == 'NON_DISCRIMINATING':
        lines.append('  the baseline is saturated, or weak enough that the margin swallows '
                     'it -- a policy scoring 0.000 would clear this bar. Ranks nothing.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run', required=True, help='run directory written by train_sb3')
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--margin', type=float, default=DEFAULT_MARGIN,
                        help='allowed shortfall in success rate against the baseline')
    parser.add_argument('--model', default=None,
                        help='weights to load; defaults to best_model, else model_final')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='report policy metrics only, with no verdict')
    parser.add_argument('--out', default=None, help='where to write eval.json')
    parser.add_argument('--eval_bounds', default=None,
                        help='yaml giving the evaluation region (see configs/eval/); '
                             'defaults to the region the policy trained on, which is '
                             'an easier question than the one that matters')
    args = parser.parse_args()

    report = evaluate(args.run, args.n_episodes, args.seed, args.margin,
                      args.model, args.skip_baseline, args.eval_bounds)
    destination = args.out or os.path.join(args.run, 'eval.json')
    with open(destination, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(render(report))
    print(f'wrote {destination}')


if __name__ == '__main__':
    main()
