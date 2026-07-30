'''Task-agnostic stable-baselines3 training.

The only module in the package permitted to import stable-baselines3; envs/ and
controllers/ stay SB3-free so inference and dataset collection never gain the
dependency.

Shaping is configuration, not code: --kv_overrides sb3_config.angle_obs=... and
sb3_config.action_repeat=... select the optional wrappers. No task is special
cased here.

    python -m safe_control_gym.experiments.train_sb3 \\
        --env_id cartpole_stabilization --algo sac --output_dir logs \\
        --overrides configs/sb3/cartpole_stabilization_sac.yaml

--env_id is an alias for --task. They mean the same registry lookup; --task is
defined once in configuration.py and shared by every entry point in the repo, so
it is not renamed, while --env_id is what a composite (system, task) id actually
is. Pass either.

Runs are written to <output_dir>/<algo>/<env_id>_<run>/, following RL Baselines3
Zoo, with the merged config, the CLI arguments and the verbatim command stored
beside the weights. <run> auto-increments, so re-running never clobbers.
config.yml is what makes a run rebuildable: eval_policy reconstructs the
environment and its wrappers from it rather than re-deriving them from flags.

--use_gpu (like every other entry point in this repo) selects SB3's device;
it defaults to CPU.

**Pass --use_gpu if a GPU is free.** Measured on an idle ilab2 (64 cores,
RTX A4500), SAC on the pendulum with `net_arch: [256, 256]`, 4000 steps, both
devices back to back on the same host with OMP/MKL threads pinned to 8:

    cpu     65.6 steps/s   ->  200k steps ~ 51 min
    cuda   111.0 steps/s   ->  200k steps ~ 30 min   (1.69x faster)

An earlier version of this docstring claimed GPU would be *slower* here, on the
general principle that small MLP policies favour CPU. That was asserted, not
measured, and it is wrong for this workload -- the first attempt to check it ran
on a box at load average 74 and measured contention rather than devices. Benchmark
on an idle host before trusting any claim of this kind, including this one.

The environment is never the bottleneck either way: the pendulum steps at
~12,500/s, so 200k steps is ~16s of simulation against tens of minutes of
gradient work.
'''
import os
import shlex
import sys

import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from safe_control_gym.envs.env_wrappers.shaping import ActionRepeat, AngleObservation
from safe_control_gym.envs.env_wrappers.vec_compat import PicklableInfo
from safe_control_gym.experiments.callbacks import SuccessRateCallback
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import set_device_from_config, set_seed_from_config

ALGOS = {'sac': SAC, 'ppo': PPO}

# CLI arguments worth recording separately from the merged config: reading
# args.yml answers "what was typed" without diffing a fully-resolved config
# against the defaults it came from.
CLI_KEYS = ('algo', 'task', 'seed', 'use_gpu', 'output_dir', 'tag', 'restore')


def apply_env_id_alias(argv=None):
    '''Rewrite --env_id to --task, which is what ConfigFactory parses.

    Returns the argv list to parse. Passing both is an error rather than a
    silent precedence rule -- they name the same thing, so disagreeing values
    mean the caller believes something untrue.
    '''
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a == '--env_id' or a.startswith('--env_id=') for a in argv):
        return argv
    if any(a == '--task' or a.startswith('--task=') for a in argv):
        raise SystemExit('Pass --env_id or --task, not both: they are the same argument.')
    return [a.replace('--env_id', '--task', 1) if a.startswith('--env_id') else a
            for a in argv]


def load_collection_bounds(path):
    '''The collection regime for one system, read from a single file.

    `configs/collection/<env_id>.yaml` records what a dataset was actually
    collected under: where episodes start, and where they terminate. Both
    training and evaluation read the same file, so the regime a policy learns in
    cannot drift from the one it is scored in -- which is the whole reason the
    numbers mean anything.

    Transcribed from each dataset's dataset_description.json, where
    initial_state_bounds and termination_thresholds are identical: collection
    starts anywhere it will not immediately die.
    '''
    if not path:
        return {}
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def apply_collection_bounds(env, env_id, bounds):
    '''Put a constructed env into the collection regime.

    Both halves matter and they are separate mechanisms: the start range is
    config the env reads at reset, while the termination box lives on the env
    object. Widening one without the other is the trap -- a wider start range
    against an unmoved kill boundary just produces episodes that end at step
    one, which reads as a hard problem rather than a broken setup.
    '''
    apply_env_attributes(env, bounds.get('env_attributes', {}))
    apply_state_space_bounds(env, env_id, bounds.get('state_space_bounds', {}))


def with_collection_init(task_config, bounds):
    '''Overlay the collection start range onto a task_config.'''
    randomization = bounds.get('init_state_randomization_info')
    if not randomization:
        return task_config
    merged = dict(task_config)
    merged['randomized_init'] = True
    merged['init_state_randomization_info'] = randomization
    return merged


def build_env(config):
    '''Registered env plus whatever shaping the config asks for.'''
    sb3_config = config.get('sb3_config', {})
    bounds = load_collection_bounds(sb3_config.get('collection_bounds'))
    task_config = with_collection_init(dict(config.task_config), bounds)
    env = make(config.task, **task_config)
    # Applied to the bare env before any wrapper, since the thresholds live on
    # the env itself.
    apply_collection_bounds(env, config.task, bounds)
    apply_env_attributes(env, sb3_config.get('env_attributes', {}))
    angle_obs = sb3_config.get('angle_obs', None)
    if angle_obs is not None:
        env = AngleObservation(env, angle_obs['angle_index'],
                               angle_obs['rate_index'], angle_obs['rate_max'])
    repeat = int(sb3_config.get('action_repeat', 1))
    if repeat > 1:
        env = ActionRepeat(env, repeat)
    return env


# State vector order per SYSTEM, for addressing state_space bounds by name.
# quadrotor.py documents these inline in _get_done; restated here because a
# config that says `theta_dot` is reviewable and one that says `[5]` is not.
#
# Keyed by system, not by env id: the state vector is a property of the system,
# and every task variant of it shares one. Keying by env id meant every new
# task -- reach was the first -- silently lost its entry and raised.
STATE_ORDER = {
    'quadrotor2d': ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot'],
    'quadrotor3d': ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                    'phi', 'theta', 'psi', 'p', 'q', 'r'],
}


def state_order(env_id):
    '''The state vector layout for whichever system this env id names.'''
    for system, order in STATE_ORDER.items():
        if env_id.startswith(system):
            return order
    return None


def apply_state_space_bounds(env, env_id, bounds):
    '''Widen or narrow the quadrotors' termination box, as their collectors do.

    The two env families bound themselves differently. cartpole has dedicated
    threshold attributes; the quadrotors terminate on `state_space`, so changing
    where they die means mutating those arrays in place --
    generate_quadrotor_2d_trajectories_rl.py:611-623 does exactly this.

    It matters most for quad3d, whose dataset samples z to 3.0 against a default
    limit of 2.0 and body rates to +/-24 against +/-8.727. Sampling that region
    without moving the bounds gives episodes that die on the first step.
    '''
    if not bounds:
        return
    order = state_order(env_id)
    if order is None:
        raise KeyError(f'No state order known for {env_id}; state_space_bounds '
                       f'is only defined for systems {sorted(STATE_ORDER)}.')
    base = env.unwrapped
    for name, (low, high) in bounds.items():
        if name not in order:
            raise KeyError(f'{env_id} has no state dimension {name!r}; expected '
                           f'one of {order}.')
        index = order.index(name)
        base.state_space.low[index] = float(low)
        base.state_space.high[index] = float(high)


def apply_env_attributes(env, attributes):
    '''Set termination attributes on a constructed env, as the collectors do.

    Not cosmetic. cartpole defaults to `x_threshold` 2.4 and
    `theta_threshold_radians` pi/2, while its dataset samples x to +/-6 and
    theta to +/-pi -- so without this, over half of those states terminate
    out-of-bounds on step one. In evaluation that reads as "this region is
    hard" when it is really "this measurement is broken"; in training it means
    the agent never sees the states it is supposed to learn.
    `generate_cartpole_trajectories.py:262-265` does the same thing.

    Lives here rather than in eval_policy because both entry points need it and
    eval_policy already imports from this module; the reverse would be circular.

    Only attributes the env already defines may be set, so a typo fails loudly
    rather than silently adding an attribute nothing reads.
    '''
    base = env.unwrapped
    for name, value in (attributes or {}).items():
        if not hasattr(base, name):
            raise AttributeError(
                f'{type(base).__name__} has no attribute {name!r}; env_attributes '
                f'may only override thresholds the env already defines.')
        setattr(base, name, float(value))


def build_train_env(config, n_envs):
    '''The training env, vectorised when asked for.

    Env stepping is the bottleneck on the quadrotors -- PyBullet at 240 Hz
    against a small MLP -- so running several in parallel is where the wall
    clock goes, not the GPU.

    SubprocVecEnv, not DummyVecEnv, whenever n_envs > 1. DummyVecEnv steps every
    env in one process, so the GIL serialises exactly the work being
    parallelised. Subprocesses also isolate each PyBullet client, which matters
    here: base_aviary's changeDynamics call targeted client 0 until it was given
    its physicsClientId, and two quadrotor envs in one process silently
    corrupted each other's dynamics (0.34 state divergence in five steps). That
    is fixed, so DummyVecEnv would be correct now -- but one env per process
    keeps it structurally impossible rather than relying on the fix holding.

    Evaluation envs stay unvectorised. SuccessRateCallback reads
    `env.unwrapped.state` to apply the goal test, which a VecEnv does not expose
    the same way, and one env is fast enough for ten episodes.
    '''
    if n_envs <= 1:
        return build_env(config)

    def factory():
        # PicklableInfo only on the vectorised path: info carries a CasADi
        # symbolic model that cannot cross a process boundary, and without
        # stripping it every worker dies on its first reset. Left off the
        # single-env path so nothing changes there.
        return PicklableInfo(build_env(config))

    # start_method='fork', explicitly. SB3 defaults to forkserver (or spawn),
    # both of which PICKLE the env constructors -- and this factory is a nested
    # closure, which pickle cannot handle, so the workers died on startup with
    # BrokenPipeError. fork inherits the closure instead of serialising it.
    #
    # Safe here because the parent has not connected to PyBullet when this runs:
    # every env, training and evaluation alike, is constructed after this call,
    # so no client handle is inherited across the fork.
    return SubprocVecEnv([factory for _ in range(n_envs)], start_method='fork')


def claim_run_dir(root, algo, env_id):
    '''<root>/<algo>/<env_id>_<n>, the lowest n not already taken.

    Claimed with exist_ok=False rather than checked-then-created, so two runs
    launched at once take different directories instead of both believing they
    own the same one.

    The parent, by contrast, must tolerate already existing: utils.mkdirs is
    os.makedirs without exist_ok, so four systems launched together raced on
    creating <root>/<algo> and two of them died with FileExistsError before
    training a single step.
    '''
    parent = os.path.join(root, algo)
    os.makedirs(parent, exist_ok=True)
    run = 1
    while True:
        candidate = os.path.join(parent, f'{env_id}_{run}')
        try:
            os.makedirs(candidate)
            return candidate
        except FileExistsError:
            run += 1


def write_run_metadata(run_dir, config):
    '''config.yml, args.yml and command.txt beside the weights.'''
    with open(os.path.join(run_dir, 'config.yml'), 'w') as handle:
        yaml.safe_dump(dict(config), handle, default_flow_style=False, sort_keys=True)
    args = {key: config.get(key) for key in CLI_KEYS if config.get(key) is not None}
    with open(os.path.join(run_dir, 'args.yml'), 'w') as handle:
        yaml.safe_dump(args, handle, default_flow_style=False, sort_keys=True)
    with open(os.path.join(run_dir, 'command.txt'), 'w') as handle:
        handle.write(shlex.join([sys.executable, '-m',
                                 'safe_control_gym.experiments.train_sb3'] + sys.argv[1:]) + '\n')


def wandb_run(config, run_dir):
    '''Start a wandb run if the config asks for one; otherwise None.

    Opt-in via an `sb3_config.wandb` block, so training never requires wandb to
    be installed, logged in, or reachable -- collection hosts and CI have no
    business needing either.

    Metrics arrive through `sync_tensorboard`, not through wandb's SB3
    integration. TensorBoard logging is on unconditionally, so the same curves
    land on disk beside the weights whether or not wandb is configured, and
    there is exactly one metric path to reason about rather than two.

    A failure here must not kill a training run that is otherwise fine: a run
    that dies at hour three because a logging backend was unreachable has
    thrown away the compute it was meant to record.
    '''
    settings = config.get('sb3_config', {}).get('wandb', None)
    if not settings:
        return None
    try:
        import wandb
    except ImportError:
        print('sb3_config.wandb set but wandb is not installed; continuing without it')
        return None
    try:
        return wandb.init(project=settings.get('project', 'safe-control-gym'),
                          entity=settings.get('entity', None),
                          group=settings.get('group', config.task),
                          name=os.path.basename(run_dir),
                          dir=run_dir,
                          mode=settings.get('mode', 'online'),
                          config=dict(config),
                          sync_tensorboard=True,
                          reinit=True)
    except Exception as error:  # noqa: BLE001 - logging must never fail a run
        print(f'wandb.init failed ({error}); continuing without it')
        return None


def train():
    '''Train and checkpoint; returns (model, run_dir).'''
    sys.argv[1:] = apply_env_id_alias()
    config = ConfigFactory().merge()
    set_seed_from_config(config)
    set_device_from_config(config)

    sb3_config = config.get('sb3_config', {})
    algo = ALGOS[config.algo]
    total_timesteps = int(sb3_config.get('total_timesteps', 100000))
    save_freq = int(sb3_config.get('save_freq', 10000))
    eval_freq = int(sb3_config.get('eval_freq', max(total_timesteps // 10, 1)))
    n_eval_episodes = int(sb3_config.get('n_eval_episodes', 5))

    run_dir = claim_run_dir(config.output_dir, config.algo, config.task)
    write_run_metadata(run_dir, config)
    print(f'device={config.device} run_dir={run_dir}')

    n_envs = int(sb3_config.get('n_envs', 1))
    # SB3's callbacks count model steps, not env steps: n_calls advances once
    # per model.step(), which consumes n_envs env steps. Left uncorrected, a
    # save_freq of 30000 with n_envs=4 fires every 120000 env steps -- measured
    # on a live run, cartpole had written zero checkpoints by 69520 steps while
    # the unvectorised pendulum had three by 62000. That silently turns ten
    # checkpoints into two, and thins the eval curve that selects best_model by
    # the same factor.
    #
    # So both frequencies are expressed in ENV steps in the config and divided
    # here, which is what SB3's own vectorised examples do.
    save_freq = max(save_freq // n_envs, 1)
    eval_freq = max(eval_freq // n_envs, 1)
    env = build_train_env(config, n_envs)
    # Two more envs, alive alongside the training env for the whole run: one for
    # SB3's reward-based EvalCallback, one for the success-rate callback. They
    # cannot share -- a callback stepping another's env mid-episode would
    # corrupt both. This was unsafe for the quadrotors until base_aviary.py's
    # changeDynamics call was given its physicsClientId, without which any of
    # them would silently corrupt the training env's dynamics. See
    # tests/test_envs/test_concurrent_pybullet_envs.py.
    eval_env = build_env(config)
    success_env = build_env(config)

    run = wandb_run(config, run_dir)
    # gradient_steps -1 means "one update per transition collected this
    # rollout", so sample efficiency does not silently drop as n_envs rises:
    # with the default of 1, eight parallel envs would collect eight
    # transitions per update instead of one.
    model = algo('MlpPolicy', env, seed=config.seed, verbose=1,
                 device=config.device,
                 tensorboard_log=run_dir,
                 gradient_steps=int(sb3_config.get('gradient_steps', -1 if n_envs > 1 else 1)),
                 policy_kwargs={'net_arch': list(sb3_config.get('net_arch', [256, 256]))})
    callbacks = [
        # Periodic checkpoints, not only best: the shipped strong/weak model pairs
        # are best-vs-intermediate checkpoints of one run, so dropping intermediates
        # would make that axis unreproducible.
        CheckpointCallback(save_freq=save_freq,
                           save_path=os.path.join(run_dir, 'checkpoints'),
                           name_prefix='step'),
        EvalCallback(eval_env, best_model_save_path=run_dir,
                     log_path=run_dir, eval_freq=eval_freq,
                     n_eval_episodes=n_eval_episodes, deterministic=True),
        # The curve that matches the acceptance bar. Without it a run can be
        # watched start to finish without ever showing the number it is judged
        # on -- EvalCallback records reward and episode length only.
        SuccessRateCallback(success_env, n_episodes=n_eval_episodes,
                            eval_freq=eval_freq),
    ]
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        model.save(os.path.join(run_dir, 'model_final'))
    finally:
        env.close()
        eval_env.close()
        success_env.close()
        if run is not None:
            run.finish()
    return model, run_dir


if __name__ == '__main__':
    train()
