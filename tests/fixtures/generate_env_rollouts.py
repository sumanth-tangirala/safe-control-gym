'''Generate golden rollout fixtures for the non-pendulum environments.

Run BEFORE the gymnasium migration, under gymnasium 0.28, and commit the
output. The migration must reproduce these trajectories exactly.

Cartpole and both quadrotors are PyBullet-backed: ``env.state`` is a
read-back of the physics client's internal joint/link state, not a value the
simulator consumes. Assigning ``env.state`` between steps is therefore a
no-op for dynamics purposes -- it does not seed the next ``step()``. Each
scenario is instead pinned by a ``reset(seed=...)`` call, and the resulting
post-reset state is recorded as ``x0`` for reference/sanity-checking only.

Run:  python tests/fixtures/generate_env_rollouts.py
'''
import json
import os

import numpy as np

from safe_control_gym.utils.registration import make

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'test_envs', 'fixtures')

# (fixture name, task id, task_config overrides, action dimension)
CASES = [
    ('cartpole_rollouts.json', 'cartpole', {}, 1),
    ('quadrotor_2d_rollouts.json', 'quadrotor', {'quad_type': 2}, 2),
    ('quadrotor_3d_rollouts.json', 'quadrotor', {
        'quad_type': 3,
        'task_info': {'stabilization_goal': [0, 0, 1], 'stabilization_goal_tolerance': 0.0},
    }, 4),
]

N_SCENARIOS = 4
N_STEPS = 25


def build(task, task_config, act_dim):
    '''Roll fixed pseudo-random action sequences and record every state.

    Each scenario resets with a fixed seed (rather than injecting a chosen
    x0) because the PyBullet-backed envs do not consume ``env.state`` as
    dynamics input -- only ``reset(seed=...)`` deterministically drives the
    underlying physics client.
    '''
    env = make(task, **task_config)
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    assert np.all(np.isfinite(low)) and np.all(np.isfinite(high)), (
        f'{task} action_space bounds are not finite; clip the sampling range explicitly.')
    rng = np.random.default_rng(0)
    scenarios = []
    for scenario in range(N_SCENARIOS):
        seed = 1000 + scenario
        env.reset(seed=seed)
        x0 = np.asarray(env.state, dtype=np.float64).tolist()
        actions, states = [], []
        for _ in range(N_STEPS):
            act = rng.uniform(low, high, size=(act_dim,))
            env.step(act)
            actions.append(np.asarray(act, dtype=np.float64).tolist())
            states.append(np.asarray(env.state, dtype=np.float64).tolist())
        scenarios.append({'seed': seed, 'x0': x0, 'actions': actions, 'states': states})
    params = {'task': task, 'task_config': task_config,
              'ctrl_freq': int(env.CTRL_FREQ), 'n_steps': N_STEPS}
    env.close()
    return {'params': params, 'scenarios': scenarios}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, task, task_config, act_dim in CASES:
        data = build(task, task_config, act_dim)
        path = os.path.join(OUT_DIR, name)
        with open(path, 'w') as handle:
            json.dump(data, handle)
        print(f'wrote {path} ({len(data["scenarios"])} scenarios)')


if __name__ == '__main__':
    main()
