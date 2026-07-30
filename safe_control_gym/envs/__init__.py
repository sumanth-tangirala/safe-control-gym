'''Register environments.'''

from safe_control_gym.utils.registration import register

register(idx='cartpole',
         entry_point='safe_control_gym.envs.gym_control.cartpole:CartPole',
         config_entry_point='safe_control_gym.envs.gym_control:cartpole.yaml')

register(idx='quadrotor',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor:Quadrotor',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor.yaml')

register(idx='inverted_pendulum',
         entry_point='safe_control_gym.envs.gym_control.inverted_pendulum:InvertedPendulum',
         config_entry_point='safe_control_gym.envs.gym_control:inverted_pendulum.yaml')

# Composite (system, task) ids.
#
# The three ids above name a *system*; the `task:` field inside their yaml names
# a task. Two axes, one flag -- so a run directory called `quadrotor_3` says
# neither which quad_type nor which task it was, and an unsupported pair fails
# somewhere inside the env rather than at lookup.
#
# One id per pair fixes both, and needs no new plumbing: configuration.py
# already resolves `--task <id>` to that id's yaml via get_config(), and
# train_sb3 splats it into make(). A composite id is therefore its base
# entry_point plus a yaml with the axes pinned.
#
# The base ids stay registered and unchanged -- every existing entry point,
# collector and example keeps working.

register(idx='inverted_pendulum_stabilization',
         entry_point='safe_control_gym.envs.gym_control.inverted_pendulum:InvertedPendulum',
         config_entry_point='safe_control_gym.envs.gym_control:inverted_pendulum_stabilization.yaml')

register(idx='cartpole_stabilization',
         entry_point='safe_control_gym.envs.gym_control.cartpole:CartPole',
         config_entry_point='safe_control_gym.envs.gym_control:cartpole_stabilization.yaml')

register(idx='quadrotor2d_stabilization',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor:Quadrotor',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor2d_stabilization.yaml')

register(idx='quadrotor3d_stabilization',
         entry_point='safe_control_gym.envs.gym_pybullet_drones.quadrotor:Quadrotor',
         config_entry_point='safe_control_gym.envs.gym_pybullet_drones:quadrotor3d_stabilization.yaml')
