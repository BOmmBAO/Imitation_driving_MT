from gym.envs.registration import register


register(
    id='MBRL-Carla-v0',
    entry_point='rl_algorithm.PETS_decision.dmbrl.env.carla:CarlaEnv'
)

