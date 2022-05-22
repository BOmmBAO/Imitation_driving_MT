import gym
import numpy as np
from trpo_mpi import TRPO
from rl_algorithm.common.policies import MlpPolicy
from rl_algorithm.common.noise import NormalActionNoise
from carla_e2e import CarlaEnv
import sys
import argparse
import tensorflow as tf
def main(load_model,ep_length, repeat_action, seed = 7):
    env = CarlaEnv(ep_length, repeat_action)
    try:
        if load_model:
            model = TRPO.load('trpo_e2e', env, action_noise = NormalActionNoise(mean=np.array([0.3, 0]), sigma=np.array([0.5, 0.1])))
        else:
            model = TRPO(MlpPolicy,
                         env,
                         verbose=0,
                         seed = seed,
                         tensorboard_log='./sem_trpo')
        model.learn(total_timesteps=20000,
                    log_interval=4,
                    tb_log_name='trpo_e2e'
                    )
        model.save("trpo_e2e")

        #del model  # remove to demonstrate saving and loading
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load', default=False, type=bool, help='whether to load existing model')
    #parser.add_argument('--map', type=str, default='Town03', help='name of carla map')
    parser.add_argument('--repeat-action', default=1, type=int, help='number of steps to repeat each action')
    parser.add_argument('--episode-length', default=200, type=int, help='maximum number of steps per episode')
    parser.add_argument('--seed', type=int, default=7, help='random seed for initialization')

    args = parser.parse_args()
    load_model = args.load
    repeat_action = args.repeat_action
    steps_per_ep = args.episode_length
    ep_length = args.episode_length
    seed = args.seed

    main(load_model,steps_per_ep, repeat_action)