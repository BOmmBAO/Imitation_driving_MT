import gym
import numpy as np
from trpo_mpi import TRPO
from rl_algorithm.common.policies import MlpPolicy
from rl_algorithm.common.noise import NormalActionNoise
from carla_e2e import CarlaEnv
import sys
import argparse

def main(model_name):
    env = CarlaEnv()

    try:
        model = TRPO.load(model_name)

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            print(reward, info)
            if done:
                obs = env.reset()
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', default='TRPO_e2e', help='name of model when saving')

    args = parser.parse_args()
    model_name = args.model_name

    main(model_name)