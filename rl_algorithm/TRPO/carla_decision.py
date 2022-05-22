import gym
from gym import spaces
import math
import carla
import argparse

try:
    import numpy as np
    import sys
    from os import path as osp
except ImportError:
    raise RuntimeError('import error!')
from carla_env.sim_carla import SimInit
from carla_env.sim_vehicle import VehicleInit
from utils.common import *
from carla_env.feature import *



class CarlaEnv(gym.Env):

    def __init__(self):
        argparser = argparse.ArgumentParser(description='Carla ArgParser practice')
        argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
        argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to')
        args = argparser.parse_args()

        self.sim = SimInit(args)
        self.car = VehicleInit(self.sim)
        self.sigma = {"sigma_pos": 0.3, "sigma_vel_upper": 0.6,
                       "sigma_vel_lower": 1.0, "sigma_yaw": 0.4}

        self.car_fea_ext = self.car.ego_car_config.fea_ext
        obs_shape = len(self.car_fea_ext.observation)
        obs_high = np.array([np.inf] * obs_shape)
        self.observation_space = spaces.Box(-obs_high, obs_high)
        self.obs_index = self.car_fea_ext.obs_index
        self.pre_obs_index = self.car_fea_ext.pre_obs_index
        self.zombie_num = self.car_fea_ext.zombie_num
        self.action_space = spaces.Discrete(5)

    def step(self, decision):
        total_step = 30 if decision == 0 else 3
        step_num = 0
        done = False
        reward = 0
        print("Step Decision:", decision)
        while step_num < total_step or self.car_fea_ext.vehicle_info.status != STATUS.FOLLOWING:
            fasle_dec = self.car.step_decision(decision)
            self.car_fea_ext = self.car.ego_car_config.fea_ext
            self.zombie_num = self.car_fea_ext.zombie_num
            self.sim.update()
            step_num += 1
            done = True if self.sim.term_check() is True or fasle_dec is True else False
            reward = self._get_reward(self.sigma)

            if done:
                break
        ob = self._get_obs()
        print("Reward:", reward)
        print("DONE?", done)
        print("Time cost:", step_num * 0.1)
        return ob, reward, done, {}

    def reset(self):
        self.sim.reset()
        self.car.reset(self.sim)
        self.sim.update()
        ob = self._get_obs()
        return ob

    def test_step(self):
        self.sim.update()
        self.car.rule_based_step()
        ob = self._get_obs()
        reward = self._get_reward(self.sigma)
        done = False if self.sim.term_check() else True
        return ob, reward, done, {}

    def _get_obs(self):
        return self.car_fea_ext.observation

    def _get_reward(self, sigmas):
        [rx, ry, ryaw, vel_des] = self.car.reference
        car_v = self.car_fea_ext.vehicle_info.v

        # velocity reward
        sigma_vel_upper, sigma_vel_lower = sigmas["sigma_vel_upper"], sigmas["sigma_vel_lower"]
        sigma_vel = sigma_vel_upper if car_v <= vel_des else sigma_vel_lower
        v_err = car_v - vel_des
        v_rewd = np.exp(-v_err**2 / (2*sigma_vel**2))

        accident_cost = -10 if self.sim.collision_event is True else 0
        reward = v_rewd + accident_cost

        return reward
