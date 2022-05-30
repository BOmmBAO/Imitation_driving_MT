
from __future__ import division
import copy
import numpy as np
import random
import time
from collections import deque

import gym
from gym import spaces
from carla_env.modules import CollisionSensor, LaneInvasionSensor

from carla_env.coordinates import train_coordinates
from carla_env.misc import _vec_decompose, delta_angle_between
from carla_env.carla_logger import *
from carla_env.feature import *
from carla_env.fplot import FeaPlot



import gym
from gym import spaces
import random
import math
import carla
import argparse
from carla_env.sim_carla import SimInit
from carla_env.sim_vehicle import VehicleInit
from carla_env.common import *
from carla_env.fplot import FeaPlot
import tensorflow as tf
from absl import logging
from carla_env.feature import STATUS


# LEVEL = {DEBUG, INFO, WARN, ERROR, FATAL}
logging.set_verbosity(logging.ERROR)
class CarlaEnv(gym.Env):

    def __init__(self, step_per_ep, target_v):
        argparser = argparse.ArgumentParser(description='Carla ArgParser practice')
        argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
        argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to')
        #argparser.add_argument('--repeat-action',default=5, type=int, help='number of steps to repeat each action')
        #argparser.add_argument('--sync', action='store_true', help='Synchronous mode execution')
        args = argparser.parse_args()
        self.desired_speed = target_v
        self.steps = 0
        self.total_steps = 200
        self.decision = False
        self.step_per_ep = step_per_ep
        self.sim = SimInit(args, self.desired_speed)
        self.current_wpt = np.array((self.sim.current_wpt.transform.location.x, self.sim.current_wpt.transform.location.y, self.sim.current_wpt.transform.rotation.yaw))
        self.car = VehicleInit(self.sim, self.decision)
        self.sigmas = {"sigma_pos": 0.03, "sigma_vel_upper": 0.16,
                       "sigma_vel_lower": 0.5, "sigma_yaw": 0.14}

        obs_shape = len(self.car.fea_ext.observation)
        # if self.action_type == 'continuous':
        obs_high = np.array([np.inf] * obs_shape)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype='float32')
        #act_high = np.array([1., 1.]) # thre, steer: [-1, 1]
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype='float32')
        self.obs_index = self.car.fea_ext.obs_index

        self.fea_plot = FeaPlot(self.car.fea_ext)
        #added from modulardecision

    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed)
        return seed


    def step(self, action):
        reward = 0
        if self.steps < self.total_steps:
            self.steps +=1
            self.car.step_action(action)
            self.sim.update()
            done, reward = self._get_reward()
        else:
            done = True
            print('Time out! Eps cost %d steps:', self.step_per_ep)
        ob = self._get_obs()
        print("Action:", action)
        print("Reward:", reward)
        print("DONE?", done)
        print(len(ob))
        return ob, reward, done, {}

    #reset environment for new episode
    def reset(self):
        print("Env Reset!")
        self.steps = 0
        self.sim.reset()
        self.car.reset(self.sim)
        self.sim.update()
        ob = self._get_obs()
        return ob

    # def test_step(self):
    #     self.sim.update()
    #     self.car.rule_based_step()
    #     ob = self._get_obs()
    #     reward = self._get_reward(self.sigma)
    #     done = False if self.sim.terminal_check() else True
    #     return ob, reward, done, {}

    def _get_obs(self):
        return self.car.fea_ext.observation

    def _plot_zombie_boxx(self, obs):
        self.fea_plot.plot_map(obs)

#TODO: add overtaking reward
    def _get_reward(self, vel_des=12):
        done, reward = self.sim.terminal_check()
        if done:
            return done, reward

        car_x, car_y, v_norm, car_yaw = self.car.fea_ext.vehicle_info.x, self.car.fea_ext.vehicle_info.y, \
                                self.car.fea_ext.vehicle_info.v, self.car.fea_ext.vehicle_info.yaw
        #lane_width = self.car.fea_ext.cur_lane_width/2
        print(v_norm)

        # sigma_pos = 0.3
        sigma_pos = self.sigmas["sigma_pos"]
        delta_yaw, wpt_yaw = self._get_delta_yaw()
        road_heading = np.array([
            np.cos(wpt_yaw / 180 * np.pi),
            np.sin(wpt_yaw / 180 * np.pi)
        ])
        pos_err_vec = np.array((car_x, car_y)) - self.current_wpt[0:2]
        lateral_dist = np.linalg.norm(pos_err_vec) * np.sign(pos_err_vec[0] * road_heading[1] - pos_err_vec[1] * road_heading[0])
        track_rewd = np.exp(-lateral_dist**2 / (2 * (sigma_pos**2)))
        print("process", track_rewd)

        # velocity reward
        sigma_vel_upper, sigma_vel_lower = self.sigmas["sigma_vel_upper"], self.sigmas["sigma_vel_lower"]
        sigma_vel = sigma_vel_upper if v_norm <= self.desired_speed else sigma_vel_lower
        delta_speed = v_norm - self.desired_speed
        v_rewd = np.exp(-delta_speed ** 2 / (2 * (sigma_vel ** 2)))
        print("process", delta_speed)

        # angle reward
        sigma_yaw = self.sigmas["sigma_yaw"]
        yaw_err = delta_yaw * np.pi / 180
        ang_rewd = np.exp(-yaw_err**2 / (2 / (sigma_yaw ** 2)))
        print("process", ang_rewd)

        if abs(lateral_dist) > 1.2:
            done = True
            reward = -100
            return done, reward

        reward = track_rewd * v_rewd * ang_rewd
        #self.ep_len +=1
        #self.last_status = self.status

        return done, reward


    def _get_delta_yaw(self):
        """
        calculate the delta yaw between ego and current waypoint
        """
        current_wpt = self.sim._map.get_waypoint(location=self.car.fea_ext.vehicle_info._location)
        if not current_wpt:
            self.logger.error('Fail to find a waypoint')
            wpt_yaw = self.current_wpt[2] % 360
        else:
            wpt_yaw = current_wpt.transform.rotation.yaw % 360
            self.current_wpt = np.array((current_wpt.transform.location.x, current_wpt.transform.location.y, current_wpt.transform.rotation.yaw))
        ego_yaw = self.car.fea_ext.vehicle_info.yaw % 360

        delta_yaw = ego_yaw - wpt_yaw
        if 180 <= delta_yaw and delta_yaw <= 360:
            delta_yaw -= 360
        elif -360 <= delta_yaw and delta_yaw <= -180:
            delta_yaw += 360

        return delta_yaw, wpt_yaw




