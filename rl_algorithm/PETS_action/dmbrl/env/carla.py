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
from carla_env.sim_carla_copy import SimInit
from carla_env.sim_vehicle import VehicleInit
from utils.common import *
from carla_env.fplot import FeaPlot
from carla_env.feature import STATUS


class CarlaEnv(gym.Env):

    def __init__(self):
        argparser = argparse.ArgumentParser(description='Carla ArgParser practice')
        argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
        argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to')
        #argparser.add_argument('--sync', action='store_true', help='Synchronous mode execution')
        args = argparser.parse_args()

        self.sim = SimInit(args)
        self.car = VehicleInit(self.sim)
        self.sigma = {"sigma_pos": 1.3, "sigma_vel_upper": 0.6,
                       "sigma_vel_lower": 1.0, "sigma_yaw": 1.4}

        obs_shape = len(self.car.fea_ext.observation)
        obs_high = np.array([np.inf] * obs_shape)
        self.observation_space = spaces.Box(-obs_high, obs_high)
        # self.observation_space = spaces.Discrete(8) # discrete decision
        act_high = np.array([1, 1]) # thre, steer: [-1, 1]
        self.action_space = gym.spaces.Box(-act_high, act_high)
        self.obs_index = self.car.fea_ext.obs_index

        self.fea_plot = FeaPlot(self.car.fea_ext)

    def step(self, action):
        self.car.step_action(action)
        self.sim.update()
        ob = self._get_obs()
        reward = self._get_reward(self.sigma)
        done = True if self.sim.term_check() else False
        print("Action:", action)
        print("Reward:", reward)
        print("DONE?", done)
        return ob, reward, done, {}

    def reset(self):
        print("Env Reset!")
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
        return self.car.fea_ext.observation

    def _plot_zombie_boxx(self, obs):
        self.fea_plot.plot_lane_andZombie_inEgoCar(obs)

    def _get_reward(self, sigmas):
        car_x, car_y, car_v, car_yaw = self.car.fea_ext.vehicle_info.x, self.car.fea_ext.vehicle_info.y, \
                                self.car.fea_ext.vehicle_info.v, self.car.fea_ext.vehicle_info.yaw
        if self.car.reference is None:
            return 0
        [rx, ry, ryaw, vel_des] = self.car.reference
        lane_width = self.car.fea_ext.cur_lane_width

        nearest_point, nearest_dist = None, 10
        for [x, y, yaw] in zip(rx, ry, ryaw):
            _dist = np.hypot(car_x-x, car_y-y)
            if _dist < nearest_dist:
                nearest_point = [x, y, yaw]
                nearest_dist = _dist

        # sigma_pos = 0.3
        sigma_pos = sigmas["sigma_pos"]
        phi = math.atan2(car_y - nearest_point[1], car_x - nearest_point[0])
        delta = pi_2_pi(ryaw[0] - phi)
        ct_err = math.sin(delta) * nearest_dist  # Cross Track Error
        track_err = abs(ct_err) / lane_width
        track_rewd = np.exp(-track_err**2 / (2 * sigma_pos**2))

        # velocity reward
        sigma_vel_upper, sigma_vel_lower = sigmas["sigma_vel_upper"], sigmas["sigma_vel_lower"]
        sigma_vel = sigma_vel_upper if car_v <= vel_des else sigma_vel_lower
        v_err = car_v - vel_des
        v_rewd = np.exp(-v_err**2 / (2*sigma_vel**2))

        # angle reward
        sigma_yaw = sigmas["sigma_yaw"]
        yaw_err = abs(pi_2_pi(nearest_point[2]-car_yaw))
        ang_rewd = np.exp(-yaw_err**2 / (2 * sigma_yaw ** 2))

        accident_cost = -10 if self.sim.collision_event or self.sim.invasion_event else 0
        reward = track_rewd * v_rewd * ang_rewd + accident_cost

        return reward







