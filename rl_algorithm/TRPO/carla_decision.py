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
from carla_env.common import *
from carla_env.feature import *



class CarlaEnv(gym.Env):

    def __init__(self, step_per_ep, target_v): #lanes_change=5):
        argparser = argparse.ArgumentParser(description='Carla ArgParser practice')
        argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
        argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to')
        args = argparser.parse_args()
        self.desired_speed = target_v
        self.steps = 0
        self.total_steps = step_per_ep
        self.decision = True
        self.sim = SimInit(args, self.desired_speed)
        self.current_wpt = np.array((self.sim.current_wpt.transform.location.x,
                                     self.sim.current_wpt.transform.location.y,
                                     self.sim.current_wpt.transform.rotation.yaw))
        self.car = VehicleInit(self.sim, self.decision)
        self.sigmas = {"sigma_pos": 0.3, "sigma_vel_upper": 1.0,
                       "sigma_vel_lower": 0.6, "sigma_yaw": 0.4}

        self.car_fea_ext = self.car.ego_car_config.fea_ext
        obs_shape = len(self.car_fea_ext.observation)
        obs_high = np.array([np.inf] * obs_shape)
        self.observation_space = spaces.Box(-obs_high, obs_high)
        self.obs_index = self.car_fea_ext.obs_index
        self.pre_obs_index = self.car_fea_ext.pre_obs_index
        self.zombie_num = self.car_fea_ext.zombie_num
        #self.action_space = spaces.MultiDiscrete([lanes_change, 7, 5])
        self.action_space = spaces.Discrete(5)
    def step(self, decision):
        self.decision = decision
        if self.steps < self.total_steps:

            total_step = 5 if decision == 0 else 3
            step_num = 0
            done = False
            reward = 0
            print("Step Decision:", decision)
            while step_num < total_step or self.car_fea_ext.vehicle_info.status != STATUS.FOLLOWING:
                fasle_dec = self.car.step_decision(decision)
                self.car_fea_ext = self.car.ego_car_config.fea_ext
                #self.zombie_num = self.car_fea_ext.zombie_num
                self.sim.update()
                step_num += 1
                done, reward = self._get_reward()

                if done:
                    break
        else:
            done = True
            print('Time out! Eps cost %d steps:', self.step_per_ep)
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


    def _get_obs(self):
        return self.car_fea_ext.observation

    def _get_reward(self, vel_des=12):
        done, reward = self.sim.terminal_check()
        if done:
            return done, reward

        car_x, car_y, v_norm, car_yaw = self.car.fea_ext.vehicle_info.x, self.car.fea_ext.vehicle_info.y, \
                                        self.car.fea_ext.vehicle_info.v, self.car.fea_ext.vehicle_info.yaw
        lane_width = self.car.fea_ext.cur_lane_width / 2
        print(v_norm)

        # sigma_pos = 0.3
        sigma_pos = self.sigmas["sigma_pos"]
        delta_yaw, wpt_yaw = self._get_delta_yaw()
        road_heading = np.array([
            np.cos(wpt_yaw / 180 * np.pi),
            np.sin(wpt_yaw / 180 * np.pi)
        ])
        pos_err_vec = np.array((car_x, car_y)) - self.current_wpt[0:2]
        lateral_dist = np.linalg.norm(pos_err_vec) * np.sign(
            pos_err_vec[0] * road_heading[1] - pos_err_vec[1] * road_heading[0]) / lane_width
        track_rewd = np.exp(-np.power(lateral_dist, 2) / 2 / sigma_pos / sigma_pos)
        print("process", track_rewd)

        # velocity reward
        sigma_vel_upper, sigma_vel_lower = self.sigmas["sigma_vel_upper"], self.sigmas["sigma_vel_lower"]
        sigma_vel = sigma_vel_upper if v_norm <= self.desired_speed else sigma_vel_lower
        delta_speed = v_norm - self.desired_speed
        v_rewd = np.exp(-np.power(delta_speed, 2) / 2 / sigma_vel / sigma_vel)
        print("process", delta_speed)

        # angle reward
        sigma_yaw = self.sigmas["sigma_yaw"]
        yaw_err = delta_yaw * np.pi / 180
        ang_rewd = np.exp(-np.power(yaw_err, 2) / 2 / sigma_yaw / sigma_yaw)

        print("process", ang_rewd)

        if abs(lateral_dist) > 1.25:
            done = True
            reward = -10
            return done, reward

        reward = track_rewd * v_rewd * ang_rewd
        # self.ep_len +=1
        # self.last_status = self.status

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

