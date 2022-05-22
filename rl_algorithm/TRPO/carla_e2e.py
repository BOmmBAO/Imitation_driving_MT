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

    def __init__(self, step_per_ep,repeat_action):
        argparser = argparse.ArgumentParser(description='Carla ArgParser practice')
        argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
        argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to')
        argparser.add_argument('--repeat-action',default=5, type=int, help='number of steps to repeat each action')
        #argparser.add_argument('--sync', action='store_true', help='Synchronous mode execution')
        args = argparser.parse_args()
        self.steps = 0
        self.total_steps = 200
        self.decision = False
        self.step_per_ep = step_per_ep
        self.repeat_action = repeat_action
        self.sim = SimInit(args)
        self.car = VehicleInit(self.sim, self.decision)
        self.sigma = {"sigma_pos": 0.3, "sigma_vel_upper": 0.6,
                       "sigma_vel_lower": 1.0, "sigma_yaw": 0.4}

        obs_shape = len(self.car.fea_ext.observation)
        # if self.action_type == 'continuous':
        obs_high = np.array([np.inf] * obs_shape)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype='float32')
        #act_high = np.array([1., 1.]) # thre, steer: [-1, 1]
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype='float32')
        self.obs_index = self.car.fea_ext.obs_index

        self.fea_plot = FeaPlot(self.car.fea_ext)
        #added from modulardecision
        self._control = carla.VehicleControl()

    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed)
        return seed


    def step(self, action):
        if self.steps <
        self.steps +=1
        total_reward = 0
        for _ in range(self.repeat_action):

            self.car.step_action(action)
            self.sim.update()
            ob = self._get_obs()
            total_reward += self._get_reward(self.sigma)
            done = True if self.sim.term_check() else False
            if done:
                break
        print("Action:", action)
        print("Reward:", total_reward)
        print("DONE?", done)
        return ob, total_reward, done, {}

    #reset environment for new episode
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
#TODO: add overtaking reward
    def _get_reward(self, sigmas, vel_des=12):
        car_x, car_y, car_v, car_yaw = self.car.fea_ext.vehicle_info.x, self.car.fea_ext.vehicle_info.y, \
                                self.car.fea_ext.vehicle_info.v, self.car.fea_ext.vehicle_info.yaw
        lane_width = self.car.fea_ext.cur_lane_width/2
        print(car_v)

        nearest_point = self.sim._map.get_waypoint(location=self.car.fea_ext.vehicle_info._location, project_to_road=True).transform
        car_point = self.sim._map.get_waypoint(location=self.car.fea_ext.vehicle_info._location, project_to_road=False).transform

        # sigma_pos = 0.3
        sigma_pos = sigmas["sigma_pos"]
        # phi = math.atan2(car_y - nearest_point[1], car_x - nearest_point[0])
        # delta = pi_2_pi(ryaw[0] - phi)
        # ct_err = math.sin(delta) * nearest_dist  # Cross Track Error
        ct_err = np.sqrt((car_point.location.x-nearest_point.location.x)**2+(car_point.location.y-nearest_point.location.y)**2)
        track_err = ct_err / lane_width
        track_rewd = np.exp(-track_err**2 / (2 * sigma_pos**2))

        # velocity reward
        sigma_vel_upper, sigma_vel_lower = sigmas["sigma_vel_upper"], sigmas["sigma_vel_lower"]
        sigma_vel = sigma_vel_upper if car_v <= vel_des else sigma_vel_lower
        v_err = car_v - vel_des
        v_rewd = np.exp(-v_err**2 / (2*sigma_vel**2))

        # angle reward
        sigma_yaw = sigmas["sigma_yaw"]
        yaw_err = abs(pi_2_pi(nearest_point.rotation.yaw-car_yaw))
        ang_rewd = np.exp(-yaw_err**2 / (2 * sigma_yaw ** 2))

        accident_cost =0
        if self.sim.collision_event:
            accident_cost -= 10
            self.sim.collision_sensor.reset()
        if len(self.sim.laneinvasion_sensor.get_history()) != 0:
            accident_cost -= 0.5
            self.sim.laneinvasion_sensor.reset()
        reward = track_rewd * v_rewd * ang_rewd + accident_cost
        #self.ep_len +=1
        #self.last_status = self.status

        return reward







