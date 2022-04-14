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
from carla_env.fplot import Plot_features


class CarlaEnv(gym.Env):

    def __init__(self, VERSION = '0.9.11'):

        args = self.default_args()
        self.sim = SimInit(args)
        self.car = VehicleInit(self.sim)
        self.sigma = {"sigma_pos": 0.3, "sigma_vel_upper": 0.6,
                       "sigma_vel_lower": 1.0, "sigma_yaw": 0.4}

        obs_shape = len(self.car.fea_ext.observation)
        obs_high = np.array([np.inf] * obs_shape)
        self.observation_space = spaces.Box(-obs_high, obs_high)
        self.obs_index = self.car.fea_ext.obs_index

        self.action_space = spaces.Discrete(5)  # discrete decision
        self.fea_plot = Plot_features(self.car.fea_ext)

    def default_args(self):
        description = ("CARLA Scenario Runner: Setup, Run and Evaluate scenarios using CARLA\n"
                       "Current version: 0.9.11")
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + "0.9.11")
        parser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
        parser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to')
        ##scenario##
        parser.add_argument('--timeout', default="10.0",
                            help='Set the CARLA client timeout value in seconds')
        parser.add_argument('--trafficManagerPort', default='8000',
                            help='Port to use for the TrafficManager (default: 8000)')
        parser.add_argument('--trafficManagerSeed', default='0',
                            help='Seed used by the TrafficManager (default: 0)')
        parser.add_argument('--sync', action='store_true',
                            help='Forces the simulation to run synchronously')
        parser.add_argument('--list', action="store_true", help='List all supported scenarios and exit')

        parser.add_argument(
            '--scenario',
            help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle')
        parser.add_argument('--openscenario', help='Provide an OpenSCENARIO definition')
        parser.add_argument(
            '--route', help='Run a route as a scenario (input: (route_file,scenario_file,[route id]))', nargs='+',
            type=str)
        #Agent used to execute the scenario. Currently only compatible with route-based scenarios
        # parser.add_argument(
        #     '--agent', help="Agent used to execute the scenario. Currently only compatible with route-based scenarios")
        # parser.add_argument('--agentConfig', type=str, help="Path to Agent's configuration file", default="")

        # parser.add_argument('--output', action="store_true", help='Provide results on stdout')
        # parser.add_argument('--file', action="store_true", help='Write results into a txt file')
        # parser.add_argument('--junit', action="store_true", help='Write results into a junit file')
        # parser.add_argument('--json', action="store_true", help='Write results into a JSON file')
        # parser.add_argument('--outputDir', default='', help='Directory for output files (default: this directory)')
        #
        # parser.add_argument('--configFile', default='',
        #                     help='Provide an additional scenario configuration file (*.xml)')
        # parser.add_argument('--additionalScenario', default='',
        #                     help='Provide additional scenario implementations (*.py)')
        #
        # parser.add_argument('--debug', action="store_true", help='Run with debug output')
        # parser.add_argument('--reloadWorld', action="store_true",
        #                     help='Reload the CARLA world before starting a scenario (default=True)')
        # parser.add_argument('--record', type=str, default='',
        #                     help='Path were the files will be saved, relative to SCENARIO_RUNNER_ROOT.\nActivates the CARLA recording feature and saves to file all the criteria information.')
        # parser.add_argument('--randomize', action="store_true", help='Scenario parameters are randomized')
        # parser.add_argument('--repetitions', default=1, type=int, help='Number of scenario executions')
        # parser.add_argument('--waitForEgo', action="store_true", help='Connect the scenario to an existing ego vehicle')

        args = parser.parse_args()
        return args

    def step(self, decision):
        self.car.step_decision(decision)
        self.sim.update()
        ob = self._get_obs()
        reward = self._get_reward(self.sigma)
        done = True if self.sim.term_check() else False
        print("Decision:", decision)
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
        # sigma_pos = sigmas["sigma_pos"]
        # phi = math.atan2(car_y - nearest_point[1], car_x - nearest_point[0])
        # delta = pi_2_pi(ryaw[0] - phi)
        # ct_err = math.sin(delta) * nearest_dist  # Cross Track Error
        # track_err = abs(ct_err) / lane_width
        # track_rewd = np.exp(-track_err**2 / (2 * sigma_pos**2))

        # velocity reward
        sigma_vel_upper, sigma_vel_lower = sigmas["sigma_vel_upper"], sigmas["sigma_vel_lower"]
        sigma_vel = sigma_vel_upper if car_v <= vel_des else sigma_vel_lower
        v_err = car_v - vel_des
        v_rewd = np.exp(-v_err**2 / (2*sigma_vel**2))

        # angle reward
        # sigma_yaw = sigmas["sigma_yaw"]
        # yaw_err = abs(pi_2_pi(nearest_point[2]-car_yaw))
        # ang_rewd = np.exp(-yaw_err**2 / (2 * sigma_yaw ** 2))

        accident_cost = -10 if self.sim.collision_event else 0
        reward = v_rewd + accident_cost

        return reward







