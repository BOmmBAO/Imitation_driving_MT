import os
import time
import numpy as np
from datetime import datetime
from typing import Any, Dict
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
from carla_env.fplot import Plot_features
from carla_env.feature import STATUS


class RotatedRectangle(object):
    """
    This class contains method to draw rectangle and find intersection point.
    """

    def __init__(self, c_x, c_y, width, height, angle):
        self.c_x = c_x
        self.c_y = c_y
        self.w = width  # pylint: disable=invalid-name
        self.h = height  # pylint: disable=invalid-name
        self.angle = angle

    def get_contour(self):
        """
        create contour
        """
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.c_x, self.c_y)

    def intersection(self, other):
        """
        Obtain a intersection point between two contour.
        """
        return self.get_contour().intersection(other.get_contour())

def detect_lane_obstacle(world, actor, extension_factor=3, margin=1.02):
    """
    This function identifies if an obstacle is present in front of the reference actor
    """
    # world = CarlaDataProvider.get_world()
    world_actors = world.get_actors().filter('vehicle.*')
    actor_bbox = actor.bounding_box
    actor_transform = actor.get_transform()
    actor_location = actor_transform.location
    actor_vector = actor_transform.rotation.get_forward_vector()
    actor_vector = np.array([actor_vector.x, actor_vector.y])
    actor_vector = actor_vector / np.linalg.norm(actor_vector)
    actor_vector = actor_vector * (extension_factor - 1) * actor_bbox.extent.x
    actor_location = actor_location + carla.Location(actor_vector[0], actor_vector[1])
    actor_yaw = actor_transform.rotation.yaw

    is_hazard = False
    for adversary in world_actors:
        if adversary.id != actor.id and \
                actor_transform.location.distance(adversary.get_location()) < 50:
            adversary_bbox = adversary.bounding_box
            adversary_transform = adversary.get_transform()
            adversary_loc = adversary_transform.location
            adversary_yaw = adversary_transform.rotation.yaw
            overlap_adversary = RotatedRectangle(
                adversary_loc.x, adversary_loc.y,
                2 * margin * adversary_bbox.extent.x, 2 * margin * adversary_bbox.extent.y, adversary_yaw)
            overlap_actor = RotatedRectangle(
                actor_location.x, actor_location.y,
                2 * margin * actor_bbox.extent.x * extension_factor, 2 * margin * actor_bbox.extent.y, actor_yaw)
            overlap_area = overlap_adversary.intersection(overlap_actor).area
            if overlap_area > 0:
                is_hazard = True
                break

    return is_hazard


#EnvType as inpt
class ScenarioEnv(gym.Env):
    """
        The observation is setted in feature. The reward is from rl_algorithm:PETS
        When created, it will initialize environment with config and Carla TCP host & port. This method will NOT create
        the simulator instance. It only creates some data structures to store information when running env.
        :Arguments:
            - cfg (Dict): Env config dict.
            - host (str, optional): Carla server IP host. Defaults to 'localhost'.
            - port (int, optional): Carla server IP port. Defaults to 9000.
            - tm_port (Optional[int], optional): Carla Traffic Manager port. Defaults to None.
        :Interfaces: reset, step, close, is_success, is_failure, render, seed
        :Properties:
            - hero_player (carla.Actor): Hero vehicle in simulator.
        """
    action_space = spaces.Dict({})
    observation_space = spaces.Dict({})

    def __init__(self, scenario_name = 'Cross_Join'):
        argparser = argparse.ArgumentParser(description='Carla ArgParser practice')
        argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
        argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to')
        args = argparser.parse_args()

        self.sim = SimInit(args)
        self.car = VehicleInit(self.sim)
        self.sigma = {"sigma_pos": 0.3, "sigma_vel_upper": 0.6,
                      "sigma_vel_lower": 1.0, "sigma_yaw": 0.4}

        obs_shape = len(self.car.fea_ext.observation)
        obs_high = np.array([np.inf] * obs_shape)
        self.observation_space = spaces.Box(-obs_high, obs_high)
        # self.observation_space = spaces.Discrete(8) # discrete decision
        act_high = np.array([1, 1])  # thre, steer: [-1, 1]
        self.action_space = gym.spaces.Box(-act_high, act_high)
        self.obs_index = self.car.fea_ext.obs_index


    def _init_carla_simulator(self) -> None:
        print()




