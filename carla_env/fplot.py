import carla
import utils.common
import math
import numpy as np
from utils import common
import matplotlib.pyplot as plt
plt.ion()
from carla_env.feature import *
from carla_env.sim_vehicle import *

from matplotlib.pyplot import MultipleLocator


class FeaPlot():

    def __init__(self, obs_index):
        self.obs_index = obs_index
        self.t = 0

    def print_info(self, obs):
        _index = self.obs_index['ego_car_vel']
        ego_car_v = obs[_index[0]: _index[1]]
        print("Ego Velocity:", ego_car_v)
        _index = self.obs_index['ego_car_acc']
        ego_car_acc = obs[_index[0]: _index[1]]

        _index = self.obs_index['zombie_cars_pos']
        zombie_pos = obs[_index[0]: _index[1]]
        print("Zombie Position:", zombie_pos)
        _index = self.obs_index['zombie_cars_v']
        zombie_v = obs[_index[0]: _index[1]]
        print("Zombie Velocity:", zombie_v)

    def plot_map(self, obs):
        obs_lane = []
        lines = ['inner_line_right', 'inner_line_left', 'outer_line_right', 'outer_line_left']
        for line_name in lines:
            _index = self.obs_index[line_name]
            obs_lane.extend(obs[_index[0]: _index[1]])

        _index = self.obs_index['zombie_cars_pos']
        obs_zombie = obs[_index[0]: _index[1]]

        plt.clf()
        fig = plt.figure(1)
        ax2 = fig.gca()

        my_ticks = np.arange(-70, 75, 5)

        plt.xticks(my_ticks)
        plt.yticks(my_ticks)
        plt.xlim((-70, 75))
        plt.ylim((-70, 75))

        for p in range(int(len(obs_lane) / 2)):
            plt.plot(obs_lane[p * 2 + 1], obs_lane[p * 2], '.', color='red', label='lane')

        for q in range(int(len(obs_zombie) / 2)):
            plt.plot(obs_zombie[q * 2 + 1], obs_zombie[q * 2], '.', color='green', label='zombie_car')

        plt.draw()
        plt.pause(0.0001)
