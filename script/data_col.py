
import numpy as np
import sys
import os
from rl_algorithm.PETS_decision.dmbrl.env.carla import CarlaEnv
from carla_env.fplot import FeaPlot
import matplotlib.pyplot as plt
from utils.common import *
import pandas as pd
from numpy import genfromtxt


def save_data(obs, dec, new_file_flag):
    path = '../Data_Set/'
    files = os.listdir(path)
    if new_file_flag:
        file_num = int(len(files) / 2)
    else:
        file_num = int(len(files) / 2) - 1

    obs_name = path + 'obs_' + str(file_num) + ".csv"
    with open(obs_name, "ab") as f:
        f.write(b"\n")
        np.savetxt(f, np.array(obs).reshape((1, -1)), fmt='%.2f', delimiter=",")

    dec_name = path + 'dec_' + str(file_num) + ".csv"
    with open(dec_name, "ab") as f:
        if new_file_flag is not True:
            f.write(b"\n")
            np.savetxt(f, np.array(dec).reshape((1, -1)), fmt='%d', delimiter=",")


def read_file(file_name):
    path = '../Data_Set/'
    full_name = path + file_name + ".csv"
    data = genfromtxt(full_name, delimiter=',')
    return data


def read_set(file_name):
    path = '../Data_Set'
    files = os.listdir(path)
    file_num = int(len(files)/2)

    obs_list, act_list = [], []
    for n in range(file_num):
        obs_name = path + 'obs_' + str(n) + ".csv"
        _data = genfromtxt(obs_name, delimiter=',')
        obs_list.append(_data)

        dec_name = path + 'dec_' + str(n) + ".csv"
        _data = genfromtxt(dec_name, delimiter=',')
        obs_list.append(_data)


class DataCollection:

    def __init__(self):
        self.dec_num = 5
        self.horizon = 200
        self.n_iter = 300
        self.env = CarlaEnv()

    def sample(self):
        ob = self.env.reset()
        save_data(ob, 0, new_file_flag=True)
        for t in range(self.horizon):
            # Generate decision randomly
            _init_prob = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
            dec = np.array([np.random.choice(self.dec_num, replace=True, p=_init_prob)])
            ob, reward, done, info = self.env.step(dec)
            if done:
                print("Collect " + str(t) + " data of rollout!")
                break
            save_data(ob, dec, new_file_flag=False)

    def collect(self):
        t = 0
        while t < self.n_iter:
            print("-------------" + str(t) + "Iteration ---------------")
            self.sample()
            t += 1


if __name__ == '__main__':

    data_col = DataCollection()
    data_col.collect()


