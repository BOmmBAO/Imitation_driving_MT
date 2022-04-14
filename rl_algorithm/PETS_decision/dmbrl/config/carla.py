from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import gym
from rl_algorithm.PETS_decision.dmbrl.env.carla import CarlaEnv
import math
import numpy as np
import tensorflow as tf
from utils.common import *
from dotmap import DotMap
import matplotlib.pyplot as plt

from rl_algorithm.PETS_decision.dmbrl.misc.DotmapUtils import get_required_argument
from rl_algorithm.PETS_decision.dmbrl.modeling.layers import FC


class CarlaConfigModule:
    ENV_NAME = "MBRL-Carla-v0"
    TASK_HORIZON = 200
    NTRAIN_ITERS = 200
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 3
    MODEL_IN, MODEL_OUT = 142, 29
    GP_NINDUCING_POINTS = 50

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 100
            },
            "CEM": {
                "popsize": 20,
                "num_elites": 8,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        return obs[:, 3:]

    @staticmethod
    def obs_postproc(obs, pred):
        new_obs = tf.add(obs[:, 0:29], pred)
        return tf.concat([new_obs, obs[:, 29:]], axis=1)

    @staticmethod
    def obs_postproc2(pre_obs_samples, obs_index):

        def old2new_localFrame(pre_obs, obs_index):

            _index = obs_index['ego_car_local_trans']
            local_trans = pre_obs[_index[0]: _index[1]]

            # position transformation
            trans_1 = [-local_trans[0], -local_trans[1], 0]
            new_ego_pos = CarlaConfigModule._rotate(CarlaConfigModule._transform(local_trans[0:2], trans_1),
                                                    local_trans[2])
            new_ego_trans = [new_ego_pos[0], new_ego_pos[1], 0]

            lane_pos, zombie_pos = [], []
            lines = ['inner_line_right', 'inner_line_left', 'outer_line_right', 'outer_line_left']

            for line_name in lines:
                _index = obs_index[line_name]
                lane_pos.extend(pre_obs[_index[0]: _index[1]])
            lane_pos = np.array(lane_pos).reshape((-1, 2))
            _index = obs_index["zombie_cars_pos"]
            zombie_pos = np.array(pre_obs[_index[0]: _index[1]]).reshape((-1, 2))

            new_lane_pos, new_zombie_pos = [], []
            for pos in lane_pos:
                new_lane_pos.append(CarlaConfigModule._rotate(CarlaConfigModule._transform(pos, trans_1),
                                                              local_trans[2]))
            for pos in zombie_pos:
                new_zombie_pos.append(CarlaConfigModule._rotate(CarlaConfigModule._transform(pos, trans_1),
                                                                local_trans[2]))

            # velocity transformation
            _index = obs_index['ego_car_vel']
            ego_vel = pre_obs[_index[0]: _index[1]]
            new_ego_vel = CarlaConfigModule._rotate(ego_vel, local_trans[2])

            _index = obs_index['zombie_cars_v']
            zombie_vel = np.array(pre_obs[_index[0]: _index[1]]).reshape((-1, 2))

            new_zombie_vel = []
            for v in zombie_vel:
                new_zombie_vel.append(CarlaConfigModule._rotate(v, local_trans[2]))

            obs_new_frame = new_ego_trans + new_ego_vel + CarlaConfigModule._flat_list(new_zombie_pos) + \
                            CarlaConfigModule._flat_list(new_zombie_vel) + CarlaConfigModule._flat_list(new_lane_pos)
            return obs_new_frame

        trans_obs = []
        for obs_sample in pre_obs_samples:
            trans_obs.append(old2new_localFrame(obs_sample, obs_index))

        return trans_obs


    @staticmethod
    def targ_proc(obs, next_obs, obs_index, zombie_num):

        def new2old_localFrame(old_obs, new_obs, obs_index):
            _index = obs_index['ego_car_world_trans']
            new_world_trans = new_obs[_index[0]: _index[1]]
            old_world_trans = old_obs[_index[0]: _index[1]]

            # position transformation
            new_ego_local_pos = [0, 0]
            _index = obs_index['zombie_cars_pos']
            new_zombie_local_pos = np.array(new_obs[_index[0]: _index[1]]).reshape((-1, 2))

            _trans = [new_world_trans[0] - old_world_trans[0], new_world_trans[1] - old_world_trans[1],
                      -new_world_trans[2]]

            new_local_ego_old_frame = CarlaConfigModule._rotate(CarlaConfigModule._transform(new_ego_local_pos, _trans),
                                                                old_world_trans[2])
            new_local_trans_pos_old_frame = [new_local_ego_old_frame[0], new_local_ego_old_frame[1],
                                             new_world_trans[2] - old_world_trans[2]]
            new_local_zombie_pos_old_frame = []

            for n in range(len(new_zombie_local_pos)):
                if n < zombie_num:
                    pos = new_zombie_local_pos[n]
                    new_world_zombie_pos = CarlaConfigModule._rotate(CarlaConfigModule._transform(pos, _trans),
                                                                     old_world_trans[2])
                    new_local_zombie_pos_old_frame.append(new_world_zombie_pos)
                else:
                    new_local_zombie_pos_old_frame.append([0., 0.])

            # velocity transformation
            _index = obs_index['ego_car_vel']
            new_ego_local_vel = new_obs[_index[0]: _index[1]]
            _index = obs_index['zombie_cars_v']
            new_zombie_local_vel = np.array(new_obs[_index[0]: _index[1]]).reshape((-1, 2))
            new_local_ego_vel_old_frame = CarlaConfigModule._rotate(new_ego_local_vel,
                                                                    old_world_trans[2] - new_world_trans[2])

            new_local_zombie_vel_old_frame = []
            for vel in new_zombie_local_vel:
                new_vel = CarlaConfigModule._rotate(vel, old_world_trans[2] - new_world_trans[2])
                new_local_zombie_vel_old_frame.append(new_vel)

            new_ob_old_frame = new_local_trans_pos_old_frame + new_local_ego_vel_old_frame + \
                               CarlaConfigModule._flat_list(new_local_zombie_pos_old_frame) + \
                               CarlaConfigModule._flat_list(new_local_zombie_vel_old_frame)
            tar = np.array(new_ob_old_frame) - np.array(old_obs[3:32])
            return tar.tolist()

        tar_list = []
        for [ob, n_ob] in zip(obs, next_obs):
            tar_list.append(new2old_localFrame(ob, n_ob, obs_index))

        return np.array(tar_list)

    @staticmethod
    def DAgger(train_in, train_targs, obs_index):

        def obs_rotation(obs, obs_index, phi):

            _index = obs_index['ego_car_local_trans']
            ego_pos = obs[_index[0]: _index[0] + 2]

            # Ego Car
            rot_ego_trans = CarlaConfigModule._rotate(ego_pos, phi)
            rot_ego_trans.append(obs[_index[1] - 1])

            _index = obs_index['ego_car_vel']
            ego_vel = obs[_index[0]: _index[1]]
            rot_ego_vel = CarlaConfigModule._rotate(ego_vel, phi)

            # Zombie Cars
            _index = obs_index['zombie_cars_pos']
            zombie_pos = np.array(obs[_index[0]: _index[1]]).reshape((-1, 2))

            _index = obs_index['zombie_cars_v']
            zombie_vel = np.array(obs[_index[0]: _index[1]]).reshape((-1, 2))

            rot_zombie_pos, rot_zombie_vel = [], []
            for n in range(len(zombie_pos)):
                rot_pos = CarlaConfigModule._rotate(zombie_pos[n], phi)
                rot_zombie_pos.append(rot_pos)
                rot_vel = CarlaConfigModule._rotate(zombie_vel[n], phi)
                rot_zombie_vel.append(rot_vel)

            lane_pos = []
            lines = ['inner_line_right', 'inner_line_left', 'outer_line_right', 'outer_line_left']
            for line_name in lines:
                _index = obs_index[line_name]
                lane_pos.extend(obs[_index[0]: _index[1]])

            lane_pos = np.array(lane_pos).reshape((-1, 2))

            rot_lane_pos = []
            for pos in lane_pos:
                rot_lane_pos.append(CarlaConfigModule._rotate(pos, phi))

            rot_obs = rot_ego_trans + rot_ego_vel + CarlaConfigModule._flat_list(rot_zombie_pos) \
                      + CarlaConfigModule._flat_list(rot_zombie_vel) + CarlaConfigModule._flat_list(rot_lane_pos)
            rot_obs.append(obs[-1])

            return np.array(rot_obs)

        def pred_rotation(obs, obs_index, phi):

            _index = obs_index['ego_car_local_trans']
            ego_pos = obs[_index[0]: _index[0] + 2]

            # Ego Car
            rot_ego_trans = CarlaConfigModule._rotate(ego_pos, phi)
            rot_ego_trans.append(obs[_index[1] - 1])

            _index = obs_index['ego_car_vel']
            ego_vel = obs[_index[0]: _index[1]]
            rot_ego_vel = CarlaConfigModule._rotate(ego_vel, phi)

            # Zombie Cars
            _index = obs_index['zombie_cars_pos']
            zombie_pos = np.array(obs[_index[0]: _index[1]]).reshape((-1, 2))

            _index = obs_index['zombie_cars_v']
            zombie_vel = np.array(obs[_index[0]: _index[1]]).reshape((-1, 2))

            rot_zombie_pos, rot_zombie_vel = [], []
            for n in range(len(zombie_pos)):
                rot_pos = CarlaConfigModule._rotate(zombie_pos[n], phi)
                rot_zombie_pos.append(rot_pos)
                rot_vel = CarlaConfigModule._rotate(zombie_vel[n], phi)
                rot_zombie_vel.append(rot_vel)

            rot_pred = rot_ego_trans + rot_ego_vel + CarlaConfigModule._flat_list(rot_zombie_pos) \
                      + CarlaConfigModule._flat_list(rot_zombie_vel)

            return np.array(rot_pred)

        phi_list = np.random.uniform(low=-np.pi/4, high=np.pi/4, size=3)
        ag_train_in = []
        for ob in train_in:
            ag_train_in.append(ob)
            for phi in phi_list:
                ag_train_in.append(obs_rotation(ob, obs_index, phi))

        ag_train_targs = []
        for targ in train_targs:
            ag_train_targs.append(targ)
            for phi in phi_list:
                ag_train_targs.append(pred_rotation(targ, obs_index, phi))

        return ag_train_in, ag_train_targs

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(500, input_dim=self.MODEL_IN, activation='swish', weight_decay=0.00001))
            model.add(FC(500, activation='swish', weight_decay=0.000025))
            model.add(FC(500, activation='swish', weight_decay=0.000025))
            model.add(FC(500, activation='swish', weight_decay=0.000025))
            model.add(FC(300, activation='swish', weight_decay=0.000025))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0005))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.000002})
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
            kernel_args=model_init_cfg.get("kernel_args", {}),
            num_inducing_points=get_required_argument(
                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
            ),
            sess=self.SESS
        ))
        return model

    @staticmethod
    def obs_cost_fn(obs_traj, obs_index, zombie_num):
        rewd_traj = 0
        vel_des = 15
        rewards = []
        CarlaConfigModule._obs_traj_display(obs_traj, obs_index, zombie_num)
        for obs in obs_traj:
            car_v = CarlaConfigModule._get_vehicle_info(obs, obs_index)
            sigma_vel = 3.0
            v_err = car_v - vel_des
            v_rewd = np.exp(-v_err**2 / (2*sigma_vel**2))

            accident_cost = -10 if CarlaConfigModule._check_collision(obs, obs_index, zombie_num) or \
                                   CarlaConfigModule._check_laneinvasion(obs, obs_index) else 0
            # _check_laneinvasion
            reward = v_rewd + accident_cost
            rewd_traj += reward
            rewards.append(reward)
            print("rewards:", reward)
        return rewd_traj, rewards

    @staticmethod
    def _obs_display(obs, obs_index, zombie_num):
        _index = obs_index['ego_car_local_trans']
        ego_car = obs[_index[0]:_index[1]]
        _index = obs_index['zombie_cars_pos']
        obs_zombie = obs[_index[0]: _index[1]]

        obs_lane = []
        lines = ['inner_line_right', 'inner_line_left', 'outer_line_right', 'outer_line_left']
        for line_name in lines:
            _index = obs_index[line_name]
            obs_lane.extend(obs[_index[0]: _index[1]])

        plt.figure(1)
        plt.xlim((ego_car[0] - 60, ego_car[0] + 60))
        plt.ylim((ego_car[1] - 60, ego_car[1] + 60))
        plt.plot(obs_lane[0::2], obs_lane[1::2], '.', color='red', label='lane')
        plt.plot(obs_zombie[0:zombie_num*2:2], obs_zombie[1:zombie_num*2:2], '.', color='green', label='zombie_car')
        plt.show()

    @staticmethod
    def _obs_traj_display(obs_traj, obs_index, zombie_num):
        plt.clf()
        plt.figure(1)
        plt.xlim((-60, 60))
        plt.ylim((-60, 60))
        ego_car, obs_zombie, obs_lane = [], [], []
        for obs in obs_traj:
            obs = obs.tolist()
            _index = obs_index['ego_car_local_trans']
            ego_car.append(obs[_index[0]:_index[1]])
            _index = obs_index['zombie_cars_pos']
            obs_zombie.append(obs[_index[0]: _index[0] + zombie_num * 2])

            lines = ['inner_line_right', 'inner_line_left', 'outer_line_right', 'outer_line_left']
            for line_name in lines:
                _index = obs_index[line_name]
                obs_lane.extend(obs[_index[0]: _index[1]])

        ego_x, ego_y = [], []
        for pos in ego_car:
            ego_x.append(pos[0])
            ego_y.append(pos[1])
        plt.plot(ego_x, ego_y, 'o-', color='blue', label='lane')
        plt.plot(obs_lane[0::2], obs_lane[1::2], '.', color='red', label='lane')

        zombie_x, zombie_y = [], []
        for pos in obs_zombie:
            zombie_x.append(pos[0])
            zombie_y.append(pos[1])

        plt.plot(zombie_x, zombie_y, '*-', color='green', label='zombie_car')
        plt.show()


    @staticmethod
    def _get_vehicle_info(obs, obs_index):
        _index = obs_index['ego_car_vel']
        _v = obs[_index[0]:_index[1]]
        v = np.hypot(_v[0], _v[1])
        return v

    @staticmethod
    def _get_waypoints(obs, obs_index):
        _index = obs_index['inner_line_right']
        inner_line_r = obs[_index[0] :_index[1]]
        _index = obs_index['inner_line_left']
        inner_line_l = obs[_index[0] :_index[1]]

        num = int(len(inner_line_l)/2)
        wp_x = [(inner_line_r[2*n]+inner_line_l[2*n])/2 for n in range(num)]
        wp_y = [(inner_line_r[2*n+1]+inner_line_l[2*n+1]) / 2 for n in range(num)]
        wp_yaw = [np.arctan2(wp_y[n+1]-wp_y[n], wp_x[n+1]-wp_x[n]) for n in range(num-1)]
        wp_yaw.append(wp_yaw[-1])
        return wp_x, wp_y, wp_yaw

    @staticmethod
    def _check_collision(obs, obs_index, zombie_num):
        _index = obs_index['zombie_cars_pos']
        zombie_cars = obs[_index[0] :_index[1]]
        _index = obs_index['ego_car_local_trans']
        ego_car = obs[_index[0] :_index[1]]

        zcars_x = [zombie_cars[2*n] for n in range(zombie_num)]
        zcars_y = [zombie_cars[2*n+1] for n in range(zombie_num)]
        dist = []
        for n in range(zombie_num):
            dist.append(np.hypot(zcars_x[n]-ego_car[0], zcars_y[n]-ego_car[1]))
        return True if any([_d < 3 for _d in dist]) else False

    @staticmethod
    def _check_laneinvasion(obs, obs_index):
        _index = obs_index['ego_car_local_trans']
        ego_car = obs[_index[0]:_index[1]]
        _index = obs_index['outer_line_right']
        outer_line_r = obs[_index[0]:_index[1]]
        _index = obs_index['outer_line_left']
        outer_line_l = obs[_index[0]:_index[1]]

        outer_lane_r_points = np.array(outer_line_r).reshape((-1, 2))
        near_point_r, min_dist = None, 100000
        for point in outer_lane_r_points:
            _dist = np.hypot(point[0] - ego_car[0], point[1] - ego_car[1])
            if _dist < min_dist:
                min_dist = _dist
                near_point_r = point

        outer_lane_l_points = np.array(outer_line_l).reshape((-1, 2))
        near_point_l, min_dist = None, 100000
        for point in outer_lane_l_points:
            _dist = np.hypot(point[0] - ego_car[0], point[1] - ego_car[1])
            if _dist < min_dist:
                min_dist = _dist
                near_point_l = point

        if near_point_l is None or near_point_r is None:
            print("near_point is None!!")

        vec_1 = np.array([near_point_l[0] - ego_car[0], near_point_l[1] - ego_car[1]])
        vec_2 = np.array([near_point_r[0] - ego_car[0], near_point_r[1] - ego_car[1]])
        theta = cal_angle(vec_1, vec_2)

        if theta < math.pi / 2:
            # print('Vehicle is out of Lane!')
            return True
        else:
            return False

    @staticmethod
    def _transform(pos, trans):
        yaw_radians = trans[2]
        P_0 = np.matrix(pos).transpose()
        P_t = np.matrix(trans[0:2]).transpose()
        R = np.matrix([[np.cos(yaw_radians), np.sin(yaw_radians)],
                       [-np.sin(yaw_radians), np.cos(yaw_radians)]])
        t_pos = R * P_0 + P_t
        return t_pos.transpose().tolist()[0]

    @staticmethod
    def _rotate(vec_2d, yaw):
        P_0 = np.matrix(vec_2d).transpose()
        R = np.matrix([[np.cos(yaw), np.sin(yaw)],
                       [-np.sin(yaw), np.cos(yaw)]])
        t_pos = R * P_0
        return t_pos.transpose().tolist()[0]

    @staticmethod
    def _flat_list(ls):
        if type(ls) == list or type(ls) == tuple:
            output = []
            for item in ls:
                output += CarlaConfigModule._flat_list(item)
            return output
        else:
            return [ls]

CONFIG_MODULE = CarlaConfigModule
