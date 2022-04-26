from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import gym
from rl_algorithm.PETS_action.dmbrl.env.carla import CarlaEnv
import math
import numpy as np
import tensorflow as tf
from utils.common import *
from dotmap import DotMap


from rl_algorithm.PETS_action.dmbrl.misc.DotmapUtils import get_required_argument
from rl_algorithm.PETS_action.dmbrl.modeling.layers import FC
from utils.common import cal_angle


class CarlaConfigModule:
    ENV_NAME = "MBRL-Carla-v0"
    TASK_HORIZON = 100
    NTRAIN_ITERS = 10
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 10
    MODEL_IN, MODEL_OUT = 140, 28
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
                "num_elites": 5,
                "max_iters": 20,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_postproc(obs, pred):
        if isinstance(obs, np.ndarray):
            return np.concatenate([pred[:, 0:28], obs[:, 28:]], axis=1)
        else:
            return tf.concat([pred[:, 0:28], obs[:, 28:]], axis=1)

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs[:, 0:28]

    @staticmethod
    def obs_cost_fn(obs, obs_index, sigmas):
        _index = obs_index['lane_width']
        lane_width = obs[0, _index[0]:_index[1]]
        _index = obs_index['ego_car_des_vel']
        vel_des = obs[0, _index[0]:_index[1]]

        rewards = None
        for n in range(obs.shape.as_list()[0]):

            car_v = CarlaConfigModule._get_vehicle_info(obs, obs_index, n)
            wp_x, wp_y, wp_yaw = CarlaConfigModule._get_waypoints(obs, obs_index, n)

            _dist = tf.sqrt(tf.add(tf.square(wp_x), tf.square(wp_y))) # hypot
            min_dist = tf.reduce_min(_dist)
            min_index = tf.arg_min(_dist, dimension=0)

            # track reward
            sigma_pos = sigmas["sigma_pos"]
            denom_t = tf.constant(2 * sigma_pos ** 2)
            phi = tf.atan2(wp_y[min_index], wp_x[min_index])
            delta = tf.subtract(wp_yaw[min_index], phi)
            ct_err = tf.divide(tf.multiply(tf.sin(delta), min_dist), lane_width)
            track_rewd = tf.exp(tf.multiply(-tf.pow(min_dist, 2), denom_t))

            # velocity reward
            sigma_vel = (sigmas["sigma_vel_upper"] + sigmas["sigma_vel_lower"])/2
            denom_v = tf.constant(2 * sigma_vel ** 2)
            v_err = tf.subtract(car_v, vel_des)
            rew_base =  0
            if tf.less(0.001, car_v) is False:
                rew_base = 0.1
            v_rewd = tf.add(tf.exp(tf.multiply(-tf.pow(v_err, 2), denom_v)), rew_base)

            # angle reward
            sigma_yaw = sigmas["sigma_yaw"]
            denom_yaw = tf.constant(2 * sigma_yaw ** 2)
            ang_rewd = tf.exp(tf.multiply(-tf.pow(wp_yaw[min_index], 2), denom_yaw))

            # accident_cost = -10 if CarlaConfigModule._check_collision(obs, obs_index, n) is True  or \
            #                        CarlaConfigModule._check_laneinvasion(obs, obs_index, n) is True else 0
            accident_cost = 0
            # rewd_sum = tf.add(tf.multiply(tf.multiply(track_rewd, v_rewd), ang_rewd), accident_cost)
            rewd_sum = tf.add(v_rewd, accident_cost)
            rewards = rewd_sum if rewards is None else tf.concat([rewards, rewd_sum], axis=0)

        return rewards


    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return 0.01 * np.sum(np.square(acs), axis=1)
        else:
            return 0.01 * tf.reduce_sum(tf.square(acs), axis=1)

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(500, input_dim=self.MODEL_IN, activation='swish', weight_decay=0.0001))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(500, activation='swish', weight_decay=0.00025))
            model.add(FC(200, activation='swish', weight_decay=0.00025))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0005))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
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
    def _get_vehicle_info(obs, obs_index, index):
        _index = obs_index['ego_car_vel']
        v = obs[index, _index[0]:_index[1]]
        _v = tf.sqrt(tf.add(tf.square(v[0]), tf.square(v[1])))

        return _v

    @staticmethod
    def _get_waypoints(obs, obs_index, index):

        _index = obs_index['inner_line_right']
        line_r = obs[index][_index[0]:_index[1]]
        _index = obs_index['inner_line_left']
        line_l = obs[index][_index[0]:_index[1]]

        num = int(line_r.shape.as_list()[0]/2)

        line_r_x = tf.gather(line_r, tf.range(0, num*2, 2), axis=0)
        line_r_y = tf.gather(line_r, tf.range(1, num*2, 2), axis=0)
        line_l_x = tf.gather(line_l, tf.range(0, num*2, 2), axis=0)
        line_l_y = tf.gather(line_l, tf.range(1, num*2, 2), axis=0)
        wp_x = tf.divide(tf.add(line_r_x, line_l_x), 2)
        wp_y = tf.divide(tf.add(line_r_y, line_l_y), 2)

        wp_yaw = tf.atan2(tf.subtract(wp_y[1:num], wp_y[0:num-1]), tf.subtract(wp_x[1:num], wp_x[0:num-1]))
        return wp_x[0:num-1], wp_y[0:num-1], wp_yaw

    @staticmethod
    def _check_collision(obs, obs_index, index):

        _index = obs_index['zombie_cars_bbox']
        _bbox = obs[index][_index[0]:_index[1]]
        num = int(_bbox.shape.as_list()[0]/2)
        bbox_x = tf.gather(_bbox, tf.range(0, num * 2, 2), axis=0)
        bbox_y = tf.gather(_bbox, tf.range(1, num * 2, 2), axis=0)
        _dist = tf.sqrt(tf.add(tf.square(bbox_x), tf.square(bbox_y)))  # hypot
        min_dist = tf.reduce_min(_dist)

        return tf.less(min_dist, 3)


    @staticmethod
    def _check_laneinvasion(obs, obs_index, index):
        _index = obs_index['outer_line_right']
        outer_line_r = obs[index][_index[0]:_index[1]]
        _index = obs_index['outer_line_left']
        outer_line_l = obs[index][_index[0]:_index[1]]

        num = int(outer_line_r.shape.as_list()[0]/2)
        right_x = tf.gather(outer_line_r, tf.range(0, num * 2, 2), axis=0)
        right_y = tf.gather(outer_line_r, tf.range(1, num * 2, 2), axis=0)
        left_x = tf.gather(outer_line_l, tf.range(0, num * 2, 2), axis=0)
        left_y = tf.gather(outer_line_l, tf.range(1, num * 2, 2), axis=0)

        _dist = tf.sqrt(tf.add(tf.square(right_x), tf.square(right_y)))
        right_min_dist = tf.reduce_min(_dist)
        right_min_index = tf.arg_min(_dist, dimension=0)

        _dist = tf.sqrt(tf.add(tf.square(left_x), tf.square(left_y)))
        left_min_dist = tf.reduce_min(_dist)
        left_min_index = tf.arg_min(_dist, dimension=0)

        def cal_angle(vec_1, vec_2):

            norm_1 = tf.sqrt(tf.add(tf.square(vec_1[0]), tf.square(vec_1[1])))
            norm_2 = tf.sqrt(tf.add(tf.square(vec_2[0]), tf.square(vec_2[1])))
            denorm = tf.multiply(norm_1, norm_2)
            nume = tf.add(tf.multiply(vec_1[0], vec_2[0]), tf.multiply(vec_1[1], vec_2[1]))
            cos_theta = tf.divide(nume, denorm)
            return tf.acos(cos_theta)

        vec_1 = tf.stack([left_x[left_min_index], left_y[left_min_index]], axis = 0)
        vec_2 = tf.stack([right_x[right_min_index], right_y[right_min_index]], axis=0)
        theta = cal_angle(vec_1, vec_2)

        return tf.greater(theta, math.pi/2)


CONFIG_MODULE = CarlaConfigModule
