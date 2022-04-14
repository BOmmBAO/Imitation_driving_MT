from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import tensorflow as tf
import numpy as np
import argparse
from dotmap import DotMap
import pprint
from scipy.io import savemat

from rl_algorithm.PETS_decision.dmbrl.controllers import Controller
from rl_algorithm.PETS_decision.dmbrl.misc.DotmapUtils import get_required_argument
from rl_algorithm.PETS_decision.dmbrl.misc.optimizers import CEMOptimizer
from rl_algorithm.PETS_decision.dmbrl.config import create_config
import matplotlib.pyplot as plt
from numpy import genfromtxt
import time


class PredictionModel():

    def __init__(self, params):
        """Creates class instance.
        Arguments:
            params
                .env (gym.env): Environment for which this controller will be used.
                .update_fns (list<func>): A list of functions that will be invoked
                    (possibly with a tensorflow session) every time this controller is reset.
                .ac_ub (np.ndarray): (optional) An array of action upper bounds.
                    Defaults to environment action upper bounds.
                .ac_lb (np.ndarray): (optional) An array of action lower bounds.
                    Defaults to environment action lower bounds.
                .per (int): (optional) Determines how often the action sequence will be optimized.
                    Defaults to 1 (reoptimizes at every call to act()).
                .prop_cfg
                    .model_init_cfg (DotMap): A DotMap of initialization parameters for the model.
                        .model_constructor (func): A function which constructs an instance of this
                            model, given model_init_cfg.
                    .model_train_cfg (dict): (optional) A DotMap of training parameters that will be passed
                        into the model every time is is trained. Defaults to an empty dict.
                    .model_pretrained (bool): (optional) If True, assumes that the model
                        has been trained upon construction.
                    .mode (str): Propagation method. Choose between [E, DS, TSinf, TS1, MM].
                        See https://arxiv.org/abs/1805.12114 for details.
                    .npart (int): Number of particles used for DS, TSinf, TS1, and MM propagation methods.
                    .ign_var (bool): (optional) Determines whether or not variance output of the model
                        will be ignored. Defaults to False unless deterministic propagation is being used.
                    .obs_preproc (func): (optional) A function which modifies observations (in a 2D matrix)
                        before they are passed into the model. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc (func): (optional) A function which returns vectors calculated from
                        the previous observations and model predictions, which will then be passed into
                        the provided cost function on observations. Defaults to lambda obs, model_out: model_out.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc2 (func): (optional) A function which takes the vectors returned by
                        obs_postproc and (possibly) modifies it into the predicted observations for the
                        next time step. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .targ_proc (func): (optional) A function which takes current observations and next
                        observations and returns the array of targets (so that the model learns the mapping
                        obs -> targ_proc(obs, next_obs)). Defaults to lambda obs, next_obs: next_obs.
                        Note: Only needs to process NumPy arrays.
                .opt_cfg
                    .mode (str): Internal optimizer that will be used. Choose between [CEM, Random].
                    .cfg (DotMap): A map of optimizer initializer parameters.
                    .plan_hor (int): The planning horizon that will be used in optimization.
                    .obs_cost_fn (func): A function which computes the cost of every observation
                        in a 2D matrix.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .ac_cost_fn (func): A function which computes the cost of every action
                        in a 2D matrix.
                .log_cfg
                    .save_all_models (bool): (optional) If True, saves models at every iteration.
                        Defaults to False (only most recent model is saved).
                        Warning: Can be very memory-intensive.
                    .log_traj_preds (bool): (optional) If True, saves the mean and variance of predicted
                        particle trajectories. Defaults to False.
                    .log_particles (bool) (optional) If True, saves all predicted particles trajectories.
                        Defaults to False. Note: Takes precedence over log_traj_preds.
                        Warning: Can be very memory-intensive
        """

        self.dO = params.env.observation_space.shape[0]
        self.dU = 1
        self.dec_num = params.env.action_space.n
        self.ac_ub, self.ac_lb = params.env.action_space.n - 1, 0
        self.update_fns = params.get("update_fns", [])
        self.per = params.get("per", 1)
        # self.rews_sigma = params.env.sigma
        self.zombie_num = params.env.zombie_num
        self.obs_index = params.env.obs_index
        self.pre_obs_index = params.env.pre_obs_index

        self.model = get_required_argument(
            params.prop_cfg.model_init_cfg, "model_constructor", "Must provide a model constructor."
        )(params.prop_cfg.model_init_cfg)
        self.model_train_cfg = params.prop_cfg.get("model_train_cfg", {})
        self.prop_mode = get_required_argument(params.prop_cfg, "mode", "Must provide propagation method.")
        self.npart = get_required_argument(params.prop_cfg, "npart", "Must provide number of particles.")
        self.ign_var = params.prop_cfg.get("ign_var", False) or self.prop_mode == "E"

        self.obs_preproc = params.prop_cfg.get("obs_preproc", lambda obs: obs)
        self.obs_postproc = params.prop_cfg.get("obs_postproc", lambda obs, model_out: model_out)
        self.obs_postproc2 = params.prop_cfg.get("obs_postproc2", lambda next_obs: next_obs)
        self.targ_proc = params.prop_cfg.get("targ_proc", lambda obs, next_obs: next_obs)
        self.dagger = params.prop_cfg.get("DAgger")

        self.opt_mode = get_required_argument(params.opt_cfg, "mode", "Must provide optimization method.")
        self.plan_hor = get_required_argument(params.opt_cfg, "plan_hor", "Must provide planning horizon.")
        self.obs_cost_fn = get_required_argument(params.opt_cfg, "obs_cost_fn", "Must provide cost on observations.")
        # self.ac_cost_fn = get_required_argument(params.opt_cfg, "ac_cost_fn", "Must provide cost on actions.")

        self.save_all_models = params.log_cfg.get("save_all_models", False)
        self.log_traj_preds = params.log_cfg.get("log_traj_preds", False)
        self.log_particles = params.log_cfg.get("log_particles", False)

        # Perform argument checks
        if self.prop_mode not in ["E", "DS", "MM", "TS1", "TSinf"]:
            raise ValueError("Invalid propagation method.")
        if self.prop_mode in ["TS1", "TSinf"] and self.npart % self.model.num_nets != 0:
            raise ValueError("Number of particles must be a multiple of the ensemble size.")
        if self.prop_mode == "E" and self.npart != 1:
            raise ValueError("Deterministic propagation methods only need one particle.")

        # training and testing variables
        self.train_num = None
        self.test_num = None
        self.train_in = np.array([]).reshape(0, self.obs_preproc(np.zeros([1, self.dO])).shape[-1] + self.dU)
        self.train_targs = np.array([]).reshape(0,
            self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO]), self.obs_index, self.zombie_num).shape[-1]
        )

        self.test_obs = np.array([]).reshape(0, self.obs_preproc(np.zeros([1, self.dO])).shape[-1])
        self.test_acs = np.array([]).reshape(0, self.dU)
        self.ground_truth = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO]), self.obs_index, self.zombie_num).shape[-1]
        )

        if self.model.is_tf_model:

            with self.model.sess.graph.as_default():
                self.pre_obs = tf.placeholder(dtype=tf.float32, shape=[20, 29])
                self.cur_obs = tf.placeholder(dtype=tf.float32, shape=[141])
                self.cur_act = tf.placeholder(dtype=tf.float32, shape=[1])

        #     self.sy_cur_obs = tf.Variable(np.zeros(self.dO-3), dtype=tf.float32)
            # self.ac_seq = tf.placeholder(shape=[1, self.plan_hor*self.dU], dtype=tf.float32)
            # self.pred_cost, self.pred_traj = self._compile_cost(self.ac_seq, get_pred_trajs=True)
            # self.model.sess.run(tf.variables_initializer([self.sy_cur_obs]))


    def run(self):
        self.reading_data()
        self.predict_graph()

        iter_num = 3000
        pos_loss, vel_loss = [], []
        for n in range(iter_num):
            print('--------------Iteration ' + str(n) + '----------------')
            time_start = time.time()
            self.model.train(self.train_in, self.train_targs, **self.model_train_cfg)
            # train_loss = self.cal_train_loss()
            test_pos_loss, test_vel_loss = self.cal_test_loss()
            print("Loss:", test_pos_loss, test_vel_loss)
            pos_loss.append(test_pos_loss)
            vel_loss.append(test_vel_loss)

            time_end = time.time()
            print('Iteration Time Cost:', time_end - time_start, 's')

        with open('./pos_loss.csv', "ab") as f:
            np.savetxt(f, np.array(pos_loss).reshape((1, -1)), fmt='%.2f', delimiter=",")
        with open('./vel_loss.csv', "ab") as f:
            np.savetxt(f, np.array(vel_loss).reshape((1, -1)), fmt='%.2f', delimiter=",")

    def reading_data(self):
        def read_iteration(index):
            obs_name = '../Data_Set/obs_' + str(index) + ".csv"
            obs_traj = genfromtxt(obs_name, delimiter=',')

            dec_name = '../Data_Set/dec_' + str(index) + ".csv"
            dec_traj = genfromtxt(dec_name, delimiter=',')
            return obs_traj, dec_traj

        path = '../Data_Set/'
        files = os.listdir(path)
        total_num = int(len(files) / 2)
        self.train_num = int(total_num * 0.9)
        self.test_num = total_num - self.train_num

        # Construct training dataset
        for n in range(self.train_num):
            obs, acs = read_iteration(index=n)
            acs = np.array(acs).reshape(-1, 1)
            new_train_in = np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1)
            new_train_targs = self.targ_proc(obs[:-1], obs[1:], self.obs_index, self.zombie_num)
            new_train_in, new_train_targs = self.dagger(new_train_in, new_train_targs, self.pre_obs_index)
            self.train_in = np.concatenate([self.train_in] + [new_train_in], axis=0)
            self.train_targs = np.concatenate([self.train_targs] + [new_train_targs], axis=0)

        # Construct testing dataset
        for n in range(self.test_num):
            obs, acs = read_iteration(index=n+self.train_num)
            acs = np.array(acs).reshape(-1, 1)
            self.test_obs = np.concatenate([self.test_obs] + [self.obs_preproc(obs[:-1])], axis=0)
            self.test_acs = np.concatenate([self.test_acs] + [acs], axis=0)
            ground_truth = self.targ_proc(obs[:-1], obs[1:], self.obs_index, self.zombie_num)
            self.ground_truth = np.concatenate([self.ground_truth] + [ground_truth], axis=0)

        # display the translated observation
        # for n in range(len(self.train_in)):
        #     cur_ob = self.train_in[n]
        #     print("Decision:", cur_ob[-1])
        #     _pre = self.train_targs[n] + cur_ob[0:29]
        #     tar_obs = np.concatenate((_pre, cur_ob[29:]), axis=0)
        #     self._obs_display_compare(cur_ob, tar_obs, self.pre_obs_index)

    # ToDo

    # ToDo
    # def cal_train_loss(self):

    def cal_test_loss(self):
        pos_loss_sum, vel_loss_sum = 0, 0
        for n in range(len(self.test_obs)):
            pred_obs = self.model.sess.run(
                [self.pre_obs],
                feed_dict={self.cur_obs: self.test_obs[n], self.cur_act: self.test_acs[n]}
            )

            pos_loss, vel_loss = self.loss_function(pred_obs[0], self.ground_truth[n])
            pos_loss_sum += pos_loss
            vel_loss_sum += vel_loss
        pos_loss_ave = pos_loss_sum / len(self.test_obs)
        vel_loss_ave = vel_loss_sum / len(self.test_obs)
        return pos_loss_ave, vel_loss_ave

    def loss_function(self, pred_obs, ground_truth):

        def loss_per_pred(pred_ob, ground_truth):
            # average predict_pos error
            index = self.pre_obs_index['ego_car_local_trans']
            pred_ego_trans = pred_ob[index[0]:index[1]]
            true_ego_trans = ground_truth[index[0]:index[1]]
            ego_pos_dif = np.array(pred_ego_trans) - np.array(true_ego_trans)
            ego_pos_error = np.hypot(ego_pos_dif[0], ego_pos_dif[1])

            index = self.pre_obs_index['zombie_cars_pos']
            pred_zombies_pos = pred_ob[index[0]:index[0] + 2 * self.zombie_num]
            true_zombies_pos = ground_truth[index[0]:index[0] + 2 * self.zombie_num]
            zombies_pos_dif = np.array(pred_zombies_pos) - np.array(true_zombies_pos)
            zombies_pos_error = 0
            for n in range(self.zombie_num):
                zombies_pos_error += np.hypot(zombies_pos_dif[2*n], zombies_pos_dif[2*n + 1])

            pos_error = ego_pos_error + zombies_pos_error

            # average predict_vel error
            index = self.pre_obs_index['ego_car_vel']
            pred_ego_vel = pred_ob[index[0]:index[1]]
            true_ego_vel = ground_truth[index[0]:index[1]]
            ego_vel_dif = np.array(pred_ego_vel) - np.array(true_ego_vel)
            ego_vel_error = np.hypot(ego_vel_dif[0], ego_vel_dif[1])

            index = self.pre_obs_index['zombie_cars_v']
            pred_zombies_vel = pred_ob[index[0]:index[0] + 2 * self.zombie_num]
            true_zombies_vel = ground_truth[index[0]:index[0] + 2 * self.zombie_num]

            zombies_vel_dif = np.array(pred_zombies_vel) - np.array(true_zombies_vel)
            zombies_vel_error = 0
            for n in range(self.zombie_num):
                zombies_vel_error += np.hypot(zombies_vel_dif[2*n], zombies_vel_dif[2*n + 1])

            vel_error = ego_vel_error + zombies_vel_error
            return pos_error, vel_error

        pos_error_sum, vel_error_sum = 0, 0

        for n in range(len(pred_obs)):
            pos_error, vel_error = loss_per_pred(pred_obs[n], ground_truth)
            pos_error_sum += pos_error
            vel_error_sum += vel_error

        pos_error_ave = pos_error_sum / len(pred_obs)
        vel_error_ave = vel_error_sum / len(pred_obs)
        # print("Loss for sample:", pos_error_ave, vel_error_ave)
        return pos_error_ave, vel_error_ave

    def predict_graph(self):
        nopt = 1
        with self.model.sess.graph.as_default():
            obs_in = tf.tile(self.cur_obs[None], [nopt * self.npart, 1])
            act_in = tf.reshape(tf.tile(self.cur_act, [nopt * self.npart]), [-1, self.dU])
            self.pre_obs = self._predict_next_obs(obs_in, act_in)


    def _predict_next_obs(self, obs, acs):
        proc_obs = obs

        if self.model.is_tf_model:
            # TS Optimization: Expand so that particles are only passed through one of the networks.
            if self.prop_mode == "TS1":
                proc_obs = tf.reshape(proc_obs, [-1, self.npart, proc_obs.get_shape()[-1]])
                sort_idxs = tf.nn.top_k(
                    tf.random_uniform([tf.shape(proc_obs)[0], self.npart]),
                    k=self.npart
                ).indices
                tmp = tf.tile(tf.range(tf.shape(proc_obs)[0])[:, None], [1, self.npart])[:, :, None]
                idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
                proc_obs = tf.gather_nd(proc_obs, idxs)
                proc_obs = tf.reshape(proc_obs, [-1, proc_obs.get_shape()[-1]])

            if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
                proc_obs, acs = self._expand_to_ts_format(proc_obs), self._expand_to_ts_format(tf.cast(acs, dtype=tf.float32))

            # Obtain model predictions
            inputs = tf.concat([proc_obs, acs], axis=-1)
            mean, var = self.model.create_prediction_tensors(inputs)

            if self.model.is_probabilistic and not self.ign_var:
                predictions = mean + tf.random_normal(shape=tf.shape(mean), mean=0, stddev=1) * tf.sqrt(var)
                if self.prop_mode == "MM":
                    model_out_dim = predictions.get_shape()[-1].value

                    predictions = tf.reshape(predictions, [-1, self.npart, model_out_dim])
                    prediction_mean = tf.reduce_mean(predictions, axis=1, keep_dims=True)
                    prediction_var = tf.reduce_mean(tf.square(predictions - prediction_mean), axis=1, keep_dims=True)
                    z = tf.random_normal(shape=tf.shape(predictions), mean=0, stddev=1)
                    samples = prediction_mean + z * tf.sqrt(prediction_var)
                    predictions = tf.reshape(samples, [-1, model_out_dim])
            else:
                predictions = mean

            # TS Optimization: Remove additional dimension
            if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
                predictions = self._flatten_to_matrix(predictions)

            if self.prop_mode == "TS1":
                predictions = tf.reshape(predictions, [-1, self.npart, predictions.get_shape()[-1]])
                sort_idxs = tf.nn.top_k(
                    - sort_idxs,
                    k=self.npart
                ).indices
                idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
                predictions = tf.gather_nd(predictions, idxs)
                predictions = tf.reshape(predictions, [-1, predictions.get_shape()[-1]])

            # return self.obs_postproc(obs, predictions)
            return predictions
        else:
            raise NotImplementedError()

    def _expand_to_ts_format(self, mat):
        dim = mat.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(mat, [-1, self.model.num_nets, self.npart // self.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [self.model.num_nets, -1, dim]
        )

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(ts_fmt_arr, [self.model.num_nets, -1, self.npart // self.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [-1, dim]
        )

    def _obs_display(self, obs, obs_index, zombie_num):
        _index = obs_index['ego_car_local_trans']
        ego_car = obs[_index[0]:_index[1]]
        _index = obs_index['zombie_cars_pos']
        obs_zombie = obs[_index[0]: _index[1]]

        obs_lane = []
        lines = ['inner_line_right', 'inner_line_left', 'outer_line_right', 'outer_line_left']
        for line_name in lines:
            _index = obs_index[line_name]
            obs_lane.extend(obs[_index[0]: _index[1]])

        plt.clf()
        plt.figure(2)
        plt.xlim((-60, 60))
        plt.ylim((-60, 60))
        plt.plot(obs_lane[0::2], obs_lane[1::2], '.', color='red', label='lane')
        plt.plot(ego_car[0], ego_car[1], 'o', color='blue', label='lane')
        plt.plot(obs_zombie[0:zombie_num*2:2], obs_zombie[1:zombie_num*2:2], '*', color='green', label='zombie_car')
        plt.show()

    def _obs_display_compare(self, cur_obs, pre_obs, obs_index):

        _index = obs_index['ego_car_local_trans']
        cur_ego = cur_obs[_index[0]: _index[1]-1]
        pre_ego = pre_obs[_index[0]: _index[1]-1]

        _index = obs_index['zombie_cars_pos']
        cur_zombie = cur_obs[_index[0]: _index[1]]
        pre_zombie = pre_obs[_index[0]:_index[1]]

        cur_lane = []
        pre_lane = []
        lines = ['inner_line_right', 'inner_line_left', 'outer_line_right', 'outer_line_left']
        for line_name in lines:
            _index = obs_index[line_name]
            cur_lane.extend(cur_obs[_index[0]: _index[1]])
            pre_lane.extend(pre_obs[_index[0]: _index[1]])

        plt.clf()
        plt.figure(1)
        plt.xlim((- 40, 40))
        plt.ylim((- 40, 40))
        plt.plot(cur_lane[0::2], cur_lane[1::2], '.', color='red', label='lane')
        plt.plot(pre_lane[0::2], pre_lane[1::2], '.', color='yellow', label='lane')
        plt.plot(cur_zombie[0::2], cur_zombie[1::2], 'o', color='green', label='zombie_car')
        plt.plot(pre_zombie[0::2], pre_zombie[1::2], '*', color='green', label='zombie_car')
        plt.plot(cur_ego[0], cur_ego[1], 'o', color='blue', label='ego_car')
        plt.plot(pre_ego[0], pre_ego[1], '*', color='blue', label='ego_car')
        plt.show()


def main(env, ctrl_type, ctrl_args, overrides, logdir):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    cfg.pprint()

    pred_model = PredictionModel(cfg.ctrl_cfg)
    pred_model.run()
    # pred_model.reading_data()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=False, default = 'carla',
                        help='Environment name: select from [carla, cartpole, reacher, pusher, halfcheetah]')
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                        help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                        help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    args = parser.parse_args()

    main(args.env, "MPC", args.ctrl_arg, args.override, args.logdir)

