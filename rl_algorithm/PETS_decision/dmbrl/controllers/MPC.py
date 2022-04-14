from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import tensorflow as tf
import numpy as np
from scipy.io import savemat

from .Controller import Controller
from rl_algorithm.PETS_decision.dmbrl.misc.DotmapUtils import get_required_argument
from rl_algorithm.PETS_decision.dmbrl.misc.optimizers import CEMOptimizer
import matplotlib.pyplot as plt

class MPC(Controller):
    optimizers = {"CEM": CEMOptimizer}

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
        super().__init__(params)
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

        self.opt_mode = get_required_argument(params.opt_cfg, "mode", "Must provide optimization method.")
        self.plan_hor = get_required_argument(params.opt_cfg, "plan_hor", "Must provide planning horizon.")
        self.obs_cost_fn = get_required_argument(params.opt_cfg, "obs_cost_fn", "Must provide cost on observations.")
        # self.ac_cost_fn = get_required_argument(params.opt_cfg, "ac_cost_fn", "Must provide cost on actions.")

        self.save_all_models = params.log_cfg.get("save_all_models", False)
        self.log_traj_preds = params.log_cfg.get("log_traj_preds", False)
        self.log_particles = params.log_cfg.get("log_particles", False)
        self.train_num = 0

        # Perform argument checks
        if self.prop_mode not in ["E", "DS", "MM", "TS1", "TSinf"]:
            raise ValueError("Invalid propagation method.")
        if self.prop_mode in ["TS1", "TSinf"] and self.npart % self.model.num_nets != 0:
            raise ValueError("Number of particles must be a multiple of the ensemble size.")
        if self.prop_mode == "E" and self.npart != 1:
            raise ValueError("Deterministic propagation methods only need one particle.")

        # Create action sequence optimizer
        opt_cfg = params.opt_cfg.get("cfg", {})
        self.optimizer = MPC.optimizers[params.opt_cfg.mode](
            hor_num=self.plan_hor,
            sol_dim=self.plan_hor * self.dU,
            dec_num=self.dec_num,
            tf_session=None if not self.model.is_tf_model else self.model.sess,
            **opt_cfg
        )

        # Controller state variables
        self.has_been_trained = params.prop_cfg.get("model_pretrained", False)
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile(0, [self.plan_hor])
        _init_prob = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
        self.init_pro = np.tile(_init_prob, (self.plan_hor, 1))
        self.train_in = np.array([]).reshape(0, self.obs_preproc(np.zeros([1, self.dO])).shape[-1] + self.dU)
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO]), self.obs_index).shape[-1]
        )

        if self.model.is_tf_model:
            self.sy_cur_obs = tf.Variable(np.zeros(self.dO-3), dtype=tf.float32)
            # self.ac_seq = tf.placeholder(shape=[1, self.plan_hor*self.dU], dtype=tf.float32)
            # self.pred_cost, self.pred_traj = self._compile_cost(self.ac_seq, get_pred_trajs=True)
            self.optimizer.setup(self._predict_horizon, self._cost)
            self.model.sess.run(tf.variables_initializer([self.sy_cur_obs]))
        else:
            raise NotImplementedError()

        print("Created an MPC controller, prop mode %s, %d particles. " % (self.prop_mode, self.npart) +
              ("Ignoring variance." if self.ign_var else ""))

        if self.save_all_models:
            print("Controller will save all models. (Note: This may be memory-intensive.")
        if self.log_particles:
            print("Controller is logging particle predictions (Note: This may be memory-intensive).")
            self.pred_particles = []
        elif self.log_traj_preds:
            print("Controller is logging trajectory prediction statistics (mean+var).")
            self.pred_means, self.pred_vars = [], []
        else:
            print("Trajectory prediction logging is disabled.")

    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """
        Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.
        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            acs_trajs: A list of action matrices, actions in rows.
            rews_trajs: A list of reward arrays.
        Returns: None.
        """
        # Construct new training points and add to training set
        new_train_in, new_train_targs = [], []
        for obs, acs in zip(obs_trajs, acs_trajs):
            new_train_in.append(np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1))
            new_train_targs.append(self.targ_proc(obs[:-1], obs[1:], self.obs_index))
        self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
        self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

        # Train the model
        self.model.train(self.train_in, self.train_targs, **self.model_train_cfg)

        # display the translated observation
        # for n in range(len(self.train_in)):
        #     self._obs_display(self.train_in[n], self.pre_obs_index, self.zombie_num)
        #     _pre = self.train_targs[n] + self.train_in[n][0:29]
        #     tar_obs = np.concatenate((_pre, self.train_in[n][29:]), axis=0)
        #     self._obs_display(tar_obs, self.pre_obs_index, self.zombie_num)

        self.train_num += 1
        if self.train_num > 1:
            self.has_been_trained = True

    # def dagger(self):
    #     self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
    #     self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).
        Returns: None
        """
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.optimizer.reset()
        if self.model.is_tf_model:
            for update_fn in self.update_fns:
                update_fn(self.model.sess)

    def act(self, obs, t, get_pred_cost=False):
        """Returns the action that this controller would take at time t given observation obs.
        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.
        Returns: An action (and possibly the predicted cost)
        """
        if not self.has_been_trained:
            _init_prob = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
            return np.array([np.random.choice(self.dec_num, replace=True, p=_init_prob)])
        # if self.ac_buf.shape[0] > 0:
        #     action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
        #     return action

        if self.model.is_tf_model:
            self.sy_cur_obs.load(obs[3:], self.model.sess)

        decision = self.optimizer.obtain_solution(self.init_pro)
        # self.prev_sol = np.concatenate([np.copy(soln)[self.per*self.dU:], np.zeros(self.per*self.dU)])
        # self.ac_buf = soln[:self.per*self.dU].reshape(-1, self.dU)
        return decision

    def dump_logs(self, primary_logdir, iter_logdir):
        """Saves logs to either a primary log directory or another iteration-specific directory.
        See __init__ documentation to see what is being logged.

        Arguments:
            primary_logdir (str): A directory path. This controller assumes that this directory
                does not change every iteration.
            iter_logdir (str): A directory path. This controller assumes that this directory
                changes every time dump_logs is called.
        Returns: None
        """
        self.model.save(iter_logdir if self.save_all_models else primary_logdir)
        if self.log_particles:
            savemat(os.path.join(iter_logdir, "predictions.mat"), {"predictions": self.pred_particles})
            self.pred_particles = []
        elif self.log_traj_preds:
            savemat(
                os.path.join(iter_logdir, "predictions.mat"),
                {"means": self.pred_means, "vars": self.pred_vars}
            )
            self.pred_means, self.pred_vars = [], []

    def _cost(self, pred_traj, ac_seqs):
        traj_samples = np.transpose(pred_traj, (1, 0, 2))
        cost, cost_list = [], []
        for n in range(len(traj_samples)):
            print("actions:", ac_seqs[int(n/20)])
            red_sum, red_list = self.obs_cost_fn(traj_samples[n], self.pre_obs_index, self.zombie_num)
            cost.append(red_sum)
            cost_list.append(red_list)
        cost_list = np.array(cost_list).reshape((20, -1))
        for i in range(20):
            print("actions:", ac_seqs[i])
            print("rewards:", cost_list[i])
        costs = np.reshape(cost, [-1, self.npart])
        cost_np = np.sum(costs, axis=1)
        return cost_np

    def _predict_horizon_tf(self, ac_seqs):
        t, nopt = tf.constant(0), tf.shape(ac_seqs)[0]
        ac_seqs = tf.reshape(ac_seqs, [-1, self.plan_hor, self.dU])
        ac_seqs = tf.reshape(tf.tile(
            tf.transpose(ac_seqs, [1, 0, 2])[:, :, None],
            [1, 1, self.npart, 1]
        ), [self.plan_hor, -1, self.dU])
        init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.npart, 1])
        pred_trajs = init_obs[None]

        def continue_prediction(t, *args):
            return tf.less(t, self.plan_hor)

        def iteration(t, cur_obs, pred_trajs):
            cur_acs = ac_seqs[t]
            next_obs = self._predict_next_obs(cur_obs, cur_acs)
            # next_obs = self.obs_postproc2(cur_obs, next_obs)
            pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)
            return t + 1, next_obs, pred_trajs

        with self.model.sess.graph.as_default():
            _, _, pred_trajs = tf.while_loop(
                cond=continue_prediction, body=iteration, loop_vars=[t, init_obs, pred_trajs],
                shape_invariants=[
                    t.get_shape(), init_obs.get_shape(), tf.TensorShape([None, None, self.dO-3])
                ]
            )
        sol = self.model.sess.run(pred_trajs)
        return sol

    def _predict_horizon(self, ac_seqs):
        nopt = tf.shape(ac_seqs)[0]
        ac_seqs = tf.reshape(ac_seqs, [-1, self.plan_hor, self.dU])
        ac_seqs = tf.reshape(tf.tile(
            tf.transpose(ac_seqs, [1, 0, 2])[:, :, None],
            [1, 1, self.npart, 1]
        ), [self.plan_hor, -1, self.dU])
        init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.npart, 1])

        def predict_iter(t, cur_obs):
            cur_acs = ac_seqs[t]
            next_obs = self._predict_next_obs(cur_obs, cur_acs)
            pre_obs = self.model.sess.run(next_obs) # in cur_obs frame
            next_obs = self.obs_postproc2(pre_obs, self.pre_obs_index) # in pred_obs frame
            t += 1
            # ToDo: waypoints should be different
            return pre_obs, next_obs

        pred_traj = []
        cur_obs = init_obs
        for t in range(self.plan_hor):
            pre_obs, cur_obs = predict_iter(t, cur_obs)
            cur_obs = tf.reshape(cur_obs, [nopt * self.npart, -1])
            pred_traj.append(pre_obs)
        return pred_traj

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

            return self.obs_postproc(obs, predictions)
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