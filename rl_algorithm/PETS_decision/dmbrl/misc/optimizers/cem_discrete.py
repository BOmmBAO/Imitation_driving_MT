from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import copy
import scipy.stats as stats
import tensorflow_probability as tfp

from .optimizer import Optimizer


class CEMOptimizer:
    """A Tensorflow-compatible CEM optimizer.
    """
    def __init__(self, sol_dim, max_iters, popsize, num_elites, hor_num, dec_num,
                 tf_session=None, threshold=0.5, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            tf_session (tf.Session): (optional) Session to be used for this optimizer. Defaults to None,
                in which case any functions passed in cannot be tf.Tensor-valued.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            threshold (float): A minimum threshold to select a action.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.max_iters, self.popsize, self.num_elites = max_iters, popsize, num_elites
        self.sol_dim, self.hor_num, self.dec_num = sol_dim, hor_num, dec_num
        self.threshold, self.alpha = threshold, alpha
        self.tf_sess = tf_session

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        if self.tf_sess is not None:
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("CEMSolver") as scope:
                    self.init_prob = tf.placeholder(dtype=tf.float32, shape=[dec_num, hor_num])
                    self.init_sol = tf.placeholder(dtype=tf.int16, shape=[sol_dim])

        self.num_opt_iters, self.cate_dist = None, None
        self.prediction_func, self.cost_func = None, None

    def setup(self, prediction_func, cost_function):
        self.prediction_func = prediction_func
        self.cost_func = cost_function

    def reset(self):
        pass

    def obtain_solution(self, init_prob):
        """
        Optimizes the cost function using the provided initial candidate distribution
        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        prob_iter, t = copy.deepcopy(init_prob), 0
        while (t < self.max_iters) and max(prob_iter[0]) < self.threshold:
            samples = None
            for i in range(self.hor_num):
                _samples = np.random.choice(self.dec_num, self.popsize, replace=True, p=prob_iter[i])
                samples = np.vstack((samples, _samples)) if samples is not None else _samples

            samples_predict = self.prediction_func(samples.T)
            costs = self.cost_func(samples_predict, samples.T)

            m = np.argsort(costs)
            l = samples.T[np.argsort(costs)]
            elites = samples.T[np.argsort(costs)][:self.num_elites]  # To Do

            for m in range(self.hor_num):
                for n in range(self.dec_num):
                    count = np.count_nonzero(elites[:, m] == n)
                    prob_iter[m, n] = (1 - self.alpha) * prob_iter[m, n] + self.alpha * count / self.num_elites
            t += 1

        sol = np.array([prob_iter[0, :].argmax(axis=0)])
        return sol



