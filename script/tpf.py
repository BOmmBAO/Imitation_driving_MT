import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy.stats as stats

sol_dim = 1
max_iters = 10
threshold = 0.5
hor_num = 10
dec_num = 10
popsize = 20
_init_prob = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
init_pro = np.tile(_init_prob, (hor_num, 1))
init_prob = np
num_elites = 5
alpha = 0.25


def continue_optimization(t, prob, best_val, best_sol):
    return tf.logical_and(tf.less(t, max_iters), tf.reduce_max(prob) < threshold)


def action_sampling(iter_prob):



    def _condition(n):
        return tf.less(n, hor_num)

    def _iteration(n):
        dist = tfp.distributions.Categorical(probs=iter_prob[n], dtype=tf.int32)
        new_sam = dist.sample([popsize])
        samples[:, n] = new_sam
        return n + 1

    samples = tf.Variable([popsize, hor_num], dtype=tf.int32)  # Todo
    t, samples = tf.while_loop(
        cond=_condition, body=_iteration,
        loop_vars=[0])
    return samples


def prob_update(old_prob, elites):

    def _condition_inner(m, n, prob_iter, counts):
        return tf.less(n, dec_num)

    def _iteration_inner (m, n, prob_iter, counts):
        prob_iter[m, n] = (1 - alpha) * prob_iter[n, m] + alpha * counts[n] / num_elites
        return m, n + 1, prob_iter, counts

    def _condition_outer(m, prob_iter):
        return tf.less(m, hor_num)

    def _iteration_outer (m, prob_iter):
        counts = tf.math.bincount(elites[m], maxlength = dec_num)
        _, prob_sol = tf.while_loop(
            cond=_condition_inner, body=_iteration_inner,
            loop_vars=[m, 0, prob_iter, counts]
        )
        return m+1, prob_iter

    _, new_prob = tf.while_loop(
        cond=_condition_outer, body=_iteration_outer,
        loop_vars=[0, old_prob]
    )
    return new_prob


def iteration(t, iter_prob, best_val, best_sol):
    samples = action_sampling(iter_prob)
    # costs = self.cost_function(samples)
    costs = tf.truncated_normal([popsize], 1, 0.2)
    values, indices = tf.nn.top_k(-costs, k = num_elites, sorted=True)
    best_val, best_sol = tf.cond(
        tf.less(-values[0], best_val),
        lambda: (-values[0], samples[indices[0]]),
        lambda: (best_val, best_sol)
    )
    elites = tf.gather(samples, indices)
    iter_prob = prob_update(iter_prob, elites)
    return t + 1, iter_prob, best_val, best_sol

tf_sess = tf.Session()
with tf_sess.graph.as_default():
    init_prob = tf.placeholder(dtype=tf.float32, shape=[dec_num, hor_num])
    init_sol = tf.placeholder(dtype=tf.int32, shape=[sol_dim])
    num_opt_iters, cate_dist, best_val, best_sol = tf.while_loop(
        cond=continue_optimization, body=iteration,
        loop_vars=[0, init_prob, float('inf'), tf.cast(init_sol, dtype=tf.int32)]
    )
tf_sess.run(cate_dist, feed_dict={init_prob: init_pro, init_sol: 0})
