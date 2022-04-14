# This is a sample Python script.

import argparse

try:
    import numpy as np
    import sys
    from os import path as osp
except ImportError:
    raise RuntimeError('import error!')


from rl_algorithm.PETS_decision.dmbrl.env.carla import CarlaEnv
from carla_env.fplot import FeaPlot
import matplotlib.pyplot as plt
from utils.common import *
plt.ioff()

old_ob_c = [3.16176971e+02, -3.50977600e+02,  4.96523493e-01,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  4.78764819e+00,  5.66744318e-02,
  1.56959010e+01,  1.02779704e+00,  1.00000000e+04,  1.00000000e+04,
  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,
  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,
  5.06149869e+00,  6.41479446e-01,  1.00000000e+04,  1.00000000e+04,
  1.00000000e+04,  1.00000000e+04,  1.00000000e+04, 1.00000000e+04,
  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,
  7.05885436e-01,  1.68172550e+00,  1.13000859e+00, 1.68726310e+00,
  1.58240812e+00,  1.69476567e+00,  1.93350417e+00,  1.70158523e+00,
  3.12232217e+00,  1.73163791e+00,  3.96409649e+00,  1.75933958e+00,
  5.42757781e+00,  1.82016427e+00,  7.69140350e+00,  1.94606207e+00,
  1.07468231e+01,  2.17738528e+00,  1.45217252e+01,  2.56111719e+00,
  2.05115262e+01,  3.39460635e+00,  2.90464046e+01, 5.06798048e+00,
  4.01935737e+01,  8.14931350e+00,  7.46400842e-01, -1.81805589e+00,
  1.18172715e+00, -1.81234052e+00,  1.64595820e+00, -1.80465355e+00,
  2.00631866e+00, -1.79764708e+00,  3.22637806e+00, -1.76681936e+00,
  4.09027026e+00, -1.73839765e+00,  5.59217560e+00, -1.67596903e+00,
  7.91548334e+00, -1.54675653e+00,  1.10511770e+01, -1.30936131e+00,
  1.49252760e+01, -9.15509605e-01,  2.10724218e+01, -6.01577554e-02,
  2.98315999e+01,  1.65718343e+00, 4.12716330e+01,  4.81947377e+00,
  5.11870110e-01,  5.17980946e+00,  8.64269562e-01,  5.18390973e+00,
  1.35767459e+00,  5.19137462e+00,  2.04833705e+00,  5.20487403e+00,
  3.01512224e+00,  5.22999376e+00,  4.36823229e+00,  5.27724269e+00,
  6.26167886e+00,  5.36720031e+00,  8.90974664e+00,  5.53950658e+00,
  1.26100757e+01,  5.87169815e+00,  1.77712055e+01,  6.51454353e+00,
  2.49447299e+01,  7.76087315e+00,  3.48461629e+01,  1.01776223e+01,
  4.83246359e+01,  1.48503553e+01,  6.57445665e-01, -8.81944301e+00,
  1.04788999e+00, -8.81488704e+00,  1.59454913e+00, -8.80663368e+00,
  2.35976292e+00, -8.79166703e+00,  3.43089520e+00, -8.76383418e+00,
  4.93005241e+00, -8.71147469e+00,  7.02784093e+00, -8.61182381e+00,
  9.96177842e+00, -8.42092587e+00,  1.40614604e+01, -8.05287306e+00,
  1.97796715e+01, -7.34064112e+00,  2.77274760e+01, -5.95979085e+00,
  3.86976273e+01, -3.28218451e+00,  5.36308718e+01,  1.89488188e+00]

new_ob_c = [3.29140137e+02, -3.42852722e+02,  6.13238458e-01,  0.00000000e+00,
  0.00000000e+00,  0.00000000e+00,  4.11352799e+00,  4.12024648e-02,
  1.55342106e+01,  9.56794762e-01,  1.00000000e+04, 1.00000000e+04,
  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,
  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,
  5.01770444e+00,  6.03541067e-01,  1.00000000e+04,  1.00000000e+04,
  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,
  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,  1.00000000e+04,
  7.12389126e-01,  1.67335318e+00,  1.00789861e+00,  1.67615586e+00,
  1.46032413e+00,  1.68171193e+00,  1.69360265e+00,  1.68517098e+00,
  2.77451694e+00,  1.70657120e+00,  3.42645007e+00,  1.72375402e+00,
  5.19120867e+00,  1.78615288e+00,  7.22353290e+00,  1.88710363e+00,
  1.07233758e+01,  2.13406356e+00,  1.49129422e+01,  2.55196532e+00,
  2.08767341e+01,  3.37914677e+00,  2.91528989e+01,  4.98864159e+00,
  4.02284644e+01,  8.01623544e+00,  7.41667209e-01, -1.82655779e+00,
  1.04495234e+00, -1.82362964e+00, 1.50926871e+00, -1.81793345e+00,
  1.74866273e+00, -1.81437475e+00,  2.85795461e+00, -1.79242080e+00,
  3.52703446e+00, -1.77484105e+00,  5.33814931e+00, -1.71073358e+00,
  7.42389740e+00, -1.60715194e+00,  1.10157004e+01, -1.35370778e+00,
  1.53153241e+01, -9.24817063e-01,  2.14358422e+01, -7.59034518e-02,
  2.99294174e+01,  1.57590693e+00,  4.12959599e+01,  4.68300808e+00,
  7.71684209e-01,  5.17403416e+00,  1.12407827e+00,  5.17770396e+00,
  1.61745349e+00,  5.18444469e+00,  2.30814912e+00, 5.19699992e+00,
  3.27495978e+00, 5.22085692e+00,  4.62818579e+00,  5.26623649e+00,
  6.52168505e+00,  5.35360546e+00,  9.17005052e+00,  5.52233633e+00,
  1.28707435e+01,  5.84953752e+00, 1.80328058e+01,  6.48533767e+00,
  2.52079924e+01,  7.72199287e+00,  3.51127327e+01,  1.01253120e+01,
  4.85974695e+01,  1.47797641e+01,  8.98309963e-01, -8.82540113e+00,
  1.28874662e+00, -8.82133689e+00,  1.83539589e+00, -8.81387498e+00,
  2.60060811e+00, -8.79994940e+00,  3.67180209e+00, -8.77353705e+00,
  5.17105164e+00, -8.72322414e+00,  7.26895566e+00, -8.62644332e+00,
  1.02031615e+01, -8.43948482e+00,  1.43032924e+01, -8.07697569e+00,
  2.00225056e+01, -7.37255243e+00,  2.79721453e+01, -6.00241734e+00,
  3.89459173e+01, -3.33968561e+00,  5.38861451e+01, 1.81714838e+00]

ob_index_full = {'ego_car_world_trans': [0, 3], 'ego_car_local_trans': [3, 6], 'ego_car_vel': [6, 8],
             'zombie_cars_pos': [8, 20], 'zombie_cars_v': [20, 32], 'inner_line_right': [32, 58],
             'inner_line_left': [58, 84], 'outer_line_right': [84, 110], 'outer_line_left': [110, 136]}

ob_index_local = {'ego_car_local_trans': [0, 3], 'ego_car_vel': [3, 5],
             'zombie_cars_pos': [5, 17], 'zombie_cars_v': [17, 29], 'inner_line_right': [29, 55],
             'inner_line_left': [55, 81], 'outer_line_right': [81, 107], 'outer_line_left': [107, 133]}

def test_in_carla():
    env = CarlaEnv()
    old_ob = None
    print("obs_index:", env.obs_index)

    while True:
        ob, reward, done, info = env.step([0])
        if old_ob is not None:
            lane_world_1, lane_world_2 = trans_to_last_frame(old_ob, ob, env.obs_index)
            plt.clf()
            plt.figure(1)

            for p in range(len(lane_world_1)):
                plt.plot(lane_world_1[p][1], lane_world_1[p][0], '.', color='red', label='lane')

            for p in range(len(lane_world_1)):
                plt.plot(lane_world_2[p][1], lane_world_2[p][0], '.', color='blue', label='lane')

            plt.show()
            plt.pause(0.0001)

        old_ob = ob

def transformation():
    # transformation in training
    # lane_world_1, lane_world_2 = trans_to_last_frame(old_ob_c, new_ob_c, ob_index_full)
    tar = new2old_localFrame(old_ob_c, new_ob_c, ob_index_full)

    # plt.clf()
    # plt.figure(1)
    #
    # for p in range(len(lane_world_1)):
    #     plt.plot(lane_world_1[p][1], lane_world_1[p][0], '.', color='red', label='lane')
    #
    # for p in range(len(lane_world_1)):
    #     plt.plot(lane_world_2[p][1], lane_world_2[p][0], '.', color='blue', label='lane')
    #
    # plt.show()
    # plt.pause(1000)

    # transformation in prediction
    _pre = np.array(tar) + np.array(old_ob_c[3:32])
    pred_ob = _pre.tolist() + old_ob_c[32:]
    rot_ob_1 = obs_rotation(pred_ob, ob_index_local, math.pi/2)
    rot_ob_2 = obs_rotation(pred_ob, ob_index_local, math.pi / 3)

    pre_ob_own_frame = old2new_localFrame(pred_ob, ob_index_local)
    # if _check_laneinvasion(pre_ob_own_frame, ob_index_local):
    #     print("Lane invasion!!")

    # _obs_display(pre_ob_own_frame, ob_index_local)
    # _obs_display(new_ob_c, ob_index_full)
    _obs_display_compare(pre_ob_own_frame, new_ob_c[3:], ob_index_local)
    # _obs_display_compare(rot_ob_1, rot_ob_2, ob_index_local)


def obs_rotation(obs, obs_index, phi):

    _index = obs_index['ego_car_local_trans']
    ego_pos = obs[_index[0]: _index[0]+2]

    # Ego Car
    rot_ego_trans = _rotate(ego_pos, phi)
    rot_ego_trans.append(obs[_index[1]-1])

    _index = obs_index['ego_car_vel']
    ego_vel = obs[_index[0]: _index[1]]
    rot_ego_vel = _rotate(ego_vel, phi)

    # Zombie Cars
    _index = obs_index['zombie_cars_pos']
    zombie_pos = np.array(obs[_index[0]: _index[1]]).reshape((-1, 2))

    _index = obs_index['zombie_cars_v']
    zombie_vel = np.array(obs[_index[0]: _index[1]]).reshape((-1, 2))

    rot_zombie_pos, rot_zombie_vel = [], []
    for n in range(len(zombie_pos)):
        rot_pos = _rotate(zombie_pos[n], phi)
        rot_zombie_pos.append(rot_pos)
        rot_vel = _rotate(zombie_vel[n], phi)
        rot_zombie_vel.append(rot_vel)

    lane_pos = []
    lines = ['inner_line_right', 'inner_line_left', 'outer_line_right', 'outer_line_left']
    for line_name in lines:
        _index = obs_index[line_name]
        lane_pos.extend(obs[_index[0]: _index[1]])

    lane_pos = np.array(lane_pos).reshape((-1, 2))

    rot_lane_pos = []
    for pos in lane_pos:
        rot_lane_pos.append(_rotate(pos, phi))

    rot_obs = rot_ego_trans + rot_ego_vel + _flat_list(rot_zombie_pos) + _flat_list(rot_zombie_vel) + _flat_list(rot_lane_pos)
    return rot_obs


def _transform(pos, trans):
    yaw_radians = trans[2]
    P_0 = np.matrix(pos).transpose()
    P_t = np.matrix(trans[0:2]).transpose()
    R = np.matrix([[np.cos(yaw_radians), np.sin(yaw_radians)],
                  [-np.sin(yaw_radians), np.cos(yaw_radians)]])
    t_pos = R * P_0 + P_t
    return t_pos.transpose().tolist()[0]


def _rotate(vec_2d, yaw):
    P_0 = np.matrix(vec_2d).transpose()
    R = np.matrix([[np.cos(yaw), np.sin(yaw)],
                  [-np.sin(yaw), np.cos(yaw)]])
    t_pos = R * P_0
    return t_pos.transpose().tolist()[0]


def _flat_list(ls):
    if type(ls) == list or type(ls) == tuple:
        output = []
        for item in ls:
            output += _flat_list(item)
        return output
    else:
        return [ls]


def trans_to_last_frame(old_obs, new_obs, obs_index):

    _index = obs_index['ego_car_world_trans']
    new_world_trans = new_obs[_index[0]: _index[1]]
    old_world_trans = old_obs[_index[0]: _index[1]]

    new_obs_lane, old_obs_lane = [], []
    lines = ['inner_line_right', 'inner_line_left', 'outer_line_right', 'outer_line_left']

    for line_name in lines:
        _index = obs_index[line_name]
        new_obs_lane.extend(new_obs[_index[0]: _index[1]])
        old_obs_lane.extend(old_obs[_index[0]: _index[1]])

    new_obs_lane_x = new_obs_lane[0:: 2]
    new_obs_lane_y = new_obs_lane[1:: 2]
    old_obs_lane_x = old_obs_lane[0:: 2]
    old_obs_lane_y = old_obs_lane[1:: 2]

    new_lane_world = []
    old_lane_world = []

    for lane_pos in zip(new_obs_lane_x, new_obs_lane_y):
        _trans = [new_world_trans[0]-old_world_trans[0], new_world_trans[1]-old_world_trans[1], -new_world_trans[2]]
        new_world_pos = _transform(lane_pos, _trans)
        new_local_pos = _rotate(new_world_pos, old_world_trans[2])
        new_lane_world.append(new_local_pos)

    for lane_pos in zip(old_obs_lane_x, old_obs_lane_y):
        old_lane_world.append(lane_pos)

    return new_lane_world, old_lane_world


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

    new_local_ego_old_frame = _rotate(_transform(new_ego_local_pos, _trans), old_world_trans[2])
    new_local_trans_pos_old_frame = [new_local_ego_old_frame[0], new_local_ego_old_frame[1],
                                     new_world_trans[2] - old_world_trans[2]]
    new_local_zombie_pos_old_frame = []
    for pos in new_zombie_local_pos:
        new_world_zombie_pos = _rotate(_transform(pos, _trans), old_world_trans[2])
        new_local_zombie_pos_old_frame.append(new_world_zombie_pos)

    # velocity transformation
    _index = obs_index['ego_car_vel']
    new_ego_local_vel = new_obs[_index[0]: _index[1]]
    _index = obs_index['zombie_cars_v']
    new_zombie_local_vel = np.array(new_obs[_index[0]: _index[1]]).reshape((-1, 2))
    new_local_ego_vel_old_frame = _rotate(new_ego_local_vel, old_world_trans[2]-new_world_trans[2])

    new_local_zombie_vel_old_frame = []
    for vel in new_zombie_local_vel:
        new_vel = _rotate(vel, old_world_trans[2] - new_world_trans[2])
        new_local_zombie_vel_old_frame.append(new_vel)

    new_ob_old_frame = new_local_trans_pos_old_frame + new_local_ego_vel_old_frame + \
                       _flat_list(new_local_zombie_pos_old_frame) + _flat_list(new_local_zombie_vel_old_frame)
    tar = np.array(new_ob_old_frame) - np.array(old_obs[3:32])
    return tar.tolist()


def old2new_localFrame(pre_obs, obs_index):

    _index = obs_index['ego_car_local_trans']
    local_trans = pre_obs[_index[0]: _index[1]]

    # position transformation
    trans_1 = [-local_trans[0], -local_trans[1], 0]
    new_ego_pos = _rotate(_transform(local_trans[0:2], trans_1), local_trans[2])
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
        new_lane_pos.append(_rotate(_transform(pos, trans_1), local_trans[2]))
    for pos in zombie_pos:
        new_zombie_pos.append(_rotate(_transform(pos, trans_1), local_trans[2]))

    # velocity transformation
    _index = obs_index['ego_car_vel']
    ego_vel = pre_obs[_index[0]: _index[1]]
    new_ego_vel = _rotate(ego_vel, local_trans[2])

    _index = obs_index['zombie_cars_v']
    zombie_vel = np.array(pre_obs[_index[0]: _index[1]]).reshape((-1, 2))

    new_zombie_vel = []
    for v in zombie_vel:
        new_zombie_vel.append(_rotate(v, local_trans[2]))

    obs_new_frame = new_ego_trans + new_ego_vel + _flat_list(new_zombie_pos) + \
                    _flat_list(new_zombie_vel) + _flat_list(new_lane_pos)

    return obs_new_frame


def _obs_display(obs, obs_index):

    _index = obs_index['ego_car_local_trans']
    ego_car = obs[_index[0]:_index[1]]
    _index = obs_index['zombie_cars_pos']
    obs_zombie = obs[_index[0]: _index[1]]

    obs_lane = []
    lines = ['inner_line_right', 'inner_line_left', 'outer_line_right', 'outer_line_left']
    for line_name in lines:
        _index = obs_index[line_name]
        obs_lane.extend(obs[_index[0]: _index[1]])

    plt.figure()
    plt.xlim((ego_car[0] - 40, ego_car[0] + 40))
    plt.ylim((ego_car[1] - 40, ego_car[1] + 40))
    plt.plot(obs_lane[0::2], obs_lane[1::2], '.', color='red', label='lane')
    plt.plot(obs_zombie[0::2], obs_zombie[1::2], '.', color='green', label='zombie_car')
    plt.show()


def _obs_display_compare(pre_obs, new_obs, obs_index):

    _index = obs_index['zombie_cars_pos']
    pre_obs_zombie = pre_obs[_index[0]: _index[1]]
    new_obs_zombie = new_obs[_index[0]:_index[1]]

    pre_obs_lane = []
    new_obs_lane = []
    lines = ['inner_line_right', 'inner_line_left', 'outer_line_right', 'outer_line_left']
    for line_name in lines:
        _index = obs_index[line_name]
        pre_obs_lane.extend(pre_obs[_index[0]: _index[1]])
        new_obs_lane.extend(new_obs[_index[0]: _index[1]])

    plt.figure()
    plt.xlim((- 40, 40))
    plt.ylim((- 40, 40))
    plt.plot(pre_obs_lane[0::2], pre_obs_lane[1::2], '.', color='red', label='lane')
    plt.plot(new_obs_lane[0::2], new_obs_lane[1::2], '.', color='yellow', label='lane')
    plt.plot(pre_obs_zombie[0::2], pre_obs_zombie[1::2], 'o', color='green', label='zombie_car')
    plt.plot(new_obs_zombie[0::2], new_obs_zombie[1::2], '*', color='blue', label='zombie_car')
    plt.show()


def _check_laneinvasion(obs, obs_index):

    _index = obs_index['ego_car_local_trans']
    ego_car = obs[_index[0]:_index[1]]
    _index = obs_index['outer_line_right']
    outer_line_r = obs[_index[0] :_index[1]]
    _index = obs_index['outer_line_left']
    outer_line_l = obs[_index[0]:_index[1]]

    outer_lane_r_points = np.array(outer_line_r).reshape((-1, 2))
    near_point_r, min_dist = None, 10000
    for point in outer_lane_r_points:
        _dist = np.hypot(point[0] - ego_car[0], point[1] - ego_car[1])
        if _dist < min_dist:
            min_dist = _dist
            near_point_r = point

    outer_lane_l_points = np.array(outer_line_l).reshape((-1, 2))
    near_point_l, min_dist = None, 10000
    for point in outer_lane_l_points:
        _dist = np.hypot(point[0] - ego_car[0], point[1] - ego_car[1])
        if _dist < min_dist:
            min_dist = _dist
            near_point_l = point

    vec_1 = np.array([near_point_l[0]-ego_car[0], near_point_l[1]-ego_car[1]])
    vec_2 = np.array([near_point_r[0]-ego_car[0], near_point_r[1]-ego_car[1]])
    theta = cal_angle(vec_1, vec_2)

    if theta < math.pi/2:
        print('Vehicle is out of Lane!')
        return True
    else:
        return False




if __name__ == '__main__':
    try:
        transformation()
        # test_in_carla()
    except KeyboardInterrupt:
        print('Exit by user')


