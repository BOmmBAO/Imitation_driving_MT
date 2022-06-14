import gym
from gym import spaces
import time
import itertools
from tools.modules import *
from config import cfg
from carla_env.misc import *
from plan_control.frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from plan_control.controller import VehiclePIDController
from tools.misc import get_speed
from plan_control.controller import IntelligentDriverModel
import argparse

try:
    import numpy as np
    import sys
    from os import path as osp
except ImportError:
    raise RuntimeError('import error!')
from carla_env.feature import *

MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_TRAFFIC = 'TRAFFIC'
TENSOR_ROW_NAMES = ['EGO', 'LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN','LLEFT', 'LLEFT_UP', 'LLEFT_DOWN',
                    'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN', 'RRIGHT', 'RRIGHT_UP', 'RRIGHT_DOWN']




class CarlaEnv(gym.Env):
    ROAD_FEATURES = dict(space=spaces.Box(low=0.0, high=1.0, shape=(1,96)), default=np.zeros(shape=1, dtype=np.float32))
    VEHICLE_FEATURES = dict(space=spaces.Box(low=0.0, high=1.0, shape=(4,)),
                            default=np.zeros(shape=4, dtype=np.float32))
    ACTOR_FEATURES = dict(space=spaces.Box(low=-1, high=1, shape=(4 + 1, 15)), efault=np.zeros(shape=(5, 15), dtype=np.float32))

    NAVIGATION_FEATURES = dict()

    def __init__(self, args): #lanes_change=5):
        self.__version__ = "0.9.12"

        # simulation
        self.verbosity = args.verbosity
        self.n_step = 0
        # self.global_route = np.load(
        #         'road_maps/global_route_town04.npy')  # track waypoints (center lane of the second lane from left)
        self.global_route = None
        self.acceleration_ = 0
        self.eps_rew = 0
        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05

        # constraints
        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)
        self.maxAcc = float(cfg.GYM_ENV.MAX_ACC)
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.N_SPAWN_CARS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_CARS)

        # frenet
        self.f_idx = 0
        self.init_s = None  # initial frenet s value - will be updated in reset function
        self.max_s = int(cfg.CARLA.MAX_S)
        self.track_length = int(cfg.GYM_ENV.TRACK_LENGTH)
        self.look_back = int(cfg.GYM_ENV.LOOK_BACK)
        self.time_step = int(cfg.GYM_ENV.TIME_STEP)
        self.loop_break = int(cfg.GYM_ENV.LOOP_BREAK)
        self.effective_distance_from_vehicle_ahead = int(cfg.GYM_ENV.DISTN_FRM_VHCL_AHD)
        self.lanechange = False
        self.is_first_path = True

        # RL
        self.w_speed = int(cfg.RL.W_SPEED)
        self.w_r_speed = int(cfg.RL.W_R_SPEED)

        self.min_speed_gain = float(cfg.RL.MIN_SPEED_GAIN)
        self.min_speed_loss = float(cfg.RL.MIN_SPEED_LOSS)
        self.lane_change_reward = float(cfg.RL.LANE_CHANGE_REWARD)
        self.lane_change_penalty = float(cfg.RL.LANE_CHANGE_PENALTY)
        self.sigma_pos = float(cfg.RL.SIGMA_POS)
        self.sigma_angle = float(cfg.RL.SIGMA_ANGLE)

        self.off_the_road_penalty = int(cfg.RL.OFF_THE_ROAD)
        self.collision_penalty = int(cfg.RL.COLLISION)

        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            self.low_state = np.array([[-1 for _ in range(self.look_back)] for _ in range(16)])
            self.high_state = np.array([[1 for _ in range(self.look_back)] for _ in range(16)])
        else:
            self.low_state = np.array(
                [[-1 for _ in range(self.look_back)] for _ in range(int(self.N_SPAWN_CARS + 1) * 2 + 1)])
            self.high_state = np.array(
                [[1 for _ in range(self.look_back)] for _ in range(int(self.N_SPAWN_CARS + 1) * 2 + 1)])


        action_low = np.array([-1, -1])
        action_high = np.array([1, 1])
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(2,), dtype=np.float32)
        # [cn, ..., c1, c0, normalized yaw angle, normalized speed error] => ci: coefficients
        self.state = np.zeros_like(self.observation_space.sample())

        # instances
        self.module_manager = ModuleManager()
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=10.0, module_manager=self.module_manager)
        self.traffic_module = TrafficManager(MODULE_TRAFFIC, module_manager=self.module_manager)
        self.module_manager.register_module(self.world_module)
        self.module_manager.register_module(self.traffic_module)
        self._get_global_rout()
        self.motionPlanner = MotionPlanner()

        # Start Modules
        self.motionPlanner.start(self.global_route)
        self.world_module.update_global_route_csp(self.motionPlanner.csp)
        self.traffic_module.update_global_route_csp(self.motionPlanner.csp)
        self.module_manager.start_modules()

        self.ego = self.world_module.hero_actor
        self.ego_los_sensor = self.world_module.los_sensor
        self.vehicleController = VehiclePIDController(self.ego, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0})
        self.IDM = IntelligentDriverModel(self.ego)

        self.module_manager.tick()  # Update carla world

        self.init_transform = self.ego.get_transform()  # ego initial transform to recover at each episode



        """
        ['EGO', 'LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN', 'LLEFT', 'LLEFT_UP',
        'LLEFT_DOWN', 'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN', 'RRIGHT', 'RRIGHT_UP', 'RRIGHT_DOWN']
        """
        self.actor_enumerated_dict = {}
        self.actor_enumeration = []
        self.side_window = 5  # times 2 to make adjacent window




    def _get_global_rout(self):
        if self.global_route is None:
            self.global_route = np.empty((0, 3))
            distance = 1
            for i in range(1520):
                wp = self.world_module.town_map.get_waypoint(carla.Location(x=406, y=-100, z=0.1),
                                                             project_to_road=True).next(distance=distance)[0]
                distance += 2
                self.global_route = np.append(self.global_route,
                                              [[wp.transform.location.x, wp.transform.location.y,
                                                wp.transform.location.z]], axis=0)
                # To visualize point clouds
                self.world_module.points_to_draw['wp {}'.format(wp.id)] = [wp.transform.location, 'COLOR_CHAMELEON_0']
            np.save('road_maps/global_route_town04', self.global_route)

    def get_vehicle_ahead(self, ego_s, ego_d, ego_init_d, ego_target_d):
        """
        This function returns the values for the leading actor in front of the ego vehicle. When there is lane-change
        it is important to consider actor in the current lane and target lane. If leading actor in the current lane is
        too close than it is considered to be vehicle_ahead other wise target lane is prioritized.
        """

        distance = self.effective_distance_from_vehicle_ahead
        others_s = [0 for _ in range(self.N_SPAWN_CARS)]
        others_d = [0 for _ in range(self.N_SPAWN_CARS)]
        for i, actor in enumerate(self.traffic_module.actors_batch):
            act_s, act_d = actor['Frenet State'][0][-1], actor['Frenet State'][1]
            others_s[i] = act_s
            others_d[i] = act_d

        init_lane_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 1.75) * (abs(np.array(others_d) - ego_init_d) < 1))[0]

        init_lane_strict_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 0.4) * (abs(np.array(others_d) - ego_init_d) < 1))[0]

        target_lane_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 3.3) * (abs(np.array(others_d) - ego_target_d) < 1))[0]

        if len(init_lane_d_idx) and len(target_lane_d_idx) == 0:
            return None  # no vehicle ahead
        else:
            init_lane_s = np.array(others_s)[init_lane_d_idx]
            init_s_idx = np.concatenate(
                (np.array(init_lane_d_idx).reshape(-1, 1), (init_lane_s - ego_s).reshape(-1, 1),)
                , axis=1)
            sorted_init_s_idx = init_s_idx[init_s_idx[:, 1].argsort()]

            init_lane_strict_s = np.array(others_s)[init_lane_strict_d_idx]
            init_strict_s_idx = np.concatenate(
                (np.array(init_lane_strict_d_idx).reshape(-1, 1), (init_lane_strict_s - ego_s).reshape(-1, 1),)
                , axis=1)
            sorted_init_strict_s_idx = init_strict_s_idx[init_strict_s_idx[:, 1].argsort()]

            target_lane_s = np.array(others_s)[target_lane_d_idx]
            target_s_idx = np.concatenate((np.array(target_lane_d_idx).reshape(-1, 1),
                                           (target_lane_s - ego_s).reshape(-1, 1),), axis=1)
            sorted_target_s_idx = target_s_idx[target_s_idx[:, 1].argsort()]

            if any(sorted_init_s_idx[:, 1][sorted_init_s_idx[:, 1] <= 10] > 0):
                vehicle_ahead_idx = int(sorted_init_s_idx[:, 0][sorted_init_s_idx[:, 1] > 0][0])
            elif any(sorted_init_strict_s_idx[:, 1][sorted_init_strict_s_idx[:, 1] <= distance] > 0):
                vehicle_ahead_idx = int(sorted_init_strict_s_idx[:, 0][sorted_init_strict_s_idx[:, 1] > 0][0])
            elif any(sorted_target_s_idx[:, 1][sorted_target_s_idx[:, 1] <= distance] > 0):
                vehicle_ahead_idx = int(sorted_target_s_idx[:, 0][sorted_target_s_idx[:, 1] > 0][0])
            else:
                return None

            # print(others_s[vehicle_ahead_idx] - ego_s, others_d[vehicle_ahead_idx], ego_d)

            return self.traffic_module.actors_batch[vehicle_ahead_idx]['Actor']

    def enumerate_actors(self):
        """
        Given the traffic actors and ego_state this fucntion enumerate actors, calculates their relative positions with
        to ego and assign them to actor_enumerated_dict.
        Keys to be updated: ['LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN', 'LLEFT', 'LLEFT_UP',
        'LLEFT_DOWN', 'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN', 'RRIGHT', 'RRIGHT_UP', 'RRIGHT_DOWN']
        """

        self.actor_enumeration = []
        ego_s = self.actor_enumerated_dict['EGO']['S'][-1]
        ego_d = self.actor_enumerated_dict['EGO']['D'][-1]

        others_s = [0 for _ in range(self.N_SPAWN_CARS)]
        others_d = [0 for _ in range(self.N_SPAWN_CARS)]
        others_id = [0 for _ in range(self.N_SPAWN_CARS)]
        for i, actor in enumerate(self.traffic_module.actors_batch):
            act_s, act_d = actor['Frenet State']
            others_s[i] = act_s[-1]
            others_d[i] = act_d
            others_id[i] = actor['Actor'].id

        def append_actor(x_lane_d_idx, actor_names=None):
            # actor names example: ['left', 'leftUp', 'leftDown']
            x_lane_s = np.array(others_s)[x_lane_d_idx]
            x_lane_id = np.array(others_id)[x_lane_d_idx]
            s_idx = np.concatenate((np.array(x_lane_d_idx).reshape(-1, 1), (x_lane_s - ego_s).reshape(-1, 1),
                                    x_lane_id.reshape(-1, 1)), axis=1)
            sorted_s_idx = s_idx[s_idx[:, 1].argsort()]

            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < self.side_window][0])] if (
                    any(abs(
                        sorted_s_idx[:, 1][abs(sorted_s_idx[:, 1]) <= self.side_window]) >= -self.side_window)) else -1)

            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > self.side_window][0])] if (
                    any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > self.side_window)) else -1)

            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < -self.side_window][-1])] if (
                    any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < -self.side_window)) else -1)

        # --------------------------------------------- ego lane -------------------------------------------------
        same_lane_d_idx = np.where(abs(np.array(others_d) - ego_d) < 1)[0]
        if len(same_lane_d_idx) == 0:
            self.actor_enumeration.append(-2)
            self.actor_enumeration.append(-2)

        else:
            same_lane_s = np.array(others_s)[same_lane_d_idx]
            same_lane_id = np.array(others_id)[same_lane_d_idx]
            same_s_idx = np.concatenate((np.array(same_lane_d_idx).reshape(-1, 1), (same_lane_s - ego_s).reshape(-1, 1),
                                         same_lane_id.reshape(-1, 1)), axis=1)
            sorted_same_s_idx = same_s_idx[same_s_idx[:, 1].argsort()]
            self.actor_enumeration.append(others_id[int(sorted_same_s_idx[:, 0][sorted_same_s_idx[:, 1] > 0][0])]
                                          if (any(sorted_same_s_idx[:, 1] > 0)) else -1)
            self.actor_enumeration.append(others_id[int(sorted_same_s_idx[:, 0][sorted_same_s_idx[:, 1] < 0][-1])]
                                          if (any(sorted_same_s_idx[:, 1] < 0)) else -1)

        # --------------------------------------------- left lane -------------------------------------------------
        left_lane_d_idx = np.where(((np.array(others_d) - ego_d) < -3) * ((np.array(others_d) - ego_d) > -4))[0]
        if ego_d < -1.75:
            self.actor_enumeration += [-2, -2, -2]

        elif len(left_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(left_lane_d_idx)

        # ------------------------------------------- two left lane -----------------------------------------------
        lleft_lane_d_idx = np.where(((np.array(others_d) - ego_d) < -6.5) * ((np.array(others_d) - ego_d) > -7.5))[0]

        if ego_d < 1.75:
            self.actor_enumeration += [-2, -2, -2]

        elif len(lleft_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(lleft_lane_d_idx)

            # ---------------------------------------------- rigth lane --------------------------------------------------
        right_lane_d_idx = np.where(((np.array(others_d) - ego_d) > 3) * ((np.array(others_d) - ego_d) < 4))[0]
        if ego_d > 5.25:
            self.actor_enumeration += [-2, -2, -2]

        elif len(right_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(right_lane_d_idx)

        # ------------------------------------------- two rigth lane --------------------------------------------------
        rright_lane_d_idx = np.where(((np.array(others_d) - ego_d) > 6.5) * ((np.array(others_d) - ego_d) < 7.5))[0]
        if ego_d > 1.75:
            self.actor_enumeration += [-2, -2, -2]

        elif len(rright_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(rright_lane_d_idx)

        # Fill enumerated actor values

        actor_id_s_d = {}
        norm_s = []
        # norm_d = []
        for actor in self.traffic_module.actors_batch:
            actor_id_s_d[actor['Actor'].id] = actor['Frenet State']

        for i, actor_id in enumerate(self.actor_enumeration):
            if actor_id >= 0:
                actor_norm_s = []
                act_s_hist, act_d = actor_id_s_d[actor_id]  # act_s_hist:list act_d:float
                for act_s, ego_s in zip(list(act_s_hist)[-self.look_back:],
                                        self.actor_enumerated_dict['EGO']['S'][-self.look_back:]):
                    actor_norm_s.append((act_s - ego_s) / self.max_s)
                norm_s.append(actor_norm_s)
            #    norm_d[i] = (act_d - ego_d) / (3 * self.LANE_WIDTH)
            # -1:empty lane, -2:no lane
            else:
                norm_s.append(actor_id)

        # How to fill actor_s when there is no lane or lane is empty. relative_norm_s to ego vehicle
        emp_ln_max = 0.03
        emp_ln_min = -0.03
        no_ln_down = -0.03
        no_ln_up = 0.004
        no_ln = 0.001

        if norm_s[0] not in (-1, -2):
            self.actor_enumerated_dict['LEADING'] = {'S': norm_s[0]}
        else:
            self.actor_enumerated_dict['LEADING'] = {'S': [emp_ln_max]}

        if norm_s[1] not in (-1, -2):
            self.actor_enumerated_dict['FOLLOWING'] = {'S': norm_s[1]}
        else:
            self.actor_enumerated_dict['FOLLOWING'] = {'S': [emp_ln_min]}

        if norm_s[2] not in (-1, -2):
            self.actor_enumerated_dict['LEFT'] = {'S': norm_s[2]}
        else:
            self.actor_enumerated_dict['LEFT'] = {'S': [emp_ln_min] if norm_s[2] == -1 else [no_ln]}

        if norm_s[3] not in (-1, -2):
            self.actor_enumerated_dict['LEFT_UP'] = {'S': norm_s[3]}
        else:
            self.actor_enumerated_dict['LEFT_UP'] = {'S': [emp_ln_max] if norm_s[3] == -1 else [no_ln_up]}

        if norm_s[4] not in (-1, -2):
            self.actor_enumerated_dict['LEFT_DOWN'] = {'S': norm_s[4]}
        else:
            self.actor_enumerated_dict['LEFT_DOWN'] = {'S': [emp_ln_min] if norm_s[4] == -1 else [no_ln_down]}

        if norm_s[5] not in (-1, -2):
            self.actor_enumerated_dict['LLEFT'] = {'S': norm_s[5]}
        else:
            self.actor_enumerated_dict['LLEFT'] = {'S': [emp_ln_min] if norm_s[5] == -1 else [no_ln]}

        if norm_s[6] not in (-1, -2):
            self.actor_enumerated_dict['LLEFT_UP'] = {'S': norm_s[6]}
        else:
            self.actor_enumerated_dict['LLEFT_UP'] = {'S': [emp_ln_max] if norm_s[6] == -1 else [no_ln_up]}

        if norm_s[7] not in (-1, -2):
            self.actor_enumerated_dict['LLEFT_DOWN'] = {'S': norm_s[7]}
        else:
            self.actor_enumerated_dict['LLEFT_DOWN'] = {'S': [emp_ln_min] if norm_s[7] == -1 else [no_ln_down]}

        if norm_s[8] not in (-1, -2):
            self.actor_enumerated_dict['RIGHT'] = {'S': norm_s[8]}
        else:
            self.actor_enumerated_dict['RIGHT'] = {'S': [emp_ln_min] if norm_s[8] == -1 else [no_ln]}

        if norm_s[9] not in (-1, -2):
            self.actor_enumerated_dict['RIGHT_UP'] = {'S': norm_s[9]}
        else:
            self.actor_enumerated_dict['RIGHT_UP'] = {'S': [emp_ln_max] if norm_s[9] == -1 else [no_ln_up]}

        if norm_s[10] not in (-1, -2):
            self.actor_enumerated_dict['RIGHT_DOWN'] = {'S': norm_s[10]}
        else:
            self.actor_enumerated_dict['RIGHT_DOWN'] = {'S': [emp_ln_min] if norm_s[10] == -1 else [no_ln_down]}

        if norm_s[11] not in (-1, -2):
            self.actor_enumerated_dict['RRIGHT'] = {'S': norm_s[11]}
        else:
            self.actor_enumerated_dict['RRIGHT'] = {'S': [emp_ln_min] if norm_s[11] == -1 else [no_ln]}

        if norm_s[12] not in (-1, -2):
            self.actor_enumerated_dict['RRIGHT_UP'] = {'S': norm_s[12]}
        else:
            self.actor_enumerated_dict['RRIGHT_UP'] = {'S': [emp_ln_max] if norm_s[12] == -1 else [no_ln_up]}

        if norm_s[13] not in (-1, -2):
            self.actor_enumerated_dict['RRIGHT_DOWN'] = {'S': norm_s[13]}
        else:
            self.actor_enumerated_dict['RRIGHT_DOWN'] = {'S': [emp_ln_min] if norm_s[13] == -1 else [no_ln_down]}

    def fix_representation(self):
        """
        Given the traffic actors fill the desired tensor with appropriate values and time_steps
        """
        self.enumerate_actors()

        self.actor_enumerated_dict['EGO']['SPEED'].extend(self.actor_enumerated_dict['EGO']['SPEED'][-1]
                                                          for _ in range(
            self.look_back - len(self.actor_enumerated_dict['EGO']['NORM_D'])))

        for act_values in self.actor_enumerated_dict.values():
            act_values['S'].extend(act_values['S'][-1] for _ in range(self.look_back - len(act_values['S'])))

        _range = np.arange(-self.look_back, -1, int(np.ceil(self.look_back / self.time_step)),
                           dtype=int)  # add last observation
        _range = np.append(_range, -1)

        lstm_obs = np.concatenate((np.array(self.actor_enumerated_dict['EGO']['SPEED'])[_range],
                                   np.array(self.actor_enumerated_dict['LEADING']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['FOLLOWING']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LEFT']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LEFT_UP']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LEFT_DOWN']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LLEFT']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LLEFT_UP']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['LLEFT_DOWN']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RIGHT']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RIGHT_UP']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RIGHT_DOWN']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RRIGHT']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RRIGHT_UP']['S'])[_range],
                                   np.array(self.actor_enumerated_dict['RRIGHT_DOWN']['S'])[_range]),
                                  axis=0)

        return lstm_obs.reshape(self.observation_space.shape[1], -1).transpose()  # state

    def step(self, action):
        self.n_step += 1

        self.actor_enumerated_dict['EGO'] = {'NORM_S': [], 'NORM_D': [], 'S': [], 'D': [], 'SPEED': []}
        if self.verbosity: print('ACTION'.ljust(15), '{:+8.6f}, {:+8.6f}'.format(float(action[0]), float(action[1])))
        if self.is_first_path:  # Episode start is bypassed
            action = [0, -1]
            self.is_first_path = False
        """
                **********************************************************************************************************************
                *********************************************** Motion Planner *******************************************************
                **********************************************************************************************************************
        """

        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
        init_speed = speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp, self.max_s]
        fpath, self.lanechange, off_the_road = self.motionPlanner.run_step_single_path(ego_state, self.f_idx,
                                                                                       df_n=action[0], Tf=5,
                                                                                       Vf_n=action[1])
        wps_to_go = len(fpath.t) - 3  # -2 bc len gives # of items not the idx of last item + 2wp controller is used
        self.f_idx = 1

        """
                **********************************************************************************************************************
                ************************************************* Controller *********************************************************
                **********************************************************************************************************************
        """
        # initialize flags
        collision = track_finished = False
        elapsed_time = lambda previous_time: time.time() - previous_time
        path_start_time = time.time()
        ego_init_d, ego_target_d = fpath.d[0], fpath.d[-1]
        # follows path until end of WPs for max 1.5 * path_time or loop counter breaks unless there is a langechange
        loop_counter = 0

        while self.f_idx < wps_to_go and (elapsed_time(path_start_time) < self.motionPlanner.D_T * 1.5 or
                                          loop_counter < self.loop_break or self.lanechange):

            loop_counter += 1
            ego_state = [self.ego.get_location().x, self.ego.get_location().y,
                         math.radians(self.ego.get_transform().rotation.yaw), 0, 0, temp, self.max_s]

            self.f_idx = closest_wp_idx(ego_state, fpath, self.f_idx)
            cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
            cmdWP2 = [fpath.x[self.f_idx + 1], fpath.y[self.f_idx + 1]]

            # overwrite command speed using IDM
            ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]  # estimated current ego_s
            ego_d = fpath.d[self.f_idx]
            vehicle_ahead = self.get_vehicle_ahead(ego_s, ego_d, ego_init_d, ego_target_d)
            cmdSpeed = self.IDM.run_step(vd=self.targetSpeed, vehicle_ahead=vehicle_ahead)

            # control = self.vehicleController.run_step(cmdSpeed, cmdWP)  # calculate control
            control = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
            self.ego.apply_control(control)  # apply control

            """
                    **********************************************************************************************************************
                    ************************************************ Update Carla ********************************************************
                    **********************************************************************************************************************
            """
            self.module_manager.tick()  # Update carla world

            collision_hist = self.world_module.get_collision_history()

            self.actor_enumerated_dict['EGO']['S'].append(ego_s)
            self.actor_enumerated_dict['EGO']['D'].append(ego_d)
            self.actor_enumerated_dict['EGO']['NORM_S'].append((ego_s - self.init_s) / self.track_length)
            self.actor_enumerated_dict['EGO']['NORM_D'].append(
                round((ego_d + self.LANE_WIDTH) / (3 * self.LANE_WIDTH), 2))
            last_speed = get_speed(self.ego)
            self.actor_enumerated_dict['EGO']['SPEED'].append(last_speed / self.maxSpeed)
            # if ego off-the road or collided
            if any(collision_hist):
                collision = True
                break

            distance_traveled = ego_s - self.init_s
            if distance_traveled < -5:
                distance_traveled = self.max_s + distance_traveled
            if distance_traveled >= self.track_length:
                track_finished = True

        """
                *********************************************************************************************************************
                *********************************************** RL Observation ******************************************************
                *********************************************************************************************************************
        """
        self.fea_ext.update()
        features = self.fea_ext.observation
        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            cars_info = self.fix_representation()
            cars_info_fla = cars_info.flatten()
            self.state = np.concatenate((cars_info_fla,features))
            print("len:", len(self.state))
            if self.verbosity == 2:
                print(3 * '---EPS UPDATE---')
                print(TENSOR_ROW_NAMES[0].ljust(15),
                      #      '{:+8.6f}  {:+8.6f}'.format(self.state[-1][1], self.state[-1][0]))
                      '{:+8.6f}'.format(cars_info[-1][0]))
                for idx in range(1, cars_info.shape[1]):
                    print(TENSOR_ROW_NAMES[idx].ljust(15), '{:+8.6f}'.format(cars_info[-1][idx]))

        if self.verbosity == 3: print(cars_info)
        """
                **********************************************************************************************************************
                ********************************************* RL Reward Function *****************************************************
                **********************************************************************************************************************
        """
        car_x, car_y, car_yaw = self.ego.get_location().x, self.ego.get_location().y, \
                                self.ego.get_transform().rotation.yaw
        print(last_speed)
        #speed reward
        e_speed = abs(self.targetSpeed - last_speed)
        r_speed = self.w_r_speed * np.exp(-e_speed ** 2 / self.maxSpeed * self.w_speed)  # 0<= r_speed <= self.w_r_speed
        #  first two path speed change increases regardless so we penalize it differently

        spd_change_percentage = (last_speed - init_speed) / init_speed if init_speed != 0 else -1
        r_laneChange = 0

        if self.lanechange and spd_change_percentage < self.min_speed_gain:
            r_laneChange = -1 * r_speed * self.lane_change_penalty  # <= 0

        elif self.lanechange:
            r_speed *= self.lane_change_reward

        positives = r_speed
        negatives = r_laneChange

        # sigma_pos = 0.3
        sigma_pos = self.sigma_pos
        delta_yaw, wpt_yaw = self._get_delta_yaw()
        road_heading = np.array([
            np.cos(wpt_yaw / 180 * np.pi),
            np.sin(wpt_yaw / 180 * np.pi)
        ])
        pos_err_vec = np.array((car_x, car_y)) - self.current_wpt[0:2]
        lateral_dist = np.linalg.norm(pos_err_vec) * np.sign(
            pos_err_vec[0] * road_heading[1] - pos_err_vec[1] * road_heading[0]) / self.LANE_WIDTH
        track_rewd = np.exp(-np.power(lateral_dist, 2) / 2 / sigma_pos / sigma_pos)
        print("_track", track_rewd)

        # angle reward
        sigma_yaw = self.sigma_angle
        yaw_err = delta_yaw * np.pi / 180
        ang_rewd = np.exp(-np.power(yaw_err, 2) / 2 / sigma_yaw / sigma_yaw)

        print("_ang", ang_rewd)


        reward = positives + negatives + track_rewd + ang_rewd  # r_speed * (1 - lane_change_penalty) <= reward <= r_speed * lane_change_reward
        # print(self.n_step, self.eps_rew)
        """
                      **********************************************************************************************************************
                      ********************************************* Episode Termination ****************************************************
                      **********************************************************************************************************************
              """

        done = False
        if collision:
            # print('Collision happened!')
            reward = self.collision_penalty
            done = True
            self.eps_rew += reward
            # print('eps rew: ', self.n_step, self.eps_rew)
            if self.verbosity: print('REWARD'.ljust(15), '{:+8.6f}'.format(reward))
            return self.state, reward, done, {'reserved': 0}

        elif track_finished:
            # print('Finished the race')
            # reward = 10
            done = True
            if off_the_road:
                reward = self.off_the_road_penalty
            self.eps_rew += reward
            # print('eps rew: ', self.n_step, self.eps_rew)
            if self.verbosity: print('REWARD'.ljust(15), '{:+8.6f}'.format(reward))
            return self.state, reward, done, {'reserved': 0}

        elif off_the_road:
            # print('Collision happened!')
            reward = self.off_the_road_penalty
            # done = True
            self.eps_rew += reward
            # print('eps rew: ', self.n_step, self.eps_rew)
            if self.verbosity: print('REWARD'.ljust(15), '{:+8.6f}'.format(reward))
            return self.state, reward, done, {'reserved': 0}

        self.eps_rew += reward
        # print(self.n_step, self.eps_rew)
        if self.verbosity: print('REWARD'.ljust(15), '{:+8.6f}'.format(reward))
        return self.state, reward, done, {'reserved': 0}


    @property
    def observation_space(self) -> spaces.Space:
        features_space = np.array([np.inf] * 171)
        # return spaces.Dict(road=self.ROAD_FEATURES['space'], vehicle=self.VEHICLE_FEATURES['space'],
        #                    navigation=self.NAVIGATION_FEATURES['space'])
        return spaces.Box(-features_space, features_space, dtype='float32')

    def reset(self):
        self.vehicleController.reset()
        self.world_module.reset()
        self.init_s = self.world_module.init_s
        init_d = self.world_module.init_d
        self.traffic_module.reset(self.init_s, init_d)
        self.motionPlanner.reset(self.init_s, self.world_module.init_d, df_n=0, Tf=4, Vf_n=0, optimal_path=False)
        self.f_idx = 0

        self.n_step = 0  # initialize episode steps count
        self.eps_rew = 0
        self.is_first_path = True
        actors_norm_s_d = []  # relative frenet consecutive s and d values wrt ego
        init_norm_d = round((init_d + self.LANE_WIDTH) / (3 * self.LANE_WIDTH), 2)
        ego_s_list = [self.init_s for _ in range(self.look_back)]
        ego_d_list = [init_d for _ in range(self.look_back)]

        self.actor_enumerated_dict['EGO'] = {'NORM_S': [0], 'NORM_D': [init_norm_d],
                                             'S': ego_s_list, 'D': ego_d_list, 'SPEED': [0]}

        self.fea_ext = FeatureExt(self.world_module, self.dt, self.ego)
        self.fea_ext.update()

        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            cars_state = self.fix_representation()
            road_info = self.fea_ext.observation
            self.state = np.concatenate((cars_state.flatten(), road_info))
            if self.verbosity == 2:
                print(3 * '---RESET---')
                print(TENSOR_ROW_NAMES[0].ljust(15),
                      #      '{:+8.6f}  {:+8.6f}'.format(self.state[-1][1], self.state[-1][0]))
                      '{:+8.6f}'.format(cars_state[-1][0]))
                for idx in range(1, cars_state.shape[1]):
                    print(TENSOR_ROW_NAMES[idx].ljust(15), '{:+8.6f}'.format(cars_state[-1][idx]))
        else:
            pass  # Could be debugged to be used
            # pad the feature lists to recover from the cases where the length of path is less than look_back time

            # self.state = self.non_fix_representation(speeds, ego_norm_s, ego_norm_d, actors_norm_s_d)

        # ---
        # Ego starts to move slightly after being relocated when a new episode starts. Probably, ego keeps a fraction of previous acceleration after
        # being relocated. To solve this, the following procedure is needed.
        self.ego.set_simulate_physics(enabled=False)
        # for _ in range(5):
        self.module_manager.tick()
        self.ego.set_simulate_physics(enabled=True)
        # ----
        return self.state


    def destroy(self):
        print('Destroying environment...')
        if self.world_module is not None:
            self.world_module.destroy()
            self.traffic_module.destroy()
############################################################################

    def _get_obs(self):
        # features
        vehicle_obs = self.fix_representation()
        road_obs = self.fea_ext.observation

        obs = dict(vehicle=vehicle_obs, road=road_obs)
        return replace_nans(obs)

    def _get_delta_yaw(self):
        """
        calculate the delta yaw between ego and current waypoint
        """
        current_wpt = self.world_module.town_map.get_waypoint(location=self.ego.get_location())
        if not current_wpt:
            #self.logger.error('Fail to find a waypoint')
            wpt_yaw = self.current_wpt[2] % 360
        else:
            wpt_yaw = current_wpt.transform.rotation.yaw % 360
            self.current_wpt = np.array((current_wpt.transform.location.x, current_wpt.transform.location.y, current_wpt.transform.rotation.yaw))
        ego_yaw = self.ego.get_transform().rotation.yaw % 360

        delta_yaw = ego_yaw - wpt_yaw
        if 180 <= delta_yaw and delta_yaw <= 360:
            delta_yaw -= 360
        elif -360 <= delta_yaw and delta_yaw <= -180:
            delta_yaw += 360

        return delta_yaw, wpt_yaw
    def _get_velocity(self):
        _v = self.ego.get_velocity()
        ego_velocity = np.array([_v.x, _v.y])
        _acc = self.ego.get_acceleration()
        ego_acc = np.array([_acc.x, _acc.y])
        v = np.linalg.norm(ego_velocity)
        acc = np.linalg.norm(ego_acc)
        return v, acc