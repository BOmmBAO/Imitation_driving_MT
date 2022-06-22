import gym
import pygame.time
from gym import spaces
import itertools
from carla_gym.envs.wrapper import *
from config import cfg
from carla_env.misc import _vec_decompose
from plan_control.path_planner import path_planner
from plan_control.controller import *
from tools.misc import get_speed
import plan_control.velocity_planner as velocity_planner
from enum import Enum
from collections import deque
from carla_gym.envs.hud import HUD
try:
    import numpy as np
    import sys
    from os import path as osp
except ImportError:
    raise RuntimeError('import error!')
from carla_env.featur_1 import *
import copy

import gym
import pygame.time
from gym import spaces
from carla_gym.envs.wrapper import *
from config import cfg
from tools.misc import get_speed
from enum import Enum
from collections import deque
from carla_gym.envs.hud import HUD
import numpy as np
import sys
from os import path as osp
from carla_env.featur_1 import *
from .carla_logger import *

class CarlaEnv(gym.Env):
    """
    action smoothing: Scalar used to smooth the incomming action signal for e2e.
                1.0 = max smoothing, 0.0 = no smoothing
    """
    def __init__(self, args, action_smoothing=0.9, synchronous=True, viewer_res=(1280, 720)): #lanes_change=5):
        self.__version__ = "0.9.9"
        self.logger = setup_carla_logger(
            "output_id", experiment_name=str(2.1))
        self.logger.info("Env running in port {}".format(1.1))
        self.args = args
        self.is_training = True
        pygame.init()
        pygame.font.init()
        # simulation
        self.task_mode = str(cfg.CARLA.TASK_MODE)
        self.max_time_episode = int(cfg.CARLA.MAX_LENGTH)
        self.clock = pygame.time.Clock
        self.width = 1280
        self.height = 720
        self.display = pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.synchronous = synchronous
        self.acceleration_ = 0
        self.global_route = None
        self.points_to_draw = []

        # constraints
        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)
        self.maxAcc = float(cfg.GYM_ENV.MAX_ACC)
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.N_SPAWN_CARS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_CARS)
        self.step_loop = int(cfg.GYM_ENV.LOOP_BREAK)

        # RL
        self.sigma_low_v = float(cfg.RL.SIGMAL_LOW_SPEED)
        self.sigma_high_v = float(cfg.RL.SIGMAL_HIGH_SPEED)
        self.sigma_pos = float(cfg.RL.SIGMA_POS)
        self.sigma_angle = float(cfg.RL.SIGMA_ANGLE)
        self._penalty = float(cfg.RL.LANE_CHANGE_PENALTY)
        self.action_space = gym.spaces.MultiDiscrete([3, 7, 5])
        self.reward = 0.0

        # instances
        self.dt = float(cfg.CARLA.DT)
        self.client = carla.Client(self.args.carla_host, self.args.carla_port)
        self.client.set_timeout(10.0)
        self.world = World(self.client)
        if self.synchronous:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.no_rendering_mode = True
            settings.fixed_delta_seconds = self.dt
            self.world.apply_settings(settings)

        # Spawn the ego vehicle at a fixed position between start and dest

        self.spawn_transform = self.world.map.get_spawn_points()[5]
        self.current_wpt = np.array((self.spawn_transform.location.x, self.spawn_transform.location.y,
                                     self.spawn_transform.rotation.yaw))
        self.actors_batch = []

        self.ego = Hero_Actor(self.world, self.spawn_transform, self.targetSpeed, on_collision_fn=lambda e: self._on_collision(e),
                              on_invasion_fn=lambda e: self._on_invasion(e), on_los_fn=True)
        self.hud = HUD(self.width, self.height)
        self.hud.set_vehicle(self.ego)
        self.world.on_tick(self.hud.on_world_tick)

        # Create cameras
        width, height = viewer_res
        self.camera = Camera(self.world, width, height,
                             transform=camera_transforms["spectator"],
                             attach_to=self.ego, on_recv_image=lambda e: self._set_viewer_image(e),
                             sensor_tick=0.0 if self.synchronous else 1.0 / 30.0)
        self.ego_los_sensor = self.ego.los_sensor
        self.terminal_state = False
        self.total_steps = 0

        # Start Modules
        # Generate waypoints along the lap
        self.pathPlanner = path_planner(self.world, self.fea_ext)
        self.vehicleController = PID_controller(self.ego, self.dt)

        self.world.tick()

    def reset(self, is_training=True):
        self.vehicleController.reset()
        self.ego.collision_sensor.reset()
        self.ego.set_transform(self.spawn_transform)
        self.ego.set_simulate_physics(False)  # Reset the car's physics
        self.ego.set_simulate_physics(True)
        self.ego.control.steer = float(0.0)
        self.ego.control.throttle = float(1.0)
        self.ego.control.brake = float(0.0)
        self.steer = 0.0
        self.throttle_or_break = 0.0
        yaw = (self.ego.get_transform().rotation.yaw) * np.pi / 180.0
        init_speed = carla.Vector3D(
            x=8.0 * np.cos(yaw),
            y=8.0 * np.sin(yaw))
        self.ego.set_velocity(init_speed)
        self.ego.tick()

        _, _, _, csp = self.pathPlanner.update( merge_dist = 8)
        self.motionPlanner.start(csp)
        self.motionPlanner.reset(0, 0 , df_n=0, Tf=4, Vf_n=0, optimal_path=False)
        self.f_idx = 0
        self.terminal_state = False  # Set to True when we want to end episode
        self.closed = False  # Set to True when ESC is pressed
        self.extra_info = []  # List of extra info shown on the HUD
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.start_t = time.time()
        self.step_count = 0
        self.is_first_path = True
        # Metrics
        self.total_reward = 0.0
        self.previous_location = self.ego.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.distance_from_center = 0.0
        self.speed_accum = 0.0
        # self.laps_completed = 0.0
        self.v_buffer = deque([], maxlen=5)
        self.fea_ext = FeatureExt(self.world, self.dt, self.ego)
        self.fea_ext.update(0.0, 1.0)
        return self.fea_ext.observation

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image


    def step(self, action):

        self.step_count += 1
        self.total_steps += 1
        if self.is_first_path:  # Episode start is bypassed
            action = [0, -1]
            self.is_first_path = False
        """
                **********************************************************************************************************************
                *********************************************** Motion Planner *******************************************************
                **********************************************************************************************************************
        """
        rx, ry, _, csp = self.pathPlanner.update(merge_dist=8)
        self.motionPlanner.start(csp)
        self.fea_ext.ref_display(rx, ry)
        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
        init_speed = speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp, 40]
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
        #ego_init_d, ego_target_d = fpath.d[0], fpath.d[-1]
        # follows path until end of WPs for max 1.5 * path_time or loop counter breaks unless there is a langechange
        loop_counter = 0

        #while self.f_idx < wps_to_go and (elapsed_time(path_start_time) < self.motionPlanner.D_T * 1.5 or loop_counter < self.loop_break or self.lanechange):
        while loop_counter <=5:
            last_speed = get_speed(self.ego)
            self.v_buffer.append(last_speed)
            loop_counter += 1
            ego_state = [self.ego.get_location().x, self.ego.get_location().y,
                         math.radians(self.ego.get_transform().rotation.yaw), 0, 0, temp, self.max_s]

            self.f_idx = closest_wp_idx(ego_state, fpath, self.f_idx)
            cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
            cmdWP2 = [fpath.x[self.f_idx + 1], fpath.y[self.f_idx + 1]]

            # overwrite command speed using IDM
            # for zombie cars
            # ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]  # estimated current ego_s
            # ego_d = fpath.d[self.f_idx]
            #vehicle_ahead = self.get_vehicle_ahead(ego_s, ego_d, ego_init_d, ego_target_d)
            cmdSpeed = self.IDM.run_step(vd=self.targetSpeed, vehicle_ahead=None)

            # control = self.vehicleController.run_step(cmdSpeed, cmdWP)  # calculate control
            control = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
            self.ego.apply_control(control)  # apply control
            self.hud.tick(self.world, self.clock)
            self.world.tick()
            # Get most recent observation and viewer image
            self.viewer_image = self._get_viewer_image()
            last_speed = get_speed(self.ego)
            self.v_buffer.append(last_speed)
            if any(self.ego.collision_sensor.get_collision_history()):
                self.terminal_state = True
                break
            elif last_speed > self.maxSpeed:
                self.terminal_state = True
                self.logger.debug('too fast')
            elif len(self.v_buffer) ==5:
                if np.mean(self.v_buffer) < 4 or np.mean(self.v_buffer) >20:
                    self.terminal_state = True


        """
                *********************************************************************************************************************
                *********************************************** RL Observation ******************************************************
                *********************************************************************************************************************
        """
        self.fea_ext.update()
        self.fea_ext.observation

        # Accumulate speed
        self.speed_accum += self.ego.get_speed()
        """
                      **********************************************************************************************************************
                      ********************************************* Episode Termination ****************************************************
                      **********************************************************************************************************************
              """

        self.center_lane_deviation += self.distance_from_center
        transform = self.ego.get_transform()
        self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location
        self.speed_accum += self.ego.get_speed()

        # if self.distance_from_center >= 1.2:
        #     print('Collision happened because of off the road!')
        #     self.terminal_state = True
        if off_the_road:
            # print('Collision happened!')
            # done = True
            self.terminal_state = True

        elif len(self.ego.collision_sensor.get_collision_history()) != 0:
            self.terminal_state = True
            print('Collision !')

        """
                       **********************************************************************************************************************
                       ********************************************* RL Reward Function *****************************************************
                       **********************************************************************************************************************
               """
        car_x = self.ego.get_location().x
        car_y = self.ego.get_location().y
        self.center_lane_deviation += self.distance_from_center
        transform = self.ego.get_transform()
        self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location
        # Calculate reward
        if not self.terminal_state:
            self.current_wpt = self._get_waypoint_xyz()
            delta_yaw, wpt_yaw = self._get_delta_yaw()
            delta_yaw = np.deg2rad(delta_yaw)
            road_heading = np.array([
                np.cos(wpt_yaw / 180 * np.pi),
                np.sin(wpt_yaw / 180 * np.pi)
            ])
            pos_err_vec = np.array((car_x, car_y)) - self.current_wpt[0:2]
            self.distance_from_center = abs(np.linalg.norm(pos_err_vec) * np.sign(
                pos_err_vec[0] * road_heading[1] - pos_err_vec[1] * road_heading[0]))
            centering_factor = np.exp(
                -np.power(self.distance_from_center / self.LANE_WIDTH, 2) / 2 / self.sigma_pos / self.sigma_pos)
            angle_factor = np.exp(-np.power(delta_yaw, 2) / 2 / self.sigma_angle / self.sigma_angle)
            sigma_vel = self.sigma_low_v if last_speed <= self.targetSpeed else self.sigma_high_v
            speed_factor = np.exp(-np.power(last_speed - self.targetSpeed, 2) / 2 / sigma_vel / sigma_vel)

            self.reward = speed_factor * centering_factor * angle_factor
        else:
            self.reward -= 10
            self.test.log({'traveled': self.distance_traveled, 'epo_reward': self.total_reward,
                           'average_speed': self.speed_accum / self.step_count})
        # Update checkpoint for training
        if self.step_count >= self.max_time_episode:
            self.logger.debug('Time out! Episode cost %d steps in route.' % self.step_count)
            self.terminal_state = True
            self.test.log({'traveled': self.distance_traveled, 'epo_reward': self.total_reward,
                           'average_speed': self.speed_accum / self.step_count})
        self.total_reward += self.reward
        self.render()
        self.test.log({'reward': self.reward})
        return self._get_obs(), self.reward, self.terminal_state, {'reserved': 0}


    @property
    def observation_space(self) -> spaces.Space:
        features_space = np.array([np.inf] * 94)
        # return spaces.Dict(road=self.ROAD_FEATURES['space'], vehicle=self.VEHICLE_FEATURES['space'],
        #                    navigation=self.NAVIGATION_FEATURES['space'])
        return spaces.Box(-features_space, features_space, dtype='float32')
    def _get_obs(self):
        self.fea_ext.update()
        return self.fea_ext.observation


    def close(self):
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):

        # Add metrics to HUD
        self.extra_info.extend([
            "Reward: % 19.2f" % self.reward,
            "Total Reward: % 19.2f" % self.total_reward,
            "",
            # "Maneuver:        % 11s" % maneuver,
            # "Laps completed:    % 7.2f %%" % (self.laps_completed * 100.0),
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (3.6 * self.speed_accum / self.step_count),
            "Total Steps:        % 9d" %(self.total_steps),
        ])
        # Blit image from spectator camera
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list

        # Render to screen
        pygame.display.flip()

        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
############################################################################

    def _on_collision(self, event):
        self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))

    def _get_delta_yaw(self):
        """
        calculate the delta yaw between ego and current waypoint
        """
        current_wpt = self.world.map.get_waypoint(location=self.ego.get_location())
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

    def _get_waypoint_xyz(self):
        """
        Get the (x,y) waypoint of current ego position
            if t != 0 and None, return the wpt of last moment
            if t == 0 and None wpt: return self.starts
        """
        waypoint = self.world.map.get_waypoint(location=self.ego.get_location())
        if waypoint:
            return np.array(
                (waypoint.transform.location.x, waypoint.transform.location.y,
                 waypoint.transform.rotation.yaw))
        else:
            return self.current_wpt

    # def get_vehicle_ahead(self, ego_s, ego_d, ego_init_d, ego_target_d):
    #     """
    #     This function returns the values for the leading actor in front of the ego vehicle. When there is lane-change
    #     it is important to consider actor in the current lane and target lane. If leading actor in the current lane is
    #     too close than it is considered to be vehicle_ahead other wise target lane is prioritized.
    #     """
    #
    #     distance = self.effective_distance_from_vehicle_ahead
    #     others_s = [0 for _ in range(self.N_SPAWN_CARS)]
    #     others_d = [0 for _ in range(self.N_SPAWN_CARS)]
    #     for i, actor in enumerate(self.traffic_module.actors_batch):
    #         act_s, act_d = actor['Frenet State'][0][-1], actor['Frenet State'][1]
    #         others_s[i] = act_s
    #         others_d[i] = act_d
    #
    #     init_lane_d_idx = \
    #         np.where((abs(np.array(others_d) - ego_d) < 1.75) * (abs(np.array(others_d) - ego_init_d) < 1))[0]
    #
    #     init_lane_strict_d_idx = \
    #         np.where((abs(np.array(others_d) - ego_d) < 0.4) * (abs(np.array(others_d) - ego_init_d) < 1))[0]
    #
    #     target_lane_d_idx = \
    #         np.where((abs(np.array(others_d) - ego_d) < 3.3) * (abs(np.array(others_d) - ego_target_d) < 1))[0]
    #
    #     if len(init_lane_d_idx) and len(target_lane_d_idx) == 0:
    #         return None  # no vehicle ahead
    #     else:
    #         init_lane_s = np.array(others_s)[init_lane_d_idx]
    #         init_s_idx = np.concatenate(
    #             (np.array(init_lane_d_idx).reshape(-1, 1), (init_lane_s - ego_s).reshape(-1, 1),)
    #             , axis=1)
    #         sorted_init_s_idx = init_s_idx[init_s_idx[:, 1].argsort()]
    #
    #         init_lane_strict_s = np.array(others_s)[init_lane_strict_d_idx]
    #         init_strict_s_idx = np.concatenate(
    #             (np.array(init_lane_strict_d_idx).reshape(-1, 1), (init_lane_strict_s - ego_s).reshape(-1, 1),)
    #             , axis=1)
    #         sorted_init_strict_s_idx = init_strict_s_idx[init_strict_s_idx[:, 1].argsort()]
    #
    #         target_lane_s = np.array(others_s)[target_lane_d_idx]
    #         target_s_idx = np.concatenate((np.array(target_lane_d_idx).reshape(-1, 1),
    #                                        (target_lane_s - ego_s).reshape(-1, 1),), axis=1)
    #         sorted_target_s_idx = target_s_idx[target_s_idx[:, 1].argsort()]
    #
    #         if any(sorted_init_s_idx[:, 1][sorted_init_s_idx[:, 1] <= 10] > 0):
    #             vehicle_ahead_idx = int(sorted_init_s_idx[:, 0][sorted_init_s_idx[:, 1] > 0][0])
    #         elif any(sorted_init_strict_s_idx[:, 1][sorted_init_strict_s_idx[:, 1] <= distance] > 0):
    #             vehicle_ahead_idx = int(sorted_init_strict_s_idx[:, 0][sorted_init_strict_s_idx[:, 1] > 0][0])
    #         elif any(sorted_target_s_idx[:, 1][sorted_target_s_idx[:, 1] <= distance] > 0):
    #             vehicle_ahead_idx = int(sorted_target_s_idx[:, 0][sorted_target_s_idx[:, 1] > 0][0])
    #         else:
    #             return None
    #
    #         # print(others_s[vehicle_ahead_idx] - ego_s, others_d[vehicle_ahead_idx], ego_d)
    #
    #         return self.traffic_module.actors_batch[vehicle_ahead_idx]['Actor']
    #
    # def enumerate_actors(self):
    #     """
    #     Given the traffic actors and ego_state this fucntion enumerate actors, calculates their relative positions with
    #     to ego and assign them to actor_enumerated_dict.
    #     Keys to be updated: ['LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN', 'LLEFT', 'LLEFT_UP',
    #     'LLEFT_DOWN', 'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN', 'RRIGHT', 'RRIGHT_UP', 'RRIGHT_DOWN']
    #     """
    #
    #     self.actor_enumeration = []
    #     ego_s = self.actor_enumerated_dict['EGO']['S'][-1]
    #     ego_d = self.actor_enumerated_dict['EGO']['D'][-1]
    #
    #     others_s = [0 for _ in range(self.N_SPAWN_CARS)]
    #     others_d = [0 for _ in range(self.N_SPAWN_CARS)]
    #     others_id = [0 for _ in range(self.N_SPAWN_CARS)]
    #     for i, actor in enumerate(self.traffic_module.actors_batch):
    #         act_s, act_d = actor['Frenet State']
    #         others_s[i] = act_s[-1]
    #         others_d[i] = act_d
    #         others_id[i] = actor['Actor'].id
    #
    #     def append_actor(x_lane_d_idx, actor_names=None):
    #         # actor names example: ['left', 'leftUp', 'leftDown']
    #         x_lane_s = np.array(others_s)[x_lane_d_idx]
    #         x_lane_id = np.array(others_id)[x_lane_d_idx]
    #         s_idx = np.concatenate((np.array(x_lane_d_idx).reshape(-1, 1), (x_lane_s - ego_s).reshape(-1, 1),
    #                                 x_lane_id.reshape(-1, 1)), axis=1)
    #         sorted_s_idx = s_idx[s_idx[:, 1].argsort()]
    #
    #         self.actor_enumeration.append(
    #             others_id[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < self.side_window][0])] if (
    #                 any(abs(
    #                     sorted_s_idx[:, 1][abs(sorted_s_idx[:, 1]) <= self.side_window]) >= -self.side_window)) else -1)
    #
    #         self.actor_enumeration.append(
    #             others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > self.side_window][0])] if (
    #                 any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > self.side_window)) else -1)
    #
    #         self.actor_enumeration.append(
    #             others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < -self.side_window][-1])] if (
    #                 any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < -self.side_window)) else -1)
    #
    #     # --------------------------------------------- ego lane -------------------------------------------------
    #     same_lane_d_idx = np.where(abs(np.array(others_d) - ego_d) < 1)[0]
    #     if len(same_lane_d_idx) == 0:
    #         self.actor_enumeration.append(-2)
    #         self.actor_enumeration.append(-2)
    #
    #     else:
    #         same_lane_s = np.array(others_s)[same_lane_d_idx]
    #         same_lane_id = np.array(others_id)[same_lane_d_idx]
    #         same_s_idx = np.concatenate((np.array(same_lane_d_idx).reshape(-1, 1), (same_lane_s - ego_s).reshape(-1, 1),
    #                                      same_lane_id.reshape(-1, 1)), axis=1)
    #         sorted_same_s_idx = same_s_idx[same_s_idx[:, 1].argsort()]
    #         self.actor_enumeration.append(others_id[int(sorted_same_s_idx[:, 0][sorted_same_s_idx[:, 1] > 0][0])]
    #                                       if (any(sorted_same_s_idx[:, 1] > 0)) else -1)
    #         self.actor_enumeration.append(others_id[int(sorted_same_s_idx[:, 0][sorted_same_s_idx[:, 1] < 0][-1])]
    #                                       if (any(sorted_same_s_idx[:, 1] < 0)) else -1)
    #
    #     # --------------------------------------------- left lane -------------------------------------------------
    #     left_lane_d_idx = np.where(((np.array(others_d) - ego_d) < -3) * ((np.array(others_d) - ego_d) > -4))[0]
    #     if ego_d < -1.75:
    #         self.actor_enumeration += [-2, -2, -2]
    #
    #     elif len(left_lane_d_idx) == 0:
    #         self.actor_enumeration += [-1, -1, -1]
    #
    #     else:
    #         append_actor(left_lane_d_idx)
    #
    #     # ------------------------------------------- two left lane -----------------------------------------------
    #     lleft_lane_d_idx = np.where(((np.array(others_d) - ego_d) < -6.5) * ((np.array(others_d) - ego_d) > -7.5))[0]
    #
    #     if ego_d < 1.75:
    #         self.actor_enumeration += [-2, -2, -2]
    #
    #     elif len(lleft_lane_d_idx) == 0:
    #         self.actor_enumeration += [-1, -1, -1]
    #
    #     else:
    #         append_actor(lleft_lane_d_idx)
    #
    #         # ---------------------------------------------- rigth lane --------------------------------------------------
    #     right_lane_d_idx = np.where(((np.array(others_d) - ego_d) > 3) * ((np.array(others_d) - ego_d) < 4))[0]
    #     if ego_d > 5.25:
    #         self.actor_enumeration += [-2, -2, -2]
    #
    #     elif len(right_lane_d_idx) == 0:
    #         self.actor_enumeration += [-1, -1, -1]
    #
    #     else:
    #         append_actor(right_lane_d_idx)
    #
    #     # ------------------------------------------- two rigth lane --------------------------------------------------
    #     rright_lane_d_idx = np.where(((np.array(others_d) - ego_d) > 6.5) * ((np.array(others_d) - ego_d) < 7.5))[0]
    #     if ego_d > 1.75:
    #         self.actor_enumeration += [-2, -2, -2]
    #
    #     elif len(rright_lane_d_idx) == 0:
    #         self.actor_enumeration += [-1, -1, -1]
    #
    #     else:
    #         append_actor(rright_lane_d_idx)
    #
    #     # Fill enumerated actor values
    #
    #     actor_id_s_d = {}
    #     norm_s = []
    #     # norm_d = []
    #     for actor in self.traffic_module.actors_batch:
    #         actor_id_s_d[actor['Actor'].id] = actor['Frenet State']
    #
    #     for i, actor_id in enumerate(self.actor_enumeration):
    #         if actor_id >= 0:
    #             actor_norm_s = []
    #             act_s_hist, act_d = actor_id_s_d[actor_id]  # act_s_hist:list act_d:float
    #             for act_s, ego_s in zip(list(act_s_hist)[-self.look_back:],
    #                                     self.actor_enumerated_dict['EGO']['S'][-self.look_back:]):
    #                 actor_norm_s.append((act_s - ego_s) / self.max_s)
    #             norm_s.append(actor_norm_s)
    #         #    norm_d[i] = (act_d - ego_d) / (3 * self.LANE_WIDTH)
    #         # -1:empty lane, -2:no lane
    #         else:
    #             norm_s.append(actor_id)
    #
    #     # How to fill actor_s when there is no lane or lane is empty. relative_norm_s to ego vehicle
    #     emp_ln_max = 0.03
    #     emp_ln_min = -0.03
    #     no_ln_down = -0.03
    #     no_ln_up = 0.004
    #     no_ln = 0.001
    #
    #     if norm_s[0] not in (-1, -2):
    #         self.actor_enumerated_dict['LEADING'] = {'S': norm_s[0]}
    #     else:
    #         self.actor_enumerated_dict['LEADING'] = {'S': [emp_ln_max]}
    #
    #     if norm_s[1] not in (-1, -2):
    #         self.actor_enumerated_dict['FOLLOWING'] = {'S': norm_s[1]}
    #     else:
    #         self.actor_enumerated_dict['FOLLOWING'] = {'S': [emp_ln_min]}
    #
    #     if norm_s[2] not in (-1, -2):
    #         self.actor_enumerated_dict['LEFT'] = {'S': norm_s[2]}
    #     else:
    #         self.actor_enumerated_dict['LEFT'] = {'S': [emp_ln_min] if norm_s[2] == -1 else [no_ln]}
    #
    #     if norm_s[3] not in (-1, -2):
    #         self.actor_enumerated_dict['LEFT_UP'] = {'S': norm_s[3]}
    #     else:
    #         self.actor_enumerated_dict['LEFT_UP'] = {'S': [emp_ln_max] if norm_s[3] == -1 else [no_ln_up]}
    #
    #     if norm_s[4] not in (-1, -2):
    #         self.actor_enumerated_dict['LEFT_DOWN'] = {'S': norm_s[4]}
    #     else:
    #         self.actor_enumerated_dict['LEFT_DOWN'] = {'S': [emp_ln_min] if norm_s[4] == -1 else [no_ln_down]}
    #
    #     if norm_s[5] not in (-1, -2):
    #         self.actor_enumerated_dict['LLEFT'] = {'S': norm_s[5]}
    #     else:
    #         self.actor_enumerated_dict['LLEFT'] = {'S': [emp_ln_min] if norm_s[5] == -1 else [no_ln]}
    #
    #     if norm_s[6] not in (-1, -2):
    #         self.actor_enumerated_dict['LLEFT_UP'] = {'S': norm_s[6]}
    #     else:
    #         self.actor_enumerated_dict['LLEFT_UP'] = {'S': [emp_ln_max] if norm_s[6] == -1 else [no_ln_up]}
    #
    #     if norm_s[7] not in (-1, -2):
    #         self.actor_enumerated_dict['LLEFT_DOWN'] = {'S': norm_s[7]}
    #     else:
    #         self.actor_enumerated_dict['LLEFT_DOWN'] = {'S': [emp_ln_min] if norm_s[7] == -1 else [no_ln_down]}
    #
    #     if norm_s[8] not in (-1, -2):
    #         self.actor_enumerated_dict['RIGHT'] = {'S': norm_s[8]}
    #     else:
    #         self.actor_enumerated_dict['RIGHT'] = {'S': [emp_ln_min] if norm_s[8] == -1 else [no_ln]}
    #
    #     if norm_s[9] not in (-1, -2):
    #         self.actor_enumerated_dict['RIGHT_UP'] = {'S': norm_s[9]}
    #     else:
    #         self.actor_enumerated_dict['RIGHT_UP'] = {'S': [emp_ln_max] if norm_s[9] == -1 else [no_ln_up]}
    #
    #     if norm_s[10] not in (-1, -2):
    #         self.actor_enumerated_dict['RIGHT_DOWN'] = {'S': norm_s[10]}
    #     else:
    #         self.actor_enumerated_dict['RIGHT_DOWN'] = {'S': [emp_ln_min] if norm_s[10] == -1 else [no_ln_down]}
    #
    #     if norm_s[11] not in (-1, -2):
    #         self.actor_enumerated_dict['RRIGHT'] = {'S': norm_s[11]}
    #     else:
    #         self.actor_enumerated_dict['RRIGHT'] = {'S': [emp_ln_min] if norm_s[11] == -1 else [no_ln]}
    #
    #     if norm_s[12] not in (-1, -2):
    #         self.actor_enumerated_dict['RRIGHT_UP'] = {'S': norm_s[12]}
    #     else:
    #         self.actor_enumerated_dict['RRIGHT_UP'] = {'S': [emp_ln_max] if norm_s[12] == -1 else [no_ln_up]}
    #
    #     if norm_s[13] not in (-1, -2):
    #         self.actor_enumerated_dict['RRIGHT_DOWN'] = {'S': norm_s[13]}
    #     else:
    #         self.actor_enumerated_dict['RRIGHT_DOWN'] = {'S': [emp_ln_min] if norm_s[13] == -1 else [no_ln_down]}