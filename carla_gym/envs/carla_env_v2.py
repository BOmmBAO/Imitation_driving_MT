import gym
import pygame.time
from gym import spaces
import time
import itertools
from carla_gym.envs.wrapper import *
from config import cfg
from carla_env.misc import *
from carla_gym.envs.planner import compute_route_waypoints
from plan_control.frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from plan_control.controller import VehiclePIDController
from tools.misc import get_speed
from plan_control.controller import IntelligentDriverModel
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
from tools.modules import *
from carla_gym.envs.wrapper import *
from config import cfg
from carla_gym.envs.coordinates import *
from tools.misc import get_speed
from enum import Enum
from collections import deque
from carla_gym.envs.hud import HUD
import numpy as np
import sys
from os import path as osp
from carla_env.featur_1 import *
from .carla_logger import *
MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_TRAFFIC = 'TRAFFIC'
TENSOR_ROW_NAMES = ['EGO', 'LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN','LLEFT', 'LLEFT_UP', 'LLEFT_DOWN',
                    'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN', 'RRIGHT', 'RRIGHT_UP', 'RRIGHT_DOWN']

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class CarlaEnv(gym.Env):
    """
    action smoothing: Scalar used to smooth the incomming action signal for e2e.
                1.0 = max smoothing, 0.0 = no smoothing
    """
    def __init__(self, args, action_smoothing=0.9, synchronous=True, viewer_res=(1280, 720), obs_res=(1280, 720)): #lanes_change=5):
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
        # self.global_route = np.load(
        #         'road_maps/global_route_town04.npy')  # track waypoints (center lane of the second lane from left)
        self.global_route = None

        # constraints
        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)
        self.maxAcc = float(cfg.GYM_ENV.MAX_ACC)
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.N_SPAWN_CARS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_CARS)

        # frenet
        self.f_idx = 0
        self.init_s = 0.0  # initial frenet s value - will be updated in reset function
        self.max_s = int(cfg.CARLA.MAX_S)
        self.track_length = int(cfg.GYM_ENV.TRACK_LENGTH)
        self.look_back = int(cfg.GYM_ENV.LOOK_BACK)
        self.time_step = int(cfg.GYM_ENV.TIME_STEP)
        self.loop_break = int(cfg.GYM_ENV.LOOP_BREAK)
        self.effective_distance_from_vehicle_ahead = int(cfg.GYM_ENV.DISTN_FRM_VHCL_AHD)
        self.lanechange = False

        # RL
        self.w_speed = int(cfg.RL.W_SPEED)
        self.w_r_speed = int(cfg.RL.W_R_SPEED)
        self.w_r_pos = int(cfg.RL.SIGMA_POS)
        self.w_r_angle = int(cfg.RL.SIGMA_ANGLE)

        self.min_speed_gain = float(cfg.RL.MIN_SPEED_GAIN)
        self.min_speed_loss = float(cfg.RL.MIN_SPEED_LOSS)
        self.lane_change_reward = float(cfg.RL.LANE_CHANGE_REWARD)
        self.lane_change_penalty = float(cfg.RL.LANE_CHANGE_PENALTY)
        self.sigma_pos = float(cfg.RL.SIGMA_POS)
        self.sigma_angle = float(cfg.RL.SIGMA_ANGLE)

        self.off_the_road_penalty = int(cfg.RL.OFF_THE_ROAD)
        self.collision_penalty = int(cfg.RL.COLLISION)
        action_high = np.array([1, 1])
        self.action_space = gym.spaces.Box(low=-action_high, high=action_high, shape=(2,), dtype=np.float32)
        #self.action_smoothing = action_smoothing #for e2e to
        self.total_reward = 0.0
        self.reward = 0.0

        # instances
        self.dt = float(cfg.CARLA.DT)
        self.client = carla.Client(self.args.carla_host, self.args.carla_port)
        self.client.set_timeout(3.0)
        self.world = World(self.args, self.client)
        if self.task_mode == 'Straight':
            world = 'Town01'
        elif self.task_mode == 'Curve':
            # self.world = self.client.load_world('Town01')
            world = 'Town05'
        elif self.task_mode == 'Long':
            world = 'Town01'
            # self.world = self.client.load_world('Town02')
        elif self.task_mode == 'Lane':
            # self.world = self.client.load_world('Town01')
            world = 'Town05'
        elif self.task_mode == 'U_curve':
            world = 'Town03'
        elif self.task_mode == 'Lane_test':
            world = 'Town03'
        self.world = World(self.client, world)
        if self.synchronous:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.dt
            self.world.apply_settings(settings)
            # Load routes
        self.starts, self.dests = train_coordinates(self.task_mode)
        # Spawn the ego vehicle at a fixed position between start and dest
        # Start and Destination
        if self.task_mode == 'Straight':
            self.route_id = 0
        elif self.task_mode == 'Curve':
            self.route_id = 1  # np.random.randint(2, 4)
        elif self.task_mode == 'Long' or self.task_mode == 'Lane' or self.task_mode == 'Lane_test':
            self.route_id = 0
        elif self.task_mode == 'U_curve':
            self.route_id = 0
        self.start = self.starts[self.route_id]
        self.dest = self.dests[self.route_id]
        self.current_wpt = np.array((self.start[0], self.start[1],
                                     self.start[5]))
        self.spawn_transform = self._set_carla_transform(self.start)
        self.ego = Hero_Actor(self.world, self.spawn_transform, on_collision_fn=lambda e: self._on_collision(e),
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
        self.init_local_route = self.local_route = self._generate_local_route(self.spawn_transform.location)
        self.motionPlanner = MotionPlanner()
        self.motionPlanner.start(self.local_route)
        self.ego.update_global_route_csp(self.motionPlanner.csp)
        self.ego_los_sensor = self.ego.los_sensor
        self.vehicleController = VehiclePIDController(self.ego, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0})
        self.IDM = IntelligentDriverModel(self.ego.actor)
        self.world.tick()

    def reset(self, is_training=True):
        self.ego.collision_sensor.reset()
        self.is_first_path = True
        # self.vehicle.control.brake = float(0.0)
        self.ego.tick()
        self.ego.set_transform(self.spawn_transform)
        self.ego.set_simulate_physics(False)  # Reset the car's physics
        self.ego.set_simulate_physics(True)
        yaw = (self.ego.get_transform().rotation.yaw) * np.pi / 180.0
        init_speed = carla.Vector3D(
            x=self.targetSpeed * np.cos(yaw),
            y=self.targetSpeed * np.sin(yaw))
        self.ego.set_velocity(init_speed)
        self.vehicleController.reset()
        self.motionPlanner.start(self.init_local_route)
        self.motionPlanner.reset(0.0, 0.0, df_n=0, Tf=4, Vf_n=0, optimal_path=False)
        self.f_idx = 0
        self.terminal_state = False  # Set to True when we want to end episode
        self.closed = False  # Set to True when ESC is pressed
        self.extra_info = []  # List of extra info shown on the HUD
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.start_t = time.time()
        self.step_count = 0

        # Metrics
        self.total_reward = 0.0
        self.previous_location = self.ego.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.distance_from_center = 0.0
        self.speed_accum = 0.0
        # self.laps_completed = 0.0
        self.fea_ext = FeatureExt(self.world, self.dt, self.ego)
        self.fea_ext.update()
        self.v_buffer = deque([], maxlen=5)
        return self._get_obs()

    def _generate_local_route(self, start):

        self.local_route = np.empty((0, 3))
        distance = 1
        for i in range(250):
            wp = self.world.map.get_waypoint(carla.Location(x=start.x, y=start.y, z=start.z),
                                                         project_to_road=True).next(distance=distance)[0]
            distance += 4
            self.local_route = np.append(self.local_route,
                                          [[wp.transform.location.x, wp.transform.location.y,
                                            wp.transform.location.z]], axis=0)

    def _set_carla_transform(self, pose):
        """Get a carla tranform object given pose.
        Args:
            pose: [x, y, z, pitch, roll, yaw].
        Returns:
            transform: the carla transform object
        """
        transform = carla.Transform()
        transform.location.x = pose[0]
        transform.location.y = pose[1]
        transform.location.z = pose[2]
        transform.rotation.pitch = pose[3]
        transform.rotation.roll = pose[4]
        transform.rotation.yaw = pose[5]
        return transform

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
            #vehicle_ahead = self.get_vehicle_ahead(ego_s, ego_d, ego_init_d, ego_target_d)
            cmdSpeed = self.IDM.run_step(vd=self.targetSpeed, vehicle_ahead=None)

            # control = self.vehicleController.run_step(cmdSpeed, cmdWP)  # calculate control
            control = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
            self.ego.apply_control(control)  # apply control

            """
                    **********************************************************************************************************************
                    ************************************************ Update Carla ********************************************************
                    **********************************************************************************************************************
            """

            self.hud.tick(self.world, self.clock)
            self.world.tick()
            # Get most recent observation and viewer image
            self.viewer_image = self._get_viewer_image()
            self._generate_local_route()
            self.motionPlanner.start(self.local_route)

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
                ********************************************* RL Reward Function *****************************************************
                **********************************************************************************************************************
        """
        car_x, car_y, car_yaw = self.ego.get_location().x, self.ego.get_location().y, \
                                self.ego.get_transform().rotation.yaw
        self.current_wpt = self._get_waypoint_xyz()
        last_speed = get_speed(self.ego)
        #speed reward
        sigmal_v = 0.6 if last_speed <= self.targetSpeed else 1.0
        r_speed = self.w_r_speed * np.exp(-(last_speed / self.targetSpeed) ** 2 / 2 / (sigmal_v ** 2)) # 0<= r_speed <= self.w_r_speed
        #  first two path speed change increases regardless so we penalize it differently

        speed_change_percentage = (last_speed - init_speed) / init_speed if init_speed != 0 else -1
        r_laneChange = 0

        if self.lanechange and speed_change_percentage < self.min_speed_gain:   # unmeaningful lanechange
            r_laneChange = -1 * r_speed * self.lane_change_penalty  # <= 0

        # elif self.lanechange:
        #     r_speed *= self.lane_change_reward

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
        self.distance_from_center = abs(np.linalg.norm(pos_err_vec) * np.sign(
            pos_err_vec[0] * road_heading[1] - pos_err_vec[1] * road_heading[0]))
        #lateral_dist = self.distance_from_center / self.LANE_WIDTH
        track_rewd = np.exp(- np.power(self.distance_from_center, 2) / 2 / sigma_pos / sigma_pos)
        print("_track", track_rewd)

        # angle reward
        sigma_yaw = self.sigma_angle
        yaw_err = delta_yaw * np.pi / 180
        ang_rewd = np.exp(- np.power(yaw_err, 2) / 2 / sigma_yaw / sigma_yaw)

        print("_ang", ang_rewd)


        self.reward = (positives + negatives) * track_rewd * ang_rewd  # r_speed * (1 - lane_change_penalty) <= reward <= r_speed * lane_change_reward
        # print(self.n_step, self.eps_rew)
        print("reward:", positives, negatives)
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

        if self.step_count >= self.max_time_episode:
            # print("Time out! Episode Done.")
            self.logger.debug('Time out! Episode cost %d steps in route %d.' %
                              (self.step_count, self.route_id))
            self.terminal_state = True

        elif self.distance_from_center >= 0.8:
            print('Collision happened because of off the road!')
            self.reward = self.off_the_road_penalty
            self.terminal_state = True

        elif last_speed < 4:
            self.terminal_state = True
            self.reward = self.off_the_road_penalty
            print('speed low!')
        elif any(self.ego.collision_sensor.get_collision_history()):
            self.terminal_state = True
            self.reward = self.off_the_road_penalty
            print('Collision !')
        # If at destination
        elif np.sqrt((car_x - self.dest[0]) ** 2 + (car_y - self.dest[1]) ** 2) < 2.0:
            print("Get destination! Episode Done.")
            self.logger.debug('Get destination! Episode cost %d steps in route %d.' % (self.step_count, self.route_id))
            # self.isSuccess = True
            self.terminal_state = True

        # Update checkpoint for training
        self.total_reward += self.reward
        self.render()
        return self._get_obs(), self.reward, self.terminal_state, {'reserved': 0}


    @property
    def observation_space(self) -> spaces.Space:
        features_space = np.array([np.inf] * 327)
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
            "Avg speed:      % 7.2f km/h" % (3.6 * self.speed_accum / self.step_count)
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