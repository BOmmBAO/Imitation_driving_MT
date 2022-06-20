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
from gym.utils import seeding
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
    def __init__(self, args, action_smoothing=0.9, synchronous=True, viewer_res=(1280, 720)): #lanes_change=5):
        self.__version__ = "0.9.9"
        self.logger = setup_carla_logger(
            "output_id", experiment_name=str(1.1))
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

        # constraints
        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)
        self.maxAcc = float(cfg.GYM_ENV.MAX_ACC)
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.N_SPAWN_CARS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_CARS)


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
        self.action_smoothing = action_smoothing
        self.total_reward = 0.0
        self.reward = 0.0


        # instances
        self.dt = float(cfg.CARLA.DT)
        self.client = carla.Client(self.args.carla_host, self.args.carla_port)
        self.client.set_timeout(3.0)
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
        self.start = self.starts[self.route_id]
        self.dest = self.dests[self.route_id]
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


    def reset(self, is_training=True):
        self.ego.collision_sensor.reset()
        self.ego.control.steer = float(0.0)
        self.ego.control.throttle = float(0.0)
        self.ego.control.brake = float(0.0)
        self.ego.tick()
        self.ego.set_transform(self.spawn_transform)
        self.ego.set_simulate_physics(False)  # Reset the car's physics
        self.ego.set_simulate_physics(True)
        yaw = (self.ego.get_transform().rotation.yaw) * np.pi / 180.0
        init_speed = carla.Vector3D(
            x=self.targetSpeed * np.cos(yaw),
            y=self.targetSpeed * np.sin(yaw))
        self.ego.set_velocity(init_speed)
        self.last_action = [0.0, 0.0]
        self.terminal_state = False  # Set to True when we want to end episode
        self.closed = False  # Set to True when ESC is pressed
        self.extra_info = []  # List of extra info shown on the HUD
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.start_t = time.time()
        self.step_count = 0
        self.is_first_path = True
        #self.start_waypoint_index = self.current_waypoint_index

        # Metrics
        self.total_reward = 0.0
        self.previous_location = self.ego.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.distance_from_center = 0.0
        self.speed_accum = 0.0
        #self.laps_completed = 0.0
        self.fea_ext = FeatureExt(self.world, self.dt, self.ego)
        self.fea_ext.update()
        self.v_buffer = deque([], maxlen=5)
        #self.render()
        # DEBUG: Draw path
        # self._draw_path(life_time=1000.0, skip=10)
        return self._get_obs()

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
            action = [0, 1.0]
            self.is_first_path = False
        steer, throttle = [float(a) for a in action]
        # steer, throttle, brake = [float(a) for a in action]
        self.ego.control.steer = self.ego.control.steer * self.action_smoothing + steer * (
                    1.0 - self.action_smoothing)
        self.ego.control.throttle = self.ego.control.throttle * self.action_smoothing + throttle * (
                    1.0 - self.action_smoothing)

        """
                **********************************************************************************************************************
                ************************************************ Update Carla ********************************************************
                **********************************************************************************************************************
        """

        self.hud.tick(self.world, self.clock)
        self.world.tick()
        # Get most recent observation and viewer image
        self.viewer_image = self._get_viewer_image()
        """
                *********************************************************************************************************************
                *********************************************** RL Observation ******************************************************
                *********************************************************************************************************************
        """
        self.fea_ext.update()
        self.fea_ext.observation
        # Get vehicle transform
        transform = self.ego.get_transform()

        # Calculate distance traveled
        self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

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
        track_rewd = np.exp(-10 * np.power(self.distance_from_center, 2) / 2 / sigma_pos / sigma_pos)
        # print("_track", track_rewd)
        # print(last_speed)

        #speed reward
        last_speed = get_speed(self.ego)
        e_speed = abs(self.targetSpeed - last_speed)
        sigmal_v = 0.6 if e_speed <= self.targetSpeed else 1.0
        r_speed = self.w_r_speed * np.exp(-e_speed/self.targetSpeed ** 2 / 2 / (sigmal_v ** 2))  # 0<= r_speed <= self.w_r_speed
        #  first two path speed change increases regardless so we penalize it differently

        #spd_change_percentage = (last_speed - init_speed) / init_speed if init_speed != 0 else -1
        r_laneChange = 0

        if 0.6 <= self.distance_from_center < 0.8 :   # unmeaningful lanechange
            r_laneChange = -1 * r_speed * self.lane_change_penalty  # <= 0

        positives = r_speed
        negatives = r_laneChange

        # angle reward
        sigma_yaw = self.sigma_angle
        yaw_err = delta_yaw * np.pi / 180
        ang_rewd = np.exp(-10 * np.power(yaw_err, 2) / 2 / sigma_yaw / sigma_yaw)
        #ang_rewd = -100*(yaw_err**2)

        # print("_ang", ang_rewd)


        #self.reward = (positives + negatives) * track_rewd * ang_rewd  # r_speed * (1 - lane_change_penalty) <= reward <= r_speed * lane_change_reward
        self.reward = (positives + negatives) * ang_rewd * track_rewd
        # print(self.n_step, self.eps_rew)
        # print("speed reward:", positives, negatives)
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

        if self.distance_from_center >= 0.8:
            print('Collision happened because of off the road!')
            self.reward = self.off_the_road_penalty
            self.terminal_state = True

        # If at destination
        dest = self.dest
        if np.sqrt((car_x-dest[0])**2+(car_y-dest[1])**2) < 2.0:
            print("Get destination! Episode Done.")
            self.logger.debug('Get destination! Episode cost %d steps in route %d.' % (self.step_count, self.route_id))
            # self.isSuccess = True
            self.terminal_state = True

        if last_speed < 4 or any(self.ego.collision_sensor.get_collision_history()):
            self.terminal_state = True
            self.reward = self.off_the_road_penalty
            print('speed low!')

        # Update checkpoint for training
        self.total_reward += self.reward
        self.render()
        return self._get_obs(), self.reward, self.terminal_state, {'reserved': 0}


    @property
    def observation_space(self) -> spaces.Space:
        features_space = np.array([np.inf] * 117)  # 12*9 + 9
        print('shape',features_space.shape)
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
        # # Get maneuver name
        # if self.current_road_maneuver == RoadOption.LANEFOLLOW:
        #     maneuver = "Follow Lane"
        # elif self.current_road_maneuver == RoadOption.LEFT:
        #     maneuver = "Left"
        # elif self.current_road_maneuver == RoadOption.RIGHT:
        #     maneuver = "Right"
        # elif self.current_road_maneuver == RoadOption.STRAIGHT:
        #     maneuver = "Straight"
        # elif self.current_road_maneuver == RoadOption.VOID:
        #     maneuver = "VOID"
        # else:
        #     maneuver = "INVALID(%i)" % self.current_road_maneuver

        # Add metrics to HUD
        self.extra_info.extend([
            "Reward: % 19.2f" % self.reward,
            "Total Reward: % 19.2f" % self.total_reward,
            "",
            # "Maneuver:        % 11s" % maneuver,
            #"Laps completed:    % 7.2f %%" % (self.laps_completed * 100.0),
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
    def _get_velocity(self):
        _v = self.ego.get_velocity()
        ego_velocity = np.array([_v.x, _v.y, _v.z])
        _acc = self.ego.get_acceleration()
        ego_acc = np.array([_acc.x, _acc.y, _acc.z])
        v = np.linalg.norm(ego_velocity)
        acc = np.linalg.norm(ego_acc)
        return v, acc

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

