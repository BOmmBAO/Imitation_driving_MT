import pygame.time
from gym import spaces
try:
    import numpy as np
    import sys
    from os import path as osp
except ImportError:
    raise RuntimeError('import error!')
import gym
import pygame.time
from gym import spaces
from carla_gym.envs.wrapper import *
from config import cfg
from local_planner.frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from low_level_controller.controller import VehiclePIDController
from agents.tools.misc import get_speed
from low_level_controller.controller import IntelligentDriverModel
from tools.misc import get_speed
from collections import deque
from carla_gym.envs.hud import HUD
from carla_env.featur_1 import *
from carla_gym.envs.utils import closest_wp_idx
from .carla_logger import *
from plan_control.path_planner import *
import plan_control.velocity_planner as velocity_planner
from plan_control.pid import *
from plan_control.linear_MPC import *
from carla_env.rule_decision import *

class CarConfig():
    def __init__(self, ego):
        self.motionPlanner = MotionPlanner()
        self.vehicleController = VehiclePIDController(ego, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0})
        self.IDM = IntelligentDriverModel(ego)
class CarlaEnv(gym.Env):
    """
    action smoothing: Scalar used to smooth the incomming action signal for e2e.
                1.0 = max smoothing, 0.0 = no smoothing
    """
    def __init__(self, test, args, action_smoothing=0.9, synchronous=True, viewer_res=(1280, 720)): #lanes_change=5):
        self.__version__ = "0.9.9"
        self.logger = setup_carla_logger(
            "output_id", experiment_name=str(2.1))
        self.logger.info("Env running in port {}".format(1.1))
        self.test = test
        self.args = args
        self.is_training = True
        pygame.init()
        pygame.font.init()
        # simulation
        self.max_time_episode = int(cfg.GYM_ENV.LOOP_BREAK)
        self.task_mode = str(cfg.CARLA.TASK_MODE)
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
        self.action_space = gym.spaces.Discrete(4)
        self.reward = 0.0

        # instances
        self.dt = float(cfg.CARLA.DT)
        self.client = carla.Client(self.args.carla_host, self.args.carla_port)
        self.client.set_timeout(10.0)
        self.world = World(self.client)
        if self.synchronous:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            #settings.no_rendering_mode = True
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
        self.terminal_state = False
        self.total_steps = 0

        # Start Modules
        # Generate waypoints along the lap
        self.ego_config = CarConfig(self.ego)
        self.csp = None
        self.world.tick()

    def reset(self, is_training=True):
        self.ego.reset()
        self.ego.control.steer = float(0.0)
        self.ego.control.throttle = float(0.0)
        self.ego.tick()
        self.ego.set_transform(self.spawn_transform)
        self.ego.set_simulate_physics(False)  # Reset the car's physics
        self.ego.set_simulate_physics(True)
        self.world.tick()
        yaw = (self.ego.get_transform().rotation.yaw) * np.pi / 180.0
        init_speed = carla.Vector3D(
            x=6.0 * np.cos(yaw),
            y=6.0 * np.sin(yaw))
        self.ego.set_velocity(init_speed)
        self.ego.tick()
        self.terminal_state = False  # Set to True when we want to end episode
        self.closed = False  # Set to True when ESC is pressed
        self.extra_info = []  # List of extra info shown on the HUD
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.step_count = 0
        # Metrics
        self.is_first = True
        self.f_idx = 0
        self.total_reward = 0.0
        self.previous_location = self.ego.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.distance_from_center = 0.0
        self.speed_accum = 0.0
        self.ego_config = CarConfig(self.ego)
        self.v_buffer = deque([], maxlen=5)
        self.fea_ext = FeatureExt(self.world, self.dt, self.ego)
        self.fea_ext.update()
        _, _, _, self.csp = path_planner(self.fea_ext).update()
        self.ego_config.motionPlanner.start(self.csp)
        self.ego_config.motionPlanner.reset(0, 0 , df_n=0, Tf=4, Vf_n=0, optimal_path=False)


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
        #print(action[0])
        self.step_count += 1
        self.total_steps += 1
        if self.is_first:
            action = 0
            self.is_first = False
        """
                **********************************************************************************************************************
                *********************************************** Motion Planner *******************************************************
                **********************************************************************************************************************
        """
        self.fea_ext = FeatureExt(self.world, self.dt, self.ego)
        self.fea_ext.update()
        _, _, _, self.csp = path_planner(self.fea_ext).update()
        self.ego_config.motionPlanner.start(self.csp)
        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
        init_speed = speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp, 70]
        self.f_idx = 0
        fpath, self.lanechange, off_the_road = self.ego_config.motionPlanner.run_step_single_path(ego_state, self.f_idx,
                                                                                       df_n=action, Tf=5,
                                                                                       Vf_n=-1)
        #print(len(fpath.t))
        wps_to_go = len(fpath.t) - 3  # -2 bc len gives # of items not the idx of last item + 2wp controller is used
        self.f_idx = 1


        """
                **********************************************************************************************************************
                ************************************************* Controller *********************************************************
                **********************************************************************************************************************
        """
        # initialize flags
        collision = track_finished = False
        # elapsed_time = lambda previous_time: time.time() - previous_time
        # path_start_time = time.time()
        ego_init_d, ego_target_d = fpath.d[0], fpath.d[-1]
        # follows path until end of WPs for max 1.5 * path_time or loop counter breaks unless there is a langechange
        loop_counter = 0

        while self.f_idx < wps_to_go and loop_counter < 30:
            loop_counter += 1
            ego_state = [self.ego.get_location().x, self.ego.get_location().y,
                         math.radians(self.ego.get_transform().rotation.yaw), 0, 0, temp, 70]

            self.f_idx = closest_wp_idx(ego_state, fpath, self.f_idx)
            cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
            cmdWP2 = [fpath.x[self.f_idx + 1], fpath.y[self.f_idx + 1]]

            # overwrite command speed using IDM
            #ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]  # estimated current ego_s
            ego_d = fpath.d[self.f_idx]
            #vehicle_ahead = self.get_vehicle_ahead(ego_s, ego_d, ego_init_d, ego_target_d)
            cmdSpeed = self.ego_config.IDM.run_step(vd=self.targetSpeed, vehicle_ahead=None)

            # control = self.vehicleController.run_step(cmdSpeed, cmdWP)  # calculate control
            control = self.ego_config.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
            self.ego.apply_control(control)  # apply control
            self.hud.tick(self.world, self.clock)
            self.world.tick()
            # Get most recent observation and viewer image
            self.viewer_image = self._get_viewer_image()

            if any(self.ego.collision_sensor.get_collision_history()):
                self.terminal_state = True
                break

        """
                *********************************************************************************************************************
                *********************************************** RL Observation ******************************************************
                *********************************************************************************************************************
        """
        self.fea_ext.update()
        # Accumulate speed
        self.speed_accum += self.ego.get_speed()
        """
                      **********************************************************************************************************************
                      ********************************************* Episode Termination ****************************************************
                      **********************************************************************************************************************
              """

        last_speed = get_speed(self.ego)
        self.v_buffer.append(last_speed)
        self.speed_accum += last_speed
        self.test.log({'distance_from_centre': self.distance_from_center})
        if self.distance_from_center >= 1.2:
            print('Collision happened because of off the road!')
            self.terminal_state = True

        elif last_speed > self.maxSpeed:
            self.terminal_state = True
            self.logger.debug('too fast')

        elif len(self.v_buffer) == 5:
            v_norm_mean = np.mean(self.v_buffer)
            if v_norm_mean < 1.0:
                self.terminal_state = True
                self.logger.debug('too low')
                print('low!')
        elif len(self.ego.collision_sensor.get_collision_history()) != 0:
            self.terminal_state = True
            self.reward = self._penalty
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
            sigma_vel = self.sigma_high_v if last_speed <= self.targetSpeed else self.sigma_low_v
            speed_factor = np.exp(-np.power(last_speed - self.targetSpeed, 2) / 2 / sigma_vel / sigma_vel)
            self.reward = speed_factor * centering_factor * angle_factor
            self.test.log({'reward': self.reward})
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
        return self.fea_ext.observation, self.reward, self.terminal_state, {'reserved': 0}


    @property
    def observation_space(self) -> spaces.Space:
        features_space = np.array([np.inf] * 79)
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

    def rule_based_step(self):
        self.fea_ext.update()
        rx, ry, ryaw, s_sum = self.path.update(self.fea_ext.vehicle_info.merge_length)
        self.fea_ext.ref_display(rx, ry)
        ref_vel = self.dec_maker.decision()
        # poly_coe = velocity_planner.speed_profile_quinticPoly(self.fea_ext.vehicle_info, ref_vel, s_sum[-1])
        poly_coe = velocity_planner.speed_profile_uniform(ref_vel)
        control_comd = self.controller.update(rx, ry, ryaw, poly_coe)
        self.ego_car.apply_control(control_comd)

    def decode_decision(self, decision, vehicle):
        '''3     4
           1  0  2'''
        merge_dist = 0
        # Change the status according to decision
        target_vel, _a = vehicle.dec_maker.car_following()
        if vehicle.fea_ext.vehicle_info.status == STATUS.FOLLOWING:
            if decision == 0:
                vehicle.fea_ext.vehicle_info.status = STATUS.FOLLOWING
                merge_dist = 0
            else:
                if decision % 2 == 0:
                    if vehicle.fea_ext.vehicle_info.status == STATUS.FOLLOWING or \
                            vehicle.fea_ext.vehicle_info.status == STATUS.LANE_CHANGING_L:
                        vehicle.fea_ext.vehicle_info.status = STATUS.START_LANE_CHANGE_R
                    merge_dist = (decision / 2) * 10

                elif decision % 2 == 1:
                    if vehicle.fea_ext.vehicle_info.status == STATUS.FOLLOWING or \
                            vehicle.fea_ext.vehicle_info.status == STATUS.LANE_CHANGING_R:
                        vehicle.fea_ext.vehicle_info.status = STATUS.START_LANE_CHANGE_L
                    merge_dist = ((decision + 1) / 2) * 10

        return merge_dist, target_vel

    # Decision with Interuption
    def update_dec(self, decision, vehicle):
        wrong_dec = False
        rx, ry, ryaw = None, None, None
        target_vel, _ = vehicle.dec_maker.car_following()

        if decision == 0:
            follow_lane = vehicle.fea_ext.cur_wp
            rx, ry, ryaw, s_sum = vehicle.path.following_path(follow_lane)
        else:
            if decision % 2 == 1:
                merge_dist = ((decision + 1) / 2) * 10

                target_vel += 3
                if vehicle.fea_ext.cur_lane.lane_change == carla.LaneChange.Left \
                        or vehicle.fea_ext.cur_lane.lane_change == carla.LaneChange.Both:
                    left_lane = vehicle.fea_ext.cur_lane.get_left_lane()
                    merge_point = vehicle.path.merge_point_calcu(left_lane, merge_dist)
                    target_lane = self.env.world.get_map().get_waypoint(merge_point)
                    rx, ry, ryaw, s_sum = vehicle.path.laneChange_path(vehicle.fea_ext.vehicle_info, target_lane,
                                                                       merge_point)
                else:
                    wrong_dec = True

            elif decision % 2 == 0:
                merge_dist = (decision / 2) * 10
                target_vel += 3
                if vehicle.fea_ext.cur_lane.lane_change == carla.LaneChange.Right \
                        or vehicle.fea_ext.cur_lane.lane_change == carla.LaneChange.Both:
                    right_lane = vehicle.fea_ext.cur_lane.get_right_lane()
                    merge_point = vehicle.path.merge_point_calcu(right_lane, merge_dist)
                    target_lane = self.env.world.get_map().get_waypoint(merge_point)
                    rx, ry, ryaw, s_sum = vehicle.path.laneChange_path(vehicle.fea_ext.vehicle_info, target_lane,
                                                                       merge_point)
                else:
                    wrong_dec = True

        return wrong_dec, [rx, ry, ryaw, target_vel]

