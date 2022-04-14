#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Summary of useful helper functions for scenarios
"""
from __future__ import print_function
import math
import shapely.geometry
import shapely.affinity
from collections import deque
import numpy as np
import random
import carla
from agents.tools.misc import vector, draw_waypoints
from agents.navigation.local_planner import RoadOption
from agents.navigation.controller import VehiclePIDController
from enum import Enum


def distance_vehicle(waypoint, vehicle_position):
    dx = waypoint[0] - vehicle_position[0]
    dy = waypoint[1] - vehicle_position[1]
    #dz = waypoint[2] - vehicle_position[2]

    return math.sqrt(dx * dx + dy * dy )

def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).
    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT

class RotatedRectangle(object):
    """
    This class contains method to draw rectangle and find intersection point.
    """

    def __init__(self, c_x, c_y, width, height, angle):
        self.c_x = c_x
        self.c_y = c_y
        self.w = width  # pylint: disable=invalid-name
        self.h = height  # pylint: disable=invalid-name
        self.angle = angle

    def get_contour(self):
        """
        create contour
        """
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.c_x, self.c_y)

    def intersection(self, other):
        """
        Obtain a intersection point between two contour.
        """
        return self.get_contour().intersection(other.get_contour())


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4


def retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.
    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).
    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT

# helper functions, mostly are planners
class WaypointFollower(object):
    """
    This is an atomic behavior to follow waypoints indefinitely
    while maintaining a given speed or if given a waypoint plan,
    follows the given plan
    """

    def __init__(self, actor, target_speed, plan=None,
                 avoid_collision=False, name="FollowWaypoints", map=None):
        """
        Set up actor and local planner
        """
        self._actor_list = []
        self._actor_list.append(actor)
        # print('\n\ninit_actor: ', actor)
        self._target_speed = target_speed
        self._local_planner_list = []
        self._plan = plan
        self._args_lateral_dict = {'K_P': 1.0, 'K_D': 0.01, 'K_I': 0.0, 'dt': 0.05}
        self._avoid_collision = avoid_collision
        self._map = map

    def setup(self, timeout=5):
        """
        Delayed one-time initialization
        """
        for actor in self._actor_list:
            # print('\n\nactor: ', actor)
            self._apply_local_planner(actor, self._map)

        return True

    def _apply_local_planner(self, actor, map):
        local_planner = WpFollowplanner(
            actor=actor,
            map=map,
            opt_dict={
                'target_speed': self._target_speed,
                'lateral_control_dict': self._args_lateral_dict})
        # if self._plan is not None:
        #     local_planner.set_global_plan(self._plan)
        self._local_planner_list.append(local_planner)

    def update(self):
        """
        Run local planner, obtain and apply control to actor
        """
        for actor, local_planner in zip(self._actor_list, self._local_planner_list):
            if actor is not None and actor.is_alive and local_planner is not None:
                control = local_planner.run_step(debug=False)
                actor.apply_control(control)


class WpFollowplanner(object):
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, actor, map, opt_dict):
        self._vehicle = actor
        self._map = map

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self._target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=600)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # initializing controller
        self.init_controller(opt_dict)

    def init_controller(self, opt_dict):
        """
        Controller initialization.
        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 0.5 / 3.6  # 0.5 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE

        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt}

        # parameters overload
        if 'dt' in opt_dict:
            self._dt = opt_dict['dt']
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'sampling_radius' in opt_dict:
            self._sampling_radius = self._target_speed * \
                                    opt_dict['sampling_radius'] / 3.6
        if 'lateral_control_dict' in opt_dict:
            args_lateral_dict = opt_dict['lateral_control_dict']
        if 'longitudinal_control_dict' in opt_dict:
            args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        self._current_waypoint = self._map.get_waypoint(
            self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

        self._global_plan = False

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))
        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.
        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def run_step(self, debug=True):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.
        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon? => add more!
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            if not self._global_plan:
                self._compute_next_waypoints(k=100)

        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        # Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        # current vehicle waypoint
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        # target waypoint
        # self._target_waypoint, self._target_road_option = self._waypoint_buffer[0]
        # move using PID controllers
        self._target_waypoint = self._current_waypoint.next(5.0)[0]
        control = self._vehicle_controller.run_step(self._target_speed, self._target_waypoint)
        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()
        if debug:
            draw_waypoints(self._vehicle.get_world(), [self._target_waypoint], self._vehicle.get_location().z + 1.0)
        return control


def detect_lane_obstacle(world, actor, extension_factor=3, margin=1.02):
    """
    This function identifies if an obstacle is present in front of the reference actor
    """
    world_actors = world.get_actors().filter('vehicle.*')
    actor_bbox = actor.bounding_box
    actor_transform = actor.get_transform()
    actor_location = actor_transform.location
    actor_vector = actor_transform.rotation.get_forward_vector()
    actor_vector = np.array([actor_vector.x, actor_vector.y])
    actor_vector = actor_vector / np.linalg.norm(actor_vector)
    actor_vector = actor_vector * (extension_factor - 1) * actor_bbox.extent.x
    actor_location = actor_location + carla.Location(actor_vector[0], actor_vector[1])
    actor_yaw = actor_transform.rotation.yaw

    is_hazard = False
    for adversary in world_actors:
        if adversary.id != actor.id and \
                actor_transform.location.distance(adversary.get_location()) < 50:
            adversary_bbox = adversary.bounding_box
            adversary_transform = adversary.get_transform()
            adversary_loc = adversary_transform.location
            adversary_yaw = adversary_transform.rotation.yaw
            overlap_adversary = RotatedRectangle(
                adversary_loc.x, adversary_loc.y,
                2 * margin * adversary_bbox.extent.x, 2 * margin * adversary_bbox.extent.y, adversary_yaw)
            overlap_actor = RotatedRectangle(
                actor_location.x, actor_location.y,
                2 * margin * actor_bbox.extent.x * extension_factor, 2 * margin * actor_bbox.extent.y, actor_yaw)
            overlap_area = overlap_adversary.intersection(overlap_actor).area
            if overlap_area > 0:
                is_hazard = True
                break

    return is_hazard


class RotatedRectangle(object):

    """
    This class contains method to draw rectangle and find intersection point.
    """

    def __init__(self, c_x, c_y, width, height, angle):
        self.c_x = c_x
        self.c_y = c_y
        self.w = width      # pylint: disable=invalid-name
        self.h = height     # pylint: disable=invalid-name
        self.angle = angle

    def get_contour(self):
        """
        create contour
        """
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.c_x, self.c_y)

    def intersection(self, other):
        """
        Obtain a intersection point between two contour.
        """
        return self.get_contour().intersection(other.get_contour())


class WaypointFollower_FullMap(object):
    """
    This is an atomic behavior to follow waypoints indefinitely
    while maintaining a given speed or if given a waypoint plan,
    follows the given plan
    """

    def __init__(self, actor, target_speed, map, world, pattern_1=None, pattern_2=None, plan=None,
                 avoid_collision=False, actor_location=None, name="FollowWaypoints"):
        """
        Set up actor and local planner
        """
        self._actor_list = []
        self._actor_list.append(actor)
        # print('\n\ninit_actor: ', actor)
        self._target_speed = target_speed
        self._local_planner_list = []
        self._plan = plan
        self._args_lateral_dict = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': 0.03}
        self._args_longitudinal_dict = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0, 'dt': 0.03}
        self._avoid_collision = avoid_collision
        self.pattern_1 = pattern_1
        self.pattern_2 = pattern_2
        self.map = map
        self.world = world
        self.actor_location = actor_location

    def setup(self, timeout=5):
        """
        Delayed one-time initialization
        """
        for actor in self._actor_list:
            self._apply_local_planner(actor)

        return True

    def _apply_local_planner(self, actor):
        local_planner = WpFollowplanner_FullMap(
            map=self.map,
            actor=actor,
            actor_location=self.actor_location,
            opt_dict={
                'target_speed': self._target_speed,
                'lateral_control_dict': self._args_lateral_dict},
            pattern_1=self.pattern_1,
            pattern_2=self.pattern_2
        )
        if self._plan is not None:
            local_planner.set_global_plan(self._plan)
        self._local_planner_list.append(local_planner)

    def update(self):
        """
        Run local planner, obtain and apply control to actor
        """
        # print('Update ...')
        for actor, local_planner in zip(self._actor_list, self._local_planner_list):
            # print(actor is not None, actor.is_alive, local_planner is not None)
            if actor is not None and actor.is_alive and local_planner is not None:
                control = local_planner.run_step(debug=False)
                if self._avoid_collision and detect_lane_obstacle(world=self.world, actor=actor):
                    control.throttle = 0.0
                    control.brake = 1.0
                actor.apply_control(control)

class WpFollowplanner_FullMap(object):
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, actor, opt_dict, map, pattern_1=None, pattern_2=None, actor_location=None):
        self.pattern_1 = pattern_1
        self.pattern_2 = pattern_2
        self._vehicle = actor
        self._map = map  # self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self._target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None
        self._road_options_list_prev = None
        self._index = None
        self.actor_location = actor_location
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=600)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        # initializing controller
        self.init_controller(opt_dict)

    def init_controller(self, opt_dict):
        """
        Controller initialization.
        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 0.5 / 3.6  # 0.5 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE

        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.01,
            'K_I': 1.4,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 1,
            'dt': self._dt}

        # parameters overload
        if 'dt' in opt_dict:
            self._dt = opt_dict['dt']
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'sampling_radius' in opt_dict:
            self._sampling_radius = self._target_speed * \
                                    opt_dict['sampling_radius'] / 3.6
        if 'lateral_control_dict' in opt_dict:
            args_lateral_dict = opt_dict['lateral_control_dict']
        if 'longitudinal_control_dict' in opt_dict:
            args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        # self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._current_waypoint = self._map.get_waypoint(self.actor_location)
        # print('self._vehicle.get_location(): ', self._current_waypoint.transform.location, self._vehicle.get_transform().location, self._vehicle, self._current_waypoint.next(1.5))
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict)

        self._global_plan = False

        self._waypoints_queue.append((self._current_waypoint.next(1.5)[0], RoadOption.LANEFOLLOW))
        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.
        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = last_waypoint.next(1.5)
            # print('next_waypoints: ', last_waypoint, next_waypoints)
            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = retrieve_options(
                    next_waypoints, self._current_waypoint)
                if self.pattern_1:
                    index = self.pattern_1.pop(0)
                    road_option = road_options_list[index]
                    next_waypoint = next_waypoints[index]
                    self.pattern_1.append(index)
                elif self.pattern_2:
                    index = self.pattern_2.pop(0)
                    if isinstance(index, int):
                        index = road_options_list.index(RoadOption(index))
                        road_option = RoadOption(index)
                        next_waypoint = next_waypoints[road_options_list.index(
                            road_option)]
                    elif isinstance(index, list):
                        next_waypoint = self._map.get_waypoint(
                            carla.Location(x=index[0], y=index[1], z=index[2]))
                        road_option = RoadOption.LANEFOLLOW
                    else:
                        raise NotImplementedError('index must be type `int` or `list`')
                    self.pattern_2.append(index)
                    print(road_options_list)
                else:  # self.pattern_1 is None and self.pattern_2 is None
                    print('self.pattern_1 is None and self.pattern_2 is None')
                # print(next_waypoint.transform.location)
            self._waypoints_queue.append((next_waypoint, road_option))

    def run_step(self, debug=True):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.
        :param debug: boolean flag to activate waypoints debugging
        :return:
        """
        # not enough waypoints in the horizon? => add more!
        if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            if not self._global_plan:
                self._compute_next_waypoints(k=100)

        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        # Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        self._target_waypoint, self._target_road_option = self._waypoint_buffer[0]

        control = self._vehicle_controller.run_step(self._target_speed, self._target_waypoint)

        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_vehicle(
                    waypoint, vehicle_transform) < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()
        if debug:
            draw_waypoints(self._vehicle.get_world(), [self._target_waypoint], self._vehicle.get_location().z + 1.0)
        return control


def _pos(_object):
    type_obj = str(type(_object))
    if 'Actor' in type_obj or 'Vehicle' in type_obj or 'TrafficLight' in type_obj:
        return [_object.get_location().x, _object.get_location().y]
    elif 'BoundingBox' in type_obj or 'Transform' in type_obj:
        return [_object.location.x, _object.location.y]
    elif 'Vector3D' in type_obj or 'Location' in type_obj:
        return [_object.x, _object.y]
    elif 'Waypoint' in type_obj:
        return [_object.transform.location.x, _object.transform.location.y]
def _pos3d(_object):
    type_obj = str(type(_object))
    if 'Actor' in type_obj or 'Vehicle' in type_obj or 'TrafficLight' in type_obj:
        return [_object.get_location().x, _object.get_location().y, _object.get_location().z]
    elif 'BoundingBox' in type_obj or 'Transform' in type_obj:
        return [_object.location.x, _object.location.y, _object.location.z]
    elif 'Vector3D' in type_obj or 'Location' in type_obj:
        return [_object.x, _object.y, _object.z]
    elif 'Waypoint' in type_obj:
        return [_object.transform.location.x, _object.transform.location.y, _object.transform.location.z]
def _dis(a, b):
    return ((b[1]-a[1])**2 + (b[0]-a[0])**2)**0.5
def _dis3d(a, b):
    return ((b[1]-a[1])**2 + (b[0]-a[0])**2 + (b[2]-a[2])**2)**0.5