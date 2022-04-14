

#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Other Leading Vehicle scenario:

The scenario realizes a common driving behavior, in which the
user-controlled ego vehicle follows a leading car driving down
a given road. At some point the leading car has to decelerate.
The ego vehicle has to react accordingly by changing lane to avoid a
collision and follow the leading car in other lane. The scenario ends
either via a timeout, or if the ego vehicle drives some distance.
"""
import carla
from carla_env.envs.scenario_utils import *
from carla_env.envs.scenario_utils import _dis, _pos
from utils import logger
from agents.navigation.basic_agent import *


class OverTake(object):
    def __init__(self, name, map, world):
        self.name = name
        self._map = map
        self.world = world
        self.speed = 0
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()

    def _scenario_init(self):
        # init hero car
        # --------------------------------------------------------
        # setup cars on a given waypoint
        hero_car_pos = [65.516594, 7.808423, 0.275307]
        wp_location = carla.Location(x=hero_car_pos[0], y=hero_car_pos[1], z=hero_car_pos[2])
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.tesla.model3'
        blueprint = self.blueprint_library.find(hero_model)
        blueprint.set_attribute('role_name', 'hero')
        self.hero_car = self.world.spawn_actor(blueprint, hero_vehicle_transform)
        #wp_location = self.hero_car.get_location()
        #wp = self._map.get_waypoint(wp_location)
        print('ego_car',self.hero_car)

        # init zombie cars
        first_vehicle_location = 44
        second_vehicle_location = first_vehicle_location + 7
        first_vehicle_waypoint = wp.next(first_vehicle_location)[0]
        second_vehicle_waypoint = wp.next(second_vehicle_location)[0].get_left_lane()
        first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location,
                                                  first_vehicle_waypoint.transform.rotation)
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
                                                   second_vehicle_waypoint.transform.rotation)

        vehicle_bp1 = self.world.get_blueprint_library().find('vehicle.audi.tt')
        vehicle_bp2 = self.world.get_blueprint_library().find('vehicle.toyota.prius')
        blueprints = [vehicle_bp1, vehicle_bp2]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')
        self.first_vehicle = self.world.spawn_actor(blueprints[0], first_vehicle_transform)
        self.second_vehicle = self.world.spawn_actor(blueprints[1], second_vehicle_transform)
        self.zombie_cars = [self.first_vehicle, self.second_vehicle]

        # --------------------------------------------------------
        # --------------------------------------------------------
        # setup local planners for zombie cars
        self._first_vehicle_speed = 36 / 3.2
        self._second_vehicle_speed = 45
        first_vehicle_planner = WaypointFollower(self.zombie_cars[0], self._first_vehicle_speed,map=self._map,
                                                         avoid_collision=True)
        second_vehicle_planner = WaypointFollower(self.zombie_cars[1], self._second_vehicle_speed,map=self._map,
                                                          avoid_collision=True)
        self.vehicle_planners = [first_vehicle_planner, second_vehicle_planner]
        #print(self.vehicle_planners)
        for planner in self.vehicle_planners:
            planner.setup()


    def _update(self):
        # update action for two local planners
        if _dis(_pos(self.hero_car), _pos(self.first_vehicle)) > 26.:
            pass
        else:
            for planner in self.vehicle_planners:
                planner.update()

    def restart(self):
        self._remove_all_actors()
        self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()