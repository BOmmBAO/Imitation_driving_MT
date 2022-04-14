import carla
from carla_env.envs.scenario_utils import *
import numpy as np
from agents.navigation.basic_agent import *
import random


class Cross_Turn_Right(object):
    def __init__(self, name, map, world, only_reset_hero=False):
        self.name = name
        self._map = map
        self.world = world
        self.speed = 0
        self.only_reset_hero = only_reset_hero
        self.blueprint_library = self.world.get_blueprint_library()
        self._scenario_init()

    def _scenario_init(self):
        # init hero car
        # --------------------------------------------------------
        # setup cars on a given waypoint
        self.hero_car_pos = [-42.350990295410156, -2.835118293762207, 1.8431016206741333]
        wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
        wp = self._map.get_waypoint(wp_location)
        hero_vehicle_transform = wp.transform
        hero_model = 'vehicle.lincoln.mkz2017'
        blueprint = random.choice(self.blueprint_library.filter(hero_model))
        blueprint.set_attribute('role_name', 'hero')
        self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)

        models = ['vehicle.nissan.patrol', 'vehicle.audi.tt',
                  'vehicle.lincoln.mkz2017', 'vehicle.volkswagen.t2',
                  'vehicle.tesla.model3', 'vehicle.nissan.micra',
                  'vehicle.audi.a2',
                  ]

        blueprints = [random.choice(self.world.get_blueprint_library().filter(model)) for model in models]
        for blueprint in blueprints:
            blueprint.set_attribute('role_name', 'scenario')

        self.blueprints = blueprints
        self.models = models
        # Not available: 135, 160
        fourth_car_pos = [-74.38717651367188, 57.531620025634766, 1.805267095565796]  # 13
        fourth_wp_location = carla.Location(x=fourth_car_pos[0], y=fourth_car_pos[1], z=fourth_car_pos[2])
        fourth_vehicle_waypoint = self._map.get_waypoint(fourth_wp_location)
        fourth_vehicle_transform = carla.Transform(fourth_vehicle_waypoint.transform.location,
                                                   fourth_vehicle_waypoint.transform.rotation)
        self.fourth_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], fourth_vehicle_transform)
        # setup local planners for zombie cars
        self._fourth_vehicle_speed = 20

        no_2_car_pos = [-95.79371643066406, 0.17835818231105804, 1.8431016206741333]  # 191 below
        no_2_wp_location = carla.Location(x=no_2_car_pos[0], y=no_2_car_pos[1], z=no_2_car_pos[2])
        no_2_vehicle_waypoint = self._map.get_waypoint(no_2_wp_location)
        no_2_vehicle_transform = carla.Transform(no_2_vehicle_waypoint.transform.location,
                                                   no_2_vehicle_waypoint.transform.rotation)
        self.no_2_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], no_2_vehicle_transform)
        # setup local planners for zombie cars
        self._no_2_vehicle_speed = 20

        no_3_car_pos = [-84.8062973022461, -25, 1.7985864877700806]  # 27
        no_3_wp_location = carla.Location(x=no_3_car_pos[0], y=no_3_car_pos[1], z=no_3_car_pos[2])
        no_3_vehicle_waypoint = self._map.get_waypoint(no_3_wp_location)
        no_3_vehicle_transform = carla.Transform(no_3_vehicle_waypoint.transform.location,
                                                 no_3_vehicle_waypoint.transform.rotation)
        self.no_3_vehicle = self.world.try_spawn_actor(blueprints[4 % len(models)], no_3_vehicle_transform)
        # setup local planners for zombie cars
        self._no_3_vehicle_speed = 20

        self.zombie_cars = [self.fourth_vehicle, self.no_2_vehicle, self.no_3_vehicle]

        fourth_vehicle_planner = WaypointFollower_FullMap(actor=self.fourth_vehicle,
                                                          target_speed=self._fourth_vehicle_speed,
                                                          actor_location=fourth_wp_location,
                                                          map=self._map,
                                                          avoid_collision=True,
                                                          pattern_1=[1, 1, 1, 1, 1, 1, 0, 0, 1, ],
                                                          world=self.world)
        no_2_vehicle_planner = WaypointFollower_FullMap(actor=self.no_2_vehicle,
                                                          target_speed=self._no_2_vehicle_speed,
                                                        actor_location=no_2_wp_location,
                                                          map=self._map,
                                                          avoid_collision=True,
                                                          pattern_1=[0, 0, 3, ],
                                                          world=self.world)
        no_3_vehicle_planner = WaypointFollower_FullMap(actor=self.no_3_vehicle,
                                                          target_speed=self._no_3_vehicle_speed,
                                                        actor_location=no_3_wp_location,
                                                          map=self._map,
                                                          avoid_collision=True,
                                                          pattern_1=[0, 0, 1, 1, 0, 0, 0, 0, 0,],
                                                          world=self.world)

        self.vehicle_planners = [fourth_vehicle_planner, no_2_vehicle_planner, no_3_vehicle_planner]

        for planner in self.vehicle_planners:
            planner.setup()

    def generate_car(self):
        additional_zombie_car = list()
        additional_zombie_car_speed = list()
        additional_pattern = list()
        additional_actor_location = list()
        all_car_pos = [[-74.38717651367188, 57.531620025634766, 1.805267095565796], [-95.79371643066406, 0.17835818231105804, 1.8431016206741333], [-85.21101379394531, -126.87477111816406, 1.7985864877700806]]
        all_pattern = [[1, 1, 1, 1, 1, 1, 0, 0, 1, ], [0, 0, 3], [0, 0, 1, 1, 0, 0, 0, 0, 0,]]

        for i, car_pos in enumerate(all_car_pos):
            if car_pos == [-1, -1, -1] or car_pos == [-2, -2, -2]:
                # car_pos == [-2, -2, -2]: get_left_lane(), speed=26
                # car_pos == [-1, -1, -1]: get_left_lane()
                car_pos = all_car_pos[i - 1]
                orig_actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(orig_actor_location).next(16)[0].get_left_lane()
                actor_location = vehicle_waypoint.transform.location
            else:
                actor_location = carla.Location(x=car_pos[0], y=car_pos[1], z=car_pos[2])
                vehicle_waypoint = self._map.get_waypoint(actor_location)

            world_actors = self.world.get_actors().filter('vehicle.*')
            flag_spawn = True
            for adversary in world_actors:
                if actor_location.distance(adversary.get_location()) < 15:
                    flag_spawn = False
            if flag_spawn:
                vehicle_transform = carla.Transform(vehicle_waypoint.transform.location,
                                                    vehicle_waypoint.transform.rotation)
                vehicle = self.world.try_spawn_actor(self.blueprints[np.random.randint(0, len(self.blueprints))],
                                                     vehicle_transform)
                if car_pos == [-2, -2, -2]:
                    _vehicle_speed = 26
                else:
                    _vehicle_speed = 25
                additional_zombie_car.append(vehicle)
                additional_zombie_car_speed.append(_vehicle_speed)
                additional_pattern.append(all_pattern[i])
                additional_actor_location.append(actor_location)
                self.zombie_cars.append(vehicle)

        for i, (one_zombie_car, one_zombie_car_speed, one_pattern, one_actor_location) in enumerate(
                zip(additional_zombie_car, additional_zombie_car_speed, additional_pattern,
                    additional_actor_location)):
            vehicle_planner = WaypointFollower_FullMap(actor=one_zombie_car, map=self._map,
                                                       actor_location=one_actor_location,
                                                       target_speed=one_zombie_car_speed,
                                                       avoid_collision=True, pattern_1=one_pattern,
                                                       world=self.world)

            self.vehicle_planners.append(vehicle_planner)
            vehicle_planner.setup()

    def _update(self):
        # update action for two local planners
        # if _dis3d(_pos3d(self.hero_car), _pos3d(self.first_vehicle)) > 26.:
        #     pass
        # else:
        #     for planner in self.vehicle_planners:
        #         planner.update()
        self.generate_car()
        for planner in self.vehicle_planners:
            planner.update()

    def restart(self):
        if self.only_reset_hero:
            wp_location = carla.Location(x=self.hero_car_pos[0], y=self.hero_car_pos[1], z=self.hero_car_pos[2])
            wp = self._map.get_waypoint(wp_location)
            hero_vehicle_transform = wp.transform
            hero_model = 'vehicle.lincoln.mkz2017'
            blueprint = random.choice(self.blueprint_library.filter(hero_model))
            blueprint.set_attribute('role_name', 'hero')
            self.hero_car = self.world.try_spawn_actor(blueprint, hero_vehicle_transform)
        else:
            self._remove_all_actors()
            self.zombie_cars = list()
            self.vehicle_planners = list()
            self._scenario_init()

    def _remove_all_actors(self):
        actors = [self.hero_car] + self.zombie_cars
        # actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()

    def _remove_zombie_cars(self):
        actors = self.zombie_cars
        for actor in actors:
            if actor.is_alive:
                actor.destroy()