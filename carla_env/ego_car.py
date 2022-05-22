import carla
import random
from carla_env.modules import CollisionSensor, LaneInvasionSensor
import numpy as np
import carla_env.common as common

class SimInit:

    def __init__(self, args):
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(10.0)
        self.dt = 0.05

        self.world = self.client.load_world('Town03')
        self._map = self.world.get_map()
        self.world.freeze_all_traffic_lights(True)

        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.dt
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        self.spectator = self.world.get_spectator()
        self.ego_car, self.zombie_cars, self.visible_zombie_cars, self.visible_zombie_cars_index = None, None, [], []
        self.collision_event = False

        self.spawn_points = self.world.get_map().get_spawn_points()
        self.init_spawn_point = self.spawn_points[1]
        self.spawn_points.remove(self.init_spawn_point)
        model3_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        model3_bp.set_attribute('color', "200, 50, 50")
        self.ego_car = self.world.spawn_actor(model3_bp, self.init_spawn_point)
        self.ego_car.set_autopilot(False)
        self.collision_sensor = CollisionSensor(self.ego_car)
        self.laneinvasion_sensor = LaneInvasionSensor(self.ego_car)
        self.world.tick()
        current_loc = self.ego_car.get_transform().location
        self.spectator.set_transform(carla.Transform(current_loc +
                                                     carla.Location(x=-20, z=70),
                                                     carla.Rotation(pitch=-70)))

    def init(self):
        self.ego_car = self.add_ego_car(self.init_spawn_point)
        self.collision_sensor = CollisionSensor(self.ego_car)
        self.laneinvasion_sensor = LaneInvasionSensor(self.ego_car)
        self.world.tick()

    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed)
        return seed

    def reset(self):
        self.destroy()
        self.collision_event = False
        self.init()

    def destroy(self):
        actors = [self.collision_sensor.sensor,
                  self.laneinvasion_sensor.sensor,
                  self.ego_car]

        for actor in actors:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            if actor.is_alive:
                actor.destroy()

    def update(self):
        self.world.tick()
        self._spec_update()

    def _spec_update(self):
        current_loc = self.ego_car.get_transform().location
        self.spectator.set_transform(carla.Transform(current_loc +
                                     carla.Location(x=-20, z=70),
                                     carla.Rotation(pitch=-70)))

    def add_ego_car(self, spawn_point, color="200, 50, 50"):
        model3_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        model3_bp.set_attribute('color', color)
        ego_car = self.world.spawn_actor(model3_bp, spawn_point)
        ego_car.set_autopilot(False)
        return ego_car

    def term_check(self):
        collision_hist = self.collision_sensor.get_history()
        if any(collision_hist):
            print("COLLISION detected from sensor!!")
            self.collision_event = True
            return True
        else:
            return False

    def on_roadcentre(self):
        actor = self.ego_car
        ego_pos = self._map.get_waypoint(location=actor.get_location(), project_to_road=False).transform.location
        lane_pos = self._map.get_waypoint(location=actor.get_location(), project_to_road=True).transform.location
        dis = np.sqrt((ego_pos.x-lane_pos.x)**2+(ego_pos.y-lane_pos.y)**2)
        return dis









