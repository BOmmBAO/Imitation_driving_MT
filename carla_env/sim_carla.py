import carla
import random
from carla_env.modules import CollisionSensor, LaneInvasionSensor
import numpy as np
import carla_env.common as common

class SimInit:

    def __init__(self, args):
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(20.0)
        self.dt = 0.1

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
        print(self.ego_car)
        self.ego_car.set_autopilot(False)
        self.zombie_cars = self.add_zombie_cars(self.spawn_points, 10)
        self.collision_sensor = CollisionSensor(self.ego_car)
        self.laneinvasion_sensor = LaneInvasionSensor(self.ego_car)
        self.world.tick()
        current_loc = self.ego_car.get_transform().location
        self.spectator.set_transform(carla.Transform(current_loc +
                                                     carla.Location(x=-20, z=70),
                                                     carla.Rotation(pitch=-70)))

    def init(self):
        self.ego_car = self.add_ego_car(self.init_spawn_point)
        self.zombie_cars = self.add_zombie_cars(self.spawn_points, 10)
        self.collision_sensor = CollisionSensor(self.ego_car)
        self.laneinvasion_sensor = LaneInvasionSensor(self.ego_car)
        self.lane_invasion_hist = []
        self.world.tick()

    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed)
        return seed

    def reset(self):
        self.collision_event = False
        self.destroy()
        self.init()

    def destroy(self):
        actors = [self.collision_sensor.sensor,
                  self.laneinvasion_sensor.sensor,
                  self.ego_car]+self.zombie_cars
        # actors = [self.collision_sensor.sensor,
        #           self.ego_car, self.lead_car] + self.zombie_cars

        for actor in actors:
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()
            if actor.is_alive:
                actor.destroy()

    def update(self):
        self.world.tick()
        # if self.zombie_cars:
        #     self._visible_zombie_cars_filter()
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

    def add_zombie_cars(self, spawn_points, count=10):
        zombie_cars = []
        random.shuffle(spawn_points)
        # blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        # blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        # blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        _bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        _bp.set_attribute('color', "50, 50, 200")
        for spawn_point in spawn_points:
            if count <= 0:
                break
            else:
                # blueprint = random.choice(blueprints)
                vehicle = self.world.try_spawn_actor(_bp, spawn_point)
                if vehicle is not None:
                    vehicle.set_simulate_physics(True)
                    vehicle.set_autopilot(True)
                    zombie_cars.append(vehicle)
                    count -= 1
        return zombie_cars

    def add_lead_car(self, event, s_point):
        if event:
            cur_lane = self.world.get_map().get_waypoint(s_point.location)
            point = cur_lane.next(20)[0]
            spawn_point = carla.Transform(point.transform.location + carla.Location(z=s_point.location.z),
                                         point.transform.rotation)
            model3_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
            model3_bp.set_attribute('color', "50, 50, 200")
            lead_car = self.world.spawn_actor(model3_bp, spawn_point)
            lead_car.set_autopilot(False)
        else:
            lead_car = None

        return lead_car

    def add_ego_cars(self, s_point, count):
        ego_cars, velocity_list = [], []
        spawn_points = []

        cur_lane = self.world.get_map().get_waypoint(s_point.location)
        right_lane = cur_lane.get_right_lane()
        left_lane = cur_lane.get_left_lane()
        lleft_lane = left_lane.get_left_lane()
        rright_lane = right_lane.get_right_lane()

        for n in range(3):
            spawn_points.append([cur_lane.next(n * 10 + 0.1)[0], 12 - n * 1])
            spawn_points.append([left_lane.next(n * 15 + 20.1)[0], 15 - n * 3])
            spawn_points.append([lleft_lane.next(n * 15 + 0.1)[0], 15 - n * 3])
            spawn_points.append([right_lane.next(n * 15 + 0.1)[0], 15 - n * 1])
            spawn_points.append([rright_lane.next(n * 15 + 0.1)[0], 15 - n * 0.5])

        _bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        _bp.set_attribute('color', "50, 50, 200")
        np.random.shuffle(spawn_points)
        for [point, v] in spawn_points:
            if count <= 0:
                break
            else:
                feasible_p = carla.Transform(point.transform.location + carla.Location(z=s_point.location.z),
                                             point.transform.rotation)
                vehicle = self.world.try_spawn_actor(_bp, feasible_p)
                if vehicle is not None:
                    velocity_list.append(v)
                    vehicle.set_simulate_physics(True)
                    vehicle.set_autopilot(False)
                    ego_cars.append(vehicle)
                    count -= 1
        return ego_cars, velocity_list

    def _visible_zombie_cars_filter(self):
        self.visible_zombie_cars.clear()
        visible_zombie_cars_index = []
        current_loc = self.ego_car.get_transform().location
        for i in range(len(self.zombie_cars)):
            car = self.zombie_cars[i]
            _pos = car.get_transform().location
            _dist = np.hypot(current_loc.x - _pos.x, current_loc.y - _pos.y)
            if _dist < 70:
                visible_zombie_cars_index.append([_dist, i])
        visible_zombie_cars_index = sorted(visible_zombie_cars_index)
        for index in visible_zombie_cars_index:
            self.visible_zombie_cars.append(self.zombie_cars[index[1]])

    def term_check(self):
        collision_hist = self.collision_sensor.get_history()
        if self._collision_check():
            print("COLLISION!!")
            self.collision_event = True
            return True
        elif any(collision_hist):
            print("COLLISION detected from sensor!!")
            self.collision_event = True
            return True
        else:
            return False

    def _collision_check(self, extension_factor=1, margin=0.9):
        """
        This function identifies if an obstacle is present in front of the reference actor
        """
        # world = CarlaDataProvider.get_world()
        actor = self.ego_car
        world_actors = self.visible_zombie_cars
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
            if actor_transform.location.distance(adversary.get_location()) < 40:
                adversary_bbox = adversary.bounding_box
                adversary_transform = adversary.get_transform()
                adversary_loc = adversary_transform.location
                adversary_yaw = adversary_transform.rotation.yaw
                overlap_adversary = common.RotatedRectangle(
                    adversary_loc.x, adversary_loc.y,
                    2 * margin * adversary_bbox.extent.x, 2 * margin * adversary_bbox.extent.y, adversary_yaw)
                overlap_actor = common.RotatedRectangle(
                    actor_location.x, actor_location.y,
                    2 * margin * actor_bbox.extent.x * extension_factor, 2 * margin * actor_bbox.extent.y, actor_yaw)
                overlap_area = overlap_adversary.intersection(overlap_actor).area
                if overlap_area > 0:
                    is_hazard = True
                    break
        return is_hazard

    def on_roadcentre(self):
        actor = self.ego_car
        ego_pos = self._map.get_waypoint(location=actor.get_location(), project_to_road=False).transform.location
        lane_pos = self._map.get_waypoint(location=actor.get_location(), project_to_road=True).transform.location
        dis = np.sqrt((ego_pos.x-lane_pos.x)**2+(ego_pos.y-lane_pos.y)**2)
        return dis









