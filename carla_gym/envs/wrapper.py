import carla
import random
import time
import collections
import math
import numpy as np
import weakref
import pygame
from config import cfg
from carla_env.misc import *
from plan_control.frenet_optimal_trajectory import frenet_to_inertial

# Colors

# We will use the color palette used in Tango Desktop Project (Each color is indexed depending on brightness level)
# See: https://en.wikipedia.org/wiki/Tango_Desktop_Project

PIXELS_PER_METER = 12

MAP_DEFAULT_SCALE = 0.1
HERO_DEFAULT_SCALE = 1.0

PIXELS_AHEAD_VEHICLE = 150



def print_transform(transform):
    print("Location(x={:.2f}, y={:.2f}, z={:.2f}) Rotation(pitch={:.2f}, yaw={:.2f}, roll={:.2f})".format(
        transform.location.x,
        transform.location.y,
        transform.location.z,
        transform.rotation.pitch,
        transform.rotation.yaw,
        transform.rotation.roll
    )
    )


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[:truncate - 1] + u"\u2026") if len(name) > truncate else name


def angle_diff(v0, v1):
    """ Calculates the signed angle difference (-pi, pi] between 2D vector v0 and v1 """
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def distance_to_line(A, B, p):
    num = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom


def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])


camera_transforms = {
    "spectator": carla.Transform(carla.Location(x=-5.5, z=50.0), carla.Rotation(pitch=-65)),
    "dashboard": carla.Transform(carla.Location(x=1.6, z=1.7))
}


# ===============================================================================
# CarlaActorBase
# ===============================================================================

class CarlaActorBase(object):
    def __init__(self, world, actor):
        self.world = world
        self.actor = actor
        self.world.actor_list.append(self)
        self.destroyed = False

    def destroy(self):
        if self.destroyed:
            raise Exception("Actor already destroyed.")
        else:
            print("Destroying ", self, "...")
            self.actor.destroy()
            self.world.actor_list.remove(self)
            self.destroyed = True

    def get_carla_actor(self):
        return self.actor

    def tick(self):
        pass

    def __getattr__(self, name):
        """Relay missing methods to underlying carla actor"""
        return getattr(self.actor, name)


# ===============================================================================
# CollisionSensor
# ===============================================================================

class CollisionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_collision_fn):
        self.on_collision_fn = on_collision_fn

        # Collision history
        self.history = []

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.collision")

        # Create and setup sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.history.append(True)
        # Call on_collision_fn
        if callable(self.on_collision_fn):
            self.on_collision_fn(event)

    def reset(self):
        self.history = []

    def get_collision_history(self):
        return self.history


# ===============================================================================
# LaneInvasionSensor
# ===============================================================================

class LaneInvasionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_invasion_fn):
        self.on_invasion_fn = on_invasion_fn

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")

        # Create sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: LaneInvasionSensor.on_invasion(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_invasion_fn
        if callable(self.on_invasion_fn):
            self.on_invasion_fn(event)
    def reset(self):
        self.history = []

class LineOfSightSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.distance = None
        self.vehicle_ahead = None
        self._parent = parent_actor
        # self.sensor_transform = carla.Transform(carla.Location(x=4, z=1.7), carla.Rotation(yaw=0)) # Put this sensor on the windshield of the car.
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.obstacle')
        bp.set_attribute('distance', '200')
        bp.set_attribute('hit_radius', '0.5')
        bp.set_attribute('only_dynamics', 'True')
        bp.set_attribute('debug_linetrace', 'False')
        bp.set_attribute('sensor_tick', '0.0')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LineOfSightSensor._on_los(weak_self, event))

    def reset(self):
        self.vehicle_ahead = None
        self.distance = None

    def destroy(self):
        self.sensor.destroy()

    def get_vehicle_ahead(self):
        return self.vehicle_ahead

    # Only works for CARLA 9.6 and above!
    def get_los_distance(self):
        return self.distance

    @staticmethod
    def _on_los(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.vehicle_ahead = event.other_actor
        self.distance = event.distance



# ===============================================================================
# Camera
# ===============================================================================

class Camera(CarlaActorBase):
    def __init__(self, world, width, height, transform=carla.Transform(),
                 sensor_tick=0.0, attach_to=None, on_recv_image=None,
                 camera_type="sensor.camera.rgb", color_converter=carla.ColorConverter.Raw):
        self.on_recv_image = on_recv_image
        self.color_converter = color_converter

        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("sensor_tick", str(sensor_tick))

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda image: Camera.process_camera_input(weak_self, image))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):
            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.on_recv_image(array)

    def destroy(self):
        super().destroy()


# ===============================================================================
# Vehicle
# ===============================================================================

class Vehicle(CarlaActorBase):
    def __init__(self, world, transform=carla.Transform(),
                 on_collision_fn=None, on_invasion_fn=None,
                 vehicle_type="vehicle.lincoln.mkz2017"):
        # Setup vehicle blueprint
        vehicle_bp = world.get_blueprint_library().find(vehicle_type)
        color = vehicle_bp.get_attribute("color").recommended_values[0]
        vehicle_bp.set_attribute("color", color)

        # Create vehicle actor
        actor = world.spawn_actor(vehicle_bp, transform)
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

        # Maintain vehicle control
        self.control = carla.VehicleControl()

        if callable(on_collision_fn):
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        if callable(on_invasion_fn):
            self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)

    def tick(self):
        self.actor.apply_control(self.control)

    def get_speed(self):
        velocity = self.get_velocity()
        return np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)


# ===============================================================================
# World
# ===============================================================================

class World():
    def __init__(self, client):
        # Set map
        self.world = client.load_world('Town04')
        self.map = self.world.get_map()

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.map = self.get_map()
        self.actor_list = []
        self.fps = 30.0
        self.dt = float(cfg.CARLA.DT)
        self.points_to_draw = {}
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)


    def tick(self):
        for actor in list(self.actor_list):
            actor.tick()
        self.world.tick()

    def destroy(self):
        print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()

    def get_carla_world(self):
        return self.world

    def get_carla_map(self):
        return self.map

    def __getattr__(self, name):
        """Relay missing methods to underlying carla object"""
        return getattr(self.world, name)

    # ===============================================================================
    # Vehicle
    # ===============================================================================

class Hero_Actor(CarlaActorBase):
    def __init__(self, world, transform=carla.Transform(),
                 on_collision_fn=None, on_invasion_fn=None, on_los_fn=True):

        # Create vehicle actor
        blueprint = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = '10,0,0'  # Red
            blueprint.set_attribute('color', color)
        actor = world.spawn_actor(blueprint, transform)
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

        # Maintain vehicle control
        self.control = carla.VehicleControl()
        self.global_csp = None
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.actors_with_transforms = []

        if callable(on_collision_fn):
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        if callable(on_invasion_fn):
            self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)
        if on_los_fn:
            self.los_sensor = LineOfSightSensor(actor)

    def tick(self):
        self.actor.apply_control(self.control)

    def get_speed(self):
        velocity = self.actor.get_velocity()
        return np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)

    def update_global_route_csp(self, global_route_csp):
        self.global_csp = global_route_csp

    def get_collision_history(self):
        return self.collision_hist

    def inertial_to_body_frame(self, xi, yi, psi):
        Xi = np.array([xi, yi])  # inertial frame
        R_psi_T = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                            [-np.sin(psi), np.cos(psi)]])
        Xt = np.array([self.actor.get_transform().location.x,  # Translation from inertial to body frame
                       self.actor.get_transform().location.y])
        Xb = np.matmul(R_psi_T, Xi - Xt)
        return Xb

    def body_to_inertial_frame(self, xb, yb, psi):
        Xb = np.array([xb, yb])  # inertial frame
        R_psi = np.array([[np.cos(psi), -np.sin(psi)],  # Rotation matrix
                          [np.sin(psi), np.cos(psi)]])
        Xt = np.array([self.actor.get_transform().location.x,  # Translation from inertial to body frame
                       self.actor.get_transform().location.y])
        Xi = np.matmul(R_psi, Xb) + Xt
        return Xi

    def reset(self):
        self.los_sensor.reset()
        self.init_s = 0
        self.init_d = 0
        x, y, z, yaw = frenet_to_inertial(self.init_s, self.init_d, self.global_csp)
        z += 0.1
        self.actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
        self.actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
        transform = carla.Transform(location=carla.Location(x=x, y=y, z=z),
                                    rotation=carla.Rotation(pitch=0.0, yaw=math.degrees(yaw), roll=0.0))
        self.actor.set_transform(transform)

    def decision_tick(self):
        actors = self.world.get_actors()
        self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
        if self.actor is not None:
            self.hero_transform = self.hero_actor.get_transform()
        self.world.tick()

class Util:

    @staticmethod
    def blits(destination_surface, source_surfaces, rect=None, blend_mode=0):
        for surface in source_surfaces:
            destination_surface.blit(surface[0], surface[1], rect, blend_mode)

    @staticmethod
    def length(v):
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    @staticmethod
    def get_bounding_box(actor):
        bb = actor.trigger_volume.extent
        corners = [carla.Location(x=-bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=-bb.y)]
        corners = [x + actor.trigger_volume.location for x in corners]
        t = actor.get_transform()
        t.transform(corners)
        return corners
