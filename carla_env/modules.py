import carla
import math
import weakref
import collections
import shapely
import numpy as np
from queue import Queue

class CollisionSensor(object):

    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_history(self):
        return self.history

    def reset(self):
        self.history = []

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.history.append(True)


class LaneInvasionSensor(object):

    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.history = []
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        self.crossed_lane_markings = 2
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    def get_history(self):
        return self.history

    def reset(self):
        self.history = []

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.history.append(event)

class CameraSensor(object):

    def __init__(self, env):
        sensor_queue = Queue()
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.sensor = self.world.spawn_actor(camera_bp,
                                   carla.Transform(carla.Location(x=-5.5, z=2.5),
                                                   carla.Rotation(pitch=8.0)),
                                   env.ego_car,
                                   carla.AttachmentType.SpringArm)
        self.sensor.listen(lambda image: sensor_queue.put(image.frame))


class Sensors(object):
    """Class to keep track of all sensors added to the vehicle"""

    def __init__(self, world, vehicle):
        super(Sensors, self).__init__()
        self.world = world
        self.vehicle = vehicle
        self.camera_queue = Queue() # queue to store images from buffer
        self.collision_flag = False # Flag for colision detection
        self.lane_crossed = False # Flag for lane crossing detection
        self.lane_crossed_type = '' # Which type of lane was crossed

        self.camera_rgb = self.add_sensors(world, vehicle, 'sensor.camera.rgb')
        self.collision = self.add_sensors(world, vehicle, 'sensor.other.collision')
        self.lane_invasion = self.add_sensors(world, vehicle, 'sensor.other.lane_invasion', sensor_tick = '0.5')

        self.sensor_list = [self.camera_rgb, self.collision, self.lane_invasion]

        self.collision.listen(lambda collisionEvent: self.track_collision(collisionEvent))
        self.camera_rgb.listen(lambda image: self.camera_queue.put(image))
        self.lane_invasion.listen(lambda event: self.on_invasion(event))

    def add_sensors(self, world, vehicle, type, sensor_tick = '0.0'):

        sensor_bp = self.world.get_blueprint_library().find(type)
        try:
            sensor_bp.set_attribute('sensor_tick', sensor_tick)
        except:
            pass
        if type == 'sensor.camera.rgb':
            sensor_bp.set_attribute('image_size_x', '100')
            sensor_bp.set_attribute('image_size_y', '100')

        sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=vehicle)
        return sensor

    def track_collision(self, collisionEvent):
        '''Whenever a collision occurs, the flag is set to True'''
        self.collision_flag = True

    def reset_sensors(self):
        '''Sets all sensor flags to False'''
        self.collision_flag = False
        self.lane_crossed = False
        self.lane_crossed_type = ''

    def on_invasion(self, event):
        '''Whenever the car crosses the lane, the flag is set to True'''
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.lane_crossed_type = text[0]
        self.lane_crossed = True

    def destroy_sensors(self):
        '''Destroy all sensors (Carla actors)'''
        for sensor in self.sensor_list:
            sensor.destroy()

    # # we also add a lidar on it
    # lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
    # lidar_bp.set_attribute('channels', str(32))
    # lidar_bp.set_attribute('points_per_second', str(90000))
    # lidar_bp.set_attribute('rotation_frequency', str(40))
    # lidar_bp.set_attribute('range', str(20))
    #
    # # set the relative location
    # lidar_location = carla.Location(0, 0, 2)
    # lidar_rotation = carla.Rotation(0, 0, 0)
    # lidar_transform = carla.Transform(lidar_location, lidar_rotation)
    # # spawn the lidar
    # self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.ego_car)
    # self.lidar.listen(
    #     lambda point_cloud: common.sensor_callback(point_cloud, self.sensor_queue, "lidar"))
