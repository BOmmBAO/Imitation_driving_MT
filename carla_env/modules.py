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
        self.history.append(True)

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

