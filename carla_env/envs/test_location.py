import carla
import numpy as np
import time
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.load_world('Town03')
_map = world.get_map()
spawn_transforms = _map.get_spawn_points()
model3_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
model3_bp.set_attribute('color', '255,255,255')
spawn_points = world.get_map().get_spawn_points()
# for p in spawn_points:
#     print(p.location)
model3_spawn_point = spawn_points[11]
model3 = world.spawn_actor(model3_bp, model3_spawn_point)
spectator = world.get_spectator()
transform = model3.get_transform()
spectator.set_transform(carla.Transform(transform.location + carla.Location(z=500),
carla.Rotation(pitch=-90)))
time.sleep(5)

def draw_waypoints(waypoints, road_id=None, life_time=50.0):
    for waypoint in waypoints:

        #if (waypoint.road_id == road_id):
        world.debug.draw_string(waypoint.location, 'O', draw_shadow=False,
                                     color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                     persistent_lines=True)


waypoints = _map.generate_waypoints(distance=1.0)
hero_car_pos = [26.509409, 7.425340, 0.275307]
#hero_car_pos = [26.509403, 7.425341, 0.001649]
        #self.hero_car_pos = [-42.350990295410156, -2.835118293762207, 1.8431016206741333]
        # self.hero_car_pos = [-74.38717651367188, 57.531620025634766, 1.805267095565796]  # 13
wp_location = carla.Location(x=hero_car_pos[0], y=hero_car_pos[1], z=hero_car_pos[2]+10)

world.debug.draw_string(wp_location, 'O', draw_shadow=True,
                                     color=carla.Color(r=0, g=255, b=0), life_time=50,
                                     persistent_lines=True)
#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

