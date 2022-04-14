import carla
import utils.common
import math
import numpy as np
from utils import common
import matplotlib.pyplot as plt
plt.ion()
import copy
import utils.logger
from utils.logger import TensorBoardOutputFormat


class STATUS:

    FOLLOWING = 0
    START_LANE_CHANGE_L = 1
    LANE_CHANGING_L = 2
    START_LANE_CHANGE_R = 3
    LANE_CHANGING_R = 4
    STOPPING = 5


class FeatureExt():

    def __init__(self, env, vehicle):
        self.autopilot = False
        self.dim = '3d'
        self.traffic_light_flag = None
        self.world = env.world
        self.vehicle = vehicle
        self.vehicle_info = VehicleInfo(vehicle)
        self.map = self.world.get_map()
        self.zombie_cars = env.zombie_cars
        self.cur_lane = None
        self.cur_lane_width = None
        self.cur_wp = None
        self.wp_list = None
        self.current_loc = None
        self.dt = env.dt

        self.waypoints_buffer = None
        self.waypoints_buffer_lane_id = None
        self.wp_ds = 0.5
        self.wp_horizon = 70
        self.distance_rate = 1.4
        self.wp_index = self.exponential_index(horizon=70)

        self.visible_zombie_cars = env.visible_zombie_cars
        self.show_dt = env.dt * 1.5
        self.is_junction = False

        self.stop_sign = False
        self.start_sign = True
        self.stop_wps = None
        self.time = 200
        self.traffic_light = None
        self.traffic_light_flag = False

        self.observation = None
        self.info_dict = {}
        self.obs_index = None
        self.pre_obs_index = None
        self.zombie_num = 0

        self.old_pos_x = None
        self.old_pos_y = None
        self.old_yaw = None

        self.i = 0

    def update(self):
        self.vehicle_info.update()
        self.current_loc = self.vehicle.get_transform().location
        self.cur_lane = self.map.get_waypoint(self.current_loc)
        self.cur_lane_width = self.cur_lane.lane_width
        self.wp_list = self.wp_list_extract(self.cur_lane)

        self.is_junction = self.cur_lane.is_junction
        self.waypoints_buffer_update()
        self.cur_wp = self.expon_down_sample()
        self.obs_update()

        # self.traffic_light_rule()

    # ==============================================================================
    # -- Functions about Waypoints Extraction --------------------------------------
    # ==============================================================================
    def waypoints_buffer_update(self):

        def uniform_wps(wp, d_s, max_sample):
            seq = d_s
            wp_l = []
            while True:
                wp_l.append(wp.next(seq)[0])
                seq += d_s

                if wp_l[-1].is_junction:
                    _lane = self.map.get_waypoint(wp_l[-1].transform.location)
                    wp_l.extend(_lane.next_until_lane_end(d_s))
                    break
                if len(wp_l) > max_sample:
                    break
            return wp_l

        def check_lane_change(cur_lane, buffer_pos):
            if cur_lane.lane_id == buffer_pos.lane_id:
                return False
            else:
                return True

        count = 100 / self.wp_ds

        if self.waypoints_buffer is None:
            _lane = self.map.get_waypoint(self.current_loc)
            self.waypoints_buffer = uniform_wps(_lane, self.wp_ds, count)

        # Find the nearest point in the waypoints buffer
        nearest_dist, index = 100, 0
        for i in range(len(self.waypoints_buffer)):
            pos = self.waypoints_buffer[i].transform.location
            _dist = np.hypot(pos.x - self.current_loc.x, pos.y - self.current_loc.y)
            if _dist < nearest_dist:
                nearest_dist = _dist
                index = i
        self.waypoints_buffer = self.waypoints_buffer[index:]

        if check_lane_change(self.cur_lane, self.waypoints_buffer[index]):
            _lane = self.map.get_waypoint(self.current_loc)
            self.waypoints_buffer = uniform_wps(_lane, self.wp_ds, count)

        # Update
        if 0 < len(self.waypoints_buffer) < count:
            _lane = self.map.get_waypoint(self.waypoints_buffer[-1].transform.location)
            self.waypoints_buffer.extend(uniform_wps(_lane, self.wp_ds,
                                                     count - len(self.waypoints_buffer)))

    def expon_down_sample(self):
        wp = []
        for index in self.wp_index:
            if index <= len(self.waypoints_buffer):
                wp.append(self.waypoints_buffer[index])
        while len(wp) < len(self.wp_index):
            print("The number of waypoints is wrong!")
            wp.append(wp[-1])
        return wp

    def find_lane_border(self, lane_wp):
        line_right, line_left = [], []
        for p in lane_wp:
            line_right.append(self.find_lanepoint_right(p))
            line_left.append(self.find_lanepoint_left(p))
        return line_right, line_left

    def wp_list_extract(self, cur_wp):

        def wp_side_extract(wp_list, wp, side):
            if side == 'right':
                while True:
                    if (wp.lane_change == carla.LaneChange.Right
                            or wp.lane_change == carla.LaneChange.Both):
                        wp = wp.get_right_lane()
                        wp_list.append(wp)
                    else:
                        break

            if side == 'left':
                while True:
                    if (wp.lane_change == carla.LaneChange.Left
                            or wp.lane_change == carla.LaneChange.Both):
                        wp = wp.get_left_lane()
                        wp_list.append(wp)
                    else:
                        break
            return wp_list

        wp_l = [cur_wp]
        if cur_wp.is_junction is False:
            wp_l = wp_side_extract(wp_l, cur_wp, 'right')
            wp_l = wp_side_extract(wp_l, cur_wp, 'left')
        return wp_l

    def exponential_index(self, horizon):
        exp_index = []
        seq = 1
        while seq < horizon:
            exp_index.append(round(seq / self.wp_ds))
            # seq *= self.distance_rate
            seq += 5
        return exp_index

    def find_road_border(self, wp_list):

        def local_wp(wp, max_distance=70):
            seq = 1.0
            wp_l = []
            while True:
                wp_l.append(wp.next(seq)[0])
                # seq *= self.distance_rate
                seq += 5
                if seq > max_distance:
                    break
            while len(wp_l) < len(self.wp_index):
                print("The number of waypoints is wrong!")
                wp_l.append(wp_l[-1])
            return wp_l

        def generate_position_list(wp_l, side='right'):
            if wp_l is None:
                return None
            else:
                pos_list = []
                if side == 'right':
                    for i in range(len(wp_l)):
                        pos_list.append(self.find_lanepoint_right(wp_l[i]))
                elif side == 'left':
                    for i in range(len(wp_l)):
                        pos_list.append(self.find_lanepoint_left(wp_l[i]))
                else:
                    return None
            return pos_list

        outer_line_r, outer_line_l = None, None
        for wp in wp_list:
            if wp.lane_change == carla.LaneChange.Right:
                outer_line_l = generate_position_list(local_wp(wp), 'left')
            elif wp.lane_change == carla.LaneChange.Left:
                outer_line_r = generate_position_list(local_wp(wp), 'right')
            elif wp.lane_change == carla.LaneChange.NONE:
                outer_line_l = generate_position_list(local_wp(wp), 'left')
                outer_line_r = generate_position_list(local_wp(wp), 'right')
        if outer_line_l is None or outer_line_r is None:
            print("Extraction for outer road lines fails!")
        return outer_line_r, outer_line_l

    def find_lanepoint_right(self, wp):
        location_drift = carla.Location(x=-np.sin(wp.transform.rotation.yaw / 180 * np.pi) * wp.lane_width / 2,
                                        y=np.cos(wp.transform.rotation.yaw / 180 * np.pi) * wp.lane_width / 2,
                                        z=0.2)
        lp = carla.Location(wp.transform.location + location_drift)
        return lp

    def find_lanepoint_left(self, wp):
        location_drift = carla.Location(x=np.sin(wp.transform.rotation.yaw / 180 * np.pi) * wp.lane_width / 2,
                                        y=-np.cos(wp.transform.rotation.yaw / 180 * np.pi) * wp.lane_width / 2,
                                        z=0.2)
        lp = carla.Location(wp.transform.location + location_drift)
        return lp

    # ==============================================================================
    # -- Features used in Rule-based Decision---------------------------------------
    # ==============================================================================
    def find_lead_car(self):
        forward_cars = []
        ego_pos = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        ego_dir = self.vehicle.get_physics_control().wheels[0].position - \
                  self.vehicle.get_physics_control().wheels[2].position
        ego_vec = [ego_dir.x, ego_dir.y]
        cur_road = self.wp_list[0]

        for i in range(len(self.zombie_cars)):
            if self.zombie_cars[i] == self.vehicle:
                continue
            z_pos = self.zombie_cars[i].get_location()
            vec_car = [z_pos.x - ego_pos[0], z_pos.y - ego_pos[1]]
            dis = np.hypot(vec_car[0], vec_car[1])
            if dis < 50:
                theta = common.cal_angle(vec_car, ego_vec)
                if theta < math.pi / 2 and self.check_onroad(self.zombie_cars[i], cur_road):
                    forward_cars.append([dis, i])

        if forward_cars:
            forward_cars.sort()
            lead_car = self.zombie_cars[forward_cars[0][1]]
            pos_x = lead_car.get_transform().location.x
            pos_y = lead_car.get_transform().location.y
            # self.bbox_display(lead_car, "red")
            length = lead_car.bounding_box.extent.x
            vel = np.hypot(lead_car.get_velocity().x, lead_car.get_velocity().y)
            return [pos_x, pos_y, length, vel]
        else:
            return None

    def find_cars_onlane(self, lane):
        forwards_cars, backwards_cars = [], []
        ego_pos = [self.vehicle.get_location().x, self.vehicle.get_location().y]
        ego_dir = self.vehicle.get_physics_control().wheels[0].position - \
                  self.vehicle.get_physics_control().wheels[2].position
        ego_vec = [ego_dir.x, ego_dir.y]
        for i in range(len(self.zombie_cars)):
            if self.zombie_cars[i] == self.vehicle:
                continue
            z_pos = self.zombie_cars[i].get_location()
            dis = np.hypot(z_pos.x - self.current_loc.x, z_pos.y - self.current_loc.y)
            if dis < 50:
                vec_car = [z_pos.x - ego_pos[0], z_pos.y - ego_pos[1]]
                theta = common.cal_angle(vec_car, ego_vec)
                if self.check_onroad(self.zombie_cars[i], lane):
                    if theta <= math.pi / 2:
                        forwards_cars.append([dis, i])
                    else:
                        backwards_cars.append([dis, i])
        if forwards_cars:
            forwards_cars.sort()
            lead_car = self.zombie_cars[forwards_cars[0][1]]
            pos_x = lead_car.get_transform().location.x
            pos_y = lead_car.get_transform().location.y
            length = lead_car.bounding_box.extent.x
            vel = np.hypot(lead_car.get_velocity().x, lead_car.get_velocity().y)
            lead_info = [pos_x, pos_y, length, vel]
        else:
            lead_info = None

        if backwards_cars:
            backwards_cars.sort()
            fol_car = self.zombie_cars[backwards_cars[0][1]]
            pos_x = fol_car.get_transform().location.x
            pos_y = fol_car.get_transform().location.y
            length = fol_car.bounding_box.extent.x
            vel = np.hypot(fol_car.get_velocity().x, fol_car.get_velocity().y)
            acc = np.hypot(fol_car.get_acceleration().x, fol_car.get_acceleration().y)
            fol_info = [pos_x, pos_y, length, vel, acc]
        else:
            fol_info = None

        return lead_info, fol_info

    def check_onroad(self, vehicle, lane):
        for i in range(4):
            p = vehicle.get_physics_control().wheels[i].position / 100
            wp = self.map.get_waypoint(p)
            if wp.lane_id == lane.lane_id:
                return True
        return False

    def traffic_light_rule(self):
        # Check the state of traffic light
        if not self.traffic_light_flag:
            self.traffic_light = self.world.get_traffic_lights_from_waypoint(self.cur_wp[0], 50)
        if self.traffic_light:
            self.traffic_light_flag = True
            if self.traffic_light[0].get_state() == carla.TrafficLightState.Red:
                self.stop_sign = True
                self.start_sign = False
                self.stop_wps = self.traffic_light[0].get_stop_waypoints()
            elif self.traffic_light[0].get_state() == carla.TrafficLightState.Green:
                print("Green Light")
                self.stop_sign = False
                self.start_sign = True
                self.traffic_light_flag = False
        else:
            self.stop_sign = False

        # Set the green light forcefully
        if self.stop_sign:
            self.time -= 1
            print("self.time", self.time)
            if self.time < 0:
                self.time = 200
                self.traffic_light[0].set_state(carla.TrafficLightState.Green)

    # ==============================================================================
    # -- Feature Vector used in Neural Networks ------------------------------------
    # ==============================================================================

    def obs_update(self):
        #feature list
        self.info_dict.clear()
        self.ext_egocar_info(self.vehicle, local_frame = True)
        self.ext_zombiecars_info(local_frame = True)
        self.ext_waypoints_info(self.wp_list, self.cur_wp, local_frame = True)
        output_info = self.dic2list(self.info_dict)
        self.observation = np.array(output_info)

        if self.obs_index is None:
            self.obs_index = self.ext_obs_index(self.info_dict)
            del(self.info_dict['ego_car_world_trans'])
            self.pre_obs_index = self.ext_obs_index(self.info_dict)


    def ext_egocar_info(self, vehicle, local_frame):
        if local_frame:
            v_world = [vehicle.get_velocity().x, vehicle.get_velocity().y]
            self.info_dict['ego_car_world_trans'] = [self.vehicle_info.x,
                                                     self.vehicle_info.y,
                                                     self.vehicle_info.yaw]
            self.info_dict['ego_car_local_trans'] = [0, 0, 0]
            self.info_dict['ego_car_vel'] = self._rotate_car(v_world)

        else:
            self.info_dict['ego_car_pos'] = [self.vehicle_info.x, self.vehicle_info.y]
            self.info_dict['ego_car_vel'] = [vehicle.get_velocity().x, vehicle.get_velocity().y]
            self.info_dict['ego_car_acc'] = [vehicle.get_acceleration().x, vehicle.get_acceleration().y]

    def ext_zombiecars_info(self, local_frame, total_cars=6):

        def _get_v_car(vehicle):
            v_world = [vehicle.get_velocity().x, vehicle.get_velocity().y]
            v_car = self._rotate_car(v_world)
            return v_car if local_frame else v_world

        def get_car_pos(vehicle):
            ego_pos = [self.current_loc.x, self.current_loc.y]
            zom_pos_world = [vehicle.get_transform().location.x, vehicle.get_transform().location.y]
            zom_pos = self._transform_car(ego_pos, zom_pos_world)
            return zom_pos if local_frame else ego_pos

        vehicles_bounding_box, vehicles_pos, vehicles_v, vehicles_acc = [], [], [], []
        for car in self.visible_zombie_cars:
            vehicles_pos.append(get_car_pos(car))
            vehicles_v.append(_get_v_car(car))
            self.zombie_num = len(self.visible_zombie_cars)
            if len(vehicles_pos) >= total_cars:
                break

        while len(vehicles_pos) < total_cars:
            fasle_pos = [0]
            vehicles_v.append(fasle_pos * 2)
            vehicles_pos.append(fasle_pos * 2)

        self.info_dict['zombie_cars_pos'] = vehicles_pos
        self.info_dict['zombie_cars_v'] = vehicles_v

    def ext_waypoints_info(self, wp_list, cur_wp, local_frame):
        # Inner lane line
        inner_line_r, inner_line_l = self.find_lane_border(cur_wp)
        # Outer road line
        if not self.is_junction:
            outer_line_r, outer_line_l = self.find_road_border(wp_list)
        else:
            outer_line_r, outer_line_l = inner_line_r, inner_line_l
        # Display inner and outer lines
        self.draw_lane_points(inner_line_r)
        self.draw_lane_points(inner_line_l)
        self.draw_lane_line(outer_line_l)
        self.draw_lane_line(outer_line_r)

        def _to_vector(ego_pos, line):
            _wp = []
            for point in line:
                res = self._transform_car(ego_pos, [point.x, point.y])
                _wp.append(res if local_frame else [point.x, point.y])
            return _wp

        ego_pos = [self.current_loc.x, self.current_loc.y]

        # Transform into vector
        self.info_dict['inner_line_right'] = _to_vector(ego_pos, inner_line_r)
        self.info_dict['inner_line_left'] = _to_vector(ego_pos, inner_line_l)
        self.info_dict['outer_line_right'] = _to_vector(ego_pos, outer_line_r)
        self.info_dict['outer_line_left'] = _to_vector(ego_pos, outer_line_l)

    def a2r(self, angle):
        return angle / 180 * np.pi

    def _rotate_car(self, pos_world):
        yaw_radians = self.vehicle_info.yaw
        pos_tran = [pos_world[0] * np.cos(yaw_radians) + pos_world[1] * np.sin(yaw_radians),
                    -pos_world[0] * np.sin(yaw_radians) + pos_world[1] * np.cos(yaw_radians)]
        return pos_tran

    def _rotate_zombie2world(self, pos_car_world, length):
        yaw_radians = self.a2r(pos_car_world.get_transform().rotation.yaw)

        def world_pos(point, vehicle_center):
            pos_tran = [vehicle_center.x + point[0] * np.cos(yaw_radians) - point[1] * np.sin(yaw_radians),
                        vehicle_center.y + point[0] * np.sin(yaw_radians) + point[1] * np.cos(yaw_radians)]
            return pos_tran

        vehicle_center = pos_car_world.get_transform().location
        box = [[length[0], length[1]],
               [length[0], -length[1]],
               [-length[0], +length[1]],
               [-length[0], -length[1]]]
        box_world = []
        for p in box:
            pos = world_pos(p, vehicle_center)
            box_world.append(pos)
        return box_world

    def _transform_car(self, pos_ego, pos_actor):
        pos_ego_tran = self._rotate_car(pos_ego)
        pos_actor_tran = self._rotate_car(pos_actor)
        pos = [pos_actor_tran[0] - pos_ego_tran[0], pos_actor_tran[1] - pos_ego_tran[1]]
        return pos

    def _flat_list(self, ls):
        if type(ls) == list or type(ls) == tuple:
            output = []
            for item in ls:
                output += self._flat_list(item)
            return output
        else:
            return [ls]

    def dic2list(self, list_dict):
        to_list = []
        for value in list_dict.values():
            new_fea = self._flat_list(value)
            to_list = to_list + new_fea
        return to_list

    def ext_obs_index(self, list_dict):
        to_list = []
        index_dict = {}
        for item in list_dict.items():
            new_fea = self._flat_list(item[1])
            index_dict[item[0]] = [len(to_list), len(to_list) + len(new_fea)]
            to_list = to_list + new_fea
        return index_dict

    # ==============================================================================
    # -- Functions about Displaying ------------------------------------------------
    # ==============================================================================
    def lane_display(self, cur_lane, wp_list):
        # Inner lane line
        inner_line_r, inner_line_l = self.find_lane_border(cur_lane)
        # Outer road line
        outer_line_r, outer_line_l = self.find_road_border(wp_list)
        # Display inner and outer lines
        self.draw_lane_points(inner_line_r)
        self.draw_lane_points(inner_line_l)
        self.draw_lane_line(outer_line_l)
        self.draw_lane_line(outer_line_r)

    def bbox_display(self, vehicle, color):
        if color == "red":
            _color = carla.Color(60, 10, 10, 0)
        elif color == "green":
            _color = carla.Color(10, 60, 10, 0)
        elif color == "blue":
            _color = carla.Color(10, 10, 60, 0)
        # Bounding Box
        bounding_box = vehicle.bounding_box
        bounding_box.location = vehicle.get_transform().location
        self.world.debug.draw_box(bounding_box, vehicle.get_transform().rotation,
                                  color=_color, life_time=self.show_dt)

    def point_display(self, p):
        self.world.debug.draw_point(p, size=0.2, color=carla.Color(0, 255, 0), life_time=self.show_dt)

    def ref_display(self, ref_x, ref_y):
        for n in range(0, len(ref_x), max(int(len(ref_x) / 20), 1)):
            p = carla.Location()
            p.x = ref_x[n]
            p.y = ref_y[n]
            self.world.debug.draw_point(p, size=0.08, color=carla.Color(0, 0, 255), life_time=self.show_dt)

    def draw_lane_points(self, points_list):
        if points_list:
            for p in points_list:
                self.world.debug.draw_point(p, life_time=0.2)

    def draw_lane_line(self, pos_list):
        if pos_list:
            for _loop in range(len(pos_list) // 2 - 1):
                self.world.debug.draw_line(pos_list[2 * _loop],
                                           pos_list[2 * _loop + 1],
                                           thickness=0.2,
                                           color=carla.Color(50, 0, 100, 0),
                                           life_time=0.2)


class VehicleInfo:

    def __init__(self, vehicle, des_vel=7):
        self.vehicle = vehicle
        self.target_vel = des_vel
        self.dt = 0.1

        self.merge_length = 0
        self.speed_max = 40
        self.acc_max = 10
        self.status = STATUS.FOLLOWING

        wb_vec = vehicle.get_physics_control().wheels[0].position - vehicle.get_physics_control().wheels[2].position
        self.wheelbase = np.sqrt(wb_vec.x ** 2 + wb_vec.y ** 2 + wb_vec.z ** 2) / 100
        self.shape = [self.vehicle.bounding_box.extent.x, self.vehicle.bounding_box.extent.y,
                      self.vehicle.bounding_box.extent.z]

        self.x = None
        self.y = None
        self.v = None
        self.acc = None
        self.yaw = None
        self.pitch = None
        self.roll = None
        self.dir_vec = None
        self.steer = None
        self.update()  # initialize

    def update(self):
        self.x = self.vehicle.get_location().x
        self.y = self.vehicle.get_location().y

        _v = self.vehicle.get_velocity()
        _acc = self.vehicle.get_acceleration()
        self.v = np.sqrt(_v.x ** 2 + _v.y ** 2)
        self.acc = np.sqrt(_acc.x ** 2 + _acc.y ** 2)
        self.steer = self.vehicle.get_control().steer
        self.dir_vec = self.vehicle.get_physics_control().wheels[0].position - \
                       self.vehicle.get_physics_control().wheels[2].position
        self.yaw = math.atan2(self.dir_vec.y, self.dir_vec.x)
        self.roll = 0
        self.pitch = 0

        self.merge_length = max(4 * self.v, 12)

