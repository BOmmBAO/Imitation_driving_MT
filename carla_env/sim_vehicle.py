import carla
from carla_env.feature import FeatureExt
from plan_control.path_planner import *
import plan_control.velocity_planner as velocity_planner

from plan_control.pid import *
from plan_control.linear_MPC import *
from carla_env.rule_decision import *



class VehicleInit():

    def __init__(self, env, decision):
        self.env = env
        self.world = env.world
        self.decision = decision
        self.ego_car = self.env.ego_car
        self.ego_car_config = CarConfig(env, self.ego_car, decision)
        #self.lead_car = env.lead_car
        #self.lead_car_config = CarConfig(env, self.lead_car)
        self.fea_ext = self.ego_car_config.fea_ext

        self.ego_car_config.fea_ext.update()
        self.last_action = np.array([1.0, 0.0])
        #self.lead_car_config.fea_ext.update()
        self.reference = None

    def step_action(self, action):
        current_action = np.array(action) + self.last_action
        current_action = np.clip(
            current_action, -1.0, 1.0, dtype=np.float32)
        throttle_or_brake, steer = current_action
        if throttle_or_brake >= 0:
            throttle = throttle_or_brake
            brake = 0
        else:
            throttle = 0
            brake = -throttle_or_brake

            # Apply control
        act = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake))
        self.ego_car.apply_control(act)
        for _ in range(1):
            self.world.tick()
        self.last_action = current_action
        self.ego_car_config.fea_ext.update()
        #self.lead_car_config.fea_ext.update()
        #rx, ry, ryaw, s_sum = self.ego_car_config.path.following_path(self.fea_ext.cur_wp)

    def step_decision(self, decision):
        self.ego_car_config.fea_ext.update()
        #self.lead_car_config.fea_ext.update()
        # is_wrong, self.reference = self.update_dec(decision[0], self.ego_car_config)
        # [rx, ry, ryaw, ref_vel] = self.reference
        merge_dist, ref_vel = self.decode_decision(decision, self.ego_car_config)
        try:
            rx, ry, ryaw, s_sum = self.ego_car_config.path.update(merge_dist)
            self.reference = [rx, ry, ryaw, ref_vel]
            poly_coe = velocity_planner.speed_profile_uniform(ref_vel)
            control_comd = self.ego_car_config.controller.update(rx, ry, ryaw, poly_coe)
            self.ego_car_config.fea_ext.ref_display(rx, ry)
            self.ego_car.apply_control(control_comd)

            #_rx, _ry, _ryaw, _s_sum = self.lead_car_config.path.update(0)
            _poly_coe = velocity_planner.speed_profile_uniform(5)
            #_control_comd = self.lead_car_config.controller.update(_rx, _ry, _ryaw, _poly_coe)
            #self.lead_car.apply_control(_control_comd)
            return False
        except:
            return True

    def reset(self, env):
        self.ego_car = env.ego_car
        self.ego_car_config = CarConfig(env, self.ego_car,self.decision)
        #self.lead_car = env.lead_car
        #self.lead_car_config = CarConfig(env, self.lead_car)
        self.ego_car_config.fea_ext.update()
        self.last_action = np.array([1.0, 0.0])
        #self.lead_car_config.fea_ext.update()
        self.reference = None


    def rule_based_step(self):
        self.fea_ext.update()
        rx, ry, ryaw, s_sum = self.ego_car_config.path.update(self.fea_ext.vehicle_info.merge_length)
        self.fea_ext.ref_display(rx, ry)
        ref_vel = self.ego_car_config.dec_maker.decision()
        # poly_coe = velocity_planner.speed_profile_quinticPoly(self.fea_ext.vehicle_info, ref_vel, s_sum[-1])
        poly_coe = velocity_planner.speed_profile_uniform(ref_vel)
        control_comd = self.ego_car_config.controller.update(rx, ry, ryaw, poly_coe)
        self.ego_car.apply_control(control_comd)

    def decode_decision(self, decision, vehicle):
        '''3     4
           1  0  2'''
        merge_dist = 0
        # Change the status according to decision
        target_vel, _a = vehicle.dec_maker.car_following()
        if vehicle.fea_ext.vehicle_info.status == STATUS.FOLLOWING:
            if decision == 0:
                vehicle.fea_ext.vehicle_info.status = STATUS.FOLLOWING
                merge_dist = 0
            else:
                if decision % 2 == 0:
                    if vehicle.fea_ext.vehicle_info.status == STATUS.FOLLOWING or \
                            vehicle.fea_ext.vehicle_info.status == STATUS.LANE_CHANGING_L:
                        vehicle.fea_ext.vehicle_info.status = STATUS.START_LANE_CHANGE_R
                    merge_dist = (decision / 2) * 10

                elif decision % 2 == 1:
                    if vehicle.fea_ext.vehicle_info.status == STATUS.FOLLOWING or \
                            vehicle.fea_ext.vehicle_info.status == STATUS.LANE_CHANGING_R:
                        vehicle.fea_ext.vehicle_info.status = STATUS.START_LANE_CHANGE_L
                    merge_dist = ((decision + 1) / 2) * 10

        return merge_dist, target_vel

    # Decision with Interuption
    def update_dec(self, decision, vehicle):
        wrong_dec = False
        rx, ry, ryaw = None, None, None
        target_vel, _ = vehicle.dec_maker.car_following()

        if decision == 0:
            follow_lane = vehicle.fea_ext.cur_wp
            rx, ry, ryaw, s_sum = vehicle.path.following_path(follow_lane)
        else:
            if decision % 2 == 1:
                merge_dist = ((decision + 1) / 2) * 10

                target_vel += 3
                if vehicle.fea_ext.cur_lane.lane_change == carla.LaneChange.Left \
                        or vehicle.fea_ext.cur_lane.lane_change == carla.LaneChange.Both:
                    left_lane = vehicle.fea_ext.cur_lane.get_left_lane()
                    merge_point = vehicle.path.merge_point_calcu(left_lane, merge_dist)
                    target_lane = self.env.world.get_map().get_waypoint(merge_point)
                    rx, ry, ryaw, s_sum = vehicle.path.laneChange_path(vehicle.fea_ext.vehicle_info, target_lane, merge_point)
                else:
                    wrong_dec = True

            elif decision % 2 == 0:
                merge_dist = (decision / 2) * 10
                target_vel += 3
                if vehicle.fea_ext.cur_lane.lane_change == carla.LaneChange.Right \
                        or vehicle.fea_ext.cur_lane.lane_change == carla.LaneChange.Both:
                    right_lane = vehicle.fea_ext.cur_lane.get_right_lane()
                    merge_point = vehicle.path.merge_point_calcu(right_lane, merge_dist)
                    target_lane = self.env.world.get_map().get_waypoint(merge_point)
                    rx, ry, ryaw, s_sum = vehicle.path.laneChange_path(vehicle.fea_ext.vehicle_info, target_lane, merge_point)
                else:
                    wrong_dec = True

        return wrong_dec, [rx, ry, ryaw, target_vel]

# ToDo zombie cars configuration
class CarConfig():
    def __init__(self, env, car, decision):
        self.fea_ext = FeatureExt(env, car)
        if decision:
            self.path = path_planner(env, self.fea_ext)
            self.dec_maker = RuleBased(self.fea_ext)
            self.controller = PID_controller(self.fea_ext, car)
            # self.controller = MPC(self.fea_ext)