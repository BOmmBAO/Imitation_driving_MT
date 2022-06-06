# ==============================================================================
# -- MultiDiscrete action space environment wrapper for the decision output-----
# ==============================================================================
# The environment wrapper for decision output, the action space in the wrapper
# is a multi-discrete class and each dim represents:
# 1) lateral goal: the lateral goal represents a lane-changing option, including
# left lane, ego car's lane, right lane, total three roads
# 2) longitudinal goal: the distancee between the car position and the goal
# position ahead of the ego car, including 0, 10, 20, 30, 40, 50, 60, 70 in meter
# 3) target velocity: including 0, 10, 20, 30, 40 in km/h
# Note: if the excute mode is short, decision will update every step, else will
# update after excute all the actions
from baselines.gail.planner.pybpp import BppPlanner
from baselines.gail.planner.commondata import TrajectoryPoint
from plan_control.pid import PID_controller
from baselines.gail import utils
import math
from collections import deque
import carla
import gym
from gym import spaces
import numpy as np

class CarlaDM(gym.Env):
    def __init__(self, world, max_length=400, stack=1, train_mode="all", test=True, region=0.8, start_v=6.4,
                 excute_mode='short', interval=0.2, lanes=3, D_skip=1,
                 overtake_curriculum=False, scenario_name=None, g_scratch_eps=120, rule_based=False):
        self.world = world
        self._control = carla.VehicleControl()
        self._stack = stack
        self._ob = deque([], maxlen=stack)
        self.v_history = deque([], maxlen=5)
        self.start_v = start_v
        self.train_mode = train_mode
        self.v_ep = []
        self.v_eps = deque([], maxlen=40)
        self.acc_ep = []
        self.acc_eps = deque([], maxlen=40)
        self.left_offset_ep = []
        self.left_offset_eps = deque([], maxlen=40)
        self.right_offset_ep = []
        self.right_offset_eps = deque([], maxlen=40)
        self.interval = interval
        self.vehicle_planner = BppPlanner(interval, 5)
        self.vehicle_controller = VehiclePIDController(self.world.vehicle)
        self.lanes = lanes
        self._eps = 0
        self._cols = 0
        self.action_space = spaces.MultiDiscrete([lanes, 7, 5])
        high = np.array([np.inf] * (self.world.shape[1] - 2))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.coordinate = utils.Coordinate(self.world.vehicle)
        self.g_scratch_eps = g_scratch_eps

        self.acc_prev = 0.
        self.ep_jerk = 0.
        self.abs_ep_jerk = 0.

        self._max_rollout_steps = max_length
        self._frames = 0
        self.last_wp = self.world.waypoints_forward_l[0][0][0]
        self.test = test
        self.region = region
        self.excute_mode = excute_mode
        self.D_skip = D_skip
        self.overtake_curriculum = overtake_curriculum
        self.rule_based = rule_based
        self.terminal = False
        if self.rule_based:
            self.rule_based_dm = RuleBased()
            self.rule_based_dm._init_condition(self.world, self.coordinate)
        if logger.scenario_name == "Merge_Env":
            # rules to choose different scenario class here
            self.scenario_name = SCENARIO_NAMES[logger.select_scenario_id]
        else:
            self.scenario_name = scenario_name

    def late_init(self):
        self.vehicle_controller = PIDController(self.world.vehicle)
        self.coordinate = utils.Coordinate(self.world.vehicle)

    def _detect_reset(self):
        def _dis(a, b):
            return ((b[1] - a[1]) ** 2 + (b[0] - a[0]) ** 2) ** 0.5

        v_norm_mean = np.mean(self.v_history)
        if len(self.v_history) == 5:
            # if self.world.lane_pos0_dotted_car_left > self.region or self.world.lane_pos0_dotted_car_right < -self.region + 0.1:
            if self.world.lane_pos0_solid_car_left > self.region or self.world.lane_pos0_solid_car_right < -self.region + 0.1:
                return True
            elif v_norm_mean < 0.05:
                return True
        else:
            return False

    def _get_scene(self):
        return self.world.road_direction

    def _update_agent_car(self):
        self.coordinate.update_vehicle(self.world.vehicle)
        self.vehicle_controller.update_vehicle(self.world.vehicle)

    def _recover_world_coordinate(self, traj, mode="3d"):
        # obtain locations
        locations_car = []
        locations_world = []
        if mode == "3d":
            for point in traj:
                locations_car.append([point.x, point.y, point.z])
            for location_car in locations_car:
                location_world = self.coordinate.transform_world3d(location_car)
                locations_world.append(location_world)
        elif mode == "2d":
            for point in traj:
                locations_car.append([point.x, point.y])
            for location_car in locations_car:
                location_world = self.coordinate.transform_world2d(location_car)
                np.append(location_world, 0.)
                locations_world.append(location_world)
        return locations_world

    def reset(self):
        self.wait_for_reset = True
        try:
            self.world.restart()
            self._ob.clear()
            self.v_history.clear()
            self._update_agent_car()
            if len(self.v_ep) != 0:
                self.v_eps.append(np.mean(self.v_ep))
                self.acc_eps.append(np.mean(self.acc_ep))
                self.left_offset_eps.append(np.mean(self.left_offset_ep))
                self.right_offset_eps.append(np.mean(self.right_offset_ep))
                self.v_ep = []
                self.acc_ep = []
                self.left_offset_ep = []
                self.right_offset_ep = []

            ## make sure it is a valid starting point
            ## if it is not, remove this starting point
            if self.world.lane_pos0_solid_car_left > self.region or self.world.lane_pos0_solid_car_right < -self.region + 0.1:
                self.reset()

            if self.world.scenario:
                if self.scenario_name != 'OtherLeadingVehicle_FullMap':
                    if self.scenario_name == 'Cross_Join' or self.scenario_name == 'Cross_Turn_Right':
                        self.fake_step(start_v=7.0, wait_steps=200)
                    elif self.scenario_name == 'Cross_Follow':
                        self.fake_step(start_v=7.0, wait_steps=50)
                    elif self.scenario_name == 'Ring_Join':
                        self.fake_step(start_v=7.0, wait_steps=150)
                    elif self.scenario_name == 'Cross_Turn_Left':
                        self.fake_step(start_v=7.0, wait_steps=150)
                    elif self.scenario_name == 'OverTake':
                        if self._eps < self.g_scratch_eps:
                            self.fake_step(start_v=9.3)
                        else:
                            self.fake_step_overtake(start_v=9.3)
                            self.world._mode = "Policy"
                    elif self.scenario_name == 'OtherLeadingVehicle':
                        self.fake_step(start_v=9.3)
                    elif self.scenario_name == 'Straight_Follow_Double' or self.scenario_name == 'Straight_Follow_Single':
                        self.fake_step(start_v=7.0)
                    else:
                        raise NotImplementedError
                else:
                    self.fake_step(start_v=self.start_v)
            else:
                self.fake_step(start_v=7.0)

            for i in range(self._stack):
                self._ob.append(self.world._observation()[:-2])

            if self.rule_based:
                self.rule_based_dm._init_condition(self.world, self.coordinate)

            self._frames = 0
            self._eps = self._eps + 1
            self.ep_jerk = 0.
            self.abs_ep_jerk = 0.

            self.last_wp = self.world.waypoints_forward_l[0][0][0]
            self.wait_for_reset = False
            return np.concatenate(self._ob, axis=0)
        except:
            print("RESET FAILED")
            try:
                self.wait_for_reset = True
                self.world.force_restart()
                self.late_init()
                # -------------------------------------------------------
                # copy without world.restart
                # -------------------------------------------------------
                self._ob.clear()
                self.v_history.clear()
                self._update_agent_car()
                if len(self.v_ep) != 0:
                    self.v_eps.append(np.mean(self.v_ep))
                    self.acc_eps.append(np.mean(self.acc_ep))
                    self.left_offset_eps.append(np.mean(self.left_offset_ep))
                    self.right_offset_eps.append(np.mean(self.right_offset_ep))
                    self.v_ep = []
                    self.acc_ep = []
                    self.left_offset_ep = []
                    self.right_offset_ep = []

                ## make sure it is a valid starting point
                ## if it is not, remove this starting point
                if self.world.lane_pos0_solid_car_left > self.region or self.world.lane_pos0_solid_car_right < -self.region + 0.1:
                    self.reset()

                # self.fake_step(start_v=self.start_v, wait_steps=self.wait_times)
                if self.world.scenario:
                    if self.scenario_name != 'OtherLeadingVehicle_FullMap':
                        if self.scenario_name == 'Cross_Join' or self.scenario_name == 'Cross_Turn_Right':
                            self.fake_step(start_v=7.0, wait_steps=200)
                        elif self.scenario_name == 'Cross_Follow':
                            self.fake_step(start_v=7.0, wait_steps=50)
                        elif self.scenario_name == 'Ring_Join':
                            self.fake_step(start_v=7.0, wait_steps=150)
                        elif self.scenario_name == 'Cross_Turn_Left':
                            self.fake_step(start_v=7.0, wait_steps=150)
                        elif self.scenario_name == 'OverTake':
                            if self._eps < self.g_scratch_eps:
                                self.fake_step(start_v=9.3)
                            else:
                                self.fake_step_overtake(start_v=9.3)
                        elif self.scenario_name == 'OtherLeadingVehicle':
                            self.fake_step(start_v=9.3)
                        elif self.scenario_name == 'Straight_Follow_Double' or self.scenario_name == 'Straight_Follow_Single':
                            self.fake_step(start_v=7.0)
                        else:
                            raise NotImplementedError
                    else:
                        self.fake_step(start_v=self.start_v)
                else:
                    self.fake_step(start_v=7.0)

                for i in range(self._stack):
                    self._ob.append(self.world._observation()[:-2])

                if self.rule_based:
                    self.rule_based_dm._init_condition(self.world, self.coordinate)

                self._frames = 0
                self._eps = self._eps + 1
                self.last_wp = self.world.waypoints_forward_l[0][0][0]
                self.wait_for_reset = False
                self.terminal = False
                return np.concatenate(self._ob, axis=0)
            except:
                print("RESET FAILED AGAIN")
                self.wait_for_reset = True
                self.world.force_restart()
                self.late_init()
                return self.reset()

    def fake_step(self, action=[0, 1], start_v=6.4, wait_steps=0):
        self.world.steer = action[0]
        fake_step_time_limit = 100 if self.world.sync else 500

        # if self.scenario_name != 'OtherLeadingVehicle':
        if self.scenario_name != 'OtherLeadingVehicle':
            fake_step_time_limit = 300 if self.world.sync else 500
        else:
            fake_step_time_limit = 100 if self.world.sync else 500

        fake_step_time = 0
        wait_steps_count = 0
        _control = carla.VehicleControl()
        while self.world.v_norm_world < start_v and fake_step_time < fake_step_time_limit:
            if wait_steps_count > wait_steps or wait_steps == 0:
                if self.train_mode == "all":
                    steer, throttle_brake = action[0], action[1]
                    steer = np.clip(steer, -1, 1)
                    throttle_brake = np.clip(throttle_brake, -1, 1)
                    if throttle_brake < 0:
                        throttle = 0
                        brake = 1
                    else:
                        throttle = np.clip(throttle_brake, 0, 1)
                        brake = 0
                elif self.train_mode == "steer":
                    steer = action[0]
                    steer = np.clip(steer, -1, 1)
                    throttle = 0.5
                    brake = 0
                else:
                    raise NotImplementedError
                action = np.array([steer, throttle, brake])
                self.terminal = False

                self._control.steer = float(action[0])
                self._control.throttle = float(action[1])
                self._control.brake = float(action[2])
                self.world.apply_control(self._control)
                fake_step_time += 1
            else:
                steer, throttle, brake = 0., 0., 0.
                _action = np.array([steer, throttle, brake])
                _control.steer = float(_action[0])
                _control.throttle = float(_action[1])
                _control.brake = float(_action[2])
                self.world.apply_control(_control)
                wait_steps_count += 1
                fake_step_time += 1
        if fake_step_time >= 500:
            self.reset()

    def fake_step_overtake(self, action=[0, 1], start_v=6.4, wait_steps=0):
        self.world.steer = action[0]
        self.world._mode = "Fake"
        fake_step_time_limit = 100 if self.world.sync else 500

        if self.scenario_name != 'OtherLeadingVehicle':
            fake_step_time_limit = 300 if self.world.sync else 500
        else:
            fake_step_time_limit = 100 if self.world.sync else 500

        fake_step_time = 0
        wait_steps_count = 0
        _control = carla.VehicleControl()
        while self.world.v_norm_world < start_v and fake_step_time < fake_step_time_limit:
            if wait_steps_count > wait_steps or wait_steps == 0:
                if self.train_mode == "all":
                    steer, throttle_brake = action[0], action[1]
                    steer = np.clip(steer, -1, 1)
                    throttle_brake = np.clip(throttle_brake, -1, 1)
                    if throttle_brake < 0:
                        throttle = 0
                        brake = 1
                    else:
                        throttle = np.clip(throttle_brake, 0, 1)
                        brake = 0
                elif self.train_mode == "steer":
                    steer = action[0]
                    steer = np.clip(steer, -1, 1)
                    throttle = 0.5
                    brake = 0
                else:
                    raise NotImplementedError
                action = np.array([steer, throttle, brake])
                terminal = False

                self._control.steer = float(action[0])
                self._control.throttle = float(action[1])
                self._control.brake = float(action[2])
                self.world.apply_control(self._control)
                fake_step_time += 1
            else:
                steer, throttle, brake = 0., 0., 0.
                _action = np.array([steer, throttle, brake])
                _control.steer = float(_action[0])
                _control.throttle = float(_action[1])
                _control.brake = float(_action[2])
                self.world.apply_control(_control)
                wait_steps_count += 1
                fake_step_time += 1

        continue_steps = logger.keypoints[self.world.scenario_now.cur_checkpoint]
        for i in range(continue_steps[0]):
            action = logger.expert_acs[i]
            steer, throttle_brake = action[0], action[1]
            steer = np.clip(steer, -1, 1)
            throttle_brake = np.clip(throttle_brake, -1, 1)
            if throttle_brake < 0:
                throttle = 0
                brake = 1
            else:
                throttle = np.clip(throttle_brake, 0, 1)
                brake = 0
            action = np.array([steer, throttle, brake])
            self._control.steer = float(action[0])
            self._control.throttle = float(action[1])
            self._control.brake = float(action[2])
            self.world.apply_control(self._control)

    def step(self, decision):
        """
        The input of decisions are transferred to a squecne of trajectory
        points by the BPP planner. Then the sequence of trajectory points
        are transferred to a squence of control signals by the PID controller
        :param decision: a list of discrete values
        :output: steering and throttle_brake values
        """

        def lateral_shift_wp(transform, shift):
            transform.rotation.yaw += 90
            shift_location = transform.location + shift * transform.get_forward_vector()
            w = self.world._map.get_waypoint(shift_location, project_to_road=False)
            return w

        def _valid_wp(wp, wp_ref):
            return not (wp.lane_id * wp_ref.lane_id < 0 or wp.lane_id == wp_ref.lane_id or wp.road_id != wp_ref.road_id)

        def _is_ahead(wp, target_pos):
            """
            Test if a target pos is ahead of the waypoint
            """
            wp_pos = _pos(wp)
            orientation = math.radians(wp.transform.rotation.yaw)
            target_vector = np.array([target_pos[0] - wp_pos[0], target_pos[1] - wp_pos[1]])
            forward_vector = np.array([np.cos(orientation), np.sin(orientation)])
            d_angle = math.degrees(math.acos(_cos(forward_vector, target_vector)))
            return d_angle < 90

        def _retrieve_goal_wp(longitudinal_goal, lateral_goal):
            def _get_ref(lateral_goal):
                ref = self.world.status + lateral_goal - 1
                return ref
                # wp_select = self.world.current_wp.next(int(longitudinal_goal))[0]

            wp_select = self.world.waypoints_queue_equal_l[0][int(longitudinal_goal)][0]
            self.world.wp_select = wp_select
            current_wp = self.world.current_wp
            # wp_refline = self.world.waypoints
            need_reset = False
            # retrieve possible lane-changing goals
            shifts = [-3.5, 0, 3.5]
            wp_select_possible = []
            current_wps_shift = []
            for shift in shifts:
                wp_select_possible.append(lateral_shift_wp(wp_select.transform, shift))
                current_wps_shift.append(lateral_shift_wp(current_wp.transform, shift))
            wp_select_shift = wp_select_possible[lateral_goal]
            current_wp_shift = current_wps_shift[lateral_goal]
            self.world.wp_select_shift = wp_select_shift
            self.world.current_wp_shift = current_wp_shift

            # candidates for visualization
            # choice_candidates = [[0, 5, 40], [1, 5, 50], [0, 10, 10], [1, 10, 40], [2, 5, 40], [2, 10, 50], [1, 15, 30], [2, 15, 30]]
            # def generate_wp_candidates(choices):
            #    wp_candidates = []
            #    for choice in choices:
            #        wp_select = self.world.waypoints_queue_equal_l[0][int(choice[1])][0]
            #        shift = shifts[choice[0]]
            #        wp_shift = lateral_shift_wp(wp_select.transform, shift)
            #        wp_candidates.append(wp_shift)
            #    return wp_candidates
            # self.world.wp_candidates = generate_wp_candidates(choice_candidates)

            if current_wp_shift is None or wp_select_shift is None:
                need_reset = True
                reset_type = "WP IS NONE"
                return None, need_reset, reset_type, current_wps_shift
            else:
                # if lateral goal is not 1 (current lane), the choosing lane must not be a solid lane
                if lateral_goal != 1 and (
                        not _valid_wp(current_wp, current_wp_shift) or not _valid_wp(wp_select, wp_select_shift)):
                    need_reset = True
                    reset_type = "INVALID LANE-CHANGING"
                    wp_goal = wp_select_shift
                elif self.world.to_intersection < 20 and lateral_goal != 1:
                    need_reset = True
                    reset_type = "INVALID LANE-CHANGING BEFORE INTERSECTION"
                    wp_goal = wp_select_shift
                elif len(self.world._global_reflines[self.world.status + (lateral_goal - 1)][0]) == 0:
                    need_reset = True
                    reset_type = "INVALID REFLINE"
                    wp_goal = wp_select_shift
                # elif _get_ref(lateral_goal) not in [1, 2, 3] and self.scenario_name == "OtherLeadingVehicle":
                elif _get_ref(lateral_goal) not in [1, 2, 3] and (
                        self.scenario_name == "OtherLeadingVehicle" or self.scenario_name == "OverTake"):
                    reset_type = "INVALID CHOICE IN OVERTAKE SCENARIO"
                    need_reset = True
                    wp_goal = wp_select_shift
                else:
                    reset_type = "VALID"
                    wp_goal = wp_select_shift
                return wp_goal, need_reset, reset_type, current_wps_shift

        def parse_goals(lateral_goal, longitudinal_goal, target_vel):
            # here target velocity is in km/h
            longitudinal_goal = longitudinal_goal * 10.0 + 10.
            if longitudinal_goal == 0.:
                longitudinal_goal = 5.0
            target_vel = target_vel * 10.0 + 10.
            return lateral_goal, longitudinal_goal, target_vel

        def generate_refline(refline_pos):
            refline = []
            for i in range(len(refline_pos) - 1):
                pos = refline_pos[i]
                pos_next = refline_pos[i + 1]
                x, y, z = pos[0], pos[1], pos[2]
                dir_x, dir_y, dir_z = (pos_next[0] - pos[0]) / 0.05, (pos_next[1] - pos[1]) / 0.05, (
                        pos_next[2] - pos[2]) / 0.05
                dir_xyz = [dir_x, dir_y, dir_z]
                dir_norm = utils._norm3d(dir_xyz)
                dir_x, dir_y, dir_z = dir_x / dir_norm, dir_y / dir_norm, dir_z / dir_norm
                point = TrajectoryPoint()
                point.x, point.y, point.z, point.dir_x, point.dir_y, point.dir_z = x, y, z, dir_x, dir_y, dir_z
                refline.append(point)
            return refline

        # ====================================================================
        # timeout handler
        # ====================================================================
        try:
            self.decision = decision
            lateral_goal, longitudinal_goal, target_vel = decision
            lateral_goal = int(lateral_goal)
            # here we implement a curriculum
            # -----------------------------------------------------
            # standard curriculum
            # -----------------------------------------------------
            if self.overtake_curriculum == 1:
                if self._eps % 1 == 0:
                    if self._frames < 100:
                        if self._frames < 23:
                            if self.world.status == 2:
                                lateral_goal = 0
                                target_vel = 3
                            elif self.world.status == 1:
                                lateral_goal = 1
                                target_vel = 3
                        elif self._frames >= 23 and self._frames < 60:
                            lateral_goal = 1
                            target_vel = 2
                        elif self._frames >= 60:
                            if self.world.status == 1:
                                lateral_goal = 2
                            elif self.world.status == 2:
                                lateral_goal = 1
                            target_vel = 2
                    else:
                        lateral_goal = 1
                        target_vel = 2
                    longitudinal_goal = random.choice([1])
            # -----------------------------------------------------
            # turn left with 25km/h and 25km/h back to the original lane
            # -----------------------------------------------------
            if self.overtake_curriculum == 2:
                if self.world.scenario_now.cur_checkpoint == "KL40" or self.world.scenario_now.cur_checkpoint == "TR":
                    lateral_goal = 2
                else:
                    lateral_goal = 1
            # -----------------------------------------------------
            # turn right curriculum
            # -----------------------------------------------------
            if self.overtake_curriculum == 3:
                if self._eps % 1 == 0:
                    if self._frames < 100:
                        if self._frames < 23:
                            if self.world.status == 2:
                                lateral_goal = 2
                                target_vel = 4
                            elif self.world.status == 1:
                                lateral_goal = 1
                                target_vel = 3
                        elif self._frames >= 23 and self._frames < 60:
                            lateral_goal = 1
                            target_vel = 3
                        elif self._frames >= 60:
                            if self.world.status == 3:
                                lateral_goal = 0
                                target_vel = 2
                            elif self.world.status == 2:
                                lateral_goal = 1
                                target_vel = 2
                    else:
                        lateral_goal = 1
                        target_vel = 2
                    longitudinal_goal = random.choice([1])
            # if self._frames > 160:
            #    lateral_goal = 2
            # -----------------------------------------------------
            lateral_goal, longitudinal_goal, target_vel = parse_goals(lateral_goal, longitudinal_goal, target_vel)
            if self.rule_based:
                lateral_goal, longitudinal_goal, target_vel = self.rule_based_dm.decision(self.world, self._frames,
                                                                                          self.coordinate)
            self.lateral_goal = lateral_goal
            self.world.longitudinal_goal, self.world.lateral_goal, self.world.target_vel = longitudinal_goal, lateral_goal, target_vel
            # generate goal position for bpp planner
            # the lateral goal is a lane-changing option
            current_wp_now = self.world.waypoints_forward_l[0][0][0]
            goal_wp, need_reset, reset_type, current_wps_shift = _retrieve_goal_wp(longitudinal_goal, lateral_goal)
            if need_reset:
                print(reset_type)
                return self.step_reset(decision)
            else:
                # ---------------------------------------------------------
                # lane-chaning detection
                lane_changing = not (
                    (
                                current_wp_now.lane_id * self.last_wp.lane_id < 0 or current_wp_now.lane_id == self.last_wp.lane_id)) \
                                and current_wp_now.road_id == self.last_wp.road_id
                if lane_changing:
                    self.world.lane_change = lane_changing
                # retrieve waypoints on the target lane
                end_idx = int(longitudinal_goal * 10)
                interval_idx = int(self.interval * 10)
                refline_pos = self.world._global_reflines[self.world.status + (lateral_goal - 1)]
                refline_car_pos = refline_pos
                refline_pos = refline_pos[0][:end_idx:interval_idx]
                refline = generate_refline(refline_pos)
                self.last_wp = current_wp_now
                # ---------------------------------------------------------
                # execute decision
                for i in range(self.D_skip):
                    self.world.goal_wp_now = goal_wp
                    goal_pos_world = utils._pos3d(goal_wp)
                    goal_pos_car = self.coordinate.transform_car3d(goal_pos_world)
                    goal_pos_car = refline_car_pos[0][-1]
                    goal_pos = goal_pos_car
                    goal_pos_last = refline_car_pos[0][-2]
                    # compute goal direction
                    goal_dir = [(goal_pos_car[0] - goal_pos_last[0]) / 0.05,
                                (goal_pos_car[1] - goal_pos_last[1]) / 0.05,
                                (goal_pos_car[2] - goal_pos_last[2]) / 0.05]
                    goal_pos_car_norm = utils._norm3d(goal_dir)
                    goal_dir = [goal_pos_car[0] / goal_pos_car_norm, goal_pos_car[1] / goal_pos_car_norm,
                                goal_pos_car[2] / goal_pos_car_norm]

                    start_pos = [0., 0., 0.]
                    if self.world.v_norm_world < 0.001:
                        start_dir = [0.00001, 0., 0.]
                    else:
                        start_dir = [1., 0., 0.]

                    # bpp planner, note the target_vel and current_vel must be in m/s
                    cur_vel = self.world.v_norm_world
                    tgt_vel = target_vel / 3.6
                    cur_acc = self.world.acc_norm_world

                    traj_ret = self.vehicle_planner.run_step(start_pos, start_dir, refline, cur_vel, tgt_vel, cur_acc)
                    # for p in traj_ret:
                    #    print(p.x, p.y)
                    # Debug
                    # traj_ret_lst = [[point.x, point.y, point.z, point.dir_x, point.dir_y, point.dir_z, point.theta, point.velocity, point.acceleration, point.curvature, point.sumdistance] for point in traj_ret]
                    # if self._frames > 20 and self._frames <= 70:
                    #     pickle.dump([start_pos, start_dir, refline, cur_vel, tgt_vel, cur_acc, traj_ret_lst], self._file)
                    # if self._frames > 80:
                    #     self._file.close()
                    #     exit()
                    self.traj_ret_world = self.world.traj_ret = self._recover_world_coordinate(traj_ret)

                    # pid controller, note the target_vel_pid must be in km/h
                    if self.excute_mode == "short":
                        # get reference position and target velocity
                        idx = 5
                        target_ref_point = traj_ret[idx]
                        target_vel_pid = traj_ret[idx].velocity * 3.6
                        self.world.pp_cur_vel, self.world.pp_tgt_vel, self.world.pp_target_vel_pid = cur_vel * 3.6, tgt_vel * 3.6, target_vel_pid
                        ref_pos = [self.traj_ret_world[idx][0], self.traj_ret_world[idx][1],
                                   self.traj_ret_world[idx][2]]
                        ref_location = carla.Location(x=ref_pos[0], y=ref_pos[1], z=ref_pos[2])
                        self.world.ref_location = ref_location

                        # get reference orientation
                        idx = 30
                        if self.rule_based:
                            idx = min(30, len(traj_ret) - 1)
                        target_ref_point = traj_ret[idx]
                        self.world.pp_ref_vel = target_ref_point.velocity * 3.6
                        target_pos_pid = [self.traj_ret_world[idx][0], self.traj_ret_world[idx][1],
                                          self.traj_ret_world[idx][2]]
                        target_location_pid = carla.Location(x=target_pos_pid[0], y=target_pos_pid[1],
                                                             z=target_pos_pid[2])
                        target_wp_pid = self.world._map.get_waypoint(target_location_pid)
                        self.world.target_wp_pid = target_wp_pid

                        # pid control
                        control = self.vehicle_controller.run_step(target_vel_pid, target_wp_pid, target_ref_point)
                        steer = control.steer
                        throttle = control.throttle
                        brake = control.brake
                        throttle_brake = throttle + -1 * brake
                        action = [steer, throttle_brake]
                        return self.step_control(action)
                    elif self.excute_mode == "long":
                        raise NotImplementedError
                    else:
                        raise NotImplementedError
        except:
            print("DECISION STEP FAILED")
            self.world.force_restart()
            self.late_init()
            return self.step_reset(self.decision)

    def step_reset(self, decision):
        control = self.world.agent_control
        v_now = self.world.v_norm_world
        acc_now = self.world.acc_norm_world
        left_offset = self.world.lane_pos0_solid_car_left
        right_offset = self.world.lane_pos0_car_right
        rew = 0.
        ob = self.get_ob(self.world._observation()[:-2])
        done, terminal = True, True
        self.terminal = terminal
        return ob, rew, done, {'scene': self._get_scene(), 'terminal': terminal,
                               'v': v_now, 'acc': acc_now, 'left_offset': left_offset,
                               'right_offset': right_offset, 'control': control,
                               'current_pos': self.world.current_pos, 'yaw': np.array([self.world.yaw])}

    def step_control(self, action):
        max_steer = self.world.delta_time / 1.5
        max_throttle = self.world.delta_time / 0.2

        if self.train_mode == "all":
            steer, throttle_brake = action[0], action[1]
            if throttle_brake < 0:
                throttle = 0
                brake = 1
            else:
                throttle = np.clip(throttle_brake, 0, 1)
                brake = 0
        elif self.train_mode == "steer":
            steer = action[0]
            time_steer = np.clip(steer, -max_steer, max_steer)
            steer = np.clip(time_steer + self.world.steer, -1, 1)
            throttle = 0.5
            brake = 0
        else:
            raise NotImplementedError
        self.world.steer = steer
        action = np.array([steer, throttle, brake])
        terminal = False

        self.v_history.append(self.world.v_norm_world)
        self.v_ep.append(self.world.v_norm_world)
        self.acc_ep.append(self.world.acc_norm_world)
        self.left_offset_ep.append(self.world.lane_pos0_solid_car_left)
        self.right_offset_ep.append(self.world.lane_pos0_car_right)

        self._control.steer = float(action[0])
        self._control.throttle = float(action[1])
        self._control.brake = float(action[2])

        self._frames += 1

        self.world.apply_control(self._control)
        rew = self.world._reward(self.lateral_goal)
        self.world.ep_rew += self.world.rew_now
        if self._detect_reset():
            done = True
            terminal = True
        elif self._frames >= self._max_rollout_steps:
            done = True
        elif self.world.collision():  # _sensor._collision:
            terminal = True
            done = True
        else:
            done = False
        self.terminal = terminal

        v_now = self.world.v_norm_world
        acc_now = self.world.acc_norm_world
        jerk = (acc_now - self.acc_prev) / 0.2
        self.ep_jerk = self.ep_jerk + jerk
        self.abs_ep_jerk = self.abs_ep_jerk + abs(jerk)
        self.acc_prev = acc_now
        # print('self.ep_jerk: ', self.ep_jerk)
        # print('self.abs_ep_jerk: ', self.abs_ep_jerk)
        left_offset = self.world.lane_pos0_solid_car_left
        right_offset = self.world.lane_pos0_car_right
        ob = self.get_ob(self.world._observation()[:-2])
        control = action[:2]
        return ob, rew, done, {'scene': self._get_scene(), 'terminal': terminal,
                               'v': v_now, 'acc': acc_now, 'left_offset': left_offset,
                               'right_offset': right_offset, 'control': control,
                               'current_pos': self.world.current_pos, 'yaw': np.array([self.world.yaw])}

    def get_ob(self, ob):
        self._ob.append(ob)
        return np.concatenate(self._ob, axis=0)

    def eprew(self):
        return self.world.ep_rew
