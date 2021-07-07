
import rospy
import numpy as np
from easydict import EasyDict as edict
import math
import copy
import time

from zzz_driver_msgs.msg import RigidBodyStateStamped
from zzz_navigation_msgs.msg import Map, Lane, LanePoint
from zzz_navigation_msgs.utils import get_lane_array, default_msg as navigation_default
from zzz_cognition_msgs.msg import MapState, LaneState, RoadObstacle
from zzz_cognition_msgs.utils import convert_tracking_box, default_msg as cognition_default
from zzz_perception_msgs.msg import TrackingBoxArray, DetectionBoxArray, ObjectSignals, DimensionWithCovariance
from zzz_common.geometry import dist_from_point_to_polyline2d, wrap_angle, dist_from_point_to_closedpolyline2d, dist_from_point_to_line2d
from zzz_common.kinematics import get_frenet_state
from zzz_cognition_msgs.msg import DrivingSpace, DynamicBoundary, DynamicBoundaryPoint
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from zzz_driver_msgs.utils import get_speed, get_yaw
from prediction import linear_predict, predict_vehicle_behavior

#jxy 20191125: first output the driving space, then use the driving space for cognition. 
#For this demo version, it will be a unified module, in future versions, this will be split into 2 modules.

DT = 0.3  # time tick [s]
STEPS = 10 # predict time steps #TODO0128: parameterize

class DrivingSpaceConstructor:
    def __init__(self, lane_dist_thres=5):
        self._static_map_buffer = None

        self._ego_vehicle_state_buffer = None

        self._surrounding_object_list_buffer = None

        self._traffic_light_detection_buffer = None

        self.dynamic_map = None

        self.virtual_lane_trigger_buffer = 0

        self._lanes_memory = []
        
        self._lane_dist_thres = lane_dist_thres

        self._ego_vehicle_distance_to_lane_head = [] # distance from vehicle to lane start
        self._ego_vehicle_distance_to_lane_tail = [] # distance from vehicle to lane end

    # ====== Data Receiver =======

    def receive_static_map(self, static_map):
        assert type(static_map) == Map
        # modify static map: virtual lane generation
        self.virtual_lane_trigger_buffer = 0

        if len(static_map.virtual_lanes) != 0 and len(static_map.next_lanes) != 0: #num must be the same, ensured in local map

            exit_junction_point_list = []
            exit_junction_direction_list = []
            if len(static_map.next_drivable_area.points) > 3:
                for i in range(len(static_map.next_lanes)):
                    point_x = static_map.next_lanes[i].central_path_points[0].position.x
                    point_y = static_map.next_lanes[i].central_path_points[0].position.y
                    next_point_x = static_map.next_lanes[i].central_path_points[1].position.x
                    next_point_y = static_map.next_lanes[i].central_path_points[1].position.y
                    exit_junction_point_list.append([point_x, point_y])
                    exit_junction_direction_list.append([next_point_x - point_x, next_point_y - point_y])

                for i in range(len(static_map.next_lanes)):
                    self.extend_junction_path(static_map, i, static_map.virtual_lanes[i].central_path_points, \
                        exit_junction_point_list[i], exit_junction_direction_list[i])

                self.virtual_lane_trigger_buffer = 1 #loaded next junction

            for i in range(len(static_map.virtual_lanes)):
                static_map.virtual_lanes[i].central_path_points.extend(static_map.next_lanes[i].central_path_points)

            self._lanes_memory = static_map.virtual_lanes

        if len(static_map.lanes) == 0: #in junction, keep the virtual lanes
            static_map.virtual_lanes = self._lanes_memory #jxy: check whether it will change as python

        self._static_map_buffer = static_map
        rospy.loginfo("Updated Local Static Map: lanes_num = %d, in_junction = %d, exit_lane_index = %d",
            len(static_map.lanes), int(static_map.in_junction), static_map.exit_lane_index[0])

    def receive_object_list(self, object_list):
        assert type(object_list) == TrackingBoxArray
        if self._ego_vehicle_state_buffer != None:
            self._surrounding_object_list_buffer = convert_tracking_box(object_list, self._ego_vehicle_state_buffer)
            #jxy: the converted objects are in the RoadObstacle() format

    def receive_ego_state(self, state):
        assert type(state) == RigidBodyStateStamped
        self._ego_vehicle_state_buffer = state
        #TODO: wrap ego vehicle just like wrapping obstacle

    def receive_traffic_light_detection(self, detection):
        assert type(detection) == DetectionBoxArray
        self._traffic_light_detection_buffer = detection

    # ====== Data Updator =======

    def update_driving_space(self):

        tstates = edict()

        if not self.init_tstates(tstates):
            return False

        self.predict_obstacles(tstates, STEPS, DT)

        self.update_static_map(tstates)

        self.update_dynamic_map_by_elements(tstates)

        self.update_dynamic_map_by_unified_boundary(tstates)

        self.output_and_visualization(tstates)
        
        rospy.logdebug("Updated driving space")

        return True

    # ====== Submodules =======

    def init_tstates(self, tstates):
        if not self._ego_vehicle_state_buffer:
            return False

        tstates.ego_vehicle_state = copy.deepcopy(self._ego_vehicle_state_buffer)

        # Update buffer information
        tstates.surrounding_object_list = copy.deepcopy(self._surrounding_object_list_buffer) or [] #jxy20201202: remove deepcopy to accelerate

        tstates.static_map = self._static_map_buffer or navigation_default(Map)
        tstates.next_drivable_area = []
        tstates.surrounding_object_list_timelist = []
        tstates.drivable_area_timelist = []
        tstates.ego_s = 0
        tstates.dynamic_map = cognition_default(MapState)

        tstates.dynamic_map.virtual_lane_trigger = copy.deepcopy(self.virtual_lane_trigger_buffer)

        return True

    def predict_obstacles(self, tstates, steps, dt):
        tstates.surrounding_object_list_timelist = linear_predict(tstates.surrounding_object_list, \
            tstates.ego_vehicle_state.state.pose.pose.position, steps, dt)

    def update_static_map(self, tstates):
        tstates.static_map_lane_path_array = get_lane_array(tstates.static_map.lanes)
        tstates.static_map_lane_tangets = [[point.tangent for point in lane.central_path_points] for lane in tstates.static_map.lanes]
        tstates.static_map_virtual_lane_path_array = get_lane_array(tstates.static_map.virtual_lanes)
        tstates.static_map_virtual_lane_tangets = [[point.tangent for point in virtual_lane.central_path_points] for virtual_lane in tstates.static_map.virtual_lanes]

    def update_dynamic_map_by_elements(self, tstates, DT=0.3, STEPS=10):
        tstates.dynamic_map.header.frame_id = "map"
        tstates.dynamic_map.header.stamp = rospy.Time.now()
        tstates.dynamic_map.ego_state = tstates.ego_vehicle_state.state
        tstates.dynamic_map.DT = DT
        tstates.dynamic_map.STEPS = STEPS
        
        if tstates.static_map.in_junction:
            rospy.logdebug("Cognition: In junction due to static map report junction location")
            tstates.dynamic_map.model = MapState.MODEL_JUNCTION_MAP
            tstates.dynamic_map.virtual_lane_trigger = 0
        else:
            tstates.dynamic_map.model = MapState.MODEL_MULTILANE_MAP

        if len(tstates.static_map.virtual_lanes) != 0: #jxy20201218: in junction model, the virtual lanes are still considered.
            for lane in tstates.static_map.virtual_lanes:
                dlane = cognition_default(LaneState)
                dlane.map_lane = lane
                tstates.dynamic_map.mmap.lanes.append(dlane)
            tstates.dynamic_map.mmap.exit_lane_index = copy.deepcopy(tstates.static_map.exit_lane_index)

        for i in range(len(tstates.surrounding_object_list)):
            front_vehicle = tstates.surrounding_object_list[i]
            tstates.dynamic_map.jmap.obstacles.append(front_vehicle)

        # Update driving_space with tstate
        if len(tstates.static_map.virtual_lanes) != 0:
            self.locate_ego_vehicle_in_lanes(tstates) #TODO: consider ego_s and front/rear vehicle in virtual lanes!!
            self.locate_surrounding_objects_in_lanes(tstates) # TODO: here lies too much repeated calculation, change it and lateral decision
            self.locate_stop_sign_in_lanes(tstates)
            self.locate_speed_limit_in_lanes(tstates)
        else:
            rospy.loginfo("virtual lanes not constructed")

        if tstates.static_map.in_junction or len(tstates.static_map.lanes) == 0:
            rospy.logdebug("In junction due to static map report junction location")
        else:
            for i in range(STEPS):
                self.locate_predicted_obstacle_in_lanes(tstates, i) #real lanes

    def update_dynamic_map_by_unified_boundary(self, tstates):
        for i in range(STEPS):
            self.calculate_drivable_areas(tstates, i)

            dynamic_boundary = DynamicBoundary()
            dynamic_boundary.header.frame_id = "map"
            dynamic_boundary.header.stamp = rospy.Time.now()
            for j in range(len(tstates.drivable_area_timelist[i])):
                drivable_area_point = tstates.drivable_area_timelist[i][j]
                boundary_point = DynamicBoundaryPoint()
                boundary_point.x = drivable_area_point[0]
                boundary_point.y = drivable_area_point[1]
                boundary_point.vx = drivable_area_point[2]
                boundary_point.vy = drivable_area_point[3]
                boundary_point.base_x = drivable_area_point[4]
                boundary_point.base_y = drivable_area_point[5]
                boundary_point.omega = drivable_area_point[6]
                boundary_point.flag = drivable_area_point[7]
                dynamic_boundary.boundary.append(boundary_point)
            tstates.dynamic_map.jmap.boundary_list.append(dynamic_boundary)

    def output_and_visualization(self, tstates):

        #jxy0511: add output to txt
        
        ego_x = tstates.ego_vehicle_state.state.pose.pose.position.x
        ego_y = tstates.ego_vehicle_state.state.pose.pose.position.y
        vel_self = np.array([[tstates.ego_vehicle_state.state.twist.twist.linear.x], [tstates.ego_vehicle_state.state.twist.twist.linear.y], [tstates.ego_vehicle_state.state.twist.twist.linear.z]])
        ego_vx = vel_self[0]
        ego_vy = vel_self[1]
        ego_v = math.sqrt(ego_vx ** 2 + ego_vy ** 2)

        total_ttc = 0
        object_x = 0 #jxy: note that only 1 vehicle is supported! If more, change here!
        object_y = 0
        object_v = 0
        if tstates.surrounding_object_list is not None:
            #here is only 1 vehicle
            for vehicle_idx, vehicle in enumerate(tstates.surrounding_object_list):
                object_x = vehicle.state.pose.pose.position.x
                object_y = vehicle.state.pose.pose.position.y
                object_vx = vehicle.state.twist.twist.linear.x
                object_vy = vehicle.state.twist.twist.linear.y
                object_v = math.sqrt(object_vx ** 2 + object_vy ** 2)

                r = np.array([object_x - ego_x, object_y - ego_y, 0])
                v = np.array([0, 0, 0])
                v[0] = object_vx - ego_vx
                v[1] = object_vy - ego_vy
                vr = np.dot(v, r) * r / (np.power(np.linalg.norm(r), 2))
                
                ttc = np.linalg.norm(r) / np.linalg.norm(vr)
                if np.dot(vr, r) > 0:
                    ttc = 100 #jxy: getting farther
                total_ttc = total_ttc + 1 / ttc
        if total_ttc is not 0:
            total_ttc = 1 / total_ttc

        t1 = time.time()

        fw = open("/home/carla/ZZZ/record_stats_pbud.txt", 'a')
        fw.write(str(t1) + " " + str(ego_x) + " " + str(ego_y) + " " + str(ego_v) + " " + \
            str(object_x) + " " + str(object_y) + " " + str(object_v) + " " + str(total_ttc))   
        fw.write("\n")
        fw.close()

        rospy.loginfo("we are at: %f %f", ego_x, ego_y)

        self.dynamic_map = tstates.dynamic_map
        
        self.visualization(tstates)

    # ========= For virtual lane =========

    def extend_junction_path(self, static_map, extension_index, path, exit_junction_point, exit_junction_direction):

        #TODO: consider strange junctions, or U turn.

        if len(exit_junction_point) == 0:
            rospy.loginfo("extension failed due to no exit junction point")
            return path

        x1 = path[-1].position.x
        y1 = path[-1].position.y
        x2 = exit_junction_point[0]
        y2 = exit_junction_point[1]
        dx1 = path[-1].position.x - path[-2].position.x
        dy1 = path[-1].position.y - path[-2].position.y
        dx2 = exit_junction_direction[0]
        dy2 = exit_junction_direction[1]

        left_vertical_dx1 = -dy1
        left_vertical_dy1 = dx1 #judge left turning or right turning, TODO: test in more scenarios, not only standard 90 turn

        left_turn_flag = 1
        if left_vertical_dx1 * dx2 + left_vertical_dy1 * dy2 < 0:
            left_turn_flag = 0

        #change coordinate, plan a 3rd polyline
        l = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
        if l == 0:
            rospy.loginfo("extension failed due to l = 0")
            return
        m = (dx1 * (x2 - x1) + dy1 * (y2 - y1)) / l
        n = (dx2 * (x2 - x1) + dy2 * (y2 - y1)) / l
        tt1 = math.sqrt(((dx1 * dx1 + dy1 * dy1) / (m * m)) - 1)
        tt2 = math.sqrt(((dx2 * dx2 + dy2 * dy2) / (n * n)) - 1)

        extended_path = static_map.virtual_lanes[extension_index].central_path_points

        for i in range(11):
            u = l - l / 10 * i
            v = -(u - l) * u * ((tt1 - tt2) / (l * l) * u + tt2 / l)
            if left_turn_flag == 0:
                v = -v
            x = ((x1 - x2) * u + (y2 - y1) * v) / l + x2
            y = ((y1 - y2) * u + (x1 - x2) * v) / l + y2
            point = LanePoint()
            point.position.x = x
            point.position.y = y
            extended_path.append(point)

    # ========= For in lane =========

    def locate_object_in_lane(self, object, tstates, dimension, dist_list=None, virtual_flag=0):
        '''
        Calculate (continuous) lane index for a object.
        Parameters: dist_list is the distance buffer. If not provided, it will be calculated
        '''

        if virtual_flag == 1:
            if not dist_list:
                dist_list = np.array([dist_from_point_to_polyline2d(
                    object.pose.pose.position.x,
                    object.pose.pose.position.y,
                    lane) for lane in tstates.static_map_virtual_lane_path_array]) # here lane is a python list of (x, y)
        else:
            if not dist_list:
                dist_list = np.array([dist_from_point_to_polyline2d(
                    object.pose.pose.position.x,
                    object.pose.pose.position.y,
                    lane) for lane in tstates.static_map_lane_path_array]) # here lane is a python list of (x, y)
        
        # Check if there's only two lanes
        if virtual_flag == 1:
            lanes_consider = tstates.static_map.virtual_lanes
        else:
            lanes_consider = tstates.static_map.lanes

        if len(lanes_consider) < 2:
            closest_lane = second_closest_lane = 0
        else:
            closest_lane, second_closest_lane = np.abs(dist_list[:, 0]).argsort()[:2]

        # Signed distance from target to two closest lane

        closest_lane_dist, second_closest_lane_dist = dist_list[closest_lane, 0], dist_list[second_closest_lane, 0]

        if abs(closest_lane_dist) > self._lane_dist_thres:
            return -1, -99, -99, -99, -99

        if virtual_flag == 1:
            lane = tstates.static_map.virtual_lanes[closest_lane]
        else:
            lane = tstates.static_map.lanes[closest_lane]
        left_boundary_array = np.array([(lbp.boundary_point.position.x, lbp.boundary_point.position.y) for lbp in lane.left_boundaries])
        right_boundary_array = np.array([(lbp.boundary_point.position.x, lbp.boundary_point.position.y) for lbp in lane.right_boundaries])

        if len(left_boundary_array) == 0:
            if virtual_flag == 1:
                ffstate = get_frenet_state(object,
                                tstates.static_map_virtual_lane_path_array[closest_lane],
                                tstates.static_map_virtual_lane_tangets[closest_lane]
                            )
            else:
                ffstate = get_frenet_state(object,
                                tstates.static_map_lane_path_array[closest_lane],
                                tstates.static_map_lane_tangets[closest_lane]
                            )
            lane_anglediff = ffstate.psi
            lane_dist_s = ffstate.s
            return closest_lane, -1, -1, lane_anglediff, lane_dist_s
        else:
            # Distance to lane considering the size of the object
            x = object.pose.pose.orientation.x
            y = object.pose.pose.orientation.y
            z = object.pose.pose.orientation.z
            w = object.pose.pose.orientation.w

            rotation_mat = np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y], [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x], [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]])
            rotation_mat_inverse = np.linalg.inv(rotation_mat) #those are the correct way to deal with quaternion

            vector_x = np.array([dimension.length_x, 0, 0])
            vector_y = np.array([0, dimension.length_y, 0])
            dx = np.matmul(rotation_mat_inverse, vector_x)
            dy = np.matmul(rotation_mat_inverse, vector_y)

            #the four corners of the object, in bird view: left front is 0, counterclockwise.
            #TODO: may consider 8 corners in the future
            corner_list_x = np.zeros(4)
            corner_list_y = np.zeros(4)
            corner_list_x[0] = object.pose.pose.position.x + dx[0]/2.0 + dy[0]/2.0
            corner_list_y[0] = object.pose.pose.position.y + dx[1]/2.0 + dy[1]/2.0
            corner_list_x[1] = object.pose.pose.position.x - dx[0]/2.0 + dy[0]/2.0
            corner_list_y[1] = object.pose.pose.position.y - dx[1]/2.0 + dy[1]/2.0
            corner_list_x[2] = object.pose.pose.position.x - dx[0]/2.0 - dy[0]/2.0
            corner_list_y[2] = object.pose.pose.position.y - dx[1]/2.0 - dy[1]/2.0
            corner_list_x[3] = object.pose.pose.position.x + dx[0]/2.0 - dy[0]/2.0
            corner_list_y[3] = object.pose.pose.position.y + dx[1]/2.0 - dy[1]/2.0

            dist_left_list_all = np.array([dist_from_point_to_polyline2d(
                    corner_list_x[i],
                    corner_list_y[i],
                    left_boundary_array) for i in range(4)])
            dist_right_list_all = np.array([dist_from_point_to_polyline2d(
                    corner_list_x[i],
                    corner_list_y[i],
                    right_boundary_array) for i in range(4)])

            dist_left_list = dist_left_list_all[:, 0]
            dist_right_list = dist_right_list_all[:, 0]

            lane_dist_left_t = -99
            lane_dist_right_t = -99

            if np.min(dist_left_list) * np.max(dist_left_list) <= 0:
                # the object is on the left boundary of lane
                lane_dist_left_t = 0
            else:
                lane_dist_left_t = np.sign(np.min(dist_left_list)) * np.min(np.abs(dist_left_list))

            if np.min(dist_right_list) * np.max(dist_right_list) <= 0:
                # the object is on the right boundary of lane
                lane_dist_right_t = 0
            else:
                lane_dist_right_t = np.sign(np.min(dist_right_list)) * np.min(np.abs(dist_right_list))

            if np.min(dist_left_list) * np.max(dist_left_list) > 0 and np.min(dist_right_list) * np.max(dist_right_list) > 0:
                if np.min(dist_left_list) * np.max(dist_right_list) >= 0:
                    # the object is out of the road
                    closest_lane = -1
            
            if virtual_flag == 1:
                ffstate = get_frenet_state(object,
                                tstates.static_map_virtual_lane_path_array[closest_lane],
                                tstates.static_map_virtual_lane_tangets[closest_lane]
                            )
            else:
                ffstate = get_frenet_state(object,
                                tstates.static_map_lane_path_array[closest_lane],
                                tstates.static_map_lane_tangets[closest_lane]
                            )
            lane_anglediff = ffstate.psi
            lane_dist_s = ffstate.s # this is also helpful in getting ego s coordinate in the road

            # Judge whether the point is outside of lanes
            if closest_lane == -1:
                # The object is at left or right most
                return closest_lane, lane_dist_left_t, lane_dist_right_t, lane_anglediff, lane_dist_s
            else:
                # The object is between center line of lanes
                a, b = closest_lane, second_closest_lane
                la, lb = abs(closest_lane_dist), abs(second_closest_lane_dist)
                if lb + la == 0:
                    lane_index_return = -1
                else:
                    lane_index_return = (b*la + a*lb)/(lb + la)
                return lane_index_return, lane_dist_left_t, lane_dist_right_t, lane_anglediff, lane_dist_s

    def locate_predicted_obstacle_in_lanes(self, tstates, time_step):
        obstacles_step = tstates.surrounding_object_list_timelist[time_step]
        if obstacles_step == None:
            return
        for obj in obstacles_step:
            if len(tstates.static_map.lanes) != 0:
                obj.lane_index, obj.lane_dist_left_t, obj.lane_dist_right_t, obj.lane_anglediff, obj.lane_dist_s = self.locate_object_in_lane(obj.state, tstates, obj.dimension)
            else:
                obj.lane_index = -1

    def locate_surrounding_objects_in_lanes(self, tstates, lane_dist_thres=3):
        lane_front_vehicle_list = [[] for _ in tstates.static_map.virtual_lanes]
        lane_rear_vehicle_list = [[] for _ in tstates.static_map.virtual_lanes]

        # TODO: separate vehicle and other objects?
        if tstates.surrounding_object_list is not None:
            for vehicle_idx, vehicle in enumerate(tstates.surrounding_object_list):
                dist_list = np.array([dist_from_point_to_polyline2d(
                    vehicle.state.pose.pose.position.x,
                    vehicle.state.pose.pose.position.y,
                    lane, return_end_distance=True)
                    for lane in tstates.static_map_virtual_lane_path_array])
                closest_dist = np.min(np.abs(dist_list[:, 0]))
                closest_lanes = []
                if len(tstates.static_map.virtual_lanes) > len(tstates.static_map.lanes) and len(tstates.static_map.lanes) != 0:
                    for i in range(len(tstates.static_map_virtual_lane_path_array)):
                        if dist_list[i, 0] == closest_dist:
                            closest_lanes.append(i)
                else:
                    closest_lanes.append(np.argmin(np.abs(dist_list[:, 0]))) #in junction, only use the closest lane

                # Determine if the vehicle is close to lane enough
                if abs(closest_dist) > lane_dist_thres:
                    continue 
                for closest_lane in closest_lanes: #copied virtual lanes are considered
                    if dist_list[closest_lane, 3] < self._ego_vehicle_distance_to_lane_head[closest_lane]:
                        # The vehicle is behind if its distance to lane start is smaller
                        lane_rear_vehicle_list[closest_lane].append((vehicle_idx, dist_list[closest_lane, 3]))
                    if dist_list[closest_lane, 4] < self._ego_vehicle_distance_to_lane_tail[closest_lane]:
                        # The vehicle is ahead if its distance to lane end is smaller
                        lane_front_vehicle_list[closest_lane].append((vehicle_idx, dist_list[closest_lane, 4]))
        
        # Put the vehicles onto lanes
        for lane_id in range(len(tstates.static_map.virtual_lanes)):
            front_vehicles = np.array(lane_front_vehicle_list[lane_id])
            rear_vehicles = np.array(lane_rear_vehicle_list[lane_id])

            if len(front_vehicles) > 0:
                # Descending sort front objects by distance to lane end
                for vehicle_row in reversed(front_vehicles[:,1].argsort()):
                    front_vehicle_idx = int(front_vehicles[vehicle_row, 0])
                    front_vehicle = tstates.surrounding_object_list[front_vehicle_idx]
                    front_vehicle.ffstate = get_frenet_state(front_vehicle.state,
                        tstates.static_map_virtual_lane_path_array[lane_id],
                        tstates.static_map_virtual_lane_tangets[lane_id]
                    )
                    # Here we use relative frenet coordinate
                    front_vehicle.ffstate.s = self._ego_vehicle_distance_to_lane_tail[lane_id] - front_vehicles[vehicle_row, 1]
                    front_vehicle.behavior = predict_vehicle_behavior(front_vehicle, tstates.static_map_virtual_lane_path_array, tstates.dynamic_map)
                    tstates.dynamic_map.mmap.lanes[lane_id].front_vehicles.append(front_vehicle)
                    break #jxy1217: only keep one, since only one is used in IDM
                
                front_vehicle = tstates.dynamic_map.mmap.lanes[lane_id].front_vehicles[0]
                rospy.logdebug("Lane index: %d, Front vehicle id: %d, behavior: %d, x:%.1f, y:%.1f, d:%.1f", 
                                lane_id, front_vehicle.uid, front_vehicle.behavior,
                                front_vehicle.state.pose.pose.position.x,front_vehicle.state.pose.pose.position.y,
                                front_vehicle.ffstate.s)

            if len(rear_vehicles) > 0:
                # Descending sort rear objects by distance to lane end
                for vehicle_row in reversed(rear_vehicles[:,1].argsort()):
                    rear_vehicle_idx = int(rear_vehicles[vehicle_row, 0])
                    rear_vehicle = tstates.surrounding_object_list[rear_vehicle_idx]
                    rear_vehicle.ffstate = get_frenet_state(rear_vehicle.state,
                        tstates.static_map_virtual_lane_path_array[lane_id],
                        tstates.static_map_virtual_lane_tangets[lane_id]
                    )
                    # Here we use relative frenet coordinate
                    rear_vehicle.ffstate.s = rear_vehicles[vehicle_row, 1] - self._ego_vehicle_distance_to_lane_head[lane_id] # negative value
                    rear_vehicle.behavior = predict_vehicle_behavior(rear_vehicle, tstates.static_map_virtual_lane_path_array, tstates.dynamic_map)
                    tstates.dynamic_map.mmap.lanes[lane_id].rear_vehicles.append(rear_vehicle)
                    break
                
                rear_vehicle = tstates.dynamic_map.mmap.lanes[lane_id].rear_vehicles[0]
                rospy.logdebug("Lane index: %d, Rear vehicle id: %d, behavior: %d, x:%.1f, y:%.1f, d:%.1f", 
                                lane_id, rear_vehicle.uid, rear_vehicle.behavior, 
                                rear_vehicle.state.pose.pose.position.x,rear_vehicle.state.pose.pose.position.y,
                                rear_vehicle.ffstate.s)

    def locate_ego_vehicle_in_lanes(self, tstates, lane_end_dist_thres=2, lane_dist_thres=5):
        dist_list_real = np.array([dist_from_point_to_polyline2d(
            tstates.ego_vehicle_state.state.pose.pose.position.x, tstates.ego_vehicle_state.state.pose.pose.position.y,
            lane, return_end_distance=True)
            for lane in tstates.static_map_lane_path_array])
        ego_dimension = DimensionWithCovariance()
        ego_dimension.length_x = 4.0
        ego_dimension.length_y = 2.0 #jxy: I don't know
        ego_dimension.length_z = 1.8

        ego_lane_index, _, _, _, ego_s = self.locate_object_in_lane(tstates.ego_vehicle_state.state, tstates, ego_dimension, virtual_flag=1)
        ego_lane_index_rounded = int(round(ego_lane_index))

        self._ego_vehicle_distance_to_lane_head = np.array([0, 0, 0, 0, 0])
        self._ego_vehicle_distance_to_lane_tail = np.array([0, 0, 0, 0, 0])
        if len(dist_list_real) != 0:
            self._ego_vehicle_distance_to_lane_head = dist_list_real[:, 3]
            self._ego_vehicle_distance_to_lane_tail = dist_list_real[:, 4]

        if len(tstates.static_map.lanes) < len(tstates.static_map.virtual_lanes):
            for i in range(len(tstates.static_map.virtual_lanes) - len(tstates.static_map.lanes)):
                self._ego_vehicle_distance_to_lane_head = np.append(self._ego_vehicle_distance_to_lane_head, self._ego_vehicle_distance_to_lane_head[-1])
                self._ego_vehicle_distance_to_lane_tail = np.append(self._ego_vehicle_distance_to_lane_tail, self._ego_vehicle_distance_to_lane_tail[-1])

        #from obstacle locator
        tstates.ego_s = ego_s
        if ego_lane_index < 0 or self._ego_vehicle_distance_to_lane_tail[ego_lane_index_rounded] <= lane_end_dist_thres:
            # Drive into junction, wait until next map
            rospy.logdebug("Cognition: Ego vehicle close to intersection, ego_lane_index = %f, dist_to_lane_tail = %f", ego_lane_index, self._ego_vehicle_distance_to_lane_tail[int(ego_lane_index)])
            tstates.dynamic_map.model = MapState.MODEL_JUNCTION_MAP
            tstates.dynamic_map.ego_ffstate = get_frenet_state(tstates.ego_vehicle_state, 
                tstates.static_map_virtual_lane_path_array[ego_lane_index_rounded],
                tstates.static_map_virtual_lane_tangets[ego_lane_index_rounded])
            tstates.dynamic_map.mmap.ego_lane_index = ego_lane_index
            # TODO: Calculate frenet coordinate here or in put_buffer?
            return
        else:
            tstates.dynamic_map.model = MapState.MODEL_MULTILANE_MAP
            tstates.dynamic_map.ego_ffstate = get_frenet_state(tstates.ego_vehicle_state, 
                tstates.static_map_virtual_lane_path_array[ego_lane_index_rounded],
                tstates.static_map_virtual_lane_tangets[ego_lane_index_rounded])
            tstates.dynamic_map.mmap.ego_lane_index = ego_lane_index
            tstates.dynamic_map.mmap.distance_to_junction = self._ego_vehicle_distance_to_lane_tail[ego_lane_index_rounded]
        rospy.logdebug("Distance to end: (lane %f) %f", ego_lane_index, self._ego_vehicle_distance_to_lane_tail[ego_lane_index_rounded])

    def locate_traffic_light_in_lanes(self, tstates):
        # TODO: Currently it's a very simple rule to locate the traffic lights
        if tstates.traffic_light_detection is None:
            return
        lights = tstates.traffic_light_detection.detections
        #jxy: demanding that the lights are in the same order as the lanes.

        total_lane_num = len(tstates.static_map.virtual_lanes)
        if len(lights) == 1:
            for i in range(total_lane_num):
                if lights[0].signal == ObjectSignals.TRAFFIC_LIGHT_RED:
                    tstates.static_map.virtual_lanes[i].map_lane.stop_state = Lane.STOP_STATE_STOP
                elif lights[0].signal == ObjectSignals.TRAFFIC_LIGHT_YELLOW:
                    tstates.static_map.virtual_lanes[i].map_lane.stop_state = Lane.STOP_STATE_YIELD
                elif lights[0].signal == ObjectSignals.TRAFFIC_LIGHT_GREEN:
                    tstates.static_map.virtual_lanes[i].map_lane.stop_state = Lane.STOP_STATE_THRU
        elif len(lights) > 1 and len(lights) == total_lane_num:
            for i in range(total_lane_num):
                if lights[i].signal == ObjectSignals.TRAFFIC_LIGHT_RED:
                    tstates.static_map.virtual_lanes[i].map_lane.stop_state = Lane.STOP_STATE_STOP
                elif lights[i].signal == ObjectSignals.TRAFFIC_LIGHT_YELLOW:
                    tstates.static_map.virtual_lanes[i].map_lane.stop_state = Lane.STOP_STATE_YIELD
                elif lights[i].signal == ObjectSignals.TRAFFIC_LIGHT_GREEN:
                    tstates.static_map.virtual_lanes[i].map_lane.stop_state = Lane.STOP_STATE_THRU
        elif len(lights) > 1 and len(lights) != total_lane_num:
            red = True
            for i in range(len(lights)):
                if lights[i].signal == ObjectSignals.TRAFFIC_LIGHT_GREEN:
                    red = False
            for i in range(total_lane_num):
                if red:
                    tstates.static_map.virtual_lanes[i].map_lane.stop_state = Lane.STOP_STATE_STOP
                else:
                    tstates.static_map.virtual_lanes[i].map_lane.stop_state = Lane.STOP_STATE_THRU
        
    def locate_stop_sign_in_lanes(self, tstates):
        '''
        Put stop sign detections into lanes
        '''
        # TODO: Implement this
        pass

    def locate_speed_limit_in_lanes(self, tstates, ref_stop_thres = 10):
        '''
        Put stop sign detections into lanes
        '''
        # TODO(zhcao): Change the speed limit according to the map or the traffic sign(perception)
        # Now we set the multilane speed limit as 40 km/h.
        total_lane_num = len(tstates.static_map.virtual_lanes)
        for i in range(total_lane_num):
            tstates.dynamic_map.mmap.lanes[i].map_lane.speed_limit = 30
    
    # ========= For drivable areas =========

    def calculate_drivable_areas(self, tstates, tt):
        '''
        A list of boundary points of drivable area
        '''

        # initialize

        tstates.drivable_area = [] #clear in every step
        ego_s = tstates.ego_s + 0

        ego_x = tstates.ego_vehicle_state.state.pose.pose.position.x
        ego_y = tstates.ego_vehicle_state.state.pose.pose.position.y

        key_node_list = []

        angle_list = [] #a boundary point is represented by 6 numbers, namely x, y, vx, vy, omega and flag
        dist_list = []
        vx_list = []
        vy_list = []
        id_list = []
        base_x_list = []
        base_y_list = []
        omega_list = []
        flag_list = []

        skip_list = [] #lane following vehicles, only when ego vehicle is in lanes
        non_skip_following_list = []

        # executing

        key_node_list = self.generate_key_nodes(tstates, tt, ego_s, ego_x, ego_y, skip_list, non_skip_following_list)

        if not self.generate_interp_points(tstates, tt, ego_s, ego_x, ego_y, key_node_list, dist_list, angle_list, vx_list, \
            vy_list, base_x_list, base_y_list, omega_list, flag_list, id_list):
            return
        
        self.consider_moving_obstacles(tstates, tt, ego_s, ego_x, ego_y, key_node_list, dist_list, angle_list, vx_list, \
            vy_list, base_x_list, base_y_list, omega_list, flag_list, id_list, skip_list, non_skip_following_list)

    def generate_key_nodes(self, tstates, tt, ego_s, ego_x, ego_y, skip_list, non_skip_following_list):
        
        key_node_list = []

        if tstates.static_map.in_junction:

            #TODO: avoid repeated calculation of static boundary

            # jxy0710: try to merge the next static boundary into the junction boundary to make one closed boundary, then add dynamic objects
            #create a list of lane section, each section is defined as (start point s, end point s)
            #calculate from the right most lane to the left most lane, drawing drivable area boundary in counterclockwise
            lane_num = len(tstates.static_map.next_lanes)

            ego_s = 0

            if len(tstates.static_map.next_lanes) > 0:
                next_lane_start_px = tstates.static_map.next_lanes[0].right_boundaries[0].boundary_point.position.x
                next_lane_start_py = tstates.static_map.next_lanes[0].right_boundaries[0].boundary_point.position.y

                ego_s = -(math.sqrt(math.pow((next_lane_start_px - ego_x), 2) + math.pow((next_lane_start_py - ego_y), 2))) #for next unit (road section)

            lane_sections = np.zeros((lane_num, 6))
            for i in range(len(tstates.static_map.next_lanes)):
                lane_sections[i, 0] = max(ego_s - 10, 0)
                lane_sections[i, 1] = min(ego_s + 50, tstates.static_map.next_lanes[i].central_path_points[-1].s)
                lane_sections[i, 2] = 0 #vx in front
                lane_sections[i, 3] = 0 #vy in front
                lane_sections[i, 4] = 0 #vx behind
                lane_sections[i, 5] = 0 #vy behind
                #TODO: projection to the vertical direction

            next_static_area = []
            
            for i in range(len(tstates.static_map.next_lanes)):
                lane = tstates.static_map.next_lanes[i]
                if i == 0:
                    self.next_lane_section_points_generation_united(lane_sections[i, 0], lane_sections[i, 1], lane_sections[i, 2], \
                        lane_sections[i, 3], lane_sections[i, 4], lane_sections[i, 5],lane.right_boundaries, next_static_area)
                
                if i != 0:
                    self.next_lane_section_points_generation_united(lane_sections[i-1, 1], lane_sections[i, 1], lane_sections[i-1, 4], \
                        lane_sections[i-1, 5], lane_sections[i, 4], lane_sections[i, 5], lane.right_boundaries, next_static_area)
                    if i != len(tstates.static_map.next_lanes) - 1:
                        self.next_lane_section_points_generation_united(lane_sections[i, 1], lane_sections[i+1, 1], lane_sections[i, 4], \
                        lane_sections[i, 5], lane_sections[i+1, 4], lane_sections[i+1, 5], lane.left_boundaries, next_static_area)
                    else:
                        self.next_lane_section_points_generation_united(lane_sections[i, 1], lane_sections[i, 0], lane_sections[i, 4], \
                        lane_sections[i, 5], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, next_static_area)

                if len(tstates.static_map.next_lanes) == 1:
                    self.next_lane_section_points_generation_united(lane_sections[i, 1], lane_sections[i, 0], lane_sections[i, 4], \
                        lane_sections[i, 5], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, next_static_area)

            # The lower part are removed.

            # joint point: the start point of the right most lane boundary of the next lanes. It is also a point in current drivable area.
            # It is the first point of the next static area.

            if len(next_static_area) > 0:
                joint_point = next_static_area[0]
                joint_point_x = joint_point[0]
                joint_point_y = joint_point[1]

                dist_array = []
                if len(tstates.static_map.drivable_area.points) >= 3:
                    for i in range(len(tstates.static_map.drivable_area.points)):
                        node_point = tstates.static_map.drivable_area.points[i]
                        node_point_x = node_point.x
                        node_point_y = node_point.y
                        dist_to_joint_point = math.sqrt(pow((node_point_x - joint_point_x), 2) + pow((node_point_y - joint_point_y), 2))
                        dist_array.append(dist_to_joint_point)

                joint_point2_index = dist_array.index(min(dist_array)) # the index of the point in drivable area that equals to the joint point
            
                key_node_list = []
                for i in range(len(tstates.static_map.drivable_area.points)):
                    j = len(tstates.static_map.drivable_area.points) - 1 - i
                    node_point = tstates.static_map.drivable_area.points[j]
                    key_node_list.append([node_point.x, node_point.y])
                    if j == joint_point2_index:
                        for k in range(len(next_static_area)):
                            if k != 0: # the joint point needs not be added again
                                key_node_list.append(next_static_area[k])

                key_node_list.reverse()

            else:
                for i in range(len(tstates.static_map.drivable_area.points)):
                    node_point = tstates.static_map.drivable_area.points[i]
                    #jxy202012: clockwise, so the counterclockwise next lanes can only be inserted in a reversed static area line
                    key_node_list.append([node_point.x, node_point.y])

        else:
            #create a list of lane section, each section is defined as (start point s, end point s)
            #calculate from the right most lane to the left most lane, drawing drivable area boundary in counterclockwise
            lane_num = len(tstates.static_map.lanes)

            lane_sections = np.zeros((lane_num, 6))
            for i in range(len(tstates.static_map.lanes)):
                lane_sections[i, 0] = max(ego_s - 50, 0)
                lane_sections[i, 1] = min(ego_s + 50, tstates.static_map.lanes[i].central_path_points[-1].s)
                lane_sections[i, 2] = 0 #vx in front
                lane_sections[i, 3] = 0 #vy in front
                lane_sections[i, 4] = 0 #vx behind
                lane_sections[i, 5] = 0 #vy behind

            for i in range(len(tstates.surrounding_object_list_timelist[tt])):
                obstacle = tstates.surrounding_object_list_timelist[tt][i]
                if obstacle.lane_index == -1:
                    continue
                else:
                    #the obstacle in on the same road as the ego vehicle
                    lane_index_rounded = int(round(obstacle.lane_index))
                    #TODO: consider those on the lane boundary
                    if obstacle.lane_dist_left_t == 0 or obstacle.lane_dist_right_t == 0: #not lane following
                        continue
                    if obstacle.lane_dist_s > ego_s and obstacle.lane_dist_s < lane_sections[lane_index_rounded, 1]:
                        if len(tstates.static_map.next_drivable_area.points) >= 3:
                            non_skip_following_list.append(i)
                            continue
                        lane_sections[lane_index_rounded, 1] = obstacle.lane_dist_s - obstacle.dimension.length_x / 2.0
                        lane_sections[lane_index_rounded, 4] = obstacle.state.twist.twist.linear.x
                        lane_sections[lane_index_rounded, 5] = obstacle.state.twist.twist.linear.y
                        skip_list.append(i)
                    elif obstacle.lane_dist_s <= ego_s and obstacle.lane_dist_s > lane_sections[lane_index_rounded, 0]:
                        lane_sections[lane_index_rounded, 0] = obstacle.lane_dist_s + obstacle.dimension.length_x / 2.0
                        lane_sections[lane_index_rounded, 2] = obstacle.state.twist.twist.linear.x
                        lane_sections[lane_index_rounded, 3] = obstacle.state.twist.twist.linear.y
                        skip_list.append(i)

            #next junction: paste with next junction, at joint point
            joint_point = tstates.static_map.lanes[0].right_boundaries[-1]
            joint_point_x = joint_point.boundary_point.position.x
            joint_point_y = joint_point.boundary_point.position.y

            dist_array = []
            next_key_node_list = [] #jxy: empty if still not loaded

            joint_point2_index = 0
                
            if len(tstates.static_map.next_drivable_area.points) >= 3:
                for i in range(len(tstates.static_map.next_drivable_area.points)):
                    node_point = tstates.static_map.next_drivable_area.points[i]
                    node_point_x = node_point.x
                    node_point_y = node_point.y
                    dist_to_joint_point = math.sqrt(pow((node_point_x - joint_point_x), 2) + pow((node_point_y - joint_point_y), 2))
                    dist_array.append(dist_to_joint_point)

                joint_point2_index = dist_array.index(min(dist_array)) # the index of the point in drivable area that equals to the joint point

                for i in range(len(tstates.static_map.next_drivable_area.points)):

                    node_point = tstates.static_map.next_drivable_area.points[i]
                    next_key_node_list.append([node_point.x, node_point.y, 0, 0, 0, 0, 0, 1])

                next_key_node_list.reverse()
                joint_point2_index = len(dist_array) - 1 - joint_point2_index

            if len(tstates.static_map.next_drivable_area.points) >= 3:
                lane = tstates.static_map.lanes[0]
                self.lane_section_points_generation(lane_sections[0, 0], lane_sections[0, 1], lane_sections[0, 2], \
                    lane_sections[0, 3], lane_sections[0, 4], lane_sections[0, 5],lane.right_boundaries, key_node_list, 1)
                for i in range(1, len(next_key_node_list)): #jxy202012: the first is the joint point, thus neglect
                    ii = i + joint_point2_index
                    if ii >= len(next_key_node_list):
                        ii = ii - len(next_key_node_list)
                    key_node_list.append(next_key_node_list[ii])
                lane = tstates.static_map.lanes[-1]
                self.lane_section_points_generation(lane_sections[lane_num-1, 1], lane_sections[lane_num-1, 0], lane_sections[lane_num-1, 4], \
                    lane_sections[lane_num-1, 5], lane_sections[lane_num-1, 2], lane_sections[lane_num-1, 3],lane.left_boundaries, key_node_list, 1)
                for j in range(len(tstates.static_map.lanes)):
                    i = len(tstates.static_map.lanes) - 1 - j
                    lane = tstates.static_map.lanes[i]                
                    if i != len(tstates.static_map.lanes) - 1:
                        self.lane_section_points_generation(lane_sections[i+1, 0], lane_sections[i, 0], lane_sections[i+1, 2], \
                            lane_sections[i+1, 3], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, key_node_list, 0)
                        if i != 0:
                            self.lane_section_points_generation(lane_sections[i, 0], lane_sections[i-1, 0], lane_sections[i, 2], \
                                lane_sections[i, 3], lane_sections[i-1, 2], lane_sections[i-1, 3], lane.right_boundaries, key_node_list, 0)
                

            else:
                for i in range(len(tstates.static_map.lanes)):
                    lane = tstates.static_map.lanes[i]
                    if i == 0:
                        self.lane_section_points_generation(lane_sections[i, 0], lane_sections[i, 1], lane_sections[i, 2], \
                            lane_sections[i, 3], lane_sections[i, 4], lane_sections[i, 5],lane.right_boundaries, key_node_list, 1)
                    
                    if i != 0:
                        self.lane_section_points_generation(lane_sections[i-1, 1], lane_sections[i, 1], lane_sections[i-1, 4], \
                            lane_sections[i-1, 5], lane_sections[i, 4], lane_sections[i, 5], lane.right_boundaries, key_node_list, 0)
                        if i != len(tstates.static_map.lanes) - 1:
                            self.lane_section_points_generation(lane_sections[i, 1], lane_sections[i+1, 1], lane_sections[i, 4], \
                            lane_sections[i, 5], lane_sections[i+1, 4], lane_sections[i+1, 5], lane.left_boundaries, key_node_list, 0)
                        else:
                            self.lane_section_points_generation(lane_sections[i, 1], lane_sections[i, 0], lane_sections[i, 4], \
                            lane_sections[i, 5], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, key_node_list, 1)

                    if len(tstates.static_map.lanes) == 1:
                        self.lane_section_points_generation(lane_sections[i, 1], lane_sections[i, 0], lane_sections[i, 4], \
                            lane_sections[i, 5], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, key_node_list, 1)

                for j in range(len(tstates.static_map.lanes)):
                    i = len(tstates.static_map.lanes) - 1 - j
                    lane = tstates.static_map.lanes[i]                
                    if i != len(tstates.static_map.lanes) - 1:
                        self.lane_section_points_generation(lane_sections[i+1, 0], lane_sections[i, 0], lane_sections[i+1, 2], \
                            lane_sections[i+1, 3], lane_sections[i, 2], lane_sections[i, 3], lane.left_boundaries, key_node_list, 0)
                        if i != 0:
                            self.lane_section_points_generation(lane_sections[i, 0], lane_sections[i-1, 0], lane_sections[i, 2], \
                            lane_sections[i, 3], lane_sections[i-1, 2], lane_sections[i-1, 3], lane.right_boundaries, key_node_list, 0)


            key_node_list.reverse()

        return key_node_list

    def generate_interp_points(self, tstates, tt, ego_s, ego_x, ego_y, key_node_list, dist_list, angle_list, vx_list, \
            vy_list, base_x_list, base_y_list, omega_list, flag_list, id_list):
        '''
        interp in key nodes by 360 degrees, deal with reverse angles
        jxy20201228: concave figure causes instability in collision check.
        This may also be helpful in fusion with LiDAR boundary detection.
        '''

        if len(key_node_list) >= 3:
            key_node_list_array = np.array(key_node_list)

            key_node_list_xy_array = key_node_list_array[:,:2]
            key_node_list_xy_array = np.vstack((key_node_list_xy_array, key_node_list_xy_array[0])) #close the figure

            dist_to_static_boundary, _, _, = dist_from_point_to_closedpolyline2d(ego_x, ego_y, key_node_list_xy_array)

            # rospy.loginfo("ego position: x: %f, y: %f", ego_x, ego_y)
            # rospy.loginfo("dist_to_static_boundary: %f", dist_to_static_boundary)

            if dist_to_static_boundary < 0 or (len(tstates.static_map.drivable_area.points) <= 3 and tstates.static_map.in_junction):
                for i in range(len(key_node_list)):
                    node_point = key_node_list[i]
                    last_node_point = key_node_list[i-1]
                    flag, vx, vy, flag_n, vx_n, vy_n = 1, 0, 0, 1, 0, 0
                    if not tstates.static_map.in_junction:
                        if node_point[7] == 3: #jxy: car following vehicle, dynamic 3
                            flag = 3
                            vx = node_point[2]
                            vy = node_point[3]
                            flag_n = 3
                            vx_n = node_point[2]
                            vy_n = node_point[3]
                    #point = [node_point.x, node_point.y]
                    #shatter the figure
                    vertex_dist = math.sqrt(pow((node_point[0] - last_node_point[0]), 2) + pow((node_point[1] - last_node_point[1]), 2))
                    if vertex_dist > 0.2:
                        #add interp points by step of 0.2m
                        for j in range(int(vertex_dist / 0.2)):
                            x = last_node_point[0] + 0.2 * (j + 1) / vertex_dist * (node_point[0] - last_node_point[0])
                            y = last_node_point[1] + 0.2 * (j + 1) / vertex_dist * (node_point[1] - last_node_point[1])
                            angle_list.append(math.atan2(y - ego_y, x - ego_x))
                            dist_list.append(math.sqrt(pow((x - ego_x), 2) + pow((y - ego_y), 2)))
                            #the velocity of static boundary is 0
                            vx_list.append(vx)
                            vy_list.append(vy)
                            base_x_list.append(0)
                            base_y_list.append(0)
                            omega_list.append(0)
                            flag_list.append(flag) #static boundary #jxy20201203: warning! FLAG of lane following vehicles should be updated!
                            id_list.append(-1) #static boundary, interp points (can be deleted)
                    
                    angle_list.append(math.atan2(node_point[1] - ego_y, node_point[0] - ego_x))
                    dist_list.append(math.sqrt(pow((node_point[0] - ego_x), 2) + pow((node_point[1] - ego_y), 2)))
                    vx_list.append(vx_n)
                    vy_list.append(vy_n)
                    base_x_list.append(0)
                    base_y_list.append(0)
                    omega_list.append(0)
                    flag_list.append(flag_n) #static boundary
                    id_list.append(-2) #static boundary, nodes (cannot be deleted)
                
            else:
                for i in range(360):
                    angle_list.append(i * math.pi / 180)
                    dist_list.append(100)
                    vx_list.append(0)
                    vy_list.append(0)
                    base_x_list.append(0)
                    base_y_list.append(0)
                    omega_list.append(0)
                    flag_list.append(0)
                    id_list.append(-1)
                
                theta_temp_list = [] #TODO: add vx and vy list
                dist_temp_list = []
                x_temp_list = []
                y_temp_list = []

                for i in range(len(key_node_list)):
                    point_temp = key_node_list[i]
                    #installation calibration, TODO: parameterize
                    x_rel = point_temp[0] - ego_x
                    y_rel = point_temp[1] - ego_y
                    theta_temp = math.atan2(y_rel, x_rel) / math.pi * 180
                    if theta_temp < 0:
                        theta_temp = theta_temp + 360
                    dist_temp = np.linalg.norm([x_rel, y_rel])
                    theta_temp_list.append(theta_temp)
                    dist_temp_list.append(dist_temp)
                    x_temp_list.append(x_rel)
                    y_temp_list.append(y_rel)

                theta_temp_list.reverse() # should be counterclockwise... TODO: simplify it
                dist_temp_list.reverse()
                x_temp_list.reverse()
                y_temp_list.reverse()
                # key_node_list.reverse() # jxy: keep it clockwise for closedpolyline2d

                reverse_start_flag = 0
                for i in range(len(key_node_list)):
                    temp_id = len(key_node_list)-i
                    if i == 0:
                        temp_id = 0
                    node_point = key_node_list[temp_id] #jxy: now counterclockwise, flag and motion are determined by the point of smaller index.
                    omega, flag, vx, vy = 0, 1, 0, 0
                    if not tstates.static_map.in_junction:
                        omega = node_point[6]
                        if node_point[7] == 3: #jxy: car following vehicle, dynamic 3
                            flag = 3
                            vx = node_point[2]
                            vy = node_point[3]

                    last_theta = theta_temp_list[i-1]
                    current_theta = theta_temp_list[i]
                    last_dist = dist_temp_list[i-1]
                    current_dist = dist_temp_list[i]
                    last_x = x_temp_list[i-1]
                    current_x = x_temp_list[i]
                    last_y = y_temp_list[i-1]
                    current_y = y_temp_list[i]
            
                    dist0, tt1, tt2, = dist_from_point_to_line2d(0, 0, last_x, last_y, current_x, current_y)
                    dist0 = abs(dist0)

                    if tt1 > 0 and tt2 < 0:
                        dist0 = -dist0

                    if current_theta > last_theta: #check if normally counterclockwise
                        if current_theta - last_theta > 180:
                            # local clockwise, will cause bug!
                            if reverse_start_flag == 0:
                                j = int(math.ceil(last_theta))
                                id = -2 - 0.001 * i #static boundary, nodes (cannot be deleted)
                                if dist_list[j] > last_dist:
                                    angle_list[j] = last_theta * math.pi / 180
                                    dist_list[j] = last_dist
                                    reverse_start_flag = 1
                                    vx_list[j] = vx
                                    vy_list[j] = vy
                                    base_x_list[j] = 0
                                    base_y_list[j] = 0
                                    omega_list[j] = omega
                                    flag_list[j] = flag
                                    id_list[j] = id
                                    reverse_start_flag = 1
                            continue

                        if math.floor(current_theta) >= math.ceil(last_theta):
                            for j in range(int(math.ceil(last_theta)), int(math.floor(current_theta)) + 1):
                                try:
                                    dist_temp = dist0 / math.cos(math.acos(dist0 / current_dist) - (current_theta - j) * math.pi / 180)
                                except:
                                    dist_temp = dist0 # caused by python calculation error when acos > 1
                                #TODO: robustness
                                if dist_list[j] > abs(dist_temp): #update!
                                    dist_list[j] = abs(dist_temp)
                                    id = -1 - 0.001 * i #static boundary, interp points (can be deleted)
                                    if abs(last_theta - j) < 1:
                                        id = -2 - 0.001 * i #static boundary, nodes (cannot be deleted)
                                        angle_list[j] = last_theta * math.pi / 180
                                        dist_list[j] = last_dist
                                    vx_list[j] = vx
                                    vy_list[j] = vy
                                    base_x_list[j] = 0
                                    base_y_list[j] = 0
                                    omega_list[j] = omega
                                    flag_list[j] = flag
                                    id_list[j] = id

                    elif current_theta - last_theta < -180: #cross pi, e.g. last theta 300, current theta 60
                        for j in range(int(math.ceil(last_theta)), 360):
                            try:
                                dist_temp = dist0 / math.cos(math.acos(dist0 / current_dist) - (current_theta - j + 360) * math.pi / 180)
                            except:
                                dist_temp = dist0 # caused by python calculation error when acos > 1
                            if dist_list[j] > abs(dist_temp):
                                dist_list[j] = abs(dist_temp)
                                id = -1 - 0.001 * i #static boundary, interp points (can be deleted)
                                if abs(last_theta - j) < 1:
                                    id = -2 - 0.001 * i #static boundary, nodes (cannot be deleted)
                                    angle_list[j] = last_theta * math.pi / 180
                                    dist_list[j] = last_dist
                                vx_list[j] = vx
                                vy_list[j] = vy
                                base_x_list[j] = 0
                                base_y_list[j] = 0
                                omega_list[j] = omega
                                flag_list[j] = flag
                                id_list[j] = id
                        for j in range(0, int(math.floor(current_theta)) + 1):
                            try:
                                dist_temp = dist0 / math.cos(math.acos(dist0 / current_dist) - (current_theta - j) * math.pi / 180)
                            except:
                                dist_temp = dist0 # caused by python calculation error when acos > 1
                            if dist_list[j] > abs(dist_temp):
                                dist_list[j] = abs(dist_temp)
                                id = -1 - 0.001 * i #static boundary, interp points (can be deleted)
                                if abs(last_theta - j) < 1:
                                    id = -2 - 0.001 * i #static boundary, nodes (cannot be deleted)
                                    angle_list[j] = last_theta * math.pi / 180
                                    dist_list[j] = last_dist
                                vx_list[j] = vx
                                vy_list[j] = vy
                                base_x_list[j] = 0
                                base_y_list[j] = 0
                                omega_list[j] = omega
                                flag_list[j] = flag
                                id_list[j] = id
                    else:
                        #local clockwise, keep one key point at reverse start point
                        if reverse_start_flag == 0:
                            j = int(math.ceil(last_theta))
                            if j >= 360:
                                j = 0
                            id = -2 - 0.001 * i #static boundary, nodes (cannot be deleted)
                            if dist_list[j] > last_dist:
                                angle_list[j] = last_theta * math.pi / 180 #TODO: check dist
                                dist_list[j] = last_dist
                                reverse_start_flag = 1
                                vx_list[j] = vx
                                vy_list[j] = vy
                                base_x_list[j] = 0
                                base_y_list[j] = 0
                                omega_list[j] = omega
                                flag_list[j] = flag
                                id_list[j] = id
                                reverse_start_flag = 1

                angle_list.reverse() #jxy: clockwise
                dist_list.reverse()
                vx_list.reverse()
                vy_list.reverse()
                base_x_list.reverse()
                base_y_list.reverse()
                omega_list.reverse()
                flag_list.reverse()
                id_list.reverse()

            return 1

        else:
            tstates.drivable_area_timelist.append([])
            rospy.loginfo("Dynamic boundary construction fail!")
            return 0

    def consider_moving_obstacles(self, tstates, tt, ego_s, ego_x, ego_y, key_node_list, dist_list, angle_list, vx_list, \
            vy_list, base_x_list, base_y_list, omega_list, flag_list, id_list, skip_list, non_skip_following_list):
        '''
        jxy202011: if in lanes, first consider the lane-keeping vehicles.
        if next junction is loaded, those ahead are all considered as free vehicles. Only those behind are considered specially.
        if next junction is not loaded, all of them should be be considered specially.
        '''

        temp_drivable_area = []

        key_node_list_array = np.array(key_node_list)

        key_node_list_xy_array = key_node_list_array[:,:2]
        key_node_list_xy_array = np.vstack((key_node_list_xy_array, key_node_list_xy_array[0])) #close the figure

        for i in range(len(angle_list)):
            if angle_list[i] > math.pi:
                angle_list[i] = angle_list[i] - 2 * math.pi

        if len(tstates.surrounding_object_list_timelist[tt]) != 0:
            for i in range(len(tstates.surrounding_object_list_timelist[tt])):
                obs = tstates.surrounding_object_list_timelist[tt][i]
                dist_to_ego = math.sqrt(math.pow((obs.state.pose.pose.position.x - tstates.ego_vehicle_state.state.pose.pose.position.x),2) 
                    + math.pow((obs.state.pose.pose.position.y - tstates.ego_vehicle_state.state.pose.pose.position.y),2))

                if i in skip_list:
                    continue

                lane_following_flag = 0
                if i in non_skip_following_list:
                    lane_following_flag = 1
                
                if dist_to_ego < 35:
                    dist_to_static_boundary, _, _, = dist_from_point_to_closedpolyline2d(obs.state.pose.pose.position.x, \
                        obs.state.pose.pose.position.y, key_node_list_xy_array)
                    if dist_to_static_boundary <= 0:
                        continue

                    self.update_boundary_obstacle(obs, i, ego_x, ego_y, dist_list, angle_list, vx_list, vy_list, base_x_list, \
                        base_y_list, omega_list, flag_list, id_list, lane_following_flag, tt)
                    
        self.merge_points(key_node_list, dist_list, angle_list, vx_list, vy_list, base_x_list, base_y_list, omega_list, flag_list, id_list)
        
        for j in range(len(angle_list)):
            x = ego_x + dist_list[j] * math.cos(angle_list[j])
            y = ego_y + dist_list[j] * math.sin(angle_list[j])
            vx = vx_list[j]
            vy = vy_list[j]
            base_x = base_x_list[j]
            base_y = base_y_list[j]
            omega = omega_list[j]
            flag = flag_list[j]
            point = [x, y, vx, vy, base_x, base_y, omega, flag]
            temp_drivable_area.append(point)
        
        #close the figure
        if len(temp_drivable_area) > 0:
            temp_drivable_area.append(temp_drivable_area[0])

        tstates.drivable_area_timelist.append(temp_drivable_area)

    def merge_points(self, key_node_list, dist_list, angle_list, vx_list, vy_list, base_x_list, base_y_list, omega_list, flag_list, id_list):
        '''
        merge the points of the same object to compress the data
        '''
        
        length_ori = len(angle_list)
        for i in range(length_ori):
            j = length_ori - 1 - i
            next_id = j + 1
            if j == len(angle_list)-1:
                next_id = 0
            if j < 0:
                break
            
            if round(id_list[j]) == -1:
                # jxy202101: adjust delete policy
                if (id_list[j-1] == id_list[j] or round(id_list[j-1]) == -2) and (id_list[next_id] == id_list[j] or round(id_list[next_id]) == -2):
                    res = (-1 - id_list[j]) * 1000
                    last_res = (round(id_list[j-1]) - id_list[j-1]) * 1000
                    next_res = (round(id_list[next_id]) - id_list[next_id]) * 1000
                    if last_res != next_res and last_res == 0:
                        last_res = len(key_node_list)
                    if round(abs(last_res - res)) <= 1 and round(abs(next_res - res)) <= 1: # python calculation error
                        del angle_list[j]
                        del dist_list[j]
                        del vx_list[j]
                        del vy_list[j]
                        del base_x_list[j]
                        del base_y_list[j]
                        del omega_list[j]
                        del flag_list[j]
                        continue #TODO: work here!
                    else:
                        flag_list[j] = 1
                        if round(abs(last_res - res)) > 1:
                            omega_list[j] = -2 #jxy0522: show shadow line
                        else:
                            omega_list[next_id] = -2
                else:
                    res = (-1 - id_list[j]) * 1000
                    last_res = (round(id_list[j-1]) - id_list[j-1]) * 1000
                    next_res = (round(id_list[next_id]) - id_list[next_id]) * 1000
                    #meet the dynamic object, add shadow line
                    if not (round(id_list[next_id]) == -1 or round(id_list[next_id]) == -2):
                        omega_list[next_id] = -2
                    elif not (round(id_list[j]) == -1 or round(id_list[j]) == -2):
                        omega_list[j] = -2

            elif round(id_list[j]) == -2:
                last_jump = 0
                next_jump = 0

                res = (-2 - id_list[j]) * 1000
                last_res = (round(id_list[j-1]) - id_list[j-1]) * 1000
                next_res = (round(id_list[next_id]) - id_list[next_id]) * 1000
                if last_res != next_res and last_res == 0:
                    next_res = 0

                if not ((round(id_list[j-1]) == -1 and round(abs(last_res - res)) <= 1) \
                    or (round(id_list[j-1]) == -2 and round(abs(last_res - res)) <= 1)):
                    last_jump = 1
                if not ((round(id_list[next_id]) == -1 and round(abs(next_res - res)) <= 1) \
                    or (round(id_list[next_id]) == -2 and round(abs(next_res - res)) <= 1)):
                    next_jump = 1

                if last_jump:
                    omega_list[j] = -2
                if next_jump:
                    omega_list[next_id] = -2

                continue
            
            elif abs(id_list[j] - id_list[j-1]) < 0.01 and abs(id_list[j] - id_list[next_id]) < 0.01:
                #jxy0525: added 0.001 for corner points
                #dynamic interp point
                del angle_list[j]
                del dist_list[j]
                del vx_list[j]
                del vy_list[j]
                del base_x_list[j]
                del base_y_list[j]
                del omega_list[j]
                del flag_list[j]
                continue

            elif abs(id_list[j] - id_list[j-1]) < 0.01 and abs(id_list[j] - id_list[next_id]) < 0.5:
                #dynamic object, if 3 points, corner point
                # angle_list[next_id] = (angle_list[next_id] + angle_list[j]) / 2
                # if abs(angle_list[next_id] - angle_list[j]) > math.pi:
                #     angle_list[next_id] = angle_list[next_id] + math.pi
                # if angle_list[next_id] > math.pi:
                #     angle_list[next_id] = angle_list[next_id] - 2 * math.pi
                # dist_list[next_id] = (dist_list[next_id] + dist_list[j]) / 2
                if id_list[j] * 100 - round(id_list[j] * 100) > 0.05:
                    angle_list[next_id] = angle_list[j]
                    dist_list[next_id] = dist_list[j] #keep this calibrated point
                del angle_list[j]
                del dist_list[j]
                del vx_list[j]
                del vy_list[j]
                del base_x_list[j]
                del base_y_list[j]
                del omega_list[j]
                del flag_list[j]
                continue

            elif abs(id_list[j] - id_list[next_id]) < 0.5 and abs(id_list[j] - id_list[j-1]) > 0.5:
                #the first dynamic point, flag set to static, since the velocity of point i refers to the velocity of the edge between i and i-1
                flag_list[j] = 1
                vx_list[j] = 0
                vy_list[j] = 0
                omega_list[j] = -2

            elif abs(id_list[j] - id_list[j-1]) < 0.5 and abs(id_list[j] - id_list[next_id]) > 0.5:
                #the last dynamic point, next_id flag set to static
                flag_list[next_id] = 1
                vx_list[next_id] = 0
                vy_list[next_id] = 0
                omega_list[next_id] = -2

    def lane_section_points_generation(self, starts, ends, startvx, startvy, endvx, endvy, lane_boundaries, outpointlist, solid_flag):

        #set the velocity of the start point to 0, since the velocity of point i refers to the velocity of the edge between i and i-1 (reversed)
        startvx = 0
        startvy = 0
        if starts <= ends:
            smalls = starts
            bigs = ends
            smallvx = startvx
            smallvy = startvy
            bigvx = endvx
            bigvy = endvy
        else:
            smalls = ends
            bigs = starts
            smallvx = endvx
            smallvy = endvy
            bigvx = startvx
            bigvy = startvy

        laneomega = -1 #TODO: mark dashed lane by this, now only for drawing, should be updated to be considered in planner
        if solid_flag == 1: #dashed lane
            laneomega = 0
        
        pointlist = []
        for j in range(len(lane_boundaries)):
            if lane_boundaries[j].boundary_point.s <= smalls:
                if j == len(lane_boundaries) - 1:
                    break
                if lane_boundaries[j+1].boundary_point.s > smalls:
                    #if s < start point s, it cannot be the last point, so +1 is ok
                    point1 = lane_boundaries[j].boundary_point
                    point2 = lane_boundaries[j+1].boundary_point

                    #projection to the longitudinal direction
                    direction = math.atan2(point2.position.y - point1.position.y, point2.position.x - point1.position.x)

                    v_value = smallvx * math.cos(direction) + smallvy * math.sin(direction)
                    vx_s = v_value * math.cos(direction)
                    vy_s = v_value * math.sin(direction)
                    flag = 0
                    if v_value != 0:
                        flag = 3
                    else:
                        flag = 1

                    pointx = point1.position.x + (point2.position.x - point1.position.x) * (smalls - point1.s) / (point2.s - point1.s)
                    pointy = point1.position.y + (point2.position.y - point1.position.y) * (smalls - point1.s) / (point2.s - point1.s)
                    point = [pointx, pointy, vx_s, vy_s, 0, 0, laneomega, flag]
                    pointlist.append(point)
            elif lane_boundaries[j].boundary_point.s > smalls and lane_boundaries[j].boundary_point.s < bigs:
                point = [lane_boundaries[j].boundary_point.position.x, lane_boundaries[j].boundary_point.position.y, 0, 0, 0, 0, laneomega, 1]
                pointlist.append(point)
            elif lane_boundaries[j].boundary_point.s >= bigs:
                if j == 0:
                    break
                if lane_boundaries[j-1].boundary_point.s < bigs:
                    point1 = lane_boundaries[j-1].boundary_point
                    point2 = lane_boundaries[j].boundary_point

                    #projection to the longitudinal direction
                    direction = math.atan2(point2.position.y - point1.position.y, point2.position.x - point1.position.x)

                    v_value = bigvx * math.cos(direction) + bigvy * math.sin(direction)
                    vx_s = v_value * math.cos(direction)
                    vy_s = v_value * math.sin(direction)
                    flag = 0
                    if v_value != 0:
                        flag = 3
                    else:
                        flag = 1
                    #the angular velocity in lanes need not be considered, so omega = 0

                    pointx = point1.position.x + (point2.position.x - point1.position.x) * (bigs - point1.s) / (point2.s - point1.s)
                    pointy = point1.position.y + (point2.position.y - point1.position.y) * (bigs - point1.s) / (point2.s - point1.s)
                    point = [pointx, pointy, vx_s, vy_s, 0, 0, laneomega, flag]
                    pointlist.append(point)

        if starts <= ends:
            for i in range(len(pointlist)):
                point = pointlist[i]
                if i == len(pointlist) - 1:
                    point[6] = 0
                outpointlist.append(point)
        else:
            # in reverse order
            for i in range(len(pointlist)):
                j = len(pointlist) - 1 - i
                point = pointlist[j]
                if j == 0:
                    point[6] = 0
                outpointlist.append(point)

    def next_lane_section_points_generation_united(self, starts, ends, startvx, startvy, endvx, endvy, lane_boundaries, next_static_area):

        #set the velocity of the start point to 0, since the velocity of point i refers to the velocity of the edge between i and i+1
        startvx = 0
        startvy = 0
        if starts <= ends:
            smalls = starts
            bigs = ends
        else:
            smalls = ends
            bigs = starts
        
        pointlist = []
        for j in range(len(lane_boundaries)):
            if lane_boundaries[j].boundary_point.s <= smalls:
                if j == len(lane_boundaries) - 1:
                    break
                if lane_boundaries[j+1].boundary_point.s > smalls:
                    #if s < start point s, it cannot be the last point, so +1 is ok
                    point1 = lane_boundaries[j].boundary_point
                    point2 = lane_boundaries[j+1].boundary_point

                    pointx = point1.position.x + (point2.position.x - point1.position.x) * (smalls - point1.s) / (point2.s - point1.s)
                    pointy = point1.position.y + (point2.position.y - point1.position.y) * (smalls - point1.s) / (point2.s - point1.s)
                    point = [pointx, pointy]
                    pointlist.append(point)
            elif lane_boundaries[j].boundary_point.s > smalls and lane_boundaries[j].boundary_point.s < bigs:
                point = [lane_boundaries[j].boundary_point.position.x, lane_boundaries[j].boundary_point.position.y]
                pointlist.append(point)
            elif lane_boundaries[j].boundary_point.s >= bigs:
                if j == 0:
                    break
                if lane_boundaries[j-1].boundary_point.s < bigs:
                    point1 = lane_boundaries[j-1].boundary_point
                    point2 = lane_boundaries[j].boundary_point

                    pointx = point1.position.x + (point2.position.x - point1.position.x) * (bigs - point1.s) / (point2.s - point1.s)
                    pointy = point1.position.y + (point2.position.y - point1.position.y) * (bigs - point1.s) / (point2.s - point1.s)
                    point = [pointx, pointy] # only static
                    pointlist.append(point)

        if starts <= ends:
            for i in range(len(pointlist)):
                point = pointlist[i]
                next_static_area.append(point)
        else:
            # in reverse order
            for i in range(len(pointlist)):
                j = len(pointlist) - 1 - i
                next_static_area.append(pointlist[j])

    def update_boundary_obstacle(self, obs, i, ego_x, ego_y, dist_list, angle_list, vx_list, vy_list, base_x_list, base_y_list, omega_list, flag_list, id_list, lane_following_flag, tt):
        #TODO: find a more robust method
        obs_x = obs.state.pose.pose.position.x
        obs_y = obs.state.pose.pose.position.y
        
        #Calculate the vertex points of the obstacle
        x = obs.state.pose.pose.orientation.x
        y = obs.state.pose.pose.orientation.y
        z = obs.state.pose.pose.orientation.z
        w = obs.state.pose.pose.orientation.w

        rotation_mat = np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y], [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x], [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]])
        rotation_mat_inverse = np.linalg.inv(rotation_mat) #those are the correct way to deal with quaternion

        vector_x = np.array([obs.dimension.length_x, 0, 0])
        vector_y = np.array([0, obs.dimension.length_y, 0])
        dx = np.matmul(rotation_mat_inverse, vector_x)
        dy = np.matmul(rotation_mat_inverse, vector_y)

        corner_list_x = np.zeros(4)
        corner_list_y = np.zeros(4)
        corner_list_x[0] = obs_x + dx[0]/2.0 + dy[0]/2.0
        corner_list_y[0] = obs_y + dx[1]/2.0 + dy[1]/2.0
        corner_list_x[1] = obs_x - dx[0]/2.0 + dy[0]/2.0
        corner_list_y[1] = obs_y - dx[1]/2.0 + dy[1]/2.0
        corner_list_x[2] = obs_x - dx[0]/2.0 - dy[0]/2.0
        corner_list_y[2] = obs_y - dx[1]/2.0 - dy[1]/2.0
        corner_list_x[3] = obs_x + dx[0]/2.0 - dy[0]/2.0
        corner_list_y[3] = obs_y + dx[1]/2.0 - dy[1]/2.0

        corner_list_angle = np.zeros(4)
        corner_list_dist = np.zeros(4)
        for j in range(4):
            corner_list_angle[j] = math.atan2(corner_list_y[j] - ego_y, corner_list_x[j] - ego_x)
            corner_list_dist[j] = math.sqrt(pow((corner_list_x[j] - ego_x), 2) + pow((corner_list_y[j] - ego_y), 2))

        small_corner_id = np.argmin(corner_list_angle)
        big_corner_id = np.argmax(corner_list_angle)

        if corner_list_angle[big_corner_id] - corner_list_angle[small_corner_id] > math.pi:
            #cross pi
            for j in range(4):
                if corner_list_angle[j] < 0:
                    corner_list_angle[j] += 2 * math.pi

        small_corner_id = np.argmin(corner_list_angle)
        big_corner_id = np.argmax(corner_list_angle)

        # add middle corner if we can see 3 corners
        smallest_dist_id = np.argmin(corner_list_dist)
        middle_corner_id = -1
        if not (small_corner_id == smallest_dist_id or big_corner_id == smallest_dist_id):
            middle_corner_id = smallest_dist_id

        for j in range(len(angle_list)):

            if (angle_list[j] < corner_list_angle[big_corner_id] and angle_list[j] > corner_list_angle[small_corner_id]):

                corner1 = -1
                corner2 = -1
                id_extra_flag = 0 # to distinguish edges in a whole object
                if middle_corner_id == -1:
                    corner1 = big_corner_id
                    corner2 = small_corner_id
                else:
                    if angle_list[j] < corner_list_angle[middle_corner_id]:
                        corner1 = middle_corner_id
                        corner2 = small_corner_id
                        id_extra_flag = 0.1
                    else:
                        corner1 = big_corner_id
                        corner2 = middle_corner_id
                        id_extra_flag = 0.2
                
                cross_position_x = corner_list_x[corner2] + (corner_list_x[corner1] - corner_list_x[corner2]) * (angle_list[j] - corner_list_angle[corner2]) / (corner_list_angle[corner1] - corner_list_angle[corner2])
                cross_position_y = corner_list_y[corner2] + (corner_list_y[corner1] - corner_list_y[corner2]) * (angle_list[j] - corner_list_angle[corner2]) / (corner_list_angle[corner1] - corner_list_angle[corner2])
                obstacle_dist = math.sqrt(pow((cross_position_x - ego_x), 2) + pow((cross_position_y - ego_y), 2))
                
                #TODO: find a more accurate method
                if dist_list[j] > obstacle_dist:

                    # Adapt to carla 0.9.8
                    vel_obs = np.array([obs.state.twist.twist.linear.x, obs.state.twist.twist.linear.y, obs.state.twist.twist.linear.z])
                    #vel_world = np.matmul(rotation_mat, vel_obs)
                    vel_world = vel_obs
                    #check if it should be reversed
                    vx = vel_world[0]
                    vy = vel_world[1]
                    omega = obs.state.twist.twist.angular.z

                    dist_list[j] = obstacle_dist
                    angle_list[j] = math.atan2(cross_position_y - ego_y, cross_position_x - ego_x) #might slightly differ
                    # rospy.loginfo("updating at angle: %f", angle_list[j])
                    # rospy.loginfo("corner1 angle: %f", corner_list_angle[corner1])
                    # rospy.loginfo("corner2 angle: %f", corner_list_angle[corner2])

                    #jxy20201231: slightly adjust
                    if abs(angle_list[j] - corner_list_angle[corner1]) < 1.0 / 180 * math.pi:
                        angle_list[j] = corner_list_angle[corner1]
                        dist_list[j] = corner_list_dist[corner1]
                        id_extra_flag = id_extra_flag + 0.001
                    elif abs(angle_list[j] - corner_list_angle[corner2]) < 1.0 / 180 * math.pi:
                        angle_list[j] = corner_list_angle[corner2]
                        dist_list[j] = corner_list_dist[corner2]
                        id_extra_flag = id_extra_flag + 0.001
                    
                    vx_list[j] = vx
                    vy_list[j] = vy
                    base_x_list[j] = obs_x
                    base_y_list[j] = obs_y
                    #omega_list[j] = omega
                    flag_list[j] = 2 + lane_following_flag #dynamic boundary
                    id_list[j] = i + id_extra_flag #mark that this point is updated by the ith obstacle

            elif (angle_list[j] + 2 * math.pi) > corner_list_angle[small_corner_id] and (angle_list[j] + 2 * math.pi) < corner_list_angle[big_corner_id]:
                
                # cross pi
                angle_list_plus = angle_list[j] + 2 * math.pi
                corner1 = -1
                corner2 = -1
                id_extra_flag = 0
                if middle_corner_id == -1:
                    corner1 = big_corner_id
                    corner2 = small_corner_id
                else:
                    if angle_list_plus < corner_list_angle[middle_corner_id]:
                        corner1 = middle_corner_id
                        corner2 = small_corner_id
                        id_extra_flag = 0.1
                    else:
                        corner1 = big_corner_id
                        corner2 = middle_corner_id
                        id_extra_flag = 0.2

                cross_position_x = corner_list_x[corner2] + (corner_list_x[corner1] - corner_list_x[corner2]) * (angle_list_plus - corner_list_angle[corner2]) / (corner_list_angle[corner1] - corner_list_angle[corner2])
                cross_position_y = corner_list_y[corner2] + (corner_list_y[corner1] - corner_list_y[corner2]) * (angle_list_plus- corner_list_angle[corner2]) / (corner_list_angle[corner1] - corner_list_angle[corner2])
                obstacle_dist = math.sqrt(pow((cross_position_x - ego_x), 2) + pow((cross_position_y - ego_y), 2))
                #TODO: find a more accurate method
                
                if dist_list[j] > obstacle_dist:

                    # Adapt to carla 0.9.8
                    vel_obs = np.array([obs.state.twist.twist.linear.x, obs.state.twist.twist.linear.y, obs.state.twist.twist.linear.z])
                    #vel_world = np.matmul(rotation_mat, vel_obs)
                    vel_world = vel_obs
                    #check if it should be reversed
                    vx = vel_world[0]
                    vy = vel_world[1]
                    omega = obs.state.twist.twist.angular.z
                    
                    #jxy0510: it is proved to be not correct only to keep the vertical velocity.
                    dist_list[j] = obstacle_dist
                    angle_list[j] = math.atan2(cross_position_y - ego_y, cross_position_x - ego_x) #might slightly differ
                    # rospy.loginfo("updating at angle (cross pi): %f", angle_list[j])
                    # rospy.loginfo("corner1 angle: %f", corner_list_angle[corner1])
                    # rospy.loginfo("corner2 angle: %f", corner_list_angle[corner2])

                    if abs(angle_list[j] + 2 * math.pi - corner_list_angle[corner1]) < 1.0 / 180 * math.pi:
                        angle_list[j] = corner_list_angle[corner1]
                        dist_list[j] = corner_list_dist[corner1]
                        id_extra_flag = id_extra_flag + 0.001 #jxy0525: mark the corner modified angle for keeping it
                    elif abs(angle_list[j] + 2 * math.pi - corner_list_angle[corner2]) < 1.0 / 180 * math.pi:
                        angle_list[j] = corner_list_angle[corner2]
                        dist_list[j] = corner_list_dist[corner2]
                        id_extra_flag = id_extra_flag + 0.001
                    vx_list[j] = vx
                    vy_list[j] = vy
                    base_x_list[j] = obs_x
                    base_y_list[j] = obs_y
                    #omega_list[j] = omega
                    flag_list[j] = 2 + lane_following_flag
                    id_list[j] = i + id_extra_flag

    # ========= Visualization =========

    def visualization(self, tstates):

        #visualization
        #1. lanes #jxy20201219: virtual lanes
        self.lanes_markerarray = MarkerArray()

        count = 0
        if len(tstates.static_map.virtual_lanes) != 0:
            biggest_id = 0 #TODO: better way to find the smallest id
            
            for lane in tstates.static_map.virtual_lanes:
                if lane.index > biggest_id:
                    biggest_id = lane.index
                tempmarker = Marker() #jxy: must be put inside since it is python
                tempmarker.header.frame_id = "map"
                tempmarker.header.stamp = rospy.Time.now()
                tempmarker.ns = "zzz/cognition"
                tempmarker.id = count
                tempmarker.type = Marker.LINE_STRIP
                tempmarker.action = Marker.ADD
                tempmarker.scale.x = 0.12
                tempmarker.color.r = 0.0
                tempmarker.color.g = 0.7
                tempmarker.color.b = 0.7
                tempmarker.color.a = 0.7
                tempmarker.lifetime = rospy.Duration(0.5)

                for lanepoint in lane.central_path_points:
                    p = Point()
                    p.x = lanepoint.position.x
                    p.y = lanepoint.position.y
                    p.z = lanepoint.position.z
                    tempmarker.points.append(p)
                self.lanes_markerarray.markers.append(tempmarker)
                count = count + 1

        #2. lane boundary line
        self.lanes_boundary_markerarray = MarkerArray()

        count = 0
        if not (tstates.static_map.in_junction):
            #does not draw lane when ego vehicle is in the junction
            
            for lane in tstates.static_map.lanes:
                if len(lane.right_boundaries) > 0 and len(lane.left_boundaries) > 0:
                    tempmarker = Marker() #jxy: must be put inside since it is python
                    tempmarker.header.frame_id = "map"
                    tempmarker.header.stamp = rospy.Time.now()
                    tempmarker.ns = "zzz/cognition"
                    tempmarker.id = count

                    #each lane has the right boundary, only the lane with the smallest id has the left boundary
                    tempmarker.type = Marker.LINE_STRIP
                    tempmarker.action = Marker.ADD
                    tempmarker.scale.x = 0.15
                    if lane.right_boundaries[0].boundary_type == 1: #broken lane is set gray
                        tempmarker.color.r = 0.6
                        tempmarker.color.g = 0.6
                        tempmarker.color.b = 0.6
                        tempmarker.color.a = 0.5
                    else:
                        tempmarker.color.r = 1.0
                        tempmarker.color.g = 1.0
                        tempmarker.color.b = 1.0
                        tempmarker.color.a = 0.5
                    tempmarker.lifetime = rospy.Duration(0.5)

                    for lb in lane.right_boundaries:
                        p = Point()
                        p.x = lb.boundary_point.position.x
                        p.y = lb.boundary_point.position.y
                        p.z = lb.boundary_point.position.z
                        tempmarker.points.append(p)
                    self.lanes_boundary_markerarray.markers.append(tempmarker)
                    count = count + 1

                    #biggest id: draw left lane
                    if lane.index == biggest_id:
                        tempmarker = Marker() #jxy: must be put inside since it is python
                        tempmarker.header.frame_id = "map"
                        tempmarker.header.stamp = rospy.Time.now()
                        tempmarker.ns = "zzz/cognition"
                        tempmarker.id = count

                        #each lane has the right boundary, only the lane with the biggest id has the left boundary
                        tempmarker.type = Marker.LINE_STRIP
                        tempmarker.action = Marker.ADD
                        tempmarker.scale.x = 0.3
                        if lane.left_boundaries[0].boundary_type == 1: #broken lane is set gray
                            tempmarker.color.r = 0.6
                            tempmarker.color.g = 0.6
                            tempmarker.color.b = 0.6
                            tempmarker.color.a = 0.5
                        else:
                            tempmarker.color.r = 1.0
                            tempmarker.color.g = 1.0
                            tempmarker.color.b = 1.0
                            tempmarker.color.a = 0.5
                        tempmarker.lifetime = rospy.Duration(0.5)

                        for lb in lane.left_boundaries:
                            p = Point()
                            p.x = lb.boundary_point.position.x
                            p.y = lb.boundary_point.position.y
                            p.z = lb.boundary_point.position.z
                            tempmarker.points.append(p)
                        self.lanes_boundary_markerarray.markers.append(tempmarker)
                        count = count + 1

        #3. obstacle
        self.obstacles_markerarray = MarkerArray()
        
        count = 0
        if tstates.surrounding_object_list_timelist[0] is not None:
            for obs in tstates.surrounding_object_list_timelist[0]:
                dist_to_ego = math.sqrt(math.pow((obs.state.pose.pose.position.x - tstates.ego_vehicle_state.state.pose.pose.position.x),2) 
                    + math.pow((obs.state.pose.pose.position.y - tstates.ego_vehicle_state.state.pose.pose.position.y),2))
                
                if dist_to_ego < 50:
                    tempmarker = Marker() #jxy: must be put inside since it is python
                    tempmarker.header.frame_id = "map"
                    tempmarker.header.stamp = rospy.Time.now()
                    tempmarker.ns = "zzz/cognition"
                    tempmarker.id = count
                    tempmarker.type = Marker.CUBE
                    tempmarker.action = Marker.ADD
                    tempmarker.pose = obs.state.pose.pose
                    tempmarker.scale.x = obs.dimension.length_x
                    tempmarker.scale.y = obs.dimension.length_y
                    tempmarker.scale.z = obs.dimension.length_z
                    if obs.lane_index == -1:
                        tempmarker.color.r = 0.5
                        tempmarker.color.g = 0.5
                        tempmarker.color.b = 0.5
                    elif obs.lane_dist_left_t == 0 or obs.lane_dist_right_t == 0:
                        # those who is on the lane boundary, warn by yellow
                        tempmarker.color.r = 1.0
                        tempmarker.color.g = 1.0
                        tempmarker.color.b = 0.0
                    else:
                        tempmarker.color.r = 1.0
                        tempmarker.color.g = 0.0
                        tempmarker.color.b = 1.0
                    if tstates.static_map.in_junction:
                        tempmarker.color.r = 1.0
                        tempmarker.color.g = 0.0
                        tempmarker.color.b = 1.0
                    tempmarker.color.a = 0.5
                    tempmarker.lifetime = rospy.Duration(0.5)

                    self.obstacles_markerarray.markers.append(tempmarker)
                    count = count + 1
            
            for obs in tstates.surrounding_object_list_timelist[0]:
                dist_to_ego = math.sqrt(math.pow((obs.state.pose.pose.position.x - tstates.ego_vehicle_state.state.pose.pose.position.x),2) 
                    + math.pow((obs.state.pose.pose.position.y - tstates.ego_vehicle_state.state.pose.pose.position.y),2))
                
                if dist_to_ego < 50:
                    tempmarker = Marker() #jxy: must be put inside since it is python
                    tempmarker.header.frame_id = "map"
                    tempmarker.header.stamp = rospy.Time.now()
                    tempmarker.ns = "zzz/cognition"
                    tempmarker.id = count
                    tempmarker.type = Marker.ARROW
                    tempmarker.action = Marker.ADD
                    tempmarker.scale.x = 0.4
                    tempmarker.scale.y = 0.7
                    tempmarker.scale.z = 0.75
                    tempmarker.color.r = 1.0
                    tempmarker.color.g = 1.0
                    tempmarker.color.b = 0.0
                    tempmarker.color.a = 0.5
                    tempmarker.lifetime = rospy.Duration(0.5)

                    #quaternion transform for obs velocity in carla 0.9.8

                    x = obs.state.pose.pose.orientation.x
                    y = obs.state.pose.pose.orientation.y
                    z = obs.state.pose.pose.orientation.z
                    w = obs.state.pose.pose.orientation.w

                    #rotation_mat = np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y], [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x], [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]])
                    #rotation_mat_inverse = np.linalg.inv(rotation_mat) #those are the correct way to deal with quaternion

                    vel_obs = np.array([obs.state.twist.twist.linear.x, obs.state.twist.twist.linear.y, obs.state.twist.twist.linear.z])
                    #vel_world = np.matmul(rotation_mat, vel_obs)
                    #vel_world = vel_obs
                    #check if it should be reversed
                    obs_vx_world = vel_obs[0]
                    obs_vy_world = vel_obs[1]
                    obs_vz_world = vel_obs[2]

                    startpoint = Point()
                    endpoint = Point()
                    startpoint.x = obs.state.pose.pose.position.x
                    startpoint.y = obs.state.pose.pose.position.y
                    startpoint.z = obs.state.pose.pose.position.z
                    endpoint.x = obs.state.pose.pose.position.x + obs_vx_world
                    endpoint.y = obs.state.pose.pose.position.y + obs_vy_world
                    endpoint.z = obs.state.pose.pose.position.z + obs_vz_world
                    tempmarker.points.append(startpoint)
                    tempmarker.points.append(endpoint)

                    self.obstacles_markerarray.markers.append(tempmarker)
                    count = count + 1

        #4. the labels of objects
        self.obstacles_label_markerarray = MarkerArray()

        count = 0
        if tstates.surrounding_object_list_timelist[0] is not None:                    
            for obs in tstates.surrounding_object_list_timelist[0]:
                dist_to_ego = math.sqrt(math.pow((obs.state.pose.pose.position.x - tstates.ego_vehicle_state.state.pose.pose.position.x),2) 
                    + math.pow((obs.state.pose.pose.position.y - tstates.ego_vehicle_state.state.pose.pose.position.y),2))
                
                if dist_to_ego < 50:
                    tempmarker = Marker() #jxy: must be put inside since it is python
                    tempmarker.header.frame_id = "map"
                    tempmarker.header.stamp = rospy.Time.now()
                    tempmarker.ns = "zzz/cognition"
                    tempmarker.id = count
                    tempmarker.type = Marker.TEXT_VIEW_FACING
                    tempmarker.action = Marker.ADD
                    hahaha = obs.state.pose.pose.position.z + 1.0
                    tempmarker.pose.position.x = obs.state.pose.pose.position.x
                    tempmarker.pose.position.y = obs.state.pose.pose.position.y
                    tempmarker.pose.position.z = hahaha
                    tempmarker.scale.z = 0.6
                    tempmarker.color.r = 1.0
                    tempmarker.color.g = 0.0
                    tempmarker.color.b = 1.0
                    tempmarker.color.a = 0.5
                    tempmarker.text = " lane_index: " + str(obs.lane_index) + "\n lane_dist_right_t: " + str(obs.lane_dist_right_t) + "\n lane_dist_left_t: " + str(obs.lane_dist_left_t) + "\n lane_anglediff: " + str(obs.lane_anglediff)
                    tempmarker.lifetime = rospy.Duration(0.5)

                    self.obstacles_label_markerarray.markers.append(tempmarker)
                    count = count + 1


        #5. ego vehicle visualization
        self.ego_markerarray = MarkerArray()

        tempmarker = Marker()
        tempmarker.header.frame_id = "map"
        tempmarker.header.stamp = rospy.Time.now()
        tempmarker.ns = "zzz/cognition"
        tempmarker.id = 1
        tempmarker.type = Marker.CUBE
        tempmarker.action = Marker.ADD
        tempmarker.pose = tstates.ego_vehicle_state.state.pose.pose
        tempmarker.scale.x = 4.0 #jxy: I don't know...
        tempmarker.scale.y = 2.0
        tempmarker.scale.z = 1.8
        tempmarker.color.r = 1.0
        tempmarker.color.g = 0.0
        tempmarker.color.b = 0.0
        tempmarker.color.a = 0.5
        tempmarker.lifetime = rospy.Duration(0.5)

        self.ego_markerarray.markers.append(tempmarker)

        #quaternion transform for ego velocity

        x = tstates.ego_vehicle_state.state.pose.pose.orientation.x
        y = tstates.ego_vehicle_state.state.pose.pose.orientation.y
        z = tstates.ego_vehicle_state.state.pose.pose.orientation.z
        w = tstates.ego_vehicle_state.state.pose.pose.orientation.w

        # rotation_mat = np.array([[1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y], [2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x], [2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y]])
        # rotation_mat_inverse = np.linalg.inv(rotation_mat) #those are the correct way to deal with quaternion

        vel_self = np.array([[tstates.ego_vehicle_state.state.twist.twist.linear.x], [tstates.ego_vehicle_state.state.twist.twist.linear.y], [tstates.ego_vehicle_state.state.twist.twist.linear.z]])
        # vel_world = np.matmul(rotation_mat_inverse, vel_self)
        # #check if it should be reversed
        ego_vx_world = vel_self[0]
        ego_vy_world = vel_self[1]
        ego_vz_world = vel_self[2]

        # ego_vx_world = self._ego_vehicle_state.state.twist.twist.linear.x
        # ego_vy_world = self._ego_vehicle_state.state.twist.twist.linear.y
        # ego_vz_world = self._ego_vehicle_state.state.twist.twist.linear.z

        tempmarker = Marker()
        tempmarker.header.frame_id = "map"
        tempmarker.header.stamp = rospy.Time.now()
        tempmarker.ns = "zzz/cognition"
        tempmarker.id = 2
        tempmarker.type = Marker.ARROW
        tempmarker.action = Marker.ADD
        tempmarker.scale.x = 0.4
        tempmarker.scale.y = 0.7
        tempmarker.scale.z = 0.75
        tempmarker.color.r = 1.0
        tempmarker.color.g = 1.0
        tempmarker.color.b = 0.0
        tempmarker.color.a = 0.5
        tempmarker.lifetime = rospy.Duration(0.5)

        startpoint = Point()
        endpoint = Point()
        startpoint.x = tstates.ego_vehicle_state.state.pose.pose.position.x
        startpoint.y = tstates.ego_vehicle_state.state.pose.pose.position.y
        startpoint.z = tstates.ego_vehicle_state.state.pose.pose.position.z
        endpoint.x = tstates.ego_vehicle_state.state.pose.pose.position.x + ego_vx_world
        endpoint.y = tstates.ego_vehicle_state.state.pose.pose.position.y + ego_vy_world
        endpoint.z = tstates.ego_vehicle_state.state.pose.pose.position.z + ego_vz_world
        tempmarker.points.append(startpoint)
        tempmarker.points.append(endpoint)

        self.ego_markerarray.markers.append(tempmarker)

        #6. drivable area
        self.drivable_area_markerarray = MarkerArray()

        count = 0
        if len(tstates.drivable_area_timelist[0]) != 0:

            for i in range(len(tstates.drivable_area_timelist[0])):
                
                #part 1: boundary section
                tempmarker = Marker() #jxy: must be put inside since it is python
                tempmarker.header.frame_id = "map"
                tempmarker.header.stamp = rospy.Time.now()
                tempmarker.ns = "zzz/cognition"
                tempmarker.id = count
                tempmarker.type = Marker.LINE_STRIP
                tempmarker.action = Marker.ADD
                tempmarker.scale.x = 0.20
                tempmarker.color.a = 0.5
                tempmarker.lifetime = rospy.Duration(0.5)

                point = tstates.drivable_area_timelist[0][i]
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = 0 #TODO: the map does not provide z value
                tempmarker.points.append(p)

                next_id = i + 1
                if next_id >= len(tstates.drivable_area_timelist[0]):
                    continue #closed line, the first point equal to the last point

                next_point = tstates.drivable_area_timelist[0][next_id]
                p = Point()
                p.x = next_point[0]
                p.y = next_point[1]
                p.z = 0 #TODO: the map does not provide z value
                tempmarker.points.append(p)

                tempmarker.color.r = 1.0
                tempmarker.color.g = 1.0
                tempmarker.color.b = 0.0

                if next_point[7] == 2:
                    tempmarker.color.r = 1.0
                    tempmarker.color.g = 0.0
                    tempmarker.color.b = 0.0
                elif next_point[7] == 3:
                    tempmarker.color.r = 0.0
                    tempmarker.color.g = 0.0
                    tempmarker.color.b = 1.0

                #jxy0522: add special color consideration
                if next_point[6] == -1:
                    #broken lane
                    tempmarker.color.r = 1.0
                    tempmarker.color.g = 1.0
                    tempmarker.color.b = 0.5
                elif next_point[6] == -2:
                    #shadow line
                    tempmarker.color.r = 0.5
                    tempmarker.color.g = 0.5
                    tempmarker.color.b = 0.5
                
                self.drivable_area_markerarray.markers.append(tempmarker)
                count = count + 1

                #part 2: boundary section motion status
                if next_point[7] == 2 or next_point[7] == 3: # and (abs(next_point[2]) + abs(next_point[3])) > 0.3:
                    tempmarker = Marker() #jxy: must be put inside since it is python
                    tempmarker.header.frame_id = "map"
                    tempmarker.header.stamp = rospy.Time.now()
                    tempmarker.ns = "zzz/cognition"
                    tempmarker.id = count
                    tempmarker.type = Marker.ARROW
                    tempmarker.action = Marker.ADD
                    tempmarker.scale.x = 0.4
                    tempmarker.scale.y = 0.7
                    tempmarker.scale.z = 0.75
                    tempmarker.color.r = 0.5
                    tempmarker.color.g = 0.5
                    tempmarker.color.b = 1.0
                    tempmarker.color.a = 0.8
                    tempmarker.lifetime = rospy.Duration(0.5)

                    startpoint = Point()
                    endpoint = Point()
                    startpoint.x = (point[0] + next_point[0]) / 2
                    startpoint.y = (point[1] + next_point[1]) / 2
                    startpoint.z = 0
                    endpoint.x = startpoint.x + next_point[2]
                    endpoint.y = startpoint.y + next_point[3]
                    endpoint.z = 0
                    tempmarker.points.append(startpoint)
                    tempmarker.points.append(endpoint)

                    self.drivable_area_markerarray.markers.append(tempmarker)
                    count = count + 1

        if len(self.drivable_area_markerarray.markers) < 100:
            for i in range(100 - len(self.drivable_area_markerarray.markers)):
                tempmarker = Marker()
                tempmarker.header.frame_id = "map"
                tempmarker.header.stamp = rospy.Time.now()
                tempmarker.ns = "zzz/cognition"
                tempmarker.id = count
                tempmarker.type = Marker.SPHERE
                tempmarker.action = Marker.ADD
                tempmarker.scale.x = 0.4
                tempmarker.scale.y = 0.7
                tempmarker.scale.z = 0.75
                tempmarker.color.r = 0.5
                tempmarker.color.g = 0.5
                tempmarker.color.b = 1.0
                tempmarker.color.a = 0.8
                tempmarker.lifetime = rospy.Duration(0.5)

                self.drivable_area_markerarray.markers.append(tempmarker)
                count = count + 1

        #7. next drivable area
        self.next_drivable_area_markerarray = MarkerArray()

        count = 0
        if len(tstates.next_drivable_area) != 0:
            
            tempmarker = Marker() #jxy: must be put inside since it is python
            tempmarker.header.frame_id = "map"
            tempmarker.header.stamp = rospy.Time.now()
            tempmarker.ns = "zzz/cognition"
            tempmarker.id = count
            tempmarker.type = Marker.LINE_STRIP
            tempmarker.action = Marker.ADD
            tempmarker.scale.x = 0.20
            tempmarker.color.r = 0.0
            tempmarker.color.g = 1.0
            tempmarker.color.b = 0.0
            tempmarker.color.a = 0.5
            tempmarker.lifetime = rospy.Duration(0.5)

            for point in tstates.next_drivable_area:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = 0 #TODO: the map does not provide z value
                tempmarker.points.append(p)
            self.next_drivable_area_markerarray.markers.append(tempmarker)
            count = count + 1

        #8. next lanes
        self.next_lanes_markerarray = MarkerArray()

        count = 0
        if len(tstates.static_map.next_lanes) != 0:
            biggest_id = 0 #TODO: better way to find the smallest id
            
            for lane in tstates.static_map.next_lanes:
                if lane.index > biggest_id:
                    biggest_id = lane.index
                tempmarker = Marker() #jxy: must be put inside since it is python
                tempmarker.header.frame_id = "map"
                tempmarker.header.stamp = rospy.Time.now()
                tempmarker.ns = "zzz/cognition"
                tempmarker.id = count
                tempmarker.type = Marker.LINE_STRIP
                tempmarker.action = Marker.ADD
                tempmarker.scale.x = 0.12
                tempmarker.color.r = 0.7
                tempmarker.color.g = 0.0
                tempmarker.color.b = 0.0
                tempmarker.color.a = 0.5
                tempmarker.lifetime = rospy.Duration(0.5)

                for lanepoint in lane.central_path_points:
                    p = Point()
                    p.x = lanepoint.position.x
                    p.y = lanepoint.position.y
                    p.z = lanepoint.position.z
                    tempmarker.points.append(p)
                self.next_lanes_markerarray.markers.append(tempmarker)
                count = count + 1

        #9. next lane boundary line
        self.next_lanes_boundary_markerarray = MarkerArray()

        count = 0
        if len(tstates.static_map.next_lanes) != 0:
            
            for lane in tstates.static_map.next_lanes:
                if len(lane.right_boundaries) > 0 and len(lane.left_boundaries) > 0:
                    tempmarker = Marker() #jxy: must be put inside since it is python
                    tempmarker.header.frame_id = "map"
                    tempmarker.header.stamp = rospy.Time.now()
                    tempmarker.ns = "zzz/cognition"
                    tempmarker.id = count

                    #each lane has the right boundary, only the lane with the smallest id has the left boundary
                    tempmarker.type = Marker.LINE_STRIP
                    tempmarker.action = Marker.ADD
                    tempmarker.scale.x = 0.15
                    
                    if lane.right_boundaries[0].boundary_type == 1: #broken lane is set gray
                        tempmarker.color.r = 0.4
                        tempmarker.color.g = 0.4
                        tempmarker.color.b = 0.4
                        tempmarker.color.a = 0.5
                    else:
                        tempmarker.color.r = 0.7
                        tempmarker.color.g = 0.7
                        tempmarker.color.b = 0.7
                        tempmarker.color.a = 0.5
                    tempmarker.lifetime = rospy.Duration(0.5)

                    for lb in lane.right_boundaries:
                        p = Point()
                        p.x = lb.boundary_point.position.x
                        p.y = lb.boundary_point.position.y
                        p.z = lb.boundary_point.position.z
                        tempmarker.points.append(p)
                    self.next_lanes_boundary_markerarray.markers.append(tempmarker)
                    count = count + 1

                    #biggest id: draw left lane
                    if lane.index == biggest_id:
                        tempmarker = Marker() #jxy: must be put inside since it is python
                        tempmarker.header.frame_id = "map"
                        tempmarker.header.stamp = rospy.Time.now()
                        tempmarker.ns = "zzz/cognition"
                        tempmarker.id = count

                        #each lane has the right boundary, only the lane with the biggest id has the left boundary
                        tempmarker.type = Marker.LINE_STRIP
                        tempmarker.action = Marker.ADD
                        tempmarker.scale.x = 0.3
                        if lane.left_boundaries[0].boundary_type == 1: #broken lane is set gray
                            tempmarker.color.r = 0.4
                            tempmarker.color.g = 0.4
                            tempmarker.color.b = 0.4
                            tempmarker.color.a = 0.5
                        else:
                            tempmarker.color.r = 0.7
                            tempmarker.color.g = 0.7
                            tempmarker.color.b = 0.7
                            tempmarker.color.a = 0.5
                        tempmarker.lifetime = rospy.Duration(0.5)

                        for lb in lane.left_boundaries:
                            p = Point()
                            p.x = lb.boundary_point.position.x
                            p.y = lb.boundary_point.position.y
                            p.z = lb.boundary_point.position.z
                            tempmarker.points.append(p)
                        self.next_lanes_boundary_markerarray.markers.append(tempmarker)
                        count = count + 1

        #10. traffic lights
        self._traffic_lights_markerarray = MarkerArray()
        #TODO: now no lights are in.
        #lights = self._traffic_light_detection.detections