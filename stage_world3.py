# import math
# import time
# import rospy
# import copy
# import tf
# import numpy as np

# from geometry_msgs.msg import Twist, Pose
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import LaserScan
# from rosgraph_msgs.msg import Clock
# from std_srvs.srv import Empty
# from std_msgs.msg import Int8
# from model.utils import get_init_pose, get_goal_point


# class StageWorld():
#     def __init__(self, beam_num, index, num_env):
#         self.index = index
#         self.num_env = num_env
#         node_name = 'StageEnv_' + str(index)
#         rospy.init_node(node_name, anonymous=None)

#         self.beam_mum = beam_num
#         self.laser_cb_num = 0
#         self.scan = None

#         # used in reset_world
#         self.self_speed = [0.0, 0.0]
#         self.step_goal = [0., 0.]
#         self.step_r_cnt = 0.

#         # used in generate goal point
#         self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m
#         self.goal_size = 0.6

#         self.robot_value = 10.
#         self.goal_value = 0.


#         # -----------Publisher and Subscriber-------------
#         cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
#         self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

#         cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
#         self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=10)

#         object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
#         self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)

#         laser_topic = 'robot_' + str(index) + '/base_scan'

#         self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

#         odom_topic = 'robot_' + str(index) + '/odom'
#         self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

#         crash_topic = 'robot_' + str(index) + '/is_crashed'
#         self.check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)


#         self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)

#         # -----------Service-------------------
#         self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)

#         # # Wait until the first callback
#         self.speed = None
#         self.state = None
#         self.speed_GT = None
#         self.state_GT = None
#         self.is_crashed = None
#         while self.scan is None or self.speed is None or self.state is None\
#                 or self.speed_GT is None or self.state_GT is None or self.is_crashed is None:
#             pass

#         rospy.sleep(1.)
#         # # What function to call when you ctrl + c
#         # rospy.on_shutdown(self.shutdown)


#     def ground_truth_callback(self, GT_odometry):
#         Quaternious = GT_odometry.pose.pose.orientation
#         Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
#         self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
#         v_x = GT_odometry.twist.twist.linear.x
#         v_y = GT_odometry.twist.twist.linear.y
#         v = np.sqrt(v_x**2 + v_y**2)
#         self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

#     def laser_scan_callback(self, scan):
#         self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
#                            scan.scan_time, scan.range_min, scan.range_max]
#         self.scan = np.array(scan.ranges)
#         self.laser_cb_num += 1


#     def odometry_callback(self, odometry):
#         Quaternions = odometry.pose.pose.orientation
#         Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
#         self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
#         self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

#     def sim_clock_callback(self, clock):
#         self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

#     def crash_callback(self, flag):
#         self.is_crashed = flag.data

#     def get_self_stateGT(self):
#         return self.state_GT

#     def get_self_speedGT(self):
#         return self.speed_GT

#     def get_laser_observation(self):
#         scan = copy.deepcopy(self.scan)
#         scan[np.isnan(scan)] = 6.0
#         scan[np.isinf(scan)] = 6.0
#         raw_beam_num = len(scan)
#         sparse_beam_num = self.beam_mum
#         step = float(raw_beam_num) / sparse_beam_num
#         sparse_scan_left = []
#         index = 0.
#         for x in range(int(sparse_beam_num / 2)):
#             sparse_scan_left.append(scan[int(index)])
#             index += step
#         sparse_scan_right = []
#         index = raw_beam_num - 1.
#         for x in range(int(sparse_beam_num / 2)):
#             sparse_scan_right.append(scan[int(index)])
#             index -= step
#         scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
#         return scan_sparse / 6.0 - 0.5


#     def get_self_speed(self):
#         return self.speed

#     def get_self_state(self):
#         return self.state

#     def get_crash_state(self):
#         return self.is_crashed

#     def get_sim_time(self):
#         return self.sim_time

#     def get_local_goal(self):
#         [x, y, theta] = self.get_self_stateGT()
#         [goal_x, goal_y] = self.goal_point
#         local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
#         local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
#         return [local_x, local_y]

#     def reset_world(self):
#         self.reset_stage()
#         self.self_speed = [0.0, 0.0]
#         self.step_goal = [0., 0.]
#         self.step_r_cnt = 0.
#         self.start_time = time.time()
#         rospy.sleep(0.5)


#     def generate_goal_point(self):
#         if self.index > 33 and self.index < 44:
#             self.goal_point = self.generate_random_goal()
#         else:
#             self.goal_point = get_goal_point(self.index)

#         self.pre_distance = 0
#         self.distance = copy.deepcopy(self.pre_distance)



#     def get_reward_and_terminate(self, t):
#         terminate = False
#         laser_scan = self.get_laser_observation()
#         laser_min = np.amin(laser_scan)
#         [x, y, theta] = self.get_self_stateGT()
#         [v, w] = self.get_self_speedGT()
#         self.pre_distance = copy.deepcopy(self.distance)
#         self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
#         reward_g = (self.pre_distance - self.distance) * 2.5
#         reward_c = 0
#         reward_w = 0
#         result = 0

#         is_crash = self.get_crash_state()

#         if self.distance < self.goal_size:
#             terminate = True
#             reward_g = 15
#             result = 'Reach Goal'

#         if is_crash == 1:
#             terminate = True
#             reward_c = -15.
#             result = 'Crashed'

#         # 新增运动质量奖励
#         linear_speed = self.speed_GT[0]
#         angular_speed = abs(self.speed_GT[1])
        
#         # 有效运动奖励（线速度与角速度的权衡）
#         movement_quality = linear_speed * math.exp(-2*angular_speed)
#         reward_m = 0.2 * movement_quality
        
#         if np.abs(w) >  1.05:
#             reward_w = -0.1 * np.abs(w)

#         reward_o = 0.0
#         if self.scan is not None and len(self.scan) > 0:
#             # 前方 180 度危险区域
#             front_angle = np.pi
#             angle_min = self.scan_param[0]
#             angle_max = self.scan_param[1]
#             angle_inc = self.scan_param[2]

#             # 计算前方光束索引范围
#             num_front_beams = int((front_angle / (angle_max - angle_min)) * len(self.scan))
#             center_idx = len(self.scan) // 2
#             start_idx = max(0, center_idx - num_front_beams // 2)
#             end_idx = min(len(self.scan) - 1, center_idx + num_front_beams // 2)
#             front_ranges = self.scan[start_idx:end_idx]

#             if len(front_ranges) > 0:
#                 min_front_range = np.min(front_ranges)
#                 linear_speed = self.speed_GT[0]
                
#                 # 动态安全距离（带上限）
#                 safe_distance = min(0.7, 0.6 + 0.1 * linear_speed)
                
#                 # 极近距离阈值（基于安全距离）
#                 critical_distance = max(0.1, 0.3 * safe_distance)
                
#                 # 分段惩罚逻辑
#                 if min_front_range < critical_distance and min_front_range >= 0.0:
#                     # 极近距离终止并高惩罚
#                     terminate = True
#                     reward_o = -20.0
#                     result = 'Critical Distance'
#                 elif min_front_range < safe_distance and min_front_range >= critical_distance:
#                     # 近距离线性惩罚
#                     reward_o = -1.5 * (safe_distance - min_front_range)

#         # # 强化角速度惩罚（分段函数）
#         # if angular_speed > 0.5:
#         #     reward_w = -0.3 * (angular_speed ** 2)
#         # elif angular_speed > 1.0:
#         #     reward_w = -0.6 * (angular_speed ** 2)
#         # else:
#         #     reward_w = 0.0
        

#         if t > 1500:
#             terminate = True
#             result = 'Time out'
#         reward = reward_g + reward_c + reward_w + reward_o
    
#         return reward, terminate, result

#     def reset_pose(self):
#         if self.index > 33 and self.index < 44:
#             reset_pose = self.generate_random_pose()
#         else:
#             reset_pose = get_init_pose(self.index)
#         rospy.sleep(0.05)
#         self.control_pose(reset_pose)
#         [x_robot, y_robot, theta] = self.get_self_stateGT()

#         while np.abs(reset_pose[0] - x_robot) > 0.2 or np.abs(reset_pose[1] - y_robot) > 0.2:
#             [x_robot, y_robot, theta] = self.get_self_stateGT()
#         rospy.sleep(0.05)


#     def control_vel(self, action):
#         move_cmd = Twist()
#         move_cmd.linear.x = action[0]
#         move_cmd.linear.y = 0.
#         move_cmd.linear.z = 0.
#         move_cmd.angular.x = 0.
#         move_cmd.angular.y = 0.
#         move_cmd.angular.z = action[1]
#         self.cmd_vel.publish(move_cmd)


#     def control_pose(self, pose):
#         pose_cmd = Pose()
#         assert len(pose)==3
#         pose_cmd.position.x = pose[0]
#         pose_cmd.position.y = pose[1]
#         pose_cmd.position.z = 0

#         qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
#         pose_cmd.orientation.x = qtn[0]
#         pose_cmd.orientation.y = qtn[1]
#         pose_cmd.orientation.z = qtn[2]
#         pose_cmd.orientation.w = qtn[3]
#         self.cmd_pose.publish(pose_cmd)


#     def generate_random_pose(self):
#         [x_robot, y_robot, theta] = self.get_self_stateGT()
#         x = np.random.uniform(9, 19)
#         y = np.random.uniform(0, 1)
#         if y <= 0.4:
#             y = -(y * 10 + 1)
#         else:
#             y = -(y * 10 + 9)
#         dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
#         while (dis_goal < 7) and not rospy.is_shutdown():
#             x = np.random.uniform(9, 19)
#             y = np.random.uniform(0, 1)
#             if y <= 0.4:
#                 y = -(y * 10 + 1)
#             else:
#                 y = -(y * 10 + 9)
#             dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
#         theta = np.random.uniform(0, 2*np.pi)
#         return [x, y, theta]

#     def generate_random_goal(self):
#         [x_robot, y_robot, theta] = self.get_self_stateGT()
#         x = np.random.uniform(9, 19)
#         y = np.random.uniform(0, 1)
#         if y <= 0.4:
#             y = -(y*10 + 1)
#         else:
#             y = -(y*10 + 9)
#         dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
#         while (dis_goal < 7) and not rospy.is_shutdown():
#             x = np.random.uniform(9, 19)
#             y = np.random.uniform(0, 1)
#             if y <= 0.4:
#                 y = -(y * 10 + 1)
#             else:
#                 y = -(y * 10 + 9)
#             dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
#         return [x, y]




import math
import time
import rospy
import copy
import tf
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8
from model.utils import test_init_pose, test_goal_point


def get_pose_info_from_file(file_path):
    """
    从指定文件中读取内容，并提取 agent 和 goal_marker 的 pose 信息。

    :param file_path: 文件的路径
    :return: 包含 agent pose 信息的列表和包含 goal_marker pose 信息的列表
    """
    agent_poses = []
    goal_marker_poses = []
    try:
        # 打开文件并读取内容
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("agent( pose"):
                    # 提取 agent 的 pose 信息
                    start = line.find("[") + 1
                    end = line.find("]")
                    pose_str = line[start:end]
                    pose = [float(num) for num in pose_str.split()]
                    pose_tmp = [pose[0], pose[1], pose[3]]
                    agent_poses.append(pose_tmp)
                elif line.startswith("goal_marker( pose"):
                    # 提取 goal_marker 的 pose 信息
                    start = line.find("[") + 1
                    end = line.find("]")
                    pose_str = line[start:end]
                    pose = [float(num) for num in pose_str.split()]
                    goal_marker_poses.append(pose[0:2])
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查文件路径。")
    except Exception as e:
        print(f"读取文件时出现错误: {e}")

    return agent_poses, goal_marker_poses

file_path = "worlds/new_world.world"
agent_poses, goal_marker_poses = get_pose_info_from_file(file_path)
# print (agent_poses)
# print (goal_marker_poses)

class StageWorld():
    def __init__(self, beam_num, index, num_env):
        self.index = index
        self.num_env = num_env
        node_name = 'StageEnv_' + str(index)
        rospy.init_node(node_name, anonymous=None)

        self.beam_num = beam_num  # 修正拼写错误
        self.laser_cb_num = 0
        self.scan = None

        # used in reset_world
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.

        # used in generate goal point
        self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m
        self.goal_size = 1.5

        self.robot_value = 10.
        self.goal_value = 0.

        # -----------Publisher and Subscriber-------------
        cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=10)

        object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)

        laser_topic = 'robot_' + str(index) + '/base_scan'
        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        odom_topic = 'robot_' + str(index) + '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        crash_topic = 'robot_' + str(index) + '/is_crashed'
        self.check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)

        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)

        # 新增：订阅障碍物的状态信息
        obstacle_odom_topic = 'obstacle_' + str(index) + '/odom'
        self.obstacle_odom_sub = rospy.Subscriber(obstacle_odom_topic, Odometry, self.obstacle_odometry_callback)
        self.obstacle_state = None
        self.obstacle_speed = None

        # 新增：发布障碍物的控制指令
        obstacle_cmd_vel_topic = 'obstacle_' + str(index) + '/cmd_vel'
        self.obstacle_cmd_vel = rospy.Publisher(obstacle_cmd_vel_topic, Twist, queue_size=10)

        # -----------Service-------------------
        self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)

        # ------------------------------
            # 新增路径发布器
        self.path_pub = rospy.Publisher('robot_{}/path'.format(index), Path, queue_size=10)
        self.current_path = Path()
        self.current_path.header.frame_id = 'map'

        # 新增：近距离惩罚参数
        self.min_obstacle_range = 1  # 前方障碍物最小安全距离（米）
        self.obstacle_penalty_coeff = 10.0  # 惩罚系数（距离每近1米惩罚值）


        # # Wait until the first callback
        self.speed = None
        self.state = None
        self.speed_GT = None
        self.state_GT = None
        self.is_crashed = None
        while self.scan is None or self.speed is None or self.state is None \
                or self.speed_GT is None or self.state_GT is None or self.is_crashed is None :
        # or self.obstacle_state is None:
            pass

        rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)



    def ground_truth_callback(self, GT_odometry):
        Quaternious = GT_odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x ** 2 + v_y ** 2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]
          # 新增路径更新逻辑
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose = GT_odometry.pose.pose
        self.current_path.poses.append(pose_stamped)
        self.path_pub.publish(self.current_path)

    def laser_scan_callback(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, 
                          scan.time_increment, scan.scan_time, scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)
        # 新增：处理无效值（NaN/inf）
        self.scan[np.isnan(self.scan)] = scan.range_max  # 替换为最大有效距离
        self.scan[np.isinf(self.scan)] = scan.range_max
        self.laser_cb_num += 1

    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crash_callback(self, flag):
        self.is_crashed = flag.data

    def obstacle_odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.obstacle_state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.obstacle_speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def get_self_stateGT(self):
        return self.state_GT

    def get_self_speedGT(self):
        return self.speed_GT

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 6.0
        scan[np.isinf(scan)] = 6.0
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_num
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
        return scan_sparse / 6.0 - 0.5

    def get_self_speed(self):
        return self.speed

    def get_self_state(self):
        return self.state

    def get_crash_state(self):
        return self.is_crashed

    def get_sim_time(self):
        return self.sim_time

    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def reset_world(self):
        self.reset_stage()
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()
        rospy.sleep(0.5)
        self.current_path = Path()  # 重置路径
        self.current_path.header.frame_id = 'map'

    def generate_goal_point(self):
        self.goal_point = goal_marker_poses[self.index]
        self.pre_distance = 0
        self.distance = copy.deepcopy(self.pre_distance)

   
    def get_reward_and_terminate(self, t):
        terminate = False
        laser_scan = self.get_laser_observation()
        laser_min = np.amin(laser_scan)
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        reward_g = (self.pre_distance - self.distance) * 2.5
        reward_c = 0
        reward_w = 0
        result = 0

        is_crash = self.get_crash_state()

        if self.distance < self.goal_size:
            terminate = True
            reward_g = 15
            result = 'Reach Goal'

        if is_crash == 1:
            terminate = True
            reward_c = -15.
            result = 'Crashed'

        # 新增运动质量奖励
        linear_speed = self.speed_GT[0]
        angular_speed = abs(self.speed_GT[1])
        
        # 有效运动奖励（线速度与角速度的权衡）
        movement_quality = linear_speed * math.exp(-2*angular_speed)
        reward_m = 0.2 * movement_quality
        
        if np.abs(w) >  1.05:
            reward_w = -0.1 * np.abs(w)

        reward_o = 0.0
        if self.scan is not None and len(self.scan) > 0:
            # 前方 180 度危险区域
            front_angle = np.pi
            angle_min = self.scan_param[0]
            angle_max = self.scan_param[1]
            angle_inc = self.scan_param[2]

            # 计算前方光束索引范围
            num_front_beams = int((front_angle / (angle_max - angle_min)) * len(self.scan))
            center_idx = len(self.scan) // 2
            start_idx = max(0, center_idx - num_front_beams // 2)
            end_idx = min(len(self.scan) - 1, center_idx + num_front_beams // 2)
            front_ranges = self.scan[start_idx:end_idx]

            if len(front_ranges) > 0:
                min_front_range = np.min(front_ranges)
                linear_speed = self.speed_GT[0]
                
                # 动态安全距离（带上限）
                safe_distance = min(0.7, 0.6 + 0.1 * linear_speed)
                
                # 极近距离阈值（基于安全距离）
                critical_distance = max(0.1, 0.3 * safe_distance)
                
                # 分段惩罚逻辑
                if min_front_range < critical_distance and min_front_range >= 0.0:
                    # 极近距离终止并高惩罚
                    terminate = True
                    reward_o = -20.0
                    result = 'Critical Distance'
                elif min_front_range < safe_distance and min_front_range >= critical_distance:
                    # 近距离线性惩罚
                    reward_o = -1.5 * (safe_distance - min_front_range)

        # # 强化角速度惩罚（分段函数）
        # if angular_speed > 0.5:
        #     reward_w = -0.3 * (angular_speed ** 2)
        # elif angular_speed > 1.0:
        #     reward_w = -0.6 * (angular_speed ** 2)
        # else:
        #     reward_w = 0.0
        

        if t > 1500:
            terminate = True
            result = 'Time out'
        reward = reward_g + reward_c + reward_w + reward_o
    
        return reward, terminate, result


    def reset_pose(self):
        # reset_pose = test_init_pose(self.index)
        reset_pose = agent_poses[self.index]
        self.control_pose(reset_pose)

    def control_vel(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)

    def control_pose(self, pose):
        pose_cmd = Pose()
        assert len(pose) == 3
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        self.cmd_pose.publish(pose_cmd)

    def generate_random_pose(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(9, 19)
        y = np.random.uniform(0, 1)
        if y <= 0.4:
            y = -(y * 10 + 1)
        else:
            y = -(y * 10 + 9)
        dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        while (dis_goal < 7) and not rospy.is_shutdown():
            x = np.random.uniform(9, 19)
            y = np.random.uniform(0, 1)
            if y <= 0.4:
                y = -(y * 10 + 1)
            else:
                y = -(y * 10 + 9)
            dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        theta = np.random.uniform(0, 2 * np.pi)
        return [x, y, theta]

    def generate_random_goal(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(9, 19)
        y = np.random.uniform(0, 1)
        if y <= 0.4:
            y = -(y * 10 + 1)
        else:
            y = -(y * 10 + 9)
        dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        while (dis_goal < 7) and not rospy.is_shutdown():
            x = np.random.uniform(9, 19)
            y = np.random.uniform(0, 1)
            if y <= 0.4:
                y = -(y * 10 + 1)
            else:
                y = -(y * 10 + 9)
            dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        return [x, y]

    # 新增：控制障碍物移动的方法
    def control_obstacle_vel(self, linear_vel, angular_vel):
        move_cmd = Twist()
        move_cmd.linear.x = linear_vel
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = angular_vel
        self.obstacle_cmd_vel.publish(move_cmd)

    def move_obstacle_randomly(self):
        # 获取障碍物当前位置
        x, y, _ = self.obstacle_state
        # 随机生成线速度和角速度
        linear_vel = np.random.uniform(0.05, 0.2)
        angular_vel = np.random.uniform(-0.5, 0.5)

        # 预测下一个位置
        dt = 0.1  # 时间步长
        new_x = x + linear_vel * np.cos(self.obstacle_state[2]) * dt
        new_y = y + linear_vel * np.sin(self.obstacle_state[2]) * dt

        # 检查新位置是否超出环境范围
        if (0 <= new_x <= self.map_size[0]) and (0 <= new_y <= self.map_size[1]):
            self.control_obstacle_vel(linear_vel, angular_vel)
        else:
            # 如果超出范围，反向旋转
            self.control_obstacle_vel(0, np.pi / 2)

