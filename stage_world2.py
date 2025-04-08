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
from model.utils import get_init_pose, get_goal_point


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
        # 课程学习参数
        self.base_safe_distance = 2.0  # 初始安全距离
        self.current_safe_distance = self.base_safe_distance             
        self.base_goal_size = 1.5      # 初始目标区域大小
        self.current_goal_size = self.base_goal_size
        self.success_counter = 0       # 成功计数器
        self.episode_count = 0         # 回合计数器
        self.difficulty_level = 0      # 当前难度等级

        # 新增运动约束参数
        self.min_linear_speed_for_turn = 0.15  # 允许转向的最小线速度
        self.max_linear_acc = 1  # 最大线加速度(m/s²)
        self.max_angular_acc = 1.0  # 最大角加速度(rad/s²)
        self.last_action_time = time.time()  # 记录上次动作时间
        
        # 优化碰撞检测参数
        self.collision_distance = 0.35  # 碰撞判定距离
        self.near_miss_distance = 0.8   # 危险距离
        self.emergency_brake_distance = 0.5  # 紧急制动距离

        # 动作平滑参数
        self.smooth_factor = 0.3  # 平滑系数(0~1)，值越小越平滑
        self.last_action = np.array([0.0, 0.0])  # 初始动作
        
        # 动作限制参数
        self.max_linear_speed = 0.8  # 最大线速度(m/s)
        self.max_angular_speed = 1 # 最大角速度(rad/s)

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
        self.goal_size = 1.0

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
        self.obstacle_penalty_coeff = 0.0001  # 惩罚系数（距离每近1米惩罚值）


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

    # 修改4：激光数据处理优化
    def laser_scan_callback(self, scan):
        # 参数有效性验证
        if not scan.ranges:
            rospy.logwarn("Empty laser scan received!")
            return
        
        try:
            self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment,
                            scan.time_increment, scan.range_min, scan.range_max]
            
            # 改进的无效值处理
            raw_scan = np.array(scan.ranges)
            valid_mask = np.logical_and(raw_scan >= scan.range_min, 
                                    raw_scan <= scan.range_max)
            
            # 使用相邻有效值填充无效值
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 0:
                raw_scan = np.interp(np.arange(len(raw_scan)), valid_indices, raw_scan[valid_indices])
            else:
                raw_scan = np.full_like(raw_scan, scan.range_max)
            
            # 优化滑动窗口滤波
            window_size = 5
            conv_window = np.hanning(window_size)  # 使用汉宁窗
            conv_window /= conv_window.sum()
            smoothed_scan = np.convolve(raw_scan, conv_window, mode='same')
            
            self.scan = np.clip(smoothed_scan, scan.range_min, scan.range_max)
            self.laser_cb_num += 1
            
        except Exception as e:
            rospy.logerr(f"Laser processing error: {str(e)}")
            self.scan = np.full(len(scan.ranges), scan.range_max)

    def update_difficulty(self, success_rate):
        """每10个成功episode更新难度"""
        # 修改更新条件为累计成功次数
        if self.success_counter >= 10:
            success_rate = self.success_counter / 10
            if success_rate > 0.7:
                self.difficulty_level += 1
                # 限制最大难度级别
                self.difficulty_level = min(self.difficulty_level, 10)  
                new_safe_dist = max(0.5, self.base_safe_distance * (0.9**self.difficulty_level))
                new_goal_size = max(0.3, self.base_goal_size * (0.95**self.difficulty_level))
                
                # 渐进式变化检测
                if abs(new_safe_dist - self.current_safe_distance) > 0.1:
                    self.current_safe_distance = new_safe_dist
                    self.current_goal_size = new_goal_size
                    print(f"Difficulty level {self.difficulty_level}: "
                        f"SafeDist={self.current_safe_distance:.2f}m, "
                        f"GoalSize={self.current_goal_size:.2f}m")
            self.success_counter = 0  # 重置计数器


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

    def control_vel(self, action):
        try:
            current_time = time.time()
            dt = current_time - self.last_action_time
            self.last_action_time = current_time
            
            # 动态平滑系数（根据速度调整）
            dynamic_smooth = 0.4 if np.linalg.norm(self.last_action) > 0.3 else 0.2
            smoothed = dynamic_smooth*action + (1-dynamic_smooth)*self.last_action
            
            # 应用加速度限制
            delta_linear = (smoothed[0] - self.last_action[0]) / dt
            delta_angular = (smoothed[1] - self.last_action[1]) / dt
            
            if abs(delta_linear) > self.max_linear_acc:
                smoothed[0] = self.last_action[0] + np.sign(delta_linear)*self.max_linear_acc*dt
                
            if abs(delta_angular) > self.max_angular_acc:
                smoothed[1] = self.last_action[1] + np.sign(delta_angular)*self.max_angular_acc*dt

            # 安全速度限制
            linear = np.clip(smoothed[0], 0, self._dynamic_speed_limit())
            angular = np.clip(smoothed[1], -self._angular_speed_limit(linear), self._angular_speed_limit(linear))
            
            # 紧急制动逻辑
            if self._emergency_check():
                linear = 0
                angular = 0
                print("EMERGENCY STOP!")

            # 更新并发布指令
            self.last_action = np.array([linear, angular])
            move_cmd = Twist()
            move_cmd.linear.x = linear
            move_cmd.angular.z = angular
            self.cmd_vel.publish(move_cmd)
            
        except Exception as e:
            rospy.logerr(f"Control error: {str(e)}")
            self._emergency_stop()

    def _dynamic_speed_limit(self):
        """根据环境复杂度动态调整最大速度"""
        if self.scan is None:
            return self.max_linear_speed
            
        # 计算前方180度区域的最小距离
        front_scan = self.scan[len(self.scan)//4 : 3*len(self.scan)//4]
        min_distance = np.min(front_scan) if len(front_scan)>0 else 6.0
        
        # 动态速度限制曲线
        if min_distance < self.emergency_brake_distance:
            return 0.0
        elif min_distance < self.near_miss_distance:
            return self.max_linear_speed * 0.3
        elif min_distance < 1.5:
            return self.max_linear_speed * 0.6
        else:
            return self.max_linear_speed

    def _angular_speed_limit(self, linear_speed):
        """根据线速度限制角速度"""
        base_limit = self.max_angular_speed
        # 线速度越高，允许的角速度越小
        return base_limit * (1 - linear_speed/self.max_linear_speed)**2

    def _emergency_check(self):
        """紧急制动条件检测"""
        if self.scan is None:
            return False
            
        # 检测前方扇形区域
        front_angle = np.pi/2  # 90度
        angle_min = self.scan_param[0]
        angle_max = self.scan_param[1]
        num_front = int(front_angle/(angle_max-angle_min)*len(self.scan))
        start = len(self.scan)//2 - num_front//2
        end = len(self.scan)//2 + num_front//2
        front_ranges = self.scan[max(0,start):min(end,len(self.scan))]
        
        if len(front_ranges)==0:
            return False
            
        min_dist = np.min(front_ranges)
        return min_dist < self.emergency_brake_distance

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
    
    # 渐进式难度训练
    def adjust_difficulty(self, success_rate):
        # 动态调整参数
        if success_rate > 0.7:
            self.min_obstacle_range *= 0.95  # 逐步缩小安全距离
            self.goal_size *= 0.95           # 提高目标精度要求
        else:
            self.min_obstacle_range = max(1.5, self.min_obstacle_range*1.05)
    

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
        # if self.index > 33 and self.index < 44:
        #         self.goal_point = self.generate_random_goal()
        # else:
        #     self.goal_point = get_goal_point(self.index)
    
        self.pre_distance = 0
        self.distance = copy.deepcopy(self.pre_distance)
    def _calculate_obstacle_reward(self):
        """基于激光数据的精细化障碍物奖励"""
        if self.scan is None:
            return 0.0
            
        # 分区域检测
        sectors = {
            'front': (np.pi/3, 2*np.pi/3),
            'left': (np.pi/6, np.pi/3),
            'right': (2*np.pi/3, 5*np.pi/6)
        }
        
        penalty = 0.0
        for sector, (start_ang, end_ang) in sectors.items():
            min_dist = self._get_sector_min_distance(start_ang, end_ang)
            if min_dist < self.collision_distance:
                penalty -= 15.0
            elif min_dist < self.near_miss_distance:
                penalty -= 8.0 * (1 - min_dist/self.near_miss_distance)
            elif min_dist < 1.5:
                penalty -= 2.0 * (1.5 - min_dist)/1.5
                
        return penalty

    def _get_sector_min_distance(self, start_ang, end_ang):
        """获取指定扇形区域的最小距离"""
        angle_min = self.scan_param[0]
        angle_inc = self.scan_param[2]
        
        start_idx = int((start_ang - angle_min) / angle_inc)
        end_idx = int((end_ang - angle_min) / angle_inc)
        sector_scan = self.scan[start_idx:end_idx]
        
        return np.min(sector_scan) if len(sector_scan)>0 else 6.0

    def get_reward_and_terminate(self, t, action=None):
        terminate = False
        result = "Running"
        [x, y, theta] = self.get_self_stateGT()
        goal_vector = np.array([self.goal_point[0]-x, self.goal_point[1]-y])
        if np.linalg.norm(goal_vector) == 0:
            goal_angle = 0.0
        else:
            goal_angle = np.arctan2(goal_vector[1], goal_vector[0])
        angle_error = np.abs(theta - goal_angle)  # 朝向误差（弧度）
        [v, w] = self.get_self_speedGT()

        # ================= 基础奖励 =================
        # 目标距离奖励（指数衰减形式）
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        diff = self.pre_distance - self.distance
        reward_g = diff * 3
        
        # 新增避障奖励项
        obstacle_reward = self._calculate_obstacle_reward()
        # ================= 碰撞惩罚 =================
        reward_c = 0.0
        is_crash = self.get_crash_state()
        # 优化碰撞惩罚
        if is_crash == 1:
            reward_c = -30.0
            result = 'Crashed'
        

        # ================= 运动惩罚 =================
        reward_w = 0.0
        # 角速度惩罚（二次函数形式）
        if np.abs(w) > 0.75:
            reward_w = -0.1 * np.abs(w)
        if self.distance > 2.0:  # 远距离时允许转向
            if angle_error > np.pi/2:  # 需大幅转向
                reward_w += 0.05 * abs(action[1])  # 奖励有效转向
            else:
                reward_w -= 0.1 * abs(action[1])  # 惩罚无效转向（如原地转圈）
        
        reward_v = 0.1 * v if v > 0.2 else -0.1       # 鼓励持续移动

        # ================= 修改障碍物惩罚计算 =================
        reward_o = 0.0
        if self.scan is not None and len(self.scan) > 0:
            # 使用动态安全距离参数
            front_angle = np.pi  # 改为60度危险区域
            angle_min = self.scan_param[0]
            angle_max = self.scan_param[1]
            angle_inc = self.scan_param[2]
            
            # 计算前方光束索引范围
            num_front_beams = int((front_angle / (angle_max - angle_min)) * len(self.scan))
            center_idx = len(self.scan) // 2
            start_idx = max(0, center_idx - num_front_beams//2)
            end_idx = min(len(self.scan)-1, center_idx + num_front_beams//2)
            front_ranges = self.scan[start_idx:end_idx]
            
            # 分级惩罚策略优化
            if len(front_ranges) > 0:
                min_front_range = np.min(front_ranges)
                if min_front_range < 1.0:
                    reward_o -= 0.1 * (1.0 - min_front_range)
                elif min_front_range < self.current_safe_distance:
                    reward_o -= 0.2 * (self.current_safe_distance - min_front_range)

        # ================= 时间惩罚 =================
        reward_t = 0 #-0.0001 * t  # 鼓励快速到达

        # ================= 修改课程学习调用逻辑 =================
        if terminate and result == 'Reach Goal':
            self.success_counter += 1

        # ================= 最终判断 =================
        # 修改终止条件判断
        if self.distance < self.current_goal_size:  # 使用动态目标尺寸
            terminate = True
            self.success_counter += 1
            reward_g += 30.0  # 最终到达奖励
            result = 'Reach Goal'

        if t > 800:  # 缩短最大步长
            terminate = True
            result = 'Time out'
            # reward_t -= 15.0

        total_reward = reward_g + reward_c + reward_w + reward_v + reward_o + reward_t

         # ================= 添加动作变化率惩罚 =================
        reward_a = 0.0
        if action is not None:
            # 计算动作变化率惩罚（使用最近两个动作）
            action_diff = np.linalg.norm(action - self.last_action)
            reward_a = -0.01 * action_diff  # 惩罚剧烈动作变化
        
        total_reward += reward_a

        # Debug信息
        # print(f"[Reward] Total:{total_reward:.2f} | Goal:{reward_g:.2f} | Collision:{reward_c:.2f} | "
        #     f"Angular:{reward_w:.2f} | Velocity:{reward_v:.2f} | Obstacle:{reward_o:.2f} | A:{reward_a:.2f}")

        return total_reward, terminate, result

    def reset_pose(self):
        # reset_pose = get_init_pose(self.index)
        reset_pose = agent_poses[self.index]
        self.control_pose(reset_pose)

    # 在控制指令发布前增加平滑处理
    def control_vel(self, action):
            # 动作平滑处理
        smoothed_action = self.smooth_factor * action + (1 - self.smooth_factor) * self.last_action
        
        # 动作限幅
        linear = np.clip(smoothed_action[0], 
                        -self.max_linear_speed, 
                        self.max_linear_speed)
        angular = np.clip(smoothed_action[1], 
                         -self.max_angular_speed, 
                         self.max_angular_speed)
        
        # 保存当前动作用于下次平滑
        self.last_action = np.array([linear, angular])
        
        # 发布控制指令
        move_cmd = Twist()
        move_cmd.linear.x = linear
        move_cmd.angular.z = angular
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

