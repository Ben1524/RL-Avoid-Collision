from circle_world import StageWorld
import os
import numpy as np
import rospy

if __name__ == "__main__":
    env = StageWorld(beam_num=360, index=0, num_env=1)
    while not rospy.is_shutdown():
        # 简单的移动逻辑，让障碍物向前移动
        env.control_obstacle_vel(0.1, 0)
        rospy.sleep(0.1)