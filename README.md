## 环境要求
- Python 2.7
- [ROS Kinetic](http://wiki.ros.org/kinetic)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- [Stage 仿真平台](http://rtv.github.io/Stage/)
- [PyTorch](http://pytorch.org/)



需要注意的是，进行阶段 2 训练的目的在于增强模型的泛化能力，使其能够在真实环境中也具备良好的表现。

请使用 `stage_ros-add_pose_and_crash` 软件包替代 ROS 提供的默认软件包。你可以按照以下步骤进行环境配置：
```bash
# 创建工作空间和源文件目录
mkdir -p catkin_ws/src
# 复制自定义的 stage_ros 包到工作空间的源目录
cp stage_ros-add_pose_and_crash catkin_ws/src
# 进入工作空间
cd catkin_ws
# 编译工作空间
catkin_make
# 加载工作空间的环境变量
source devel/setup.bash
```

###  训练
同样，你可以按需调整 `ppo_stage2.py` 中的超参数，之后运行以下命令开启训练：
```bash
rosrun stage_ros_add_pose_and_crash stageros -g worlds/stage2.world
mpiexec -np 44 python ppo_stage2.py
```

## 测试方法
运行以下命令进行测试：
```bash
rosrun stage_ros_add_pose_and_crash stageros worlds/circle.world
mpiexec --allow-run-as-root --use-hwthread-cpus --oversubscribe -np 6 python circle_test.py
```

