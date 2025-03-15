import random
import math

# 地图范围
MAP_SIZE = 80
MIN_DISTANCE = 2

shapes = {
    "rect": {
        "define": "define dynamic_obstacle_rect model\n(\n    color \"red\"\n    size [1 1 1]\n    gui_nose 0\n    obstacle_return 1\n    ranger_return 1\n)",
        "instance": "dynamic_obstacle_rect( pose [{x:.2f} {y:.2f} 0.0 {angle:.1f}])"
    },
    "circle_approx": {
        "define": """define dynamic_obstacle_circle_approx model
(
    color "red"
    size [1 1 1]
    gui_nose 0
    obstacle_return 1
    ranger_return 1
    block(
        points 12
        {points}
        z [0 0.21]
    )
)""",
        "instance": "dynamic_obstacle_circle_approx( pose [{x:.2f} {y:.2f} 0.0 {angle:.1f}])"
    }
}

def generate_circle_points(radius=0.5, num_points=12):
    points_str = ""
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points_str += f"point[{i}] [{x:.2f} {y:.2f}]\n        "
    return points_str

def is_overlapping(new_x, new_y, existing_obstacles):
    for x, y in existing_obstacles:
        distance = math.sqrt((new_x - x) ** 2 + (new_y - y) ** 2)
        if distance < MIN_DISTANCE:
            return True
    if abs(new_x) > MAP_SIZE / 2 or abs(new_y) > MAP_SIZE / 2:
        return True
    return False

def generate_uniform_pos(existing_obstacles, retry=1000):
    for _ in range(retry):
        x = random.uniform(-MAP_SIZE/2 + 5, MAP_SIZE/2 - 5)  # 留边5m
        y = random.uniform(-MAP_SIZE/2 + 5, MAP_SIZE/2 - 5)
        if not is_overlapping(x, y, existing_obstacles):
            return x, y
    raise Exception("无法生成有效位置")

def generate_non_overlapping_obstacles(num_obstacles):
    existing_obstacles = []
    obstacle_configs = []
    shape_names = list(shapes.keys())

    for _ in range(num_obstacles):
        x, y = generate_uniform_pos(existing_obstacles)
        angle = random.uniform(0, 360)
        existing_obstacles.append((x, y))
        
        shape_name = random.choice(shape_names)
        shape = shapes[shape_name]
        if shape_name == "circle_approx":
            points_str = generate_circle_points()
            shape["define"] = shape["define"].format(points=points_str)
        config = shape["instance"].format(x=x, y=y, angle=angle)
        obstacle_configs.append(config)
    return existing_obstacles, obstacle_configs

def generate_non_overlapping_agents(num_agents, existing_obstacles):
    agent_configs = []
    agent_positions = []
    
    for _ in range(num_agents):
        x, y = generate_uniform_pos(existing_obstacles)
        angle = random.uniform(0, 360)
        existing_obstacles.append((x, y))
        
        agent_config = f"agent( pose [{x:.2f} {y:.2f} 0.0 {angle:.1f}])"
        agent_configs.append(agent_config)
        agent_positions.append((x, y, angle))
    return agent_configs, agent_positions

def generate_non_overlapping_goal_points(agent_positions, existing_obstacles):
    all_positions = existing_obstacles + [pos[:2] for pos in agent_positions]
    goal_configs = []
    goal_positions = []

    for _ in range(len(agent_positions)):
        x, y = generate_uniform_pos(all_positions)
        angle = random.uniform(0, 360)
        all_positions.append((x, y))
        goal_positions.append((x, y, angle))

        goal_config = f"goal_marker( pose [{x:.2f} {y:.2f} 0.0 {angle:.1f}])"
        goal_configs.append(goal_config)

    return goal_configs, goal_positions

# 生成配置
num_obstacles = 30
existing_obstacles, obstacle_configs = generate_non_overlapping_obstacles(num_obstacles)

num_agents = 24
agent_configs, agent_positions = generate_non_overlapping_agents(num_agents, existing_obstacles)

goal_configs, _ = generate_non_overlapping_goal_points(agent_positions, existing_obstacles)

# 组装world文件
world_content = """
show_clock 0
show_clock_interval 10000
resolution 0.01
threads 12
speedup 1

define sicklaser ranger
(
  sensor(
    pose [ 0 0 0.1 0 ]
    fov 180
    range [ 0.0 6.0 ]
    samples 512
  )
  color "random"
  block(
    points 4
    point[0] [0 0]
    point[1] [0 1]
    point[2] [1 1]
    point[3] [1 0]
    z [0 0.21]
  )
)

define floorplan model
(
  color "gray30"
  boundary 1
  gui_nose 0
  gui_grid 0
  gui_move 0
  gui_outline 0
  gripper_return 0
  fiducial_return 0
  ranger_return 1
  obstacle_return 1
)

floorplan
(
  name "blank"
  bitmap "rect.png"
  size [80.000 80.000 1.000]
  pose [0.000 0.000 0.000 0.000]
  trail 10000
)
window
(
  size [1300 1300]
  scale 40
  center [0 0]
  rotate [0 0]
  show_occupancy 1
  show_data 1
  show_flags 1
  show_blocks 1
  show_clock 1
  show_footprints 1
  show_grid 1
  show_trailarrows 0
  show_trailrise 0
  show_trailfast 0
  show_occupancy 0
)
define agent position
(
  size [0.88 0.76 0.44]
  origin [0 0 0 0]
  gui_nose 1
  color "random"
  drive "diff"
  obstacle_return 1
  ranger_return 0.5
  blob_return 1
  fiducial_return 1
  sicklaser(
    pose [ 0 0 0 0 ]
  )
)
"""

# 添加形状定义
for shape in shapes.values():
    world_content += shape["define"] + "\n"

world_content += """
define goal_marker model
(
    color "green"
    size [0.5 0.5 0.2]
    gui_nose 0
    obstacle_return 0
    ranger_return 0
    block(
        points 5
        point[0] [0.0 0.0]
        point[1] [0.0 0.5]
        point[2] [0.0 -0.5]
        point[3] [0.5 0.0]
        point[4] [-0.5 0.0]
        z [0 0.2]
    )
)
"""

# 添加智能体、目标点、障碍物
world_content += "\n".join(agent_configs) + "\n"
world_content += "\n".join(goal_configs) + "\n"
world_content += "\n".join(obstacle_configs)

# 保存文件
with open("worlds/new_world.world", "w") as f:
    f.write(world_content)

print("新的 world 文件已生成：new_world.world")