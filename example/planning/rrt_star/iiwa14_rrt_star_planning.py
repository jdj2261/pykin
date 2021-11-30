import numpy as np
import sys, os
import yaml
import trimesh

parent_dir = os.path.dirname(os.getcwd())
pykin_path = parent_dir + "/../../"
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.planners.rrt_star_planner import RRTStarPlanner
from pykin.collision.collision_manager import CollisionManager
from pykin.kinematics.transform import Transform
from pykin.utils.obstacle_utils import Obstacle
from pykin.utils import plot_utils as plt
from pykin.utils.collision_utils import apply_robot_to_collision_manager, apply_robot_to_scene

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi= 100)

file_path = '../../../asset/urdf/iiwa14/iiwa14.urdf'
mesh_path = pykin_path+"/asset/urdf/iiwa14/"
yaml_path = '../../../asset/config/iiwa14_init_params.yaml'

with open(yaml_path) as f:
    controller_config = yaml.safe_load(f)

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0], pos=[0, 0, 0]))
robot.setup_link_name("iiwa14_link_0", "iiwa14_right_hand")

##################################################################
init_qpos = controller_config["init_qpos"]
init_fk = robot.forward_kin(init_qpos)
init_eef_pose = robot.get_eef_pose(init_fk)
goal_eef_pose = controller_config["goal_pose"]

c_manager = CollisionManager(mesh_path)
c_manager.filter_contact_names(robot, init_fk)
c_manager = apply_robot_to_collision_manager(c_manager, robot, init_fk)

milk_path = pykin_path+"/asset/objects/meshes/milk.stl"
milk_mesh = trimesh.load_mesh(milk_path)

obs = Obstacle()
o_manager = CollisionManager(milk_path)
for i in range(9):
    name = "miik_" + str(i)
    if i < 3:
        obs_pos = [0.3, -0.5 + i * 0.5, 0.3] 
    elif 3 <= i < 6:
        obs_pos = [0.3, -0.5 + (i-3) * 0.5, 0.9] 
    else:
        obs_pos = [0.3, -0.5 + (i-6) * 0.5, -0.3]

    o_manager.add_object(name, gtype="mesh", gparam=milk_mesh, transform=Transform(pos=obs_pos).h_mat)
    obs(name=name, gtype="mesh", gparam=milk_mesh, gpose=Transform(pos=obs_pos))

##################################################################

planner = RRTStarPlanner(
    robot=robot,
    self_collision_manager=c_manager,
    obstacle_collision_manager=o_manager,
    delta_distance=0.03,
    epsilon=0.2, 
    max_iter=1000,
    gamma_RRT_star=10,
    dimension=7
)

joint_path = planner.get_path_in_joinst_space(cur_q=init_qpos, goal_pose=goal_eef_pose)

if joint_path is None :
    print("Cannot Visulization Path")
    exit()

joint_trajectory = []
eef_poses = []
resolution = 0.5
for step, joint in enumerate(joint_path):
    if step % (1/resolution) == 0 or step == len(joint_path)-1:
        transformations = robot.forward_kin(joint)
        joint_trajectory.append(transformations)
        eef_poses.append(transformations[robot.eef_name].pos)

plt.plot_animation(
    robot,
    joint_trajectory, 
    fig, 
    ax,
    eef_poses=eef_poses,
    obstacles=obs,
    visible_obstacles=True,
    visible_collision=True, 
    interval=1, 
    repeat=True)

