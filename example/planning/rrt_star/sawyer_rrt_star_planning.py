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
from pykin.objects.object_manager import ObjectManager
from pykin.utils import plot_utils as plt

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi= 100)

file_path = '../../../asset/urdf/sawyer/sawyer.urdf'
mesh_path = pykin_path+"/asset/urdf/sawyer/"
yaml_path = '../../../asset/config/sawyer_init_params.yaml'

with open(yaml_path) as f:
    controller_config = yaml.safe_load(f)

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0], pos=[0, 0, 0]))
robot.setup_link_name("sawyer_base", "sawyer_right_hand")

##################################################################
init_qpos = controller_config["init_qpos"]

init_fk = robot.forward_kin(np.concatenate((np.zeros(1), init_qpos)))
init_eef_pose = robot.get_eef_pose(init_fk)
goal_eef_pose = controller_config["goal_pose"]

c_manager = CollisionManager(mesh_path)
c_manager.setup_robot_collision(robot, init_fk)

milk_path = pykin_path+"/asset/objects/meshes/milk.stl"
milk_mesh = trimesh.load_mesh(milk_path)

obs = ObjectManager()
o_manager = CollisionManager(milk_path)
for i in range(9):
    name = "miik_" + str(i)
    if i < 3:
        obs_pos = [0.3, -0.5 + i * 0.5, 0.3] 
    elif 3 <= i < 6:
        obs_pos = [0.3, -0.5 + (i-3) * 0.5, 0.9] 
    else:
        obs_pos = [0.3, -0.5 + (i-6) * 0.5, -0.3]

    o_manager.add_object(name, gtype="mesh", gparam=milk_mesh, h_mat=Transform(pos=obs_pos).h_mat)
    obs(name=name, gtype="mesh", gparam=milk_mesh, h_mat=Transform(pos=obs_pos).h_mat)
##################################################################

planner = RRTStarPlanner(
    robot=robot,
    delta_distance=0.05,
    epsilon=0.2, 
    gamma_RRT_star=0.5,
    dimension=7
)

planner.run(
    cur_q=init_qpos, 
    goal_pose=goal_eef_pose,
    robot_col_manager=c_manager,
    object_col_manager=o_manager,
    max_iter=1000)

interpolated_path = planner.get_joint_path(n_step=5)

joint_trajectory = []
eef_poses = []
for step, joint in enumerate(interpolated_path):
    fk = robot.forward_kin(np.concatenate((np.zeros(1),joint)))
    joint_trajectory.append(fk)
    eef_poses.append(fk[robot.eef_name].pos)

plt.plot_animation(
    robot,
    joint_trajectory, 
    fig, 
    ax,
    eef_poses=eef_poses,
    objects=obs,
    visible_objects=True,
    visible_collision=True, 
    interval=1, 
    repeat=True)