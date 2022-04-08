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


file_path = '../../../asset/urdf/panda/panda.urdf'
mesh_path = pykin_path+"/asset/urdf/panda/"
yaml_path = '../../../asset/config/panda_init_params.yaml'

with open(yaml_path) as f:
    controller_config = yaml.safe_load(f)

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0], pos=[0, 0, 0]))
robot.setup_link_name("panda_link_0", "panda_right_hand")

##################################################################
init_qpos = controller_config["init_qpos"]
init_fk = robot.forward_kin(init_qpos)
goal_eef_pose = controller_config["goal_pose"]

robot_c_manager = CollisionManager(mesh_path)
robot_c_manager.setup_robot_collision(robot, init_fk, "visual")
robot_c_manager.show_collision_info()

milk_path = pykin_path+"/asset/objects/meshes/milk.stl"
milk_mesh = trimesh.load_mesh(milk_path)

objs = ObjectManager()
obj_c_manager = CollisionManager(milk_path)
for i in range(6):
    name = "milk_" + str(i)
    obs_pos = [0.5, -0.2+i*0.1, 0.3]

    obj_c_manager.add_object(name, gtype="mesh", gparam=milk_mesh, h_mat=Transform(pos=obs_pos).h_mat)
    objs(name=name, gtype="mesh", gparam=milk_mesh, h_mat=Transform(pos=obs_pos).h_mat)
##################################################################

# objs.remove_object("milk_1")
planner = RRTStarPlanner(
    robot=robot,
    delta_distance=0.1,
    epsilon=0.2, 
    gamma_RRT_star=3,
    dimension=7
)
planner.run(
    cur_q=init_qpos, 
    goal_pose=goal_eef_pose,
    robot_col_manager=robot_c_manager,
    object_col_manager=obj_c_manager,
    max_iter=1000)

print(planner.tree.nodes[planner.goal_node][planner.COST])

interpolated_path = planner.get_joint_path(n_step=10)

if not interpolated_path:
    print("Cannot Visulization Path")
    exit()

joint_trajectory = []
eef_poses = []

# print(planner._cur_qpos, interpolated_path[0])

for step, joint in enumerate(interpolated_path):
    fk = robot.forward_kin(joint)
    joint_trajectory.append(fk)
    eef_poses.append(fk[robot.eef_name].pos)

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi= 100)
plt.plot_animation(
    robot,
    joint_trajectory, 
    fig, 
    ax,
    eef_poses=eef_poses,
    objects=objs,
    visible_objects=True,
    visible_collision=True, 
    interval=1, 
    repeat=True)