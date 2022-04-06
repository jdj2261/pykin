import numpy as np
import argparse
import sys, os
import yaml

parent_dir = os.path.dirname(os.getcwd())
pykin_path = parent_dir + "/../../"
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.utils import plot_utils as plt
from pykin.planners.cartesian_planner import CartesianPlanner
from pykin.collision.collision_manager import CollisionManager

help_str = "python iiwa14_cartesian_planning.py"\
            " --timesteps 500 --damping 0.03 --resolution 0.2 --pos-sensitivity 0.03"

parser = argparse.ArgumentParser(usage=help_str)
parser.add_argument("--timesteps", type=int, default=500)
parser.add_argument("--damping", type=float, default=0.03)
parser.add_argument("--resolution", type=float, default=0.2)
parser.add_argument("--pos-sensitivity", type=float, default=0.05)
args = parser.parse_args()

file_path = pykin_path+'/asset/urdf/iiwa14/iiwa14.urdf'
mesh_path = pykin_path+"/asset/urdf/iiwa14/"
yaml_path = pykin_path+'/asset/config/iiwa14_init_params.yaml'

with open(yaml_path) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0], pos=[0, 0, 0]))
robot.setup_link_name("iiwa14_link_0", "iiwa14_right_hand")

##################################################################
init_fk = robot.forward_kin(init_qpos)
init_eef_pose = robot.get_eef_pose(init_fk)
goal_eef_pose = controller_config["goal_pos"]
##################################################################

c_manager = CollisionManager(mesh_path)
c_manager.setup_robot_collision(robot, init_fk)

task_plan = CartesianPlanner(
    robot, 
    n_step=args.timesteps,
    dimension=7,
    damping=args.damping,
    pos_sensitivity=args.pos_sensitivity)

joint_path, target_poses = task_plan.get_joint_path(
    cur_q=init_qpos,
    goal_pose=goal_eef_pose,
    robot_col_manager=c_manager,
    object_col_manager=None,
    resolution=args.resolution)

if joint_path is None and target_poses is None:
    print("Cannot Visulization Path")
    exit()

joint_trajectory = []
for joint in joint_path:
    fk = robot.forward_kin(joint)
    joint_trajectory.append(fk)

print(f"Computed Goal Position : {joint_trajectory[-1][robot.eef_name].pose}")
print(f"Desired Goal position : {target_poses[-1]}")

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi= 100)

plt.plot_animation(
    robot,
    joint_trajectory,
    fig=fig, 
    ax=ax,
    visible_collision=True,
    eef_poses=target_poses,
    objects=[],
    repeat=True)
