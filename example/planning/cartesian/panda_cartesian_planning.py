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
from pykin.utils.collision_utils import apply_robot_to_collision_manager, apply_robot_to_scene

help_str = "python panda_cartesian_planning.py"\
            " --timesteps 500 --damping 0.03 --resolution 0.2 --pos-sensitivity 0.03"

parser = argparse.ArgumentParser(usage=help_str)
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--damping", type=float, default=0.03)
parser.add_argument("--resolution", type=float, default=0.05)
parser.add_argument("--pos-sensitivity", type=float, default=0.04)
args = parser.parse_args()

file_path = pykin_path+'asset/urdf/panda/panda.urdf'
mesh_path = pykin_path+"/asset/urdf/panda/"
yaml_fpath = pykin_path+'/asset/config/panda_init_params.yaml'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("panda_link0", "panda_link7")

with open(yaml_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]
fk = robot.forward_kin(np.array(init_qpos))

init_eef_pose = robot.get_eef_pose(fk)
goal_eef_pose = controller_config["goal_pos"]
##################################################################

c_manager = CollisionManager(mesh_path)
c_manager.filter_contact_names(robot, fk)
c_manager = apply_robot_to_collision_manager(c_manager, robot, fk)

task_plan = CartesianPlanner(
    robot, 
    self_collision_manager=c_manager,
    obstacle_collision_manager=None,
    n_step=args.timesteps,
    dimension=7)

joint_path, target_poses = task_plan.get_path_in_joinst_space(
    current_q=init_qpos,
    goal_pose=goal_eef_pose,
    resolution=args.resolution, 
    damping=args.damping,
    pos_sensitivity=args.pos_sensitivity)

if joint_path is None and target_poses is None:
    print("Cannot Visulization Path")
    exit()

joint_trajectory = []
for joint in joint_path:
    transformations = robot.forward_kin(joint)
    joint_trajectory.append(transformations)

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
    obstacles=[],
    repeat=True)
