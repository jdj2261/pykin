import numpy as np
import sys, os
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../../" )
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.utils import plot_utils as plt
from pykin.planners.cartesian_planner import CartesianPlanner

file_path = '../../../asset/urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("base", "right_l6")

##################################################################
init_joints = [0, -np.pi/4, 0, 0, np.pi/4, 0, 0, 0]
init_fk = robot.forward_kin(init_joints)

target_joints = [0, np.pi/4, np.pi/4, np.pi/4, 0, 0, 0, 0]
goal_transformations = robot.forward_kin(target_joints)

init_eef_pose = robot.get_eef_pose(init_fk)
goal_eef_pose = robot.get_eef_pose(goal_transformations)
##################################################################

task_plan = CartesianPlanner(
    robot, 
    obstacles=[],
    current_pose=init_eef_pose,
    goal_pose=goal_eef_pose,
    n_step=500,
    arm_type="right")

waypoints = task_plan.get_waypoints()

joint_path, target_poses = task_plan.get_path_in_joinst_space(
    waypoints, 
    epsilon=float(1e-6),
    resolution=0.1, 
    damping=0.5)

joint_trajectory = []
for joint in joint_path:
    transformations = robot.forward_kin(np.concatenate((np.zeros(1),joint)))
    joint_trajectory.append(transformations)

print(f"cur_pose : {joint_trajectory[-1][robot.eef_name].pose}")
print(f"target_pose : {target_poses[-1]}")

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi= 100)

plt.plot_animation(
    robot,
    joint_trajectory,
    fig=fig, 
    ax=ax,
    visible_collision=False,
    eef_poses=target_poses,
    obstacles=[],
    repeat=True)
