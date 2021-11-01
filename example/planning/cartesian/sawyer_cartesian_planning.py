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
init_joints = [0, np.pi/5, 0, 0, 0, 0, 0, 0]
init_fk = robot.forward_kin(init_joints)

target_joints = [0, np.pi/6, 0, 0, 0, 0, 0, 0]
goal_transformations = robot.forward_kin(target_joints)

init_eef_pose = robot.compute_eef_pose(init_fk)
goal_eef_pose = robot.compute_eef_pose(goal_transformations)
##################################################################

ik_init = robot.inverse_kin(np.random.randn(7), init_eef_pose)
ik_goal = robot.inverse_kin(np.random.randn(7), goal_eef_pose)

current_joints = ik_init
cur_fk = robot.forward_kin(np.hstack((np.zeros(1),current_joints)))
goal_fk = robot.forward_kin(np.hstack((np.zeros(1),ik_goal)))

cur_eef_pose = robot.compute_eef_pose(cur_fk)
goal_eef_pose = robot.compute_eef_pose(goal_fk)

task_plan = CartesianPlanner(robot, obstacles=[])

waypoints = task_plan.get_path_in_cartesian_space(
    current_pose=cur_eef_pose,
    goal_pose=goal_eef_pose,
    n_step=5000
)

task_plan.setup_init_joint(current_joints, cur_fk)

joint_path, target_poses = task_plan.get_path_in_joinst_space(
    waypoints, 
    resolution=0.01, 
    damping=0.5)

joint_trajectory = []
for joint in joint_path:
    transformations = robot.forward_kin(np.concatenate((np.zeros(1),joint)))
    joint_trajectory.append(transformations)

print(goal_eef_pose, joint_trajectory[-1][robot.eef_name].pose)

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi= 100)

plt.plot_animation(
    robot,
    joint_trajectory,
    fig=fig, 
    ax=ax,
    visible_collision=True,
    eef_poses=target_poses,
    obstacles=[])
