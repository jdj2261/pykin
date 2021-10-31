import numpy as np
import sys, os
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

from pykin.robots.single_arm import SingleArm
# from pykin.planners.cartesian_planner import CartesianPlanner
from pykin.kinematics.transform import Transform
from pykin.utils.obstacle_utils import Obstacle
# from pykin.utils.kin_utils import get_linear_interpoation
from pykin.utils import plot_utils as plt
import pykin.utils.transform_utils as t_utils
import pykin.utils.kin_utils as k_utils
import pykin.kinematics.jacobian as jac


file_path = '../../asset/urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name("base", "right_l6")

init_joints = [0, 0, 0, 0, 0, 0, 0, 0]
cur_fk = robot.forward_kin(init_joints)

target_joints = [0, np.pi/3, np.pi/3, np.pi/3, -np.pi/3, 0, 0, 0]
goal_transformations = robot.forward_kin(target_joints)

init_eef_pose = robot.compute_eef_pose(cur_fk)
goal_eef_pose = robot.compute_eef_pose(goal_transformations)

ik_init = robot.inverse_kin(np.random.randn(7), init_eef_pose)
ik_goal = robot.inverse_kin(np.random.randn(7), goal_eef_pose)

init_T = t_utils.get_homogeneous_matrix(init_eef_pose[:3], init_eef_pose[3:])
goal_T = t_utils.get_homogeneous_matrix(goal_eef_pose[:3], goal_eef_pose[3:])

cur_T = init_T
# cur_fk = init_transformations
current_joints = ik_init
# current_joints = np.random.randn(7)

n_step = 100
lamb = 0.5
dof = len(current_joints)
joints = []
cur_fk = robot.forward_kin(np.hstack((np.zeros(1),current_joints)))
goal_fk = robot.forward_kin(np.hstack((np.zeros(1),ik_goal)))
goal_eef_pose = robot.compute_eef_pose(goal_fk)

joints.append(np.hstack((np.zeros(1),current_joints)))

trajectory = []
for step in range(1, n_step+1):

    delta_t = step/n_step
    
    p = t_utils.get_linear_interpoation(init_eef_pose[:3], goal_eef_pose[:3], delta_t)
    o = t_utils.get_quaternion_slerp(init_eef_pose[3:], goal_eef_pose[3:], delta_t)
    tar_T = t_utils.get_homogeneous_matrix(p, o)

    err_pose = k_utils.calc_pose_error(tar_T, cur_T, float(1e-15))

    dX = err_pose

    J = jac.calc_jacobian(robot.desired_frames, cur_fk, dof)
    Jh = np.dot(np.linalg.inv(np.dot(J.T, J) + lamb*np.identity(dof)), J.T)
    # print(Jh)
    dq = lamb * np.dot(Jh, dX)
   
    print(np.linalg.norm(err_pose), dq)
    # print(current_joints, dq)
    current_joints = np.array([(current_joints[i] + dq[i]) for i in range(dof)]).reshape(7,)
    
    if step % 5 == 0:
        trajectory.append(p)
        joints.append(np.hstack((np.zeros(1),current_joints)))

    cur_fk = robot.forward_kin(np.hstack((np.zeros(1),current_joints)))
    cur_T = cur_fk[robot.eef_name].homogeneous_matrix

joint_trajectory = []
for joint in joints:
    transformations = robot.forward_kin([0]+joint)
    joint_trajectory.append(transformations)
print(joint_trajectory[-1][robot.eef_name])
fig, ax = plt.init_3d_figure(figsize=(12,6), dpi= 100)

plt.plot_animation(
    robot,
    joint_trajectory,
    fig=fig, 
    ax=ax,
    eef_poses=trajectory,
    obstacles=[],
    visible_obstacles=False,
    visible_collision=True, 
    visible_text=False,
    visible_scatter=False,
    interval=1, 
    repeat=False,
    result=None)

