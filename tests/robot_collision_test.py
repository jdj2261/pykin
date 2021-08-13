import sys
import os
import numpy as np
from pprint import pprint
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
sys.path.append(pykin_path)
from pykin.utils import plot as plt
from pykin.kinematics import transform as tf
from pykin.robot import Robot
from pykin import robot
from pykin.utils.shell_color import ShellColors as scolors
file_path = '../asset/urdf/baxter/baxter.urdf'
# file_path = '../asset/urdf/sawyer.urdf'

robot = Robot(file_path, tf.Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]), joint_safety=True)

robot.set_desired_tree("base", "left_wrist")
head_thetas = np.zeros(1)
if robot.joint_safety:
    left_arm_thetas = np.clip(np.random.randn(
        7), robot.joint_limits_lower[:7], robot.joint_limits_upper[:7])

left_arm_thetas = np.array([-np.pi/4, 0, 0, np.pi, 0, 0, 0])
left_arm_fk = robot.forward_kinematics(left_arm_thetas)
target_pos = left_arm_fk["left_wrist"].matrix()

_, ax = plt.init_3d_figure("Target")
plt.plot_robot(robot, left_arm_fk, ax, "Left arm", visible_collision=True,
               visible_mesh=True, mesh_path='../asset/urdf/baxter/')
ax.legend()
# plt.show_figure()
init_left_thetas = np.random.randn(7)

if robot.joint_safety:
    init_left_thetas = np.clip(np.random.randn(
        7), robot.joint_limits_lower, robot.joint_limits_upper)

target_l_pose = np.concatenate(
    (left_arm_fk["left_wrist"].pos, left_arm_fk["left_wrist"].rot))

ik_left_LM_result = robot.inverse_kinematics(
    init_left_thetas, target_l_pose, method="LM", maxIter=100)

result_fk_LM = robot.forward_kinematics(ik_left_LM_result)

# eps = float(1e-12)
# for i, (link, transform) in enumerate(left_arm_fk.items()):
#     cur_link = link
#     cur_position = transform.pos
#     cur_radius = float(robot.tree.links[cur_link].radius)
#     for j in range(i+1, len(left_arm_fk.items())):

#         if "left_lower" in cur_link and j == i+1:
#             continue

#         next_link = list(left_arm_fk.keys())[j]
#         next_position = list(left_arm_fk.values())[j].pos
#         next_radius = float(robot.tree.links[next_link].radius)

#         diff = np.zeros(3)
#         diff[0] = cur_position[0] - next_position[0]
#         diff[1] = cur_position[1] - next_position[1]
#         diff[2] = cur_position[2] - next_position[2]

#         distance = np.linalg.norm(diff)

#         if distance < eps:
#             continue

#         if distance > cur_radius + next_radius:
#             print(cur_link, next_link, cur_radius, next_radius, end=" ")
#             print(f"{scolors.OKCYAN}Not Collision{scolors.ENDC}")
#         else:
#             print(cur_link, next_link, distance, cur_radius, next_radius, end=" ")
#             print(f"{scolors.FAIL}Collision{scolors.ENDC}")
#         # print(cur_link, next_link, np.linalg.norm(diff))
#         # print(cur_link, next_link, cur_radius, next_radius)
#     print()

l_pose_new_LM = result_fk_LM["left_wrist"].matrix()

left_error_LM = np.linalg.norm(
    np.dot(l_pose_new_LM, np.linalg.inv(target_pos)) - np.mat(np.eye(4)))


# print(f"{scolors.OKCYAN}Target Left wrist Pose{scolors.ENDC}: \n{target_pos}")
# print(f"{scolors.OKCYAN}Current Left wrist Pose{scolors.ENDC}: \n{l_pose_new_LM}")
# print(f"\n{scolors.WARNING}LM Method Error: {scolors.ENDC}: {left_error_LM}")

# _, ax = plt.init_3d_figure("IK Result")
# plt.plot_robot(robot, result_fk_LM, ax, "Left arm", visible_collision=True,
#                visible_mesh=False, mesh_path='../asset/urdf/baxter/')
# ax.legend()
plt.show_figure()
