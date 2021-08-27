import numpy as np

from pykin.robot import Robot
from pykin.kinematics.transform import Transform
from pykin.utils import plot_utils as plt
from pykin.utils.kin_utils import ShellColors as scolors

file_path = '../asset/urdf/baxter/baxter.urdf'

robot = Robot(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

visible_collision = True
visible_visual = True

# set target joints angle
head_thetas =  np.zeros(1)
right_arm_thetas = np.random.randn(7)
left_arm_thetas = np.random.randn(7)

print(f"{scolors.OKBLUE}Target Right arm Angle{scolors.ENDC}: \n{right_arm_thetas}")
print(f"{scolors.OKBLUE}Target Left arm Angle{scolors.ENDC}: \n{left_arm_thetas}")
print()
#################################################################################
#                                Forward Kinematics                             #
#################################################################################
thetas = np.concatenate((head_thetas ,right_arm_thetas ,left_arm_thetas))

# caculuate FK
fk = robot.kin.forward_kinematics(thetas)

# show FK graph
_, ax = plt.init_3d_figure("FK Result")
plt.plot_robot(robot, fk, ax, "baxter", 
                visible_visual=visible_visual, 
                visible_collision=visible_collision )
ax.legend()
# plt.show_figure()

#################################################################################
#                                Inverse Kinematics                             #
#################################################################################
init_right_thetas = np.random.randn(7)
init_left_thetas = np.random.randn(7)

# Set desired frame (root, end)
robot.set_desired_frame("base", "right_wrist")
right_arm_fk = robot.kin.forward_kinematics(right_arm_thetas)
target_r_pose = np.concatenate((right_arm_fk["right_wrist"].pos, right_arm_fk["right_wrist"].rot))

# Right's arm IK solution by LM
ik_right_LM_result, _ = robot.kin.inverse_kinematics(init_right_thetas, target_r_pose, method="LM", maxIter=100)

# Right's arm IK solution by NR
ik_right_NR_result, _ = robot.kin.inverse_kinematics(init_right_thetas, target_r_pose, method="NR", maxIter=100)

# Set desired link (root, end)
robot.set_desired_frame("base", "left_wrist")
left_arm_fk = robot.kin.forward_kinematics(left_arm_thetas)
target_l_pose = np.concatenate((left_arm_fk["left_wrist"].pos, left_arm_fk["left_wrist"].rot))

# Left's arm IK solution by LM
ik_left_LM_result, _= robot.kin.inverse_kinematics(init_left_thetas, target_l_pose, method="LM", maxIter=100)

# Left's arm IK solution by NR
ik_left_NR_result, _ = robot.kin.inverse_kinematics(init_left_thetas, target_l_pose, method="NR", maxIter=100)

print(f"\n{scolors.HEADER}LM Method: Current Right arm Angles{scolors.ENDC}: \n{ik_right_LM_result}")
print(f"{scolors.HEADER}LM Method: Current Left arm Angles{scolors.ENDC}: \n{ik_left_LM_result}")

print(f"\n{scolors.HEADER}NR Method: Current Right arm Angles{scolors.ENDC}: \n{ik_right_NR_result}")
print(f"{scolors.HEADER}NR Method: Current Left arm Angles{scolors.ENDC}: \n{ik_left_NR_result}")


thetas_LM = np.concatenate((head_thetas, ik_right_LM_result, ik_left_LM_result))
robot.reset_desired_frames()
result_fk_LM = robot.kin.forward_kinematics(thetas_LM)

_, ax = plt.init_3d_figure("LM IK Result")
plt.plot_robot(robot, result_fk_LM, ax,
               "baxter",
               visible_visual=visible_visual, 
               visible_collision=visible_collision)
ax.legend()


goal_r_pose_LM = np.concatenate((result_fk_LM["right_wrist"].pos, result_fk_LM["right_wrist"].rot))
goal_l_pose_LM = np.concatenate((result_fk_LM["left_wrist"].pos, result_fk_LM["left_wrist"].rot))

r_pose = right_arm_fk["right_wrist"].matrix()
l_pose = left_arm_fk["left_wrist"].matrix()

r_pose_new_LM = result_fk_LM["right_wrist"].matrix()
l_pose_new_LM = result_fk_LM["left_wrist"].matrix()

thetas_NR = np.concatenate((head_thetas, ik_right_NR_result, ik_left_NR_result))
robot.reset_desired_frames()
result_fk_NR = robot.kin.forward_kinematics(thetas_NR)

_, ax = plt.init_3d_figure("NR IK Result")
plt.plot_robot(robot, result_fk_NR, ax,
               "baxter",
               visible_visual=visible_visual, 
               visible_collision=visible_collision)
ax.legend()

goal_r_pose_NR = np.concatenate((result_fk_NR["right_wrist"].pos, result_fk_NR["right_wrist"].rot))
goal_l_pose_NR = np.concatenate((result_fk_NR["left_wrist"].pos, result_fk_NR["left_wrist"].rot))

r_pose_new_NR = result_fk_NR["right_wrist"].matrix()
l_pose_new_NR = result_fk_NR["left_wrist"].matrix()


print(f"\n{scolors.OKGREEN}Target Right wrist Pose{scolors.ENDC}: \n{r_pose}")
print(f"{scolors.OKGREEN}LM Method: Current Right wrist Pose{scolors.ENDC}: \n{r_pose_new_LM}")
print(f"{scolors.OKGREEN}NR Method: Current Right wrist Pose{scolors.ENDC}: \n{r_pose_new_NR}")
print(f"\n{scolors.OKCYAN}Target Left wrist Pose{scolors.ENDC}: \n{l_pose}")
print(f"{scolors.OKCYAN}LM Method: Current Left wrist Pose{scolors.ENDC}: \n{l_pose_new_LM}")
print(f"{scolors.OKCYAN}NR Method: Current Left wrist Pose{scolors.ENDC}: \n{l_pose_new_NR}")

right_error_LM = np.linalg.norm(np.dot(r_pose_new_LM, np.linalg.inv(r_pose)) - np.mat(np.eye(4)))
left_error_LM = np.linalg.norm(np.dot(l_pose_new_LM, np.linalg.inv(l_pose)) - np.mat(np.eye(4)))

right_error_NR = np.linalg.norm(np.dot(r_pose_new_NR, np.linalg.inv(r_pose)) - np.mat(np.eye(4)))
left_error_NR = np.linalg.norm(np.dot(l_pose_new_NR, np.linalg.inv(l_pose)) - np.mat(np.eye(4)))

print(f"\n{scolors.WARNING}LM Method Error: {scolors.ENDC}: {right_error_LM}, {left_error_LM}")
print(f"{scolors.WARNING}NR Method Error: {scolors.ENDC}: {right_error_NR}, {left_error_NR}")

plt.show_figure()
