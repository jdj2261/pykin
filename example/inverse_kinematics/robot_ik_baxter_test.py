import numpy as np

from pykin.robots.bimanual import Bimanual
from pykin.kinematics.transform import Transform
from pykin.utils import plot_utils as plt
from pykin.utils.transform_utils import compute_pose_error


file_path = '../../asset/urdf/baxter/baxter.urdf'

robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

visible_collision = True

# set target joints angle
head_thetas =  np.zeros(1)
right_arm_thetas = np.array([-np.pi/4 , 0, 0, 0, 0 , 0 ,0])
left_arm_thetas = np.array([np.pi/4 , 0, 0, 0, 0 , 0 ,0])

thetas = np.concatenate((head_thetas ,right_arm_thetas ,left_arm_thetas))

robot.setup_link_name("base", "right_wrist")
robot.setup_link_name("base", "left_wrist")

#################################################################################
#                                Set target pose                                #
#################################################################################
target_transformations = robot.forward_kin(thetas)
_, ax = plt.init_3d_figure("Target Pose")
plt.plot_robot(robot, 
               ax=ax,
               transformations=target_transformations, 
               visible_collision=visible_collision)

#################################################################################
#                                Inverse Kinematics                             #
#################################################################################
init_thetas = np.random.randn(7)
target_pose = { "right": robot.get_eef_pose(target_transformations)["right"], 
                "left" : robot.get_eef_pose(target_transformations)["left"]}

ik_LM_result = robot.inverse_kin(
    init_thetas, 
    target_pose, 
    method="LM", 
    maxIter=100)

ik_NR_result = robot.inverse_kin(
    init_thetas, 
    target_pose, 
    method="NR", 
    maxIter=100)

thetas_LM = np.concatenate((head_thetas, ik_LM_result["right"], ik_LM_result["left"]))
result_fk_LM = robot.forward_kin(thetas_LM)
_, ax = plt.init_3d_figure("LM IK Result")
plt.plot_robot(robot, ax, result_fk_LM, 
               visible_collision=visible_collision)

thetas_NR = np.concatenate((head_thetas, ik_NR_result["right"], ik_NR_result["left"]))
result_fk_NR = robot.forward_kin(thetas_NR)
_, ax = plt.init_3d_figure("NR IK Result")
plt.plot_robot(robot, ax, result_fk_NR,
               visible_collision=visible_collision)

err = {}
for arm in robot.arm_type:
    err[arm+"_NR_error"] = compute_pose_error(
        target_transformations[robot.eef_name[arm]].h_mat,
        result_fk_NR[robot.eef_name[arm]].h_mat)

    err[arm+"_LM_error"] = compute_pose_error(
        target_transformations[robot.eef_name[arm]].h_mat,
        result_fk_LM[robot.eef_name[arm]].h_mat)

print(err)

plt.show_figure()
