import numpy as np

from pykin.robots.bimanual import Bimanual
from pykin.kinematics.transform import Transform
from pykin.utils import plot_utils as p_utils

file_path = 'urdf/baxter/baxter.urdf'

robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

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

target_fk = robot.forward_kin(thetas)
robot.set_transform(thetas)
_, ax = p_utils.init_3d_figure("Target Pose")
p_utils.plot_robot(robot=robot, 
               ax=ax,
               only_visible_geom=True)

#################################################################################
#                                Inverse Kinematics                             #
#################################################################################
init_thetas = np.random.randn(7)
target_pose = { "right": robot.compute_eef_pose(target_fk)["right"], 
                "left" : robot.compute_eef_pose(target_fk)["left"]}

ik_LM_result = robot.inverse_kin(
    init_thetas, 
    target_pose, 
    method="LM", 
    max_iter=100)

ik_NR_result = robot.inverse_kin(
    init_thetas, 
    target_pose, 
    method="NR", 
    max_iter=100)

thetas_LM = np.concatenate((head_thetas, ik_LM_result["right"], ik_LM_result["left"]))
robot.set_transform(thetas_LM)
_, ax = p_utils.init_3d_figure("LM IK Result")
p_utils.plot_robot(robot=robot, 
               ax=ax,
               only_visible_geom=True)

thetas_NR = np.concatenate((head_thetas, ik_NR_result["right"], ik_NR_result["left"]))
robot.set_transform(thetas_NR)
_, ax = p_utils.init_3d_figure("NR IK Result")
p_utils.plot_robot(robot=robot, 
               ax=ax,
               geom="visual",
               only_visible_geom=True)
p_utils.show_figure()