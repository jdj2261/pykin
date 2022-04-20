import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(pykin_path)

from pykin.robots.bimanual import Bimanual
from pykin.kinematics.transform import Transform
from pykin.utils import plot_utils as plt

file_path = '../../../asset/urdf/baxter/baxter.urdf'

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
_, ax = plt.init_3d_figure("Target Pose")
plt.plot_robot(robot=robot, 
               ax=ax,
               visible_geom=True)

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
_, ax = plt.init_3d_figure("LM IK Result")
plt.plot_robot(robot=robot, 
               ax=ax,
               visible_geom=True)

thetas_NR = np.concatenate((head_thetas, ik_NR_result["right"], ik_NR_result["left"]))
robot.set_transform(thetas_NR)
_, ax = plt.init_3d_figure("NR IK Result")
plt.plot_robot(robot=robot, 
               ax=ax,
               visible_geom=True)
plt.show_figure()