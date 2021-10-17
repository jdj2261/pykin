import os, sys
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)

import numpy as np

from pykin.robots.bimanual import Bimanual
from pykin.kinematics.transform import Transform
from pykin.utils import plot_utils as plt
from pykin.utils.kin_utils import ShellColors as scolors

file_path = '../../asset/urdf/baxter/baxter.urdf'

robot = Bimanual(
    file_path, 
    Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]),
)

robot.setup_link_name("base", "right_wrist")
robot.setup_link_name("base", "left_wrist")

head_thetas =  np.zeros(1)
right_arm_thetas = np.random.randn(7)
left_arm_thetas = np.random.randn(7)

thetas = np.concatenate((head_thetas ,right_arm_thetas ,left_arm_thetas))

# # caculuate FK
fk = robot.forward_kin(thetas)

_, ax = plt.init_3d_figure("FK")
plt.plot_robot(robot, ax)

init_right_thetas = np.random.randn(7)

target_pose = { "right": robot.eef_pose["right"], 
                "left" : robot.eef_pose["left"]}

print(target_pose)
ik_right_LM_result = robot.inverse_kin(init_right_thetas, target_pose, method="LM", maxIter=100)
thetas = np.concatenate((head_thetas ,ik_right_LM_result["right"] ,ik_right_LM_result["left"]))

result_fk = robot.forward_kin(thetas)

print(ik_right_LM_result)
_, ax = plt.init_3d_figure("IK")
plt.plot_robot(
    robot, 
    ax)

plt.show_figure()
