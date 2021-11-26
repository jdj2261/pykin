import numpy as np
import sys, os

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)
from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/sawyer/sawyer.urdf'

robot = SingleArm(file_path, Transform(
    rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
robot.setup_link_name(eef_name="sawyer_right_hand")

target_thetas = [0, np.pi/2, 0, 0, 0, 0, 0, 0]
robot_transformations = robot.forward_kin(target_thetas)

_, ax = plt.init_3d_figure("FK")
plt.plot_robot(robot,
               ax=ax, 
               transformations=robot_transformations,
               visible_collision=True)
plt.show_figure()
