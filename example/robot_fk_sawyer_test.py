import sys
import os
import numpy as np
from pprint import pprint
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robot import Robot
from pykin.utils import plot_utils as plt

file_path = '../asset/urdf/sawyer/sawyer.urdf'

robot = Robot(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
print(robot)
robot.set_desired_frame("base", "right_wrist")

# sawyer_example
target_thetas = [np.pi/3, 0, 0, 0, 0, 0, 0, 0]
init_thetas = np.random.randn(8)
robot_transformations = robot.kin.forward_kinematics(target_thetas)

_, ax = plt.init_3d_figure("FK")
plt.plot_robot(robot,
               transformations=robot_transformations,
               ax=ax, 
               name="sawyer",
               visible_visual=False,
               visible_collision=True,
               mesh_path='../asset/urdf/sawyer/')

ax.legend()
plt.show_figure()
