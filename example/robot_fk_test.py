import sys
import os
import numpy as np
from pprint import pprint
# pykin_path = os.path.abspath(os.path.dirname(__file__)+"../")
# sys.path.append(pykin_path)

from pykin import robot
from pykin.robot import Robot
from pykin.kinematics.transform import Transform
from pykin.utils import plot as plt
from pykin.utils.shell_color import ShellColors as sc


# baxter_example
file_path = '../asset/urdf/baxter/baxter.urdf'
robot = Robot(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

head_thetas = [0.0]
right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]

thetas = head_thetas + right_arm_thetas + left_arm_thetas
fk = robot.forward_kinematics(thetas)


"""
If you want to know transformations of all links,
you don't have to write get_desired_tree and desired_frame.
"""
for link, transform in fk.items():
    print(f"{sc.HEADER}{link}{sc.ENDC}, {transform.rot}, {transform.pos}")

"""
If you want to know transformation of desired link,
you must write get_desried_tree.
"""
robot.set_desired_tree("base", "left_wrist")
fk = robot.forward_kinematics(left_arm_thetas)
for link, T in fk.items():
    print(f"link: {link}, pose:{np.concatenate((T.pos, T.rot))} ")

_, ax = plt.init_3d_figure()
plt.plot_robot(robot, fk, ax, "left_wrist", visible_collision=True)
ax.legend()
plt.show_figure()

"""
If you want to reknow transformations of all links,
you must write desired_frame.
"""
robot.desired_frame = None
fk = robot.forward_kinematics(thetas)
pprint(fk)

"""
If you want to see baxter robot plot,
you must write "baxter" in plot_robot method
Otherwise, you can't see correct result plot
"""
_, ax = plt.init_3d_figure()
plt.plot_robot(robot, fk, ax, "baxter", visible_collision=True)
ax.legend()
plt.show_figure()
