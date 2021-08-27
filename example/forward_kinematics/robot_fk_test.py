import numpy as np

from pykin.robot import Robot
from pykin.kinematics.transform import Transform
from pykin.utils import plot_utils as plt
from pykin.utils.kin_utils import ShellColors as sc


# baxter_example
file_path = '../../asset/urdf/baxter/baxter.urdf'
robot = Robot(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))

head_thetas = [0.0]
right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]

thetas = head_thetas + right_arm_thetas + left_arm_thetas
fk = robot.kin.forward_kinematics(thetas)

"""
If you want to know transformations of all links,
you don't have to write set_desired_tree and desired_tree.
"""
for link, transform in fk.items():
    print(f"{sc.HEADER}{link}{sc.ENDC}, {transform.rot}, {transform.pos}")

"""
If you want to know transformation of desired link,
you must write set_desried_frame.
"""
robot.set_desired_frame("base", "left_wrist")
robot_transformations = robot.kin.forward_kinematics(left_arm_thetas)
for link, T in robot_transformations.items():
    print(f"link: {link}, pose:{np.concatenate((T.pos, T.rot))} ")

_, ax = plt.init_3d_figure()
plt.plot_robot(robot,
               transformations=robot_transformations,
               ax=ax,
               name="baxter_left_arm"
               )
              
ax.legend()
plt.show_figure()

"""
If you want to reknow transformations of all links,
you must write reset_desired_frames.
"""
robot.reset_desired_frames()
fk = robot.kin.forward_kinematics(thetas)

"""
If you want to see baxter robot plot,
you must write "baxter" in plot_robot method
Otherwise, you can't see correct result plot
"""
_, ax = plt.init_3d_figure()
plt.plot_robot(robot,
               transformations=fk,
               ax=ax, 
               name="baxter", 
               visible_collision=True)
ax.legend()
plt.show_figure()
