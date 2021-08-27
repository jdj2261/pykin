import sys

from pykin.robot import Robot
from pykin.utils import plot_utils as plt

file_path = '../../asset/urdf/baxter/baxter.urdf'

if len(sys.argv) > 1:
    robot_name = sys.argv[1]
    file_path = '../../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'
robot = Robot(file_path)

fig, ax = plt.init_3d_figure("URDF")

"""
Only baxter and sawyer robots can see collisions.
It is not visible unless sphere, cylinder, and box are defined in collision/geometry tags in urdf.
"""
# If visible_collision is True, visualize collision
plt.plot_robot(robot, 
               transformations=robot.transformations,
               ax=ax, 
               name=robot.robot_name,
               visible_visual=False, 
               visible_collision=True, 
               mesh_path='../asset/urdf/baxter/')
ax.legend()
plt.show_figure()