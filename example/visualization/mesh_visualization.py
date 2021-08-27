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
# If visible_visual is True, visualize mesh
# and you have to input mesh_path
plt.plot_robot(robot, 
               transformations=robot.transformations,
               ax=ax, 
               name=robot.robot_name,
               visible_visual=True, 
               visible_collision=False, 
               mesh_path='../../asset/urdf/'+robot.robot_name+'/')
"""
The mesh file doesn't use matplotlib, 
so it's okay to comment out the line below.
"""
# ax.legend()
# plt.show_figure()