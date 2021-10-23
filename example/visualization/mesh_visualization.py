import sys

file_path = '../../asset/urdf/baxter/baxter.urdf'

if len(sys.argv) > 1:
    robot_name = sys.argv[1]
    file_path = '../../asset/urdf/' + robot_name + '/' + robot_name + '.urdf'

if "baxter" in file_path:
    from pykin.robots.bimanual import Bimanual
    robot = Bimanual(file_path)
else:
    from pykin.robots.single_arm import SingleArm
    robot = SingleArm(file_path)
from pykin.utils import plot_utils as plt
fig, ax = plt.init_3d_figure("URDF")

"""
Only baxter and sawyer robots can see collisions.
It is not visible unless sphere, cylinder, and box are defined in collision/geometry tags in urdf.
"""
# If visible_visual is True, visualize mesh
# and you have to input mesh_path
plt.plot_robot(robot, 
               ax=ax, 
               visible_visual=True, 
               visible_collision=False, 
               mesh_path='../../asset/urdf/'+robot.robot_name+'/')
"""
The mesh file doesn't use matplotlib, 
so it's okay to comment out the line below.
"""

plt.show_figure()