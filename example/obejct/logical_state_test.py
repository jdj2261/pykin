import numpy as np
import sys, os
import trimesh

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.objects.gripper import GripperManager
from pykin.objects.object_manager import ObjectManager
from pykin.objects.object_info import ObjectInfo
import pykin.utils.plot_utils as plt

fig, ax = plt.init_3d_figure(figsize=(10,6), dpi=120)

#################################### Objects ####################################
red_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77]))
blue_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77 + 0.06]))
green_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77 + 0.12]))

support_box_pose = Transform(pos=np.array([0.6, -0.2, 0.77]), rot=np.array([0, np.pi/2, 0]))
table_pose = Transform(pos=np.array([0.4, 0.24, 0.0]))

cube_mesh = trimesh.load(pykin_path+'/asset/objects/meshes/ben_cube.stl')
box_goal_mesh = trimesh.load(pykin_path+'/asset/objects/meshes/box_goal.stl')
table_mesh = trimesh.load(pykin_path+'/asset/objects/meshes/custom_table.stl')

cube_mesh.apply_scale(0.06)
box_goal_mesh.apply_scale(0.001)
table_mesh.apply_scale(0.01)

object_mngr = ObjectManager()
red_box = ObjectInfo(name="red_box", gtype="mesh", color=[1, 0, 0], gparam=cube_mesh, h_mat=red_box_pose.h_mat)
blue_box = ObjectInfo(name="blue_box", gtype="mesh", color=[0, 0, 1], gparam=cube_mesh, h_mat=blue_box_pose.h_mat)
green_box = ObjectInfo(name="green_box", gtype="mesh", color=[0, 1, 0], gparam=cube_mesh, h_mat=green_box_pose.h_mat)
goal_box = ObjectInfo(name="box", gtype="mesh", color=[1, 0, 1], gparam=box_goal_mesh, h_mat=support_box_pose.h_mat)
table = ObjectInfo(name="table", gtype="mesh", color=[0.39, 0.263, 0.129], gparam=table_mesh, h_mat=table_pose.h_mat)

#################################### Gripper ####################################
file_path = '../../asset/urdf/panda/panda.urdf'
mesh_path = pykin_path+"/asset/urdf/panda/"
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))

configures = {}
configures["gripper_names"] = ["right_gripper", "leftfinger", "rightfinger"]
configures["gripper_max_width"] = 0.08
configures["gripper_max_depth"] = 0.035
configures["tcp_position"] = np.array([0, 0, 0.097])

gripper = GripperManager(robot, mesh_path, **configures)

#################################### Logical state ####################################
red_box.logical_state = {"on" : table}
blue_box.logical_state = {"on" : red_box}
green_box.logical_state = {"on" : blue_box}
goal_box.logical_state = {"on" : table, "goal": True}
table.logical_state = {"static" : True}
gripper.logical_state = {"holding": None}

object_mngr.add_object(obj_info=table)
object_mngr.add_object(obj_info=goal_box)
object_mngr.add_object(obj_info=red_box)
object_mngr.add_object(obj_info=blue_box)
object_mngr.add_object(obj_info=green_box)
object_mngr.add_gripper(gripper)

print(object_mngr.get_logical_all_states())

object_mngr.gripper_manager.set_eef_transform(blue_box_pose.h_mat)

# object_mngr.add_object(name="red_box", gtype="mesh", color=[1, 0, 0], gparam=cube_mesh, h_mat=red_box_pose.h_mat, for_grasp=True)
# object_mngr.add_object(name="blue_box", gtype="mesh", color=[0, 0, 1], gparam=cube_mesh, h_mat=blue_box_pose.h_mat, for_grasp=True)
# object_mngr.add_object(name="green_box", gtype="mesh", color=[0, 1, 0], gparam=cube_mesh, h_mat=green_box_pose.h_mat, for_grasp=True)
# object_mngr.add_object(name="box", gtype="mesh", color=[1, 0, 1], gparam=box_goal_mesh, h_mat=support_box_pose.h_mat, for_support=True)
# object_mngr.add_object(name="table", gtype="mesh", color=[0.39, 0.263, 0.129], gparam=table_mesh, h_mat=table_pose.h_mat)

# object_mngr.objects["red_box"].logical_state

object_mngr.visualize_all_objects(ax, 0.3)
plt.show_figure()