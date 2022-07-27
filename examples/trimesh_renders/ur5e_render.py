import numpy as np
import trimesh
import os
import yaml

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.kin_utils import apply_robot_to_scene

current_file_path = os.path.abspath(os.path.dirname(__file__))

from pykin.utils import plot_utils as p_utils


urdf_path = 'urdf/ur5e/ur5e_with_robotiq140.urdf'
robot = SingleArm(
    urdf_path, 
    Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]),
    has_gripper=True,
    gripper_name="robotiq140_gripper")
robot.setup_link_name("ur5e_base_link", "ur5e_right_hand")

c_manager = CollisionManager(is_robot=True)
c_manager.setup_robot_collision(robot, geom="visual")
c_manager.show_collision_info()

custom_fpath = current_file_path + '/../../pykin/assets/config/ur5e_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]
robot.set_transform(np.array(init_qpos))


for link, info in robot.info[c_manager.geom].items():
    if link in c_manager._objs:
        c_manager.set_transform(name=link, h_mat=info[3])
        
milk_path = current_file_path + "/../../pykin/assets/objects/meshes/milk.stl"
test_mesh = trimesh.load_mesh(milk_path)

o_manager = CollisionManager()
o_manager.add_object("milk1", gtype="mesh", gparam=test_mesh, h_mat=Transform(pos=[0.1, 0, 0.4]).h_mat)
o_manager.add_object("milk2", gtype="mesh", gparam=test_mesh, h_mat=Transform(pos=[0.4, 0, 0.4]).h_mat)

scene = trimesh.Scene()
scene = apply_robot_to_scene(trimesh_scene=scene, robot=robot, geom=c_manager.geom)
scene.set_camera(np.array([np.pi/2, 0, np.pi/2]), 5, resolution=(1024, 512))

scene.add_geometry(test_mesh, node_name="milk1", transform=Transform(pos=[0.1, 0, 0.4]).h_mat)
scene.add_geometry(test_mesh, node_name="milk2", transform=Transform(pos=[0.4, 0, 0.4]).h_mat)

result, name = c_manager.in_collision_internal(return_names=True)
print(result, name)

result, name = c_manager.in_collision_other(o_manager, return_names=True)
print(result, name)

scene.show()