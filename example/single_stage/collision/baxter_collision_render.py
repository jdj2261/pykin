import numpy as np
import trimesh
import sys, os
import yaml

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(pykin_path)

from pykin.robots.bimanual import Bimanual
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.kin_utils import apply_robot_to_scene
from pykin.utils import plot_utils as plt


file_path = '../../../asset/urdf/baxter/baxter.urdf'
robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]))

custom_fpath = '../../../asset/config/baxter_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)

init_qpos = controller_config["init_qpos"]
init_qpos = np.concatenate((np.zeros(1), np.array(init_qpos)))
robot.set_transform(init_qpos)

c_manager = CollisionManager(is_robot=True)
c_manager.setup_robot_collision(robot, geom="collision")
c_manager.show_collision_info()

# _, ax = plt.init_3d_figure("FK")
# plt.plot_robot(ax=ax, 
#                robot=robot,
#                geom="collision",
#                only_visible_geom=True)
# plt.show_figure()

for link, info in robot.info[c_manager.geom].items():
    if link in c_manager._objs:
        c_manager.set_transform(name=link, h_mat=info[3])
        
milk_path = pykin_path+"/asset/objects/meshes/milk.stl"
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