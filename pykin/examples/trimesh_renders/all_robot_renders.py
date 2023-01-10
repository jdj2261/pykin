import numpy as np
import trimesh
import yaml
import os

from pykin.robots.single_arm import SingleArm
from pykin.robots.bimanual import Bimanual
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.kin_utils import apply_robot_to_scene

current_file_path = os.path.abspath(os.path.dirname(__file__))
##################################################################################################
# panda
file_path = 'urdf/panda/panda.urdf'
panda_robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[0, 0, 0]))

print(current_file_path)
custom_fpath = current_file_path + '/../../../pykin/assets/config/panda_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]
panda_robot.set_transform(np.array(init_qpos))

c_manager = CollisionManager(is_robot=True)
c_manager.setup_robot_collision(panda_robot, geom="visual")

for link, info in panda_robot.info[c_manager.geom].items():
    if link in c_manager._objs:
        c_manager.set_transform(name=link, h_mat=info[3])

scene = trimesh.Scene()
scene = apply_robot_to_scene(trimesh_scene=scene, robot=panda_robot, geom=c_manager.geom)
scene.set_camera(np.array([np.pi/2, 0, np.pi]), 5, resolution=(1024, 512))
##################################################################################################
# doosan
file_path = 'urdf/doosan/doosan_with_robotiq140.urdf'
doosan_robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[-1, 0, 0]))
custom_fpath = current_file_path + '/../../../pykin/assets/config/doosan_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]
doosan_robot.set_transform(np.array(init_qpos))

c_manager = CollisionManager(is_robot=True)
c_manager.setup_robot_collision(doosan_robot, geom="visual")

for link, info in doosan_robot.info[c_manager.geom].items():
    if link in c_manager._objs:
        c_manager.set_transform(name=link, h_mat=info[3])

scene = apply_robot_to_scene(trimesh_scene=scene, robot=doosan_robot, geom=c_manager.geom)
##################################################################################################
# iiwa14
file_path = 'urdf/iiwa14/iiwa14.urdf'
iiwa14 = SingleArm(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[-2, 0, 0]))

custom_fpath = current_file_path + '/../../../pykin/assets/config/iiwa14_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]
iiwa14.set_transform(np.array(init_qpos))

c_manager = CollisionManager(is_robot=True)
c_manager.setup_robot_collision(iiwa14, geom="visual")

for link, info in iiwa14.info[c_manager.geom].items():
    if link in c_manager._objs:
        c_manager.set_transform(name=link, h_mat=info[3])

scene = apply_robot_to_scene(trimesh_scene=scene, robot=iiwa14, geom=c_manager.geom)
##################################################################################################
# ur5e
file_path = 'urdf/ur5e/ur5e.urdf'
ur5e_robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[-3, 0, 0]))
custom_fpath = current_file_path + '/../../../pykin/assets/config/ur5e_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]
ur5e_robot.set_transform(np.array(init_qpos))

c_manager = CollisionManager(is_robot=True)
c_manager.setup_robot_collision(ur5e_robot, geom="visual")

for link, info in ur5e_robot.info[c_manager.geom].items():
    if link in c_manager._objs:
        c_manager.set_transform(name=link, h_mat=info[3])

scene = apply_robot_to_scene(trimesh_scene=scene, robot=ur5e_robot, geom=c_manager.geom)
##################################################################################################
# sawyer
file_path = 'urdf/sawyer/sawyer.urdf'
sawyer_robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[1, 0, 0]))
custom_fpath = current_file_path + '/../../../pykin/assets/config/sawyer_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]
sawyer_robot.set_transform(np.array(init_qpos))

c_manager = CollisionManager(is_robot=True)
c_manager.setup_robot_collision(sawyer_robot, geom="visual")

for link, info in sawyer_robot.info[c_manager.geom].items():
    if link in c_manager._objs:
        c_manager.set_transform(name=link, h_mat=info[3])

scene = apply_robot_to_scene(trimesh_scene=scene, robot=sawyer_robot, geom=c_manager.geom)
##################################################################################################
# baxter
file_path = 'urdf/baxter/baxter.urdf'
baxter_robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[2, 0, 0]))
custom_fpath = current_file_path + '/../../../pykin/assets/config/baxter_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]
baxter_robot.set_transform(np.array(np.concatenate((np.zeros(1), np.array(init_qpos)))))

c_manager = CollisionManager(is_robot=True)
c_manager.setup_robot_collision(baxter_robot, geom="visual")

for link, info in baxter_robot.info[c_manager.geom].items():
    if link in c_manager._objs:
        c_manager.set_transform(name=link, h_mat=info[3])

scene = apply_robot_to_scene(trimesh_scene=scene, robot=baxter_robot, geom=c_manager.geom)

scene.show()