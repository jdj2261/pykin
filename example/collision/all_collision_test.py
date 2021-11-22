import numpy as np
import trimesh
import yaml
import sys, os

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)
from pykin.robots.single_arm import SingleArm
from pykin.robots.bimanual import Bimanual
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.collision_utils import apply_robot_to_collision_manager, apply_robot_to_scene

# iiwa
file_path = '../../asset/urdf/iiwa14/iiwa14.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[0, 0, 0]))

custom_fpath = '../../asset/config/iiwa14_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]

fk = robot.forward_kin(np.array(init_qpos))
mesh_path = pykin_path+"/asset/urdf/iiwa14/"

c_manager = CollisionManager(mesh_path)
c_manager.filter_contact_names(robot, fk)
c_manager = apply_robot_to_collision_manager(c_manager, robot, fk)
test, name, data = c_manager.in_collision_internal(return_names=True, return_data=True)

scene = trimesh.Scene()
scene = apply_robot_to_scene(scene=scene, mesh_path=mesh_path, robot=robot, fk=fk)
scene.set_camera(np.array([np.pi/2, 0, np.pi/2]), 5, resolution=(1024, 512))

# panda
file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[1, 0, 0]))
custom_fpath = '../../asset/config/panda_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = [0.0, np.pi/6, 0.0, -np.pi*12/24, 0.0, np.pi*5/8,0.0]
fk = robot.forward_kin(np.array(init_qpos))

mesh_path = pykin_path+"/asset/urdf/panda/"
c_manager = CollisionManager(mesh_path)
c_manager.filter_contact_names(robot, fk)
c_manager = apply_robot_to_collision_manager(c_manager, robot, fk)

result, objs_in_collision, contact_data = c_manager.in_collision_internal(return_names=True, return_data=True)
print(result, objs_in_collision, len(contact_data))
scene = apply_robot_to_scene(scene=scene, mesh_path=mesh_path, robot=robot, fk=fk)

# sawyer
custom_fpath = '../../asset/config/sawyer_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]

file_path = '../../asset/urdf/sawyer/sawyer.urdf'
robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[2, 0, 0]))
fk = robot.forward_kin(np.array(np.concatenate((np.zeros(1), init_qpos))))

mesh_path = pykin_path+"/asset/urdf/sawyer/"
c_manager = CollisionManager(mesh_path)
c_manager.filter_contact_names(robot)
c_manager = apply_robot_to_collision_manager(c_manager, robot, fk)
test, name, data = c_manager.in_collision_internal(return_names=True, return_data=True)
scene = apply_robot_to_scene(scene=scene, mesh_path=mesh_path, robot=robot, fk=fk)

# baxter
file_path = '../../asset/urdf/baxter/baxter.urdf'
robot = Bimanual(file_path, Transform(rot=[0.0, 0.0, np.pi/2], pos=[-1, 0, 0]))

custom_fpath = '../../asset/config/baxter_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]
fk = robot.forward_kin(np.concatenate((np.zeros(1), np.array(init_qpos))))

mesh_path = pykin_path+"/asset/urdf/baxter/"
c_manager = CollisionManager(mesh_path)
c_manager.filter_contact_names(robot, fk)
c_manager = apply_robot_to_collision_manager(c_manager, robot, fk)
scene = apply_robot_to_scene(scene=scene, mesh_path=mesh_path, robot=robot, fk=fk)

scene.show()