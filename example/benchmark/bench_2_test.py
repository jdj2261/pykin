import numpy as np
import sys, os



from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene_manager import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
import pykin.utils.plot_utils as p_utils

file_path = 'urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")
robot.init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])

shelf_pose = Transform(pos=np.array([0.9, 0, 1.41725156]),rot=np.array([0, 0, np.pi/2]))
bin_pose = Transform(pos=np.array([0.0, 1.0, 0.3864222]))

bottle_meshes = []
for i in range(6):
    bottle_meshes.append(get_object_mesh('bottle.stl'))
bottle_pose1 = Transform(pos=np.array([1.0, 0, 1.29]))
bottle_pose2 = Transform(pos=np.array([0.95, 0.05, 1.29]))
bottle_pose3 = Transform(pos=np.array([0.95, -0.05,1.29]))
bottle_pose4 = Transform(pos=np.array([0.90, 0.1, 1.29]))
bottle_pose5 = Transform(pos=np.array([0.90, 0, 1.29]))
bottle_pose6 = Transform(pos=np.array([0.90, -0.1, 1.29]))

benchmark_config = {2 : None}
scene_mngr = SceneManager("visual", is_pyplot=False, benchmark=benchmark_config)

"""
13, 8, 0
15,  9
17, 16, 2
"""


for i in range(20):
    shelf_name = 'shelf_' + str(i)
    shelf_mesh_test = get_object_mesh(shelf_name + '.stl', scale=0.9)
    scene_mngr.add_object(name=shelf_name, gtype="mesh", h_mat=shelf_pose.h_mat, gparam=shelf_mesh_test, color=[0.39, 0.263, 0.129])

for i in range(20):
    bin_name = 'bin_' + str(i)
    bin_mesh_test = get_object_mesh(bin_name + '.stl', scale=0.9)
    scene_mngr.add_object(name=bin_name, gtype="mesh", h_mat=bin_pose.h_mat, gparam=bin_mesh_test, color=[0.8 + i*0.01, 0.8 + i*0.01, 0.8 + i*0.01])

scene_mngr.add_object(name="bottle_1", gtype="mesh", h_mat=bottle_pose1.h_mat, gparam=bottle_meshes[0], color=[1., 0., 0.])
scene_mngr.add_object(name="bottle_2", gtype="mesh", h_mat=bottle_pose2.h_mat, gparam=bottle_meshes[1], color=[0., 1., 0.])
scene_mngr.add_object(name="bottle_3", gtype="mesh", h_mat=bottle_pose3.h_mat, gparam=bottle_meshes[2], color=[0., 1., 0.])
scene_mngr.add_object(name="bottle_4", gtype="mesh", h_mat=bottle_pose4.h_mat, gparam=bottle_meshes[3], color=[0., 1., 0.])
scene_mngr.add_object(name="bottle_5", gtype="mesh", h_mat=bottle_pose5.h_mat, gparam=bottle_meshes[4], color=[0., 1., 0.])
scene_mngr.add_object(name="bottle_6", gtype="mesh", h_mat=bottle_pose6.h_mat, gparam=bottle_meshes[5], color=[0., 1., 0.])
scene_mngr.add_robot(robot, robot.init_qpos)

scene_mngr.set_logical_state("bottle_1", ("on", "shelf_9"))
scene_mngr.set_logical_state("bottle_2", ("on", "shelf_9"))
scene_mngr.set_logical_state("bottle_3", ("on", "shelf_9"))
scene_mngr.set_logical_state("bottle_4", ("on", "shelf_9"))
scene_mngr.set_logical_state("bottle_5", ("on", "shelf_9"))
scene_mngr.set_logical_state("bottle_6", ("on", "shelf_9"))

for i in range(20):
    scene_mngr.set_logical_state(f"shelf_"+str(i), (scene_mngr.scene.logical_state.static, True))
    scene_mngr.set_logical_state(f"bin_"+str(i), (scene_mngr.scene.logical_state.static, True))
scene_mngr.set_logical_state(scene_mngr.gripper_name, (scene_mngr.scene.logical_state.holding, None))
scene_mngr.update_logical_states()

scene_mngr.show_logical_states()

fig, ax = p_utils.init_3d_figure(name="Benchmark 2")
result, names = scene_mngr.collide_objs_and_robot(return_names=True)
scene_mngr.render_scene(ax)
scene_mngr.show()