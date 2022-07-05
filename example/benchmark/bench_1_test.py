import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene_manager import SceneManager
from pykin.utils.mesh_utils import get_object_mesh
import pykin.utils.plot_utils as p_utils

file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")
robot.init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])

A_box_pose = Transform(pos=np.array([0.6, 0.05, 0.77]))
B_box_pose = Transform(pos=np.array([0.6, 0.15, 0.77]))
C_box_pose = Transform(pos=np.array([0.6, 0.25, 0.77]))
D_box_pose = Transform(pos=np.array([0.5, 0.05, 0.77]))
E_box_pose = Transform(pos=np.array([0.5, 0.15, 0.77]))
F_box_pose = Transform(pos=np.array([0.5, 0.25, 0.77]))
goal_box_pose = Transform(pos=np.array([0.6, -0.2, 0.77]), rot=np.array([0, np.pi/2, 0]))
table_pose = Transform(pos=np.array([1.0, -0.4, -0.03]))
ceiling_pose = Transform(pos=np.array([1.0, -0.4, 1.5]))
tray_red_pose = Transform(pos=np.array([0.6, -0.5-0.3, 0.8]))
tray_blue_pose = Transform(pos=np.array([0.6, 0.5, 0.8]))

box_meshes = []
for i in range(6):
    box_meshes.append(get_object_mesh('ben_cube.stl', 0.06))
goal_box_mesh = get_object_mesh('goal_box.stl', 0.001)
table_mesh = get_object_mesh('ben_table.stl')
ceiling_mesh = get_object_mesh('ben_table_ceiling.stl')
tray_red_mesh = get_object_mesh('ben_tray_red.stl')
tray_blue_mesh = get_object_mesh('ben_tray_blue.stl')

param = {'stack_num' : 6, 'goal_box':'tray_red'}
benchmark_config = {1 : param}

scene_mngr = SceneManager("visual", is_pyplot=False, benchmark=benchmark_config)
scene_mngr.add_object(name="A_box", gtype="mesh", gparam=box_meshes[0], h_mat=A_box_pose.h_mat, color=[1.0, 0.0, 0.0])
scene_mngr.add_object(name="B_box", gtype="mesh", gparam=box_meshes[1], h_mat=B_box_pose.h_mat, color=[0.0, 1.0, 0.0])
scene_mngr.add_object(name="C_box", gtype="mesh", gparam=box_meshes[2], h_mat=C_box_pose.h_mat, color=[0.0, 0.0, 1.0])
scene_mngr.add_object(name="D_box", gtype="mesh", gparam=box_meshes[3], h_mat=D_box_pose.h_mat, color=[1.0, 1.0, 0.0])
scene_mngr.add_object(name="E_box", gtype="mesh", gparam=box_meshes[4], h_mat=E_box_pose.h_mat, color=[0.0, 1.0, 1.0])
scene_mngr.add_object(name="F_box", gtype="mesh", gparam=box_meshes[5], h_mat=F_box_pose.h_mat, color=[1.0, 0.0, 1.0])
# scene_mngr.add_object(name="goal_box", gtype="mesh", gparam=goal_box_mesh, h_mat=goal_box_pose.h_mat, color=[1.0, 1.0, 1.0])
scene_mngr.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="ceiling", gtype="mesh", gparam=ceiling_mesh, h_mat=ceiling_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="tray_red", gtype="mesh", gparam=tray_red_mesh, h_mat=tray_red_pose.h_mat, color=[1.0, 0, 0])
scene_mngr.add_object(name="tray_blue", gtype="mesh", gparam=tray_blue_mesh, h_mat=tray_blue_pose.h_mat, color=[0, 0, 1.0])
scene_mngr.add_robot(robot, robot.init_qpos)

scene_mngr.set_logical_state("A_box", ("on", "table"))
scene_mngr.set_logical_state("B_box", ("on", "table"))
scene_mngr.set_logical_state("C_box", ("on", "table"))
scene_mngr.set_logical_state("D_box", ("on", "table"))
scene_mngr.set_logical_state("E_box", ("on", "table"))
scene_mngr.set_logical_state("F_box", ("on", "table"))
# scene_mngr.set_logical_state("goal_box", ("on", "table"))
scene_mngr.set_logical_state("table", (scene_mngr.scene.logical_state.static, True))
scene_mngr.set_logical_state(scene_mngr.gripper_name, (scene_mngr.scene.logical_state.holding, None))
scene_mngr.update_logical_states()

fig, ax = p_utils.init_3d_figure(name="Benchmark 1")
result, names = scene_mngr.collide_objs_and_robot(return_names=True)
print(names)
scene_mngr.render_scene(ax)
scene_mngr.show()