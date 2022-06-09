import numpy as np
import sys, os
import yaml

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene_manager import SceneManager
from pykin.scene.scene import Scene
from pykin.utils.mesh_utils import get_object_mesh
from pykin.search.mcts import MCTS
import pykin.utils.plot_utils as plt


file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")

custom_fpath = '../../asset/config/panda_init_params.yaml'
with open(custom_fpath) as f:
    controller_config = yaml.safe_load(f)
init_qpos = controller_config["init_qpos"]

red_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77]))
blue_box_pose = Transform(pos=np.array([0.6, 0.35, 0.77]))
green_box_pose = Transform(pos=np.array([0.6, 0.05, 0.77]))
support_box_pose = Transform(pos=np.array([0.6, -0.2, 0.77]), rot=np.array([0, np.pi/2, 0]))
table_pose = Transform(pos=np.array([0.4, 0.24, 0.0]))

red_cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
blue_cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
green_cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
box_goal_mesh = get_object_mesh('box_goal.stl', 0.001)
table_mesh = get_object_mesh('custom_table.stl', 0.01)

scene_mngr = SceneManager("collision", is_pyplot=True, benchmark=1)
scene_mngr.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="A_box", gtype="mesh", gparam=red_cube_mesh, h_mat=red_box_pose.h_mat, color=[1.0, 0.0, 0.0])
scene_mngr.add_object(name="B_box", gtype="mesh", gparam=blue_cube_mesh, h_mat=blue_box_pose.h_mat, color=[0.0, 0.0, 1.0])
scene_mngr.add_object(name="C_box", gtype="mesh", gparam=green_cube_mesh, h_mat=green_box_pose.h_mat, color=[0.0, 1.0, 0.0])
scene_mngr.add_object(name="goal_box", gtype="mesh", gparam=box_goal_mesh, h_mat=support_box_pose.h_mat, color=[1.0, 0, 1.0])
scene_mngr.add_robot(robot, init_qpos)
############################# Logical State #############################

scene_mngr.scene.logical_states["A_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["B_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["C_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["goal_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["table"] = {scene_mngr.scene.logical_state.static : True}
scene_mngr.scene.logical_states[scene_mngr.gripper_name] = {scene_mngr.scene.logical_state.holding : None}

scene_mngr.update_logical_states()
scene_mngr.show_scene_info()
scene_mngr.show_logical_states()

# test = scene_mngr.scene.get_objs_chain_list("A_box")
# print(test)
# scene_mngr.scene.pick_obj_name = "C_box"
# print(scene_mngr.scene.is_terminal_state())


mcts = MCTS(scene_mngr)
nodes = mcts.do_planning()
best_nodes = mcts.get_best_node(cur_node=0)
print(best_nodes)
# nodes.reverse()
# print(nodes)
for node in best_nodes:
    fig, ax = plt.init_3d_figure(name="Level wise 1")
    scene:Scene = mcts.tree.nodes[node]['state']
    if mcts.tree.nodes[node]['type'] == "action":
        continue
    print(node, scene)
    # scene.show_logical_states()
    scene_mngr.render_objects_and_gripper(ax, scene)
    scene_mngr.show()



############################# Scene Info #############################
# print(scene_mngr.get_objs_info())
# print(scene_mngr.get_gripper_info())
# print(scene_mngr.get_robot_info())

# scene_mngr.render_objects_and_gripper(ax)
# scene_mngr.show()