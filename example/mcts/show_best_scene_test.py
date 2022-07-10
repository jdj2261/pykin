import numpy as np
import sys, os

pykin_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene_manager import SceneManager
from pykin.scene.scene import Scene
from pykin.utils.mesh_utils import get_object_mesh
from pykin.search.mcts import MCTS
import pykin.utils.plot_utils as p_utils

file_path = '../../asset/urdf/panda/panda.urdf'
robot = SingleArm(
    f_name=file_path, 
    offset=Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), 
    has_gripper=True)
robot.setup_link_name("panda_link_0", "panda_right_hand")
robot.init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, -np.pi/4])

red_box_pose = Transform(pos=np.array([0.6, 0.2, 0.77]))
blue_box_pose = Transform(pos=np.array([0.6, 0.35, 0.77]))
green_box_pose = Transform(pos=np.array([0.6, 0.05, 0.77]))
test1_box_pose = Transform(pos=np.array([0.5, 0.2, 0.77]))
test2_box_pose = Transform(pos=np.array([0.5, 0.35, 0.77]))
test3_box_pose = Transform(pos=np.array([0.5, 0.05, 0.77]))

support_box_pose = Transform(pos=np.array([0.6, -0.2, 0.77]), rot=np.array([0, np.pi/2, 0]))
table_pose = Transform(pos=np.array([0.4, 0.24, 0.0]))

red_cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
blue_cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
green_cube_mesh = get_object_mesh('ben_cube.stl', 0.06)
goal_box_mesh = get_object_mesh('goal_box.stl', 0.001)
table_mesh = get_object_mesh('custom_table.stl', 0.01)

param = {'stack_num' : 4}
benchmark_config={1 : param}

scene_mngr = SceneManager("collision", is_pyplot=True, benchmark=benchmark_config)
scene_mngr.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="A_box", gtype="mesh", gparam=red_cube_mesh, h_mat=red_box_pose.h_mat, color=[1.0, 0.0, 0.0])
scene_mngr.add_object(name="B_box", gtype="mesh", gparam=blue_cube_mesh, h_mat=blue_box_pose.h_mat, color=[0.0, 0.0, 1.0])
scene_mngr.add_object(name="C_box", gtype="mesh", gparam=green_cube_mesh, h_mat=green_box_pose.h_mat, color=[0.0, 1.0, 0.0])
scene_mngr.add_object(name="goal_box", gtype="mesh", gparam=goal_box_mesh, h_mat=support_box_pose.h_mat, color=[1.0, 0, 1.0])
scene_mngr.add_object(name="D_box", gtype="mesh", gparam=green_cube_mesh, h_mat=test1_box_pose.h_mat, color=[1.0, 1.0, 0.0])
# scene_mngr.add_object(name="E_box", gtype="mesh", gparam=green_cube_mesh, h_mat=test2_box_pose.h_mat, color=[0.0, 1.0, 1.0])
# scene_mngr.add_object(name="F_box", gtype="mesh", gparam=green_cube_mesh, h_mat=test3_box_pose.h_mat, color=[1.0, 0.0, 1.0])
scene_mngr.add_robot(robot, robot.init_qpos)
############################# Logical State #############################

scene_mngr.scene.logical_states["A_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["B_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["C_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.scene.logical_states["goal_box"] = {scene_mngr.scene.logical_state.on : scene_mngr.scene.objs["table"]}
scene_mngr.set_logical_state("D_box", ("on", "table"))
# scene_mngr.set_logical_state("E_box", ("on", "table"))
# scene_mngr.set_logical_state("F_box", ("on", "table"))
scene_mngr.scene.logical_states["table"] = {scene_mngr.scene.logical_state.static : True}
scene_mngr.scene.logical_states[scene_mngr.gripper_name] = {scene_mngr.scene.logical_state.holding : None}

scene_mngr.update_logical_states()
scene_mngr.show_scene_info()
scene_mngr.show_logical_states()

mcts = MCTS(scene_mngr, sampling_method='greedy')
mcts.debug_mode = False
mcts.budgets = 100
mcts.max_depth = 10
mcts.exploration_c = 1.4
nodes = mcts.do_planning()
best_nodes = mcts.get_best_node(cur_node=0)

# nodes.reverse()
# print(nodes)
init_theta = None
init_scene = None
success_pnp = True
pnp_joint_all_pathes = []
place_all_object_poses = []
pick_all_objects = []
test = []
test2 = []
test3 = []
for node in best_nodes:
    fig, ax = p_utils.init_3d_figure(name="Level wise 1")
    scene:Scene = mcts.tree.nodes[node]['state']
    if mcts.tree.nodes[node]['type'] == "action":
        continue
    
    action = mcts.tree.nodes[node].get(mcts.node_data.ACTION)
    scene.show_scene_info()
    scene_mngr.render_objects_and_gripper(ax, scene)
    scene_mngr.show()
