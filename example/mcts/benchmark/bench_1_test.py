import numpy as np
import sys, os
import matplotlib.pyplot as plt

pykin_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.robots.single_arm import SingleArm
from pykin.scene.scene_manager import SceneManager
from pykin.scene.scene import Scene
from pykin.utils.mesh_utils import get_object_mesh
from pykin.search.mcts import MCTS
import pykin.utils.plot_utils as p_utils

file_path = '../../../asset/urdf/panda/panda.urdf'
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
ceiling_pose = Transform(pos=np.array([1.1, -0.4, 1.5]))
tray_red_pose = Transform(pos=np.array([0.6, -0.4-0.3, 0.9]))
tray_blue_pose = Transform(pos=np.array([0.6, 0.4, 0.9]))

box_meshes = []
for i in range(6):
    box_meshes.append(get_object_mesh('ben_cube.stl', 0.06))
goal_box_mesh = get_object_mesh('goal_box.stl', 0.001)
table_mesh = get_object_mesh('ben_table.stl')
ceiling_mesh = get_object_mesh('ben_table_ceiling.stl')
tray_red_mesh = get_object_mesh('ben_tray_red.stl')
tray_blue_mesh = get_object_mesh('ben_tray_blue.stl')

param = {'stack_num' : 3, 'goal_object' : "tray_red"}
benchmark_config = {1 : param}

scene_mngr = SceneManager("collision", is_pyplot=True, benchmark=benchmark_config)
scene_mngr.add_object(name="A_box", gtype="mesh", gparam=box_meshes[0], h_mat=A_box_pose.h_mat, color=[1.0, 0.0, 0.0])
scene_mngr.add_object(name="B_box", gtype="mesh", gparam=box_meshes[1], h_mat=B_box_pose.h_mat, color=[0.0, 1.0, 0.0])
scene_mngr.add_object(name="C_box", gtype="mesh", gparam=box_meshes[2], h_mat=C_box_pose.h_mat, color=[0.0, 0.0, 1.0])
# scene_mngr.add_object(name="D_box", gtype="mesh", gparam=box_meshes[3], h_mat=D_box_pose.h_mat, color=[1.0, 1.0, 0.0])
# scene_mngr.add_object(name="E_box", gtype="mesh", gparam=box_meshes[4], h_mat=E_box_pose.h_mat, color=[0.0, 1.0, 1.0])
# scene_mngr.add_object(name="F_box", gtype="mesh", gparam=box_meshes[5], h_mat=F_box_pose.h_mat, color=[1.0, 0.0, 1.0])
scene_mngr.add_object(name="table", gtype="mesh", gparam=table_mesh, h_mat=table_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="ceiling", gtype="mesh", gparam=ceiling_mesh, h_mat=ceiling_pose.h_mat, color=[0.39, 0.263, 0.129])
scene_mngr.add_object(name="tray_red", gtype="mesh", gparam=tray_red_mesh, h_mat=tray_red_pose.h_mat, color=[1.0, 0, 0])
scene_mngr.add_object(name="tray_blue", gtype="mesh", gparam=tray_blue_mesh, h_mat=tray_blue_pose.h_mat, color=[0, 0, 1.0])
scene_mngr.add_robot(robot, robot.init_qpos)

scene_mngr.set_logical_state("A_box", ("on", "table"))
scene_mngr.set_logical_state("B_box", ("on", "table"))
scene_mngr.set_logical_state("C_box", ("on", "table"))
# scene_mngr.set_logical_state("D_box", ("on", "table"))
# scene_mngr.set_logical_state("E_box", ("on", "table"))
# scene_mngr.set_logical_state("F_box", ("on", "table"))
scene_mngr.set_logical_state("ceiling", (scene_mngr.scene.logical_state.static, True))
scene_mngr.set_logical_state("tray_red", (scene_mngr.scene.logical_state.static, True))
scene_mngr.set_logical_state("tray_blue", (scene_mngr.scene.logical_state.static, True))

scene_mngr.set_logical_state("table", (scene_mngr.scene.logical_state.static, True))
scene_mngr.set_logical_state(scene_mngr.gripper_name, (scene_mngr.scene.logical_state.holding, None))
scene_mngr.update_logical_states()

# fig, ax = p_utils.init_3d_figure(name="Benchmark 1")
# result, names = scene_mngr.collide_objs_and_robot(return_names=True)
# print(names)
# scene_mngr.render_scene(ax)
# scene_mngr.show()

mcts = MCTS(scene_mngr)
mcts.debug_mode = False

# 최대부터
mcts.budgets = 100
mcts.max_depth = 20
# mcts.exploration_c = 30
mcts.exploration_c = 300
# mcts.sampling_method = 'bai_ucb' # 405
mcts.sampling_method = 'bai_perturb' # 58
# mcts.sampling_method = 'uct' # 369
nodes = mcts.do_planning()


subtree = mcts.get_subtree()
mcts.visualize_tree("MCTS", subtree)

best_nodes = mcts.get_best_node(subtree)

rewards = mcts.rewards
max_iter = np.argmax(rewards)
print(max_iter)
plt.plot(rewards)
plt.show()


# Do planning
best_nodes = mcts.get_best_node(subtree)
if best_nodes:
    for node in best_nodes:
        mcts.show_logical_action(node)

    # mcts.visualize_tree("Subtree", subtree)

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
        # fig, ax = p_utils.init_3d_figure(name="Level wise 1")
        scene:Scene = mcts.tree.nodes[node]['state']
        if mcts.tree.nodes[node]['type'] == "action":
            continue
        
        action = mcts.tree.nodes[node].get(mcts.node_data.ACTION)
        # scene_mngr.render_objects_and_gripper(ax, scene)
        # scene_mngr.show()

        if action:
            if list(action.keys())[0] == 'grasp':
                success_pick = False
                pick_scene:Scene = mcts.tree.nodes[node]['state']
                # ik_solve, grasp_poses = mcts.pick_action.get_possible_ik_solve_level_2(scene=pick_scene, grasp_poses=pick_scene.grasp_poses)
                # if ik_solve:
                print("pick")
                if init_theta is None:
                    init_theta = mcts.pick_action.scene_mngr.scene.robot.init_qpos
                pick_joint_path = mcts.pick_action.get_possible_joint_path_level_3(
                    scene=pick_scene, 
                    grasp_poses=pick_scene.grasp_poses,
                    init_thetas=init_theta)
                if pick_joint_path:
                    # pick_all_objects.append([pick_scene.robot.gripper.attached_obj_name])
                    init_theta = pick_joint_path[-1][mcts.pick_action.move_data.MOVE_default_grasp][-1]
                    success_pick = True
                else:
                    print("Pick joint Fail")
                    success_pnp = False
                    break
            else:
                success_place = False
                place_scene:Scene = mcts.tree.nodes[node]['state']
                # ik_solve, release_poses = mcts.place_action.get_possible_ik_solve_level_2(scene=place_scene, release_poses=place_scene.release_poses)
                # if ik_solve:
                print("place")
                place_joint_path = mcts.place_action.get_possible_joint_path_level_3(
                    scene=place_scene, 
                    release_poses=place_scene.release_poses, 
                    init_thetas=init_theta)
                if place_joint_path:
                    success_place = True
                    init_theta = place_joint_path[-1][mcts.place_action.move_data.MOVE_default_release][-1]
                    if success_pick and success_place:
                        test += pick_joint_path + place_joint_path
                        test2.append(pick_scene.robot.gripper.attached_obj_name)
                        test3.append(place_scene.objs[place_scene.pick_obj_name].h_mat)
                        print("Success pnp")
                        success_pnp = True
                    else:
                        print("PNP Fail")
                        success_pnp = False
                        break
                else:
                    print("Place joint Fail")
                    success_pnp = False
                    break
        else:
            init_scene = scene

    if success_pnp:
        pnp_joint_all_pathes.append((test))
        pick_all_objects.append(test2)
        place_all_object_poses.append(test3)
        for pnp_joint_all_path, pick_all_object, place_all_object_pose in zip(pnp_joint_all_pathes, pick_all_objects, place_all_object_poses):
            # fig, ax = p_utils.init_3d_figure( name="Level wise 3")
            result_joint = []
            eef_poses = []
            attach_idxes = []
            detach_idxes = []

            attach_idx = 0
            detach_idx = 0

            grasp_task_idx = 0
            post_grasp_task_idx = 0

            release_task_idx = 0
            post_release_task_idx = 0
            cnt = 0
            for pnp_joint_path in pnp_joint_all_path:        
                for j, (task, joint_path) in enumerate(pnp_joint_path.items()):
                    for k, joint in enumerate(joint_path):
                        cnt += 1
                        
                        if task == mcts.pick_action.move_data.MOVE_grasp:
                            grasp_task_idx = cnt
                        if task == mcts.pick_action.move_data.MOVE_post_grasp:
                            post_grasp_task_idx = cnt
                            
                        if post_grasp_task_idx - grasp_task_idx == 1:
                            attach_idx = grasp_task_idx
                            attach_idxes.append(attach_idx)

                        if task == mcts.place_action.move_data.MOVE_release:
                            release_task_idx = cnt
                        if task == mcts.place_action.move_data.MOVE_post_release:
                            post_release_task_idx = cnt
                        if post_release_task_idx - release_task_idx == 1:
                            detach_idx = release_task_idx
                            detach_idxes.append(detach_idx)
                        
                        result_joint.append(joint)
                        fk = mcts.pick_action.scene_mngr.scene.robot.forward_kin(joint)
                        eef_poses.append(fk[mcts.place_action.scene_mngr.scene.robot.eef_name].pos)

        for node in best_nodes:
            mcts.show_logical_action(node)

        fig, ax = p_utils.init_3d_figure( name="Level wise 3")
        mcts.place_action.scene_mngr.animation(
            ax,
            fig,
            init_scene=scene_mngr.scene,
            joint_path=result_joint,
            eef_poses=None,
            visible_gripper=True,
            visible_text=True,
            alpha=1.0,
            interval=50,
            repeat=False,
            pick_object = pick_all_object,
            attach_idx = attach_idxes,
            detach_idx = detach_idxes,
            place_obj_pose= place_all_object_pose)