import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

import pykin.utils.plot_utils as p_utils
import pykin.utils.sampler as sampler

from pykin.search.node_data import NodeData 
from pykin.scene.scene import Scene
from pykin.scene.scene_manager import SceneManager
from pykin.action.pick import PickAction
from pykin.action.place import PlaceAction
from pykin.utils.kin_utils import ShellColors as sc


class MCTS:
    def __init__(
        self,
        scene_mngr:SceneManager,
        sampling_method:str='uct',
        budgets:int=500, 
        exploration_constant:float=100000,
        max_depth:int=20,
        gamma:float=1,
        eps:float=0.01,
        debug_mode=False,
    ):
        self.node_data = NodeData
        self.scene_mngr = scene_mngr
        self.state = scene_mngr.scene
        self.pick_action = PickAction(scene_mngr, n_contacts=0, n_directions=1)
        self.place_action = PlaceAction(scene_mngr, n_samples_held_obj=0, n_samples_support_obj=10)

        self._sampling_method = sampling_method
        self._budgets = budgets
        self.exploration_c = exploration_constant
        self.max_depth = max_depth
        self.gamma = gamma
        self.eps = eps
        self.debug_mode = debug_mode

        self.tree = self._create_tree(self.state)
        self.nodes = None
        self.infeasible_reward = -3
        self.goal_reward = 3
        self.rewards = []
        self._config = {}
        
    def _create_tree(self, state:Scene):
        tree = nx.DiGraph()
        tree.add_node(0)
        tree.update(
            nodes=[(0, {NodeData.DEPTH: 0,
                        NodeData.STATE: state,
                        NodeData.ACTION: None,
                        NodeData.REWARD: 0,
                        NodeData.VALUE: -np.inf,
                        NodeData.VALUE_HISTORY: [],
                        NodeData.VISIT: 0,
                        NodeData.NUMBER: 0,
                        NodeData.TYPE: 'state',
                        NodeData.GOAL: False})])
        return tree

    def do_planning(self):
        for i in range(self._budgets):
            print(f"{sc.HEADER}=========== Search iteration : {i+1} ==========={sc.ENDC}")
            self._search(state_node=0, depth=0)
            self.rewards.append(self.get_max_reward())

            # if (i+1) % 40 == 0:
            #     self.visualize_tree('Test', tree=self.tree)
    def _search(self, state_node, depth):
        cur_state_node = state_node
        cur_state:Scene = self.tree.nodes[cur_state_node][NodeData.STATE]

        #? Check Current State
        #*======================================================================================================================== #
        if self._is_terminal(cur_state):
            print(f"{sc.OKBLUE}Success!!!!!{sc.ENDC}")
            reward = self._get_reward(cur_state, depth=depth, is_terminal=True)
            self.tree.nodes[state_node][NodeData.GOAL] = True
            self._update_value(cur_state_node, reward)
            return reward
        
        if depth == self.max_depth:
            reward = self.infeasible_reward
            self._update_value(cur_state_node, reward)
            print(f"{sc.WARNING}Exceeded the maximum depth!!{sc.ENDC}")
            return reward
        
        #? Select Logical Action
        #*======================================================================================================================== #
        cur_logical_action_node = self._select_logical_action_node(cur_state_node, cur_state, depth, self._sampling_method)
        cur_logical_action = None
        
        #! [DEBUG]
        if self.debug_mode:
            self.visualize_tree("Next Logical Node", self.tree)
        
        next_state_node = None
        next_state = None
        
        if cur_logical_action_node is not None:
            cur_logical_action = self.tree.nodes.get(cur_logical_action_node).get(NodeData.ACTION)
            
            if cur_logical_action[self.pick_action.info.TYPE] == "pick":
                print(f"{sc.COLOR_BROWN}[Action]{sc.ENDC} {sc.OKGREEN}Pick {cur_logical_action[self.pick_action.info.PICK_OBJ_NAME]}{sc.ENDC}")
            if cur_logical_action[self.pick_action.info.TYPE] == "place":
                print(f"{sc.COLOR_BROWN}[Action]{sc.ENDC} {sc.OKGREEN}Place {cur_logical_action[self.pick_action.info.HELD_OBJ_NAME]} on {cur_logical_action[self.pick_action.info.PLACE_OBJ_NAME]}{sc.ENDC}")

            #? Select Next State
            #*======================================================================================================================== #
            next_state_node = self._select_next_state_node(cur_logical_action_node, cur_state, cur_logical_action, depth, self._sampling_method)
            # assert next_state_node is not None, f"Next state node is None... Why??"
            if next_state_node is not None:
                next_state = self.tree.nodes.get(next_state_node).get(NodeData.STATE)

        #! [DEBUG]
        if self.debug_mode:
            self.render_state("cur_state", cur_state)
            self.render_state("next_state", next_state)
        
        #? Get reward
        #*======================================================================================================================== #
        reward = self._get_reward(cur_state, cur_logical_action, next_state, depth)
        print(f"{sc.MAGENTA}[Reward]{sc.ENDC} S({cur_state_node}) -> A({cur_logical_action_node}) -> S'({next_state_node}) Reward : {sc.UNDERLINE}{np.round(reward,3)}{sc.ENDC}")

        if cur_logical_action_node is None or next_state_node is None:
            value = reward
        else:
            discount_value = -0.1
            value = reward + discount_value + self.gamma * self._search(next_state_node, depth+1)

        self._update_value(cur_state_node, value)
        # print(f"{sc.MAGENTA}[Backpropagation]{sc.ENDC} Cur state Node : {cur_state_node}, Value : {np.round(value,3)}")
        if self.debug_mode:
            self.visualize_tree("Backpropagation", self.tree)

        return value

    def _select_logical_action_node(self, cur_state_node, cur_state, depth, exploration_method="bai_ucb"):
        # e-greedy, softmax
        cur_Visit = self.tree.nodes[cur_state_node][NodeData.VISIT]
        eps = np.maximum(np.minimum(1., 1 / np.maximum(cur_Visit, 1)), self.eps)
        self._config["eps"] = eps 

        children = [child for child in self.tree.neighbors(cur_state_node)]
        logical_action_node = None

        if not children:
            visit = self.tree.nodes[cur_state_node][NodeData.VISIT]
            if visit == 0:
                # print(f"Current state node {cur_state_node} is a leaf node, So expand")
                self._expand_action_node(cur_state_node, cur_state, depth)
            expanded_children = [child for child in self.tree.neighbors(cur_state_node)]
            if not expanded_children:
                return logical_action_node
            logical_action_node = random.choice(expanded_children)
        else:
            # print(f"Current state node has children {children}")
            logical_action_node = self._sample_child_node(children, exploration_method)

        return logical_action_node

    def _expand_action_node(self, cur_state_node, cur_state:Scene, depth):
        is_holding = cur_state.logical_states[cur_state.robot.gripper.name][cur_state.logical_state.holding] is not None

        if not is_holding:
            possible_actions = list(self.pick_action.get_possible_actions_level_1(cur_state))
            # self.render_action("Pick Action", cur_state, possible_actions, is_holding)
        else:
            possible_actions = list(self.place_action.get_possible_actions_level_1(cur_state))
            # self.render_action("Place Action", cur_state, possible_actions, is_holding)
        
        for possible_action in possible_actions:
            action_node = self.tree.number_of_nodes()
            self.tree.add_node(action_node)        
            self.tree.update(nodes=[(action_node, { NodeData.DEPTH: depth+1,
                                                NodeData.STATE: cur_state,
                                                NodeData.ACTION: possible_action,
                                                NodeData.VALUE: -np.inf,
                                                NodeData.VALUE_HISTORY: [],
                                                NodeData.VISIT: 0,
                                                NodeData.NUMBER: action_node,
                                                NodeData.TYPE: 'action',
                                                NodeData.GOAL: False})])
            self.tree.add_edge(cur_state_node, action_node)

    def _select_next_state_node(self, cur_logical_action_node:int, cur_state:Scene, cur_logical_action:dict, depth, exploration_method="bai_ucb"):
        next_state_node = None

        children = [child for child in self.tree.neighbors(cur_logical_action_node)]
        
        if not children:
            visit = self.tree.nodes[cur_logical_action_node][NodeData.VISIT]
            if visit == 0:
                # print(f"Logical action node {cur_logical_action_node} is a leaf node, So expand")
                self._expand_next_state_node(cur_logical_action_node, cur_state, cur_logical_action, depth)
            expanded_children = [child for child in self.tree.neighbors(cur_logical_action_node)]
            if not expanded_children:
                return next_state_node
            next_state_node = random.choice(expanded_children)
        else:
            # print(f"Logical action node has children {children}")
            next_state_node = self._sample_child_node(children, exploration_method)
    
        return next_state_node

    def _expand_next_state_node(self, cur_logical_action_node, cur_state:Scene, cur_logical_action, depth):
        logical_action_type = cur_logical_action[self.pick_action.info.TYPE]
        if logical_action_type == "pick":
            next_states = list(self.pick_action.get_possible_transitions(cur_state, cur_logical_action)) 

        if logical_action_type == "place":
            next_states = list(self.place_action.get_possible_transitions(cur_state, cur_logical_action))

        for next_state in next_states:
            next_node = self.tree.number_of_nodes()
            next_scene:Scene = next_state

            if logical_action_type == "pick":
                cur_geometry_action = next_scene.grasp_poses
            if logical_action_type == "place":
                cur_geometry_action = next_scene.release_poses

            self.tree.add_node(next_node)        
            self.tree.update(nodes=[(next_node, { NodeData.NUMBER: next_node,
                                                  NodeData.VISIT: 0,
                                                  NodeData.DEPTH: depth+1,
                                                  NodeData.STATE: next_state,
                                                  NodeData.ACTION: cur_geometry_action,
                                                  NodeData.VALUE: -np.inf,
                                                  NodeData.VALUE_HISTORY: [],
                                                  NodeData.TYPE: 'state',
                                                  NodeData.GOAL: False})])
            self.tree.add_edge(cur_logical_action_node, next_node)

    def _sample_child_node(self, children, exploration_method):
        assert len(children) != 0
        if exploration_method == "random":
            best_idx = sampler.find_best_idx_from_random(self.tree, children)
        if exploration_method == "greedy":
            best_idx = sampler.find_idx_from_greedy(self.tree, children)
        if exploration_method == "uct":
            best_idx = sampler.find_idx_from_uct(self.tree, children, self.exploration_c)
        if exploration_method == "bai_ucb":
            best_idx = sampler.find_idx_from_bai_ucb(self.tree, children, self.exploration_c)
        if exploration_method == "bai_perturb":
            best_idx = sampler.find_idx_from_bai_perturb(self.tree, children, self.exploration_c)
        
        child_node = children[best_idx]
        return child_node

    def _update_value(self, cur_state_node, value):
        self._update_node(cur_state_node, value)
        if cur_state_node != 0:
            action_node = [node for node in self.tree.predecessors(cur_state_node)][0]
            self._update_node(action_node, value)
        
    def _update_node(self, node, reward):
        if node != 0:
            parent_node = [node for node in self.tree.predecessors(node)][0]
            if self.tree.nodes[node][NodeData.GOAL] is True:
                self.tree.nodes[parent_node][NodeData.GOAL] = True

        self.tree.nodes[node][NodeData.VISIT] += 1
        self.tree.nodes[node][NodeData.VALUE_HISTORY].append(reward)
        if reward > self.tree.nodes[node][NodeData.VALUE]:
            self.tree.nodes[node][NodeData.VALUE] = reward

    @staticmethod
    def _is_terminal(state:Scene):
        if state.is_terminal_state():
            return True
        return False

    def _get_reward(self, cur_state:Scene=None, cur_logical_action:dict={}, next_state:Scene=None, depth=None, is_terminal:bool=False) -> float:
        reward = -0.05
        if is_terminal:
            print(f"Terminal State! Reward is {self.goal_reward}")
            return self.goal_reward

        if cur_state is None:
            print(f"Current state is None.. Reward is {self.infeasible_reward}")
            return self.infeasible_reward
        
        if cur_logical_action is None:
            print(f"Current logical action is None.. Reward is {self.infeasible_reward}")
            return self.infeasible_reward

        if next_state is None:
            print(f"Next state is None.. Reward is {self.infeasible_reward}")
            return self.infeasible_reward

        if self.scene_mngr.scene.bench_num == 1:
            logical_action_type = cur_logical_action[self.pick_action.info.TYPE]

            if logical_action_type == 'place':
                prev_succes_stacked_box_num = cur_state.success_stacked_box_num
                next_state_is_success = next_state.check_success_stacked_bench_1()
                
                if next_state_is_success:
                    if next_state.stacked_box_num - prev_succes_stacked_box_num == 1:
                        print(f"{sc.COLOR_CYAN}Good Action{sc.ENDC}")
                        return abs(reward)
                    if next_state.stacked_box_num - prev_succes_stacked_box_num == -1:
                        print(f"{sc.FAIL}Bad Action{sc.ENDC}")
                        return reward * 2
                else:
                    print(f"{sc.WARNING}Wrong Action{sc.ENDC}")
                    return reward
        
        if self.scene_mngr.scene.bench_num == 2:
            return 0
        return 0

    def get_nodes_from_leaf_node(self, leaf_node):
        parent_nodes = [node for node in self.tree.predecessors(leaf_node)]
        if not parent_nodes:
            return [leaf_node]
        else:
            parent_node = parent_nodes[0]
            return [leaf_node] + self.get_nodes_from_leaf_node(parent_node)

    def get_best_node(self, tree=None, cur_node=0):
        if tree is None:
            tree = self.tree
        
        if not tree:
            return 
        children = [child for child in tree.neighbors(cur_node)]
        
        if not children:
            return [cur_node]
        else:
            # print(f"Node : {[tree.nodes[child][NodeData.NUMBER] for child in children]}")
            # print(f"Q : {[tree.nodes[child][NodeData.VALUE] for child in children]}")
            # print(f"Visit: {[tree.nodes[child][NodeData.VISIT] for child in children]}")
        
            best_idx = np.argmax([tree.nodes[child][NodeData.VALUE] for child in children])
            next_node = children[best_idx]
            return [cur_node] + self.get_best_node(tree, next_node)

    def get_subtree(self):
        visited_nodes = [n for n in self.tree.nodes if self.tree.nodes[n][NodeData.GOAL] is True]
        
        subtree:nx.DiGraph = self.tree.subgraph(visited_nodes)
        return subtree
        
    def get_leaf_nodes(self, tree:nx.DiGraph):
        leaf_nodes = [node for node in tree.nodes if not [c for c in tree.neighbors(node)]]
        leaf_nodes.sort()
        return leaf_nodes

    def get_max_reward(self):
        root_max_reward = self.tree.nodes[0][NodeData.VALUE]
        return root_max_reward

    def show_logical_action(self, node):
        logical_action = self.tree.nodes[node][NodeData.ACTION]
        if logical_action is not None:
            if self.tree.nodes[node][NodeData.TYPE] == "action":
                if logical_action[self.pick_action.info.TYPE] == "pick":
                    print(f"Action Node: {node} {sc.OKGREEN}Action: Pick {logical_action[self.pick_action.info.PICK_OBJ_NAME]}{sc.ENDC}")
                if logical_action[self.pick_action.info.TYPE] == "place":
                    print(f"Action Node: {node}  {sc.OKGREEN}Action: Place {logical_action[self.pick_action.info.HELD_OBJ_NAME]} on {logical_action[self.pick_action.info.PLACE_OBJ_NAME]}{sc.ENDC}")
    
    def visualize_tree(self, title, tree=None):
        if tree is None:
            tree = self.tree
        labels = {}
        for n in tree.nodes:
            if tree.nodes[n][NodeData.ACTION] is not None:
                if tree.nodes[n][NodeData.TYPE] == "action":
                    if self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE] == 'pick':
                        labels.update({ n: 'Type:{}\nNode:{:d}\nDepth:{:d}\nVisit:{:d}\nValue:{:.2f}\nAction:({} {})\nGOAL:{}'.format(
                            tree.nodes[n][NodeData.TYPE],
                            tree.nodes[n][NodeData.NUMBER],
                            tree.nodes[n][NodeData.DEPTH],
                            tree.nodes[n][NodeData.VISIT],
                            tree.nodes[n][NodeData.VALUE],
                            tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE],
                            tree.nodes[n][NodeData.ACTION][self.pick_action.info.PICK_OBJ_NAME],
                            tree.nodes[n][NodeData.GOAL],)})

                    if tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE] == 'place':
                        labels.update({ n: 'Type:{}\nNode:{:d}\nDepth:{:d}\nVisit:{:d}\nValue:{:.2f}\nAction:({} {} on {})\nGOAL:{}'.format(
                            tree.nodes[n][NodeData.TYPE],
                            tree.nodes[n][NodeData.NUMBER],
                            tree.nodes[n][NodeData.DEPTH],
                            tree.nodes[n][NodeData.VISIT],
                            tree.nodes[n][NodeData.VALUE],
                            tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE],
                            tree.nodes[n][NodeData.ACTION][self.pick_action.info.HELD_OBJ_NAME],
                            tree.nodes[n][NodeData.ACTION][self.pick_action.info.PLACE_OBJ_NAME],
                            tree.nodes[n][NodeData.GOAL],)})
                
                if tree.nodes[n][NodeData.TYPE] == "state":
                    labels.update({ n: 'Type:{}\nNode:{:d}\nDepth:{:d}\nVisit:{:d}\nValue:{:.2f}\nGOAL:{}'.format(
                        tree.nodes[n][NodeData.TYPE],
                        tree.nodes[n][NodeData.NUMBER],
                        tree.nodes[n][NodeData.DEPTH],
                        tree.nodes[n][NodeData.VISIT],
                        tree.nodes[n][NodeData.VALUE],
                        tree.nodes[n][NodeData.GOAL],)})
            else:
                labels.update({ n: 'Type:{}\nNode:{:d}\nDepth:{:d}\nVisit:{:d}\nValue:{:.2f}\nGOAL:{}'.format(
                    tree.nodes[n][NodeData.TYPE],
                    tree.nodes[n][NodeData.NUMBER],
                    tree.nodes[n][NodeData.DEPTH],
                    tree.nodes[n][NodeData.VISIT],
                    tree.nodes[n][NodeData.VALUE],
                    tree.nodes[n][NodeData.GOAL],)})

        plt.figure(title, figsize=(14, 10),)
        pos = graphviz_layout(tree, prog='dot')
        nx.draw(tree, pos, labels=labels, node_shape="s", node_color="none",
                bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.1'))
        plt.show()

    def render_state(self, title, state):

        ax = None
        if self.scene_mngr.is_pyplot is True:
            fig, ax = p_utils.init_3d_figure(name=title)

        self.pick_action.scene_mngr.render_objects_and_gripper(ax, state)
        self.pick_action.show()

    def render_action(self, title, scene, actions, is_holding):
        fig, ax = p_utils.init_3d_figure(name=title)

        if not is_holding:
            for pick_action in actions:
                for grasp_pose in pick_action[self.pick_action.info.GRASP_POSES]:
                    self.pick_action.scene_mngr.render.render_axis(ax, grasp_pose[self.pick_action.move_data.MOVE_grasp])
                    self.pick_action.scene_mngr.render_gripper(ax, pose=grasp_pose[self.pick_action.move_data.MOVE_grasp])
        else:   
            for place_action in actions:
                for release_pose, obj_pose in place_action[self.place_action.info.RELEASE_POSES]:
                    self.place_action.scene_mngr.render.render_axis(ax, release_pose[self.place_action.move_data.MOVE_release])
                    self.place_action.scene_mngr.render.render_object(ax, self.place_action.scene_mngr.scene.objs[self.place_action.scene_mngr.scene.robot.gripper.attached_obj_name], obj_pose, alpha=0.3)
                    self.place_action.scene_mngr.render_gripper(ax, pose=release_pose[self.place_action.move_data.MOVE_release])

        self.pick_action.scene_mngr.render_objects(ax, scene)
        p_utils.plot_basis(ax)
        self.pick_action.show()

    def simulate_path(self, nodes):
        if nodes:
            for node in nodes:
                self.show_logical_action(node)

            init_theta = None
            success_pnp = True
            pnp_joint_all_pathes = []
            place_all_object_poses = []
            pick_all_objects = []
            test = []
            test2 = []
            test3 = []
            for node in nodes:
                if self.tree.nodes[node]['type'] == "action":
                    continue
                action = self.tree.nodes[node].get(self.node_data.ACTION)

                if action:
                    if list(action.keys())[0] == 'grasp':
                        success_pick = False
                        pick_scene:Scene = self.tree.nodes[node]['state']
                        # ik_solve, grasp_poses = mcts.pick_action.get_possible_ik_solve_level_2(scene=pick_scene, grasp_poses=pick_scene.grasp_poses)
                        # if ik_solve:
                        print("pick")
                        if init_theta is None:
                            init_theta = self.pick_action.scene_mngr.scene.robot.init_qpos
                        pick_joint_path = self.pick_action.get_possible_joint_path_level_3(
                            scene=pick_scene, 
                            grasp_poses=pick_scene.grasp_poses,
                            init_thetas=init_theta)
                        if pick_joint_path:
                            # pick_all_objects.append([pick_scene.robot.gripper.attached_obj_name])
                            init_theta = pick_joint_path[-1][self.pick_action.move_data.MOVE_default_grasp][-1]
                            success_pick = True
                        else:
                            print("Pick joint Fail")
                            success_pnp = False
                            break
                    else:
                        success_place = False
                        place_scene:Scene = self.tree.nodes[node]['state']
                        # ik_solve, release_poses = mcts.place_action.get_possible_ik_solve_level_2(scene=place_scene, release_poses=place_scene.release_poses)
                        # if ik_solve:
                        print("place")
                        place_joint_path = self.place_action.get_possible_joint_path_level_3(
                            scene=place_scene, 
                            release_poses=place_scene.release_poses, 
                            init_thetas=init_theta)
                        if place_joint_path:
                            success_place = True
                            init_theta = place_joint_path[-1][self.place_action.move_data.MOVE_default_release][-1]
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
                                
                                if task == self.pick_action.move_data.MOVE_grasp:
                                    grasp_task_idx = cnt
                                if task == self.pick_action.move_data.MOVE_post_grasp:
                                    post_grasp_task_idx = cnt
                                    
                                if post_grasp_task_idx - grasp_task_idx == 1:
                                    attach_idx = grasp_task_idx
                                    attach_idxes.append(attach_idx)

                                if task == self.place_action.move_data.MOVE_release:
                                    release_task_idx = cnt
                                if task == self.place_action.move_data.MOVE_post_release:
                                    post_release_task_idx = cnt
                                if post_release_task_idx - release_task_idx == 1:
                                    detach_idx = release_task_idx
                                    detach_idxes.append(detach_idx)
                                
                                result_joint.append(joint)
                                fk = self.pick_action.scene_mngr.scene.robot.forward_kin(joint)
                                eef_poses.append(fk[self.place_action.scene_mngr.scene.robot.eef_name].pos)

                fig, ax = p_utils.init_3d_figure( name="Level wise 3")
                self.scene_mngr.animation(
                    ax,
                    fig,
                    init_scene=self.scene_mngr.scene,
                    joint_path=result_joint,
                    eef_poses=None,
                    visible_gripper=True,
                    visible_text=True,
                    alpha=1.0,
                    interval=50, #ms
                    repeat=False,
                    pick_object = pick_all_object,
                    attach_idx = attach_idxes,
                    detach_idx = detach_idxes,
                    place_obj_pose= place_all_object_pose)

    @property
    def sampling_method(self):
        return self._sampling_method

    @sampling_method.setter
    def sampling_method(self, sampling_method):
        self._sampling_method = sampling_method

    @property
    def n_iters(self):
        return self._n_iters

    @n_iters.setter
    def n_iters(self, n_iters):
        self._n_iters = n_iters

    @property
    def budgets(self):
        return self._budgets

    @budgets.setter
    def budgets(self, budgets):
        self._budgets = budgets


if __name__ == "__main__":
    pass