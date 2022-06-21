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
        sampling_method:dict={},
        budgets:int=500, 
        exploration_constant:float=100000,
        max_depth:int=20,
        gamma:float=1,
        eps:float=0.01,
        visible_graph=False
    ):
        self.node_data = NodeData
        self.state = scene_mngr.scene
        self.pick_action = PickAction(scene_mngr, n_contacts=1, n_directions=1)
        self.place_action = PlaceAction(scene_mngr, n_samples_held_obj=1, n_samples_support_obj=5)

        self._sampling_method = sampling_method
        self._budgets = budgets
        self.exploration_c = exploration_constant
        self.max_depth = max_depth
        self.gamma = gamma
        self.eps = eps
        self.visible = visible_graph
        self.tree = self._create_tree(self.state)
        self.nodes = None
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
                        NodeData.TYPE: 'state'})])
        return tree

    def do_planning(self):
        for i in range(self._budgets):
            print(f"{sc.HEADER}=========== Search iteration : {i+1} ==========={sc.ENDC}")
            self._search(state_node=0, depth=0)

            # if self.nodes:
            #     break
            # if self.visible:
            #     if (i+1) % self._budgets == 0:
            #         self.visualize("Backpropagatge")
            #         print("==="*20)
        # return self.nodes
        # return self.get_best_node(root_node=0)

    def _search(self, state_node, depth):
        cur_state_node = state_node
        cur_state:Scene = self.tree.nodes[cur_state_node][NodeData.STATE]
    
        if depth >= self.max_depth:
            print(f"{sc.WARNING}Exceeded the maximum depth!!{sc.ENDC}")
            reward = 0
            # self._update_reward(cur_state_node, reward)
            return reward

        if self._is_terminal(cur_state):
            print(f"{sc.OKBLUE}Success!!!!!{sc.ENDC}")
            # reward = self._get_reward(cur_state, is_terminal=True)
            reward = 1e+6
            self._update_reward(cur_state_node, reward)
            return reward

        cur_logical_action_node = self._select_logical_action_node(cur_state_node, cur_state, depth)
        
        if cur_logical_action_node is None:
            print(f"{sc.WARNING}Not possible action{sc.ENDC}")
            # reward = self._get_reward(cur_state, cur_logical_action=None)
            reward = -1e+4
            self._update_reward(cur_state_node, reward)
            return reward
        cur_logical_action = self.tree.nodes[cur_logical_action_node][NodeData.ACTION]

        next_state_node = self._select_next_state_node(cur_logical_action_node, cur_state, cur_logical_action, depth)
        if next_state_node is None:
            print(f"{sc.FAIL}Not possible state{sc.ENDC}")
            # reward = self._get_reward(cur_state, cur_logical_action, next_state=None)
            reward = -1e+4
            self._update_reward(cur_logical_action_node, reward)
            return reward

        next_state = self.tree.nodes[next_state_node][NodeData.STATE]

        #### For Debug ######################################################################################################################################################
        if cur_logical_action[self.pick_action.info.TYPE] == "pick":
            print(f"Currenct State Node: {cur_state_node} Currenct Action Node: {cur_logical_action_node} Next State Node: {next_state_node} {sc.OKGREEN}Action: Pick {cur_logical_action[self.pick_action.info.PICK_OBJ_NAME]}{sc.ENDC}")
        if cur_logical_action[self.pick_action.info.TYPE] == "place":
            print(f"Currenct State Node: {cur_state_node} Currenct Action Node: {cur_logical_action_node} Next State Node: {next_state_node} {sc.OKGREEN}Action: Place {cur_logical_action[self.pick_action.info.HELD_OBJ_NAME]} on {cur_logical_action[self.pick_action.info.PLACE_OBJ_NAME]}{sc.ENDC}")
        # self.visualize_tree("Next Scene", self.tree)
        # self.render_state("next_state", next_state)
        ########################################################################################################################################################################

        reward = self._get_reward(cur_state, cur_logical_action, next_state, is_terminal=False)  
        # reward -= 100.0
        
        value = reward + self.gamma * self._search(next_state_node, depth+1)
        self._update_value(cur_state_node, cur_logical_action_node, value)
        return value

    def _select_logical_action_node(self, cur_state_node, cur_state, depth, exploration_method="uct"):
        # e-greedy, softmax
        cur_Visit = self.tree.nodes[cur_state_node][NodeData.VISIT]
        eps = np.maximum(np.minimum(1., 1 / np.maximum(cur_Visit, 1)), self.eps)
        self._config["eps"] = eps 

        children = [child for child in self.tree.neighbors(cur_state_node)]
        logical_action_node = None
        if not children:
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
                                                NodeData.TYPE: 'action'})])
            self.tree.add_edge(cur_state_node, action_node)

    def _select_next_state_node(self, cur_logical_action_node:int, cur_state:Scene, cur_logical_action:dict, depth, exploration_method="uct"):
        next_state_node = None
        children = [child for child in self.tree.neighbors(cur_logical_action_node)]

        if not children:
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
                                                  NodeData.TYPE: 'state'})])
            self.tree.add_edge(cur_logical_action_node, next_node)

    def _sample_child_node(self, children, exploration_method):
        assert len(children) != 0
        if exploration_method == "random":
            best_idx = sampler.find_best_idx_from_random(self.tree, children)
        if exploration_method == "greedy":
            best_idx = sampler.find_idx_from_greedy(self.tree, children)
        if exploration_method == "ucb1":
            best_idx = sampler.find_idx_from_ucb1(self.tree, children)
        if exploration_method == "uct":
            best_idx = sampler.find_idx_from_uct(self.tree, children, self.exploration_c)
        if exploration_method == "bai_ucb":
            best_idx = sampler.find_idx_from_bai_ucb(self.tree, children)
        if exploration_method == "bai_perturb":
            best_idx = sampler.find_idx_from_bai_perturb(self.tree, children)
        
        child_node = children[best_idx]
        return child_node

    def _update_reward(self, node, reward):
        self.tree.nodes[node][NodeData.VISIT] += 1
        self.tree.nodes[node][NodeData.VALUE_HISTORY].append(reward)
        if reward > self.tree.nodes[node][NodeData.VALUE]:
            self.tree.nodes[node][NodeData.VALUE] = reward

    def _update_value(self, cur_state_node, cur_logical_action_node, value):
        if self.tree.nodes[cur_state_node][NodeData.TYPE] == "state":
            self._update_reward(cur_state_node, value)

        if self.tree.nodes[cur_logical_action_node][NodeData.TYPE] == "action":
            self._update_reward(cur_logical_action_node, value)

    @staticmethod
    def _is_terminal(state:Scene):
        if state.is_terminal_state():
            return True
        return False
    
    def _get_reward(self, cur_state:Scene=None, cur_logical_action:dict={}, next_state:Scene=None, is_terminal:bool=False) -> float:
        reward = -1e+2
        if cur_state is not None:
            if is_terminal:
                reward = 1e+8
                return reward
            else:
                if cur_logical_action is None:
                    reward = -1e+4
                    return reward

                if next_state is None:
                    reward = -1e+4
                    return reward

                if next_state is not None:
                    logical_action_type = cur_logical_action[self.pick_action.info.TYPE]
                    if logical_action_type == 'place':
                        success_stacked_objs_num = next_state.check_state_bench_1()
                        reward = 1e+4 * success_stacked_objs_num -1e+2
                        if success_stacked_objs_num != 0:
                            # if cur_logical_action[self.pick_action.info.TYPE] == "pick":
                            #     print(f"{sc.OKGREEN}Action: Pick {cur_logical_action[self.pick_action.info.PICK_OBJ_NAME]}{sc.ENDC}")
                            # if cur_logical_action[self.pick_action.info.TYPE] == "place":
                            #     print(f"{sc.OKGREEN}Action: Place {cur_logical_action[self.pick_action.info.HELD_OBJ_NAME]} on {cur_logical_action[self.pick_action.info.PLACE_OBJ_NAME]}{sc.ENDC}")
                            print(f"Reward: {reward}")
                        return reward
        return reward

    def get_nodes_from_leaf_node(self, leaf_node):
        parent_nodes = [node for node in self.tree.predecessors(leaf_node)]
        if not parent_nodes:
            return [leaf_node]
        else:
            parent_node = parent_nodes[0]
            return [leaf_node] + self.get_nodes_from_leaf_node(parent_node)

    def get_best_node(self, cur_node=0):
        children = [child for child in self.tree.neighbors(cur_node)]
        
        if not children:
            return [cur_node]
        else:
            print(f"Node : {[self.tree.nodes[child][NodeData.NUMBER] for child in children]}")
            print(f"Q : {[self.tree.nodes[child][NodeData.VALUE] for child in children]}")
            print(f"Visit: {[self.tree.nodes[child][NodeData.VISIT] for child in children]}")
        
            best_idx = np.argmax([self.tree.nodes[child][NodeData.VISIT] for child in children])
            next_node = children[best_idx]
            return [cur_node] + self.get_best_node(next_node)

    def get_subtree(self):
        visited_nodes = [n for n in self.tree.nodes if self.tree.nodes[n][NodeData.VALUE] > 0]
        subtree:nx.DiGraph = self.tree.subgraph(visited_nodes)
        return subtree
        
    def get_leaf_nodes(self, tree:nx.DiGraph):
        leaf_nodes = [node for node in tree.nodes if not [c for c in tree.neighbors(node)]]
        leaf_nodes.sort()
        return leaf_nodes

    def show_logical_action(self, node):
        logical_action = self.tree.nodes[node][NodeData.ACTION]
        if logical_action[self.pick_action.info.TYPE] == "pick":
            print(f"Currenct Node: {node} {sc.OKGREEN}Action: Pick {logical_action[self.pick_action.info.PICK_OBJ_NAME]}{sc.ENDC}")
        if logical_action[self.pick_action.info.TYPE] == "place":
            print(f"Currenct Node: {node}  {sc.OKGREEN}Action: Place {logical_action[self.pick_action.info.HELD_OBJ_NAME]} on {logical_action[self.pick_action.info.PLACE_OBJ_NAME]}{sc.ENDC}")
    
    def visualize_tree(self, title, tree):
        labels = {}
        for n in tree.nodes:
            if tree.nodes[n][NodeData.ACTION] is not None:
                if tree.nodes[n][NodeData.TYPE] == "action":
                    if self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE] == 'pick':
                        labels.update({ n: 'Type:{}\nNode:{:d}\nDepth:{:d}\nValueVisi:{:d}\nValue:{:.2f}\nAction:({} {})'.format(
                            tree.nodes[n][NodeData.TYPE],
                            tree.nodes[n][NodeData.NUMBER],
                            tree.nodes[n][NodeData.DEPTH],
                            tree.nodes[n][NodeData.VISIT],
                            tree.nodes[n][NodeData.VALUE],
                            tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE],
                            tree.nodes[n][NodeData.ACTION][self.pick_action.info.PICK_OBJ_NAME])})

                    if tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE] == 'place':
                        labels.update({ n: 'Type:{}\nNode:{:d}\nDepth:{:d}\nVisit:{:d}\nValue:{:.2f}\nAction:({} {} on {})'.format(
                            tree.nodes[n][NodeData.TYPE],
                            tree.nodes[n][NodeData.NUMBER],
                            tree.nodes[n][NodeData.DEPTH],
                            tree.nodes[n][NodeData.VISIT],
                            tree.nodes[n][NodeData.VALUE],
                            tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE],
                            tree.nodes[n][NodeData.ACTION][self.pick_action.info.HELD_OBJ_NAME],
                            tree.nodes[n][NodeData.ACTION][self.pick_action.info.PLACE_OBJ_NAME])})
                
                if tree.nodes[n][NodeData.TYPE] == "state":
                    labels.update({ n: 'Type:{}\nNode:{:d}\nDepth:{:d}\nVisit:{:d}\nValue:{:.2f}'.format(
                        tree.nodes[n][NodeData.TYPE],
                        tree.nodes[n][NodeData.NUMBER],
                        tree.nodes[n][NodeData.DEPTH],
                        tree.nodes[n][NodeData.VISIT],
                        tree.nodes[n][NodeData.VALUE],)})
            else:
                labels.update({ n: 'Type:{}\nNode:{:d}\nDepth:{:d}\nVisit:{:d}\nValue:{:.2f}'.format(
                    tree.nodes[n][NodeData.TYPE],
                    tree.nodes[n][NodeData.NUMBER],
                    tree.nodes[n][NodeData.DEPTH],
                    tree.nodes[n][NodeData.VISIT],
                    tree.nodes[n][NodeData.VALUE],)})

        plt.figure(title, figsize=(14, 10),)
        pos = graphviz_layout(tree, prog='dot')
        nx.draw(tree, pos, labels=labels, node_shape="s", node_color="none",
                bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.1'))
        plt.show()

    def render_state(self, title, state):
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