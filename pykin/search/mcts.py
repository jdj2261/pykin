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
        self.place_action = PlaceAction(scene_mngr, n_samples_held_obj=0, n_samples_support_obj=5)

        self._sampling_method = sampling_method
        self._budgets = budgets
        self.exploration_c = exploration_constant
        self.max_depth = max_depth
        self.gamma = gamma
        self.eps = eps
        self.debug_mode = debug_mode

        self.tree = self._create_tree(self.state)
        self.nodes = None
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

    def _search(self, state_node, depth):
        cur_state_node = state_node
        cur_state:Scene = self.tree.nodes[cur_state_node][NodeData.STATE]
    
        #? Check Current State
        #*======================================================================================================================== #
        if self._is_terminal(cur_state):
            print(f"{sc.OKBLUE}Success!!!!!{sc.ENDC}")
            reward = self._get_reward(cur_state, depth=depth, is_terminal=True)
            # reward = 100 * depth
            self.tree.nodes[state_node][NodeData.GOAL] = True
            self._update_value(cur_state_node, reward)
            return reward
        
        if depth >= self.max_depth:
            print(f"{sc.WARNING}Exceeded the maximum depth!!{sc.ENDC}")
            # reward = self._get_reward(cur_state, depth=depth, is_terminal=False)
            reward = -10 * self.max_depth / 10
            self._update_value(cur_state_node, reward)
            return 0
        #? Select Logical Action
        #*======================================================================================================================== #
        cur_logical_action_node = self._select_logical_action_node(cur_state_node, cur_state, depth, self._sampling_method)
        
        #! [DEBUG]
        if self.debug_mode:
            print(f"{sc.MAGENTA}[Select Action]{sc.ENDC} action node : {cur_logical_action_node}")
            self.visualize_tree("Next Logical Node", self.tree)
        
        if cur_logical_action_node is None:
            print(f"{sc.WARNING}Not possible action{sc.ENDC}")
            reward = self._get_reward(cur_state, depth=depth, cur_logical_action=None)
            # reward = -100
            self._update_value(cur_state_node, reward)
            return reward
        cur_logical_action = self.tree.nodes[cur_logical_action_node][NodeData.ACTION]
        
        #! [DEBUG]
        if cur_logical_action[self.pick_action.info.TYPE] == "pick":
            print(f"{sc.MAGENTA}[Action]{sc.ENDC} {sc.OKGREEN}Action: Pick {cur_logical_action[self.pick_action.info.PICK_OBJ_NAME]}{sc.ENDC}")
        if cur_logical_action[self.pick_action.info.TYPE] == "place":
            print(f"{sc.MAGENTA}[Action]{sc.ENDC} {sc.OKGREEN}Action: Place {cur_logical_action[self.pick_action.info.HELD_OBJ_NAME]} on {cur_logical_action[self.pick_action.info.PLACE_OBJ_NAME]}{sc.ENDC}")
        
        #? Select Next State
        #*======================================================================================================================== #
        next_state_node = self._select_next_state_node(cur_logical_action_node, cur_state, cur_logical_action, depth, self._sampling_method)
        
        #! [DEBUG]
        if self.debug_mode:
            print(f"{sc.MAGENTA}[Select Next State]{sc.ENDC} next state node : {next_state_node}")
            self.visualize_tree("Next State Node", self.tree)
        
        if next_state_node is None:
            print(f"{sc.FAIL}Not possible state{sc.ENDC}")
            reward = self._get_reward(cur_state, cur_logical_action, depth=depth, next_state=None)
            # reward = -100
            self._update_value(cur_logical_action_node, reward)
            return reward
        next_state = self.tree.nodes[next_state_node][NodeData.STATE]
        
        ##########################################################################################################################################################
        # # ![DEBUG]
        # if cur_logical_action[self.pick_action.info.TYPE] == "pick":
        #     print(f"{sc.MAGENTA}[Result]{sc.ENDC} Currenct State Node: {cur_state_node} Currenct Action Node: {cur_logical_action_node} Next State Node: {next_state_node} {sc.OKGREEN}Action: Pick {cur_logical_action[self.pick_action.info.PICK_OBJ_NAME]}{sc.ENDC}")
        # if cur_logical_action[self.pick_action.info.TYPE] == "place":
        #     print(f"{sc.MAGENTA}[Result]{sc.ENDC} Currenct State Node: {cur_state_node} Currenct Action Node: {cur_logical_action_node} Next State Node: {next_state_node} {sc.OKGREEN}Action: Place {cur_logical_action[self.pick_action.info.HELD_OBJ_NAME]} on {cur_logical_action[self.pick_action.info.PLACE_OBJ_NAME]}{sc.ENDC}")
        # self.visualize_tree("Next Scene", self.tree)
        # if self.debug_mode:
            # self.render_state("next_state", next_state)
        ##########################################################################################################################################################

        #? Get reward
        #*======================================================================================================================== #
        reward = self._get_reward(cur_state, cur_logical_action, next_state, depth) - 0.1
        print(f"{sc.MAGENTA}[Reward]{sc.ENDC} S({cur_state_node}) -> A({cur_logical_action_node}) -> S'({next_state_node}) Reward : {np.round(reward,3)}")
        value = reward + self.gamma * self._search(next_state_node, depth+1)
        # self._update_value(cur_state_node, value, cur_logical_action_node)
        self._update_value(cur_state_node, value)
        if self.debug_mode:
            print(f"{sc.MAGENTA}[Backpropagation]{sc.ENDC} Cur state Node : {cur_state_node}, Value : {value}")
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
            
            children = [node for node in self.tree.neighbors(cur_state_node)]
            if children:
                pass

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
        reward = 0
        scaler = 1000
        if cur_state is not None:
            if is_terminal:
                # reward = 10 * (self.max_depth - depth)
                reward = 1000 * (self.max_depth - depth + 1)
                return reward / scaler
            else:
                if cur_logical_action is None:
                    # reward = -1 * depth
                    reward = -10 * depth
                    return reward / scaler

                if next_state is None:
                    # reward = -1 * depth
                    reward = -10 * depth
                    return reward / scaler

                if next_state is not None:
                    logical_action_type = cur_logical_action[self.pick_action.info.TYPE]
                    
                    if logical_action_type == 'pick':
                        prev_success_cnt = cur_state.success_cnt
                        next_is_success, next_stacked_num = next_state.check_success_stacked_bench_1()

                        if next_is_success:
                            if next_stacked_num - prev_success_cnt == 1:
                                print("Good Action")
                                # reward = 10 * (self.max_depth - depth)
                                reward = np.exp(-depth) * 50000
                                # reward = (depth - self.max_depth)**4
                                print(reward, 10.0 * (self.max_depth - depth))
                                reward = np.clip(reward, 10.0 * (self.max_depth - depth), float('inf'))
                            if next_stacked_num - prev_success_cnt == -1:
                                print("Bad Action") 
                                reward = -5 * depth
                                # reward = (-5 * depth) / scaler
                        else:
                            print("Wrong Action")
                            reward = -5 * depth
                        return reward / scaler
        return reward / scaler

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