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


class MCTS(NodeData):
    def __init__(
        self,
        scene_mngr:SceneManager,
        sampling_method:dict={},
        n_iters:int=1200, 
        exploration_constant:float=1.414,
        max_depth:int=20,
        gamma:float=1,
        eps:float=0.01,
        visible_graph=False
    ):
        self.state = scene_mngr.scene
        self.pick_action = PickAction(scene_mngr, n_contacts=5, n_directions=10)
        self.place_action = PlaceAction(scene_mngr, n_samples_held_obj=100, n_samples_support_obj=100)

        self._sampling_method = sampling_method
        self._n_iters = n_iters
        self.c = exploration_constant
        self._max_depth = max_depth
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
                        NodeData.Q: -np.inf,
                        NodeData.V: -np.inf,
                        NodeData.VISITS: 0,
                        NodeData.NUMBER: 0,
                        NodeData.Q_HISTORY: [],
                        NodeData.V_HISTORY: [],
                        NodeData.TYPE: 'state'})])
        return tree

    def do_planning(self):
        for i in range(self._n_iters):
            print(f"{sc.HEADER}=========== Search iteration : {i+1} ==========={sc.ENDC}")
            self._search(cur_node=0, depth=0)

            if self.nodes:
                break
            # if self.visible:
            #     if (i+1) % self._n_iters == 0:
            #         self.visualize("Backpropagatge")
            #         print("==="*20)
        return self.nodes
        # return self._get_best_action(root_node=0)

    def get_nodes(self, leaf_node, nodes=[]):
            parent_nodes = [node for node in self.tree.predecessors(leaf_node)]
            if not parent_nodes:
                return
            parent_node = parent_nodes[0]
            nodes.append(parent_node)
            self.get_nodes(parent_node, nodes)
            return [leaf_node] + nodes

    def _search(self, cur_node, depth):
        state:Scene = self.tree.nodes[cur_node][NodeData.STATE]
    
        reward = 0
        if depth >= self._max_depth:
            return reward

        if self._is_terminal(state):
            if state is not None:
                print("Success!!!!!")
                self.nodes = self.get_nodes(cur_node, [])

            reward = 10
            self.tree.nodes[cur_node][NodeData.VISITS] += 1
            self.tree.nodes[cur_node][NodeData.REWARD] = reward
            self.tree.nodes[cur_node][NodeData.V] = reward
            self.tree.nodes[cur_node][NodeData.V_HISTORY].append(reward)
            return reward

        action, action_node = self._select_action(cur_node, state, depth)

        if action is None:
            print("Not possible action")
            return reward
        # self.visualize(f"SelectAction Node: {cur_node} Depth: {depth}")

        next_state, state_node = self._simulate(action_node, state, action, depth)
        reward = self._get_reward(next_state)
        
        if next_state is None:
            print("Not possible state")
            self.tree.nodes[cur_node][NodeData.V] = reward
            return reward

        #### For Debug ######################################################################################################################################################
        if action[self.pick_action.info.TYPE] == "pick":
            print(f"Simulate Node: {cur_node} Depth: {depth} {sc.OKGREEN}Action: Pick {action[self.pick_action.info.PICK_OBJ_NAME]}{sc.ENDC}")
        if action[self.pick_action.info.TYPE] == "place":
            print(f"Simulate Node: {cur_node} Depth: {depth} {sc.OKGREEN}Action: Place {action[self.pick_action.info.HELD_OBJ_NAME]} on {action[self.pick_action.info.PLACE_OBJ_NAME]}{sc.ENDC}")
        self.visualize("Simulate")
        self.render_state("next_state", next_state)
        ########################################################################################################################################################################
        
        V_sum = reward + self.gamma * self._search(state_node, depth+1)
        print(state_node, V_sum)
        self._update_value(state_node, action_node, V_sum)

        # print(f"Backpropagation Node: {cur_node} Depth: {depth}")
        # self.visualize("Backpropagation")

        return V_sum

    def _select_action(self, cur_node, state, depth, exploration_method="random"):
        # e-greedy, softmax
        cur_visits = self.tree.nodes[cur_node][NodeData.VISITS]
        eps = np.maximum(np.minimum(1., 1 / np.maximum(cur_visits, 1)), self.eps)
        self._config["eps"] = eps 

        children = [child for child in self.tree.neighbors(cur_node)]

        if not children:
            # print(f"Cur node {cur_node} is a leaf node, So expand")
            self._expand_action_node(cur_node, state, depth)
            expanded_children = [child for child in self.tree.neighbors(cur_node)]
            if not expanded_children:
                return None, None
            next_node = random.choice(expanded_children)
        else:
            # print(f"Cur node has children {children}")
            next_node = self._sample_child_node(children, exploration_method, NodeData.Q)
        
        action = self.tree.nodes[next_node][NodeData.ACTION]
        # print(action[self.pick_action.info.PICK_OBJ_NAME])
        # print(f"Get best action node is {next_node}, and Action is {action}")
        return action, next_node

    def _expand_action_node(self, cur_node, state:Scene, depth):
        
        is_holding = state.logical_states[state.robot.gripper.name][state.logical_state.holding] is not None

        if not is_holding:
            possible_actions = list(self.pick_action.get_possible_actions_level_1(state))
        else:
            possible_actions = list(self.place_action.get_possible_actions_level_1(state))

        for possible_action in possible_actions:
            next_node = self.tree.number_of_nodes()
            self.tree.add_node(next_node)        
            self.tree.update(nodes=[(next_node, { NodeData.DEPTH: depth+1,
                                                NodeData.STATE: state,
                                                NodeData.ACTION: possible_action,
                                                NodeData.REWARD: 0,
                                                NodeData.Q: -np.inf,
                                                NodeData.VISITS: 0,
                                                NodeData.NUMBER: next_node,
                                                NodeData.Q_HISTORY: [],
                                                NodeData.TYPE: 'action'})])
            self.tree.add_edge(cur_node, next_node)

    # TODO
    def _simulate(self, next_node:int, state:Scene, action:dict, depth, exploration_method="random"):
        next_state = None
        reward = -10

        children = [child for child in self.tree.neighbors(next_node)]

        if not children:
            # print(f"Cur node {cur_node} is a leaf node, So expand")
            self._expand_state_node(next_node, state, action, depth)
            expanded_children = [child for child in self.tree.neighbors(next_node)]
            if not expanded_children:
                return next_state, reward
            next_node = random.choice(expanded_children)
        else:
            # print(f"Cur node has children {children}")
            next_node = self._sample_child_node(children, exploration_method, NodeData.V)
        
        next_state = self.tree.nodes[next_node][NodeData.STATE]
        return next_state, next_node

    def _expand_state_node(self, cur_node, state:Scene, action, depth):

        if action[self.pick_action.info.TYPE] == "pick":
            next_states = list(self.pick_action.get_possible_transitions(state, action)) 

        if action[self.pick_action.info.TYPE] == "place":
            next_states = list(self.place_action.get_possible_transitions(state, action))

        for next_state in next_states:
            next_node = self.tree.number_of_nodes()
            self.tree.add_node(next_node)        
            self.tree.update(nodes=[(next_node, { NodeData.NUMBER: next_node,
                                                  NodeData.VISITS: 0,
                                                  NodeData.DEPTH: depth+1,
                                                  NodeData.STATE: next_state,
                                                  NodeData.ACTION: action,
                                                  NodeData.REWARD: 0,
                                                  NodeData.V: -np.inf,
                                                  NodeData.V_HISTORY: [],
                                                  NodeData.TYPE: 'state'})])
            self.tree.add_edge(cur_node, next_node)

    def _sample_child_node(self, children, exploration_method, value):
        assert len(children) != 0
        if exploration_method == "random":
            best_idx = sampler.find_best_idx_from_random(self.tree, children, value)
        if exploration_method == "greedy":
            best_idx = sampler.find_idx_from_greedy(self.tree, children)
        if exploration_method == "ucb1":
            best_idx = sampler.find_idx_from_ucb1(self.tree, children, self.c)
        if exploration_method == "uct":
            best_idx = sampler.find_idx_from_uct(self.tree, children, self.c)
        if exploration_method == "bai_ucb":
            best_idx = sampler.find_idx_from_bai_ucb(self.tree, children)
        if exploration_method == "bai_perturb":
            best_idx = sampler.find_idx_from_bai_perturb(self.tree, children)
        
        child_node = children[best_idx]
        return child_node

    def _update_value(self, state_node, action_node, V_sum):
        self.tree.nodes[state_node][NodeData.VISITS] += 1
        self.tree.nodes[action_node][NodeData.VISITS] += 1

        if self.tree.nodes[state_node][NodeData.TYPE] == "state":
            if V_sum > self.tree.nodes[state_node][NodeData.V]:
                self.tree.nodes[state_node][NodeData.V] = V_sum
            self.tree.nodes[state_node][NodeData.V_HISTORY].append(self.tree.nodes[state_node][NodeData.V])

        if self.tree.nodes[action_node][NodeData.TYPE] == "action":
            V_list = [self.tree.nodes[child][NodeData.V] for child in self.tree.neighbors(action_node)]
            Q = np.max(V_list)
            self.tree.nodes[action_node][NodeData.Q] = Q
            self.tree.nodes[action_node][NodeData.Q_HISTORY].append(Q)

    def _is_terminal(self, state:Scene):
        if state is None:
            print("Not next_state")
            return True
            
        if state.is_terminal_state():
            return True
        return False
    
    def _get_reward(self, cur_state:Scene):
        reward = -1
        if cur_state is None:
            reward = -100
        return reward

    def _get_best_action(self, root_node=0):
        children = [child for child in self.tree.neighbors(root_node)]
        best_idx = np.argmax([self.tree.nodes[child][NodeData.Q] for child in children])
        
        print(f"Reward: {[self.tree.nodes[child][NodeData.REWARD] for child in children]}")
        print(f"Q : {[self.tree.nodes[child][NodeData.Q] for child in children]}")
        print(f"Visits: {[self.tree.nodes[child][NodeData.VISITS] for child in children]}")
        
        return self.tree.nodes[children[best_idx]][NodeData.STATE]

    def visualize(self, title):
        labels = {}
        for n in self.tree.nodes:
            if self.tree.nodes[n][NodeData.ACTION] is not None:
                if self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE] == 'pick':
                    if self.tree.nodes[n][NodeData.TYPE] == 'action':
                        labels.update({ n: 'Node:{:d}\nDepth:{:d}\nVisits:{:d}\nQ(s,a):{:.2f}\nAction:({} {})'.format(
                            self.tree.nodes[n][NodeData.NUMBER],
                            self.tree.nodes[n][NodeData.DEPTH],
                            self.tree.nodes[n][NodeData.VISITS],
                            self.tree.nodes[n][NodeData.Q],
                            self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE],
                            self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.PICK_OBJ_NAME])})
                    if self.tree.nodes[n][NodeData.TYPE] == 'state':
                        labels.update({ n: 'Node:{:d}\nDepth:{:d}\nVisits:{:d}\nReward:{:d}\nV(s):{:.2f}\nAction:({} {})'.format(
                            self.tree.nodes[n][NodeData.NUMBER],
                            self.tree.nodes[n][NodeData.DEPTH],
                            self.tree.nodes[n][NodeData.VISITS],
                            self.tree.nodes[n][NodeData.REWARD],
                            self.tree.nodes[n][NodeData.V],
                            self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE],
                            self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.PICK_OBJ_NAME])})

                if self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE] == 'place':
                    if self.tree.nodes[n][NodeData.TYPE] == 'action':
                        labels.update({ n: 'Node:{:d}\nDepth:{:d}\nVisits:{:d}\nQ(s,a):{:.2f}\nAction:({} {} on {})'.format(
                            self.tree.nodes[n][NodeData.NUMBER],
                            self.tree.nodes[n][NodeData.DEPTH],
                            self.tree.nodes[n][NodeData.VISITS],
                            self.tree.nodes[n][NodeData.Q],
                            self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE],
                            self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.HELD_OBJ_NAME],
                            self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.PLACE_OBJ_NAME])})
                    if self.tree.nodes[n][NodeData.TYPE] == 'state':
                        labels.update({ n: 'Node:{:d}\nDepth:{:d}\nVisits:{:d}\nReward:{:d}\nV(s):{:.2f}\nAction:({} {} on {})'.format(
                            self.tree.nodes[n][NodeData.NUMBER],
                            self.tree.nodes[n][NodeData.DEPTH],
                            self.tree.nodes[n][NodeData.VISITS],
                            self.tree.nodes[n][NodeData.REWARD],
                            self.tree.nodes[n][NodeData.V],
                            self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.TYPE],
                            self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.HELD_OBJ_NAME],
                            self.tree.nodes[n][NodeData.ACTION][self.pick_action.info.PLACE_OBJ_NAME])})
            else:
                labels.update({ n: 'Node:{:d}\nDepth:{:d}\nVisits:{:d}\nReward:{:d}\nQ(s,a):{:.2f}\nV(s):{:.2f}'.format(
                    self.tree.nodes[n][NodeData.NUMBER],
                    self.tree.nodes[n][NodeData.DEPTH],
                    self.tree.nodes[n][NodeData.VISITS],
                    self.tree.nodes[n][NodeData.REWARD],
                    self.tree.nodes[n][NodeData.Q],
                    self.tree.nodes[n][NodeData.V],)})

        plt.figure(title, figsize=(14, 10),)
        pos = graphviz_layout(self.tree, prog='dot')
        nx.draw(self.tree, pos, labels=labels, node_shape="s", node_color="none",
                bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.1'))
        plt.show()

    def render_state(self, title, state):
        fig, ax = p_utils.init_3d_figure(name=title)
        self.pick_action.scene_mngr.render_objects_and_gripper(ax, state)
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