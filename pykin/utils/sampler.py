import numpy as np
import networkx as nx

from pykin.search.node_data import NodeData

def find_best_idx_from_random(tree, children):
    # eps = self._config["eps"]
    # if eps > np.random.uniform():
    best_node_idx = np.random.choice(len([tree.nodes[child][NodeData.VALUE] for child in children]))
    return best_node_idx

def find_idx_from_greedy(tree, children):
    best_node_idx = 0
    max_value = -float('inf')
    for idx, child in enumerate(children):
        value = tree.nodes[child][NodeData.VALUE]
        n = tree.nodes[child][NodeData.VISIT]
        if n == 0:
            best_node_idx = idx
        else:
            if value > max_value:
                max_value = value
                best_node_idx = idx
    
    # print(len(children), best_node_idx)
    # best_node_idx = np.argmax([tree.nodes[child][NodeData.VALUE] for child in children if n = tree.nodes[child][NodeData.VISIT] == 0])
    return best_node_idx

# def find_idx_from_uct(tree, children, c):
#     ucts = []
#     for child in children:
#         values = tree.nodes[child][NodeData.VALUE_HISTORY]
#         n = tree.nodes[child][NodeData.VISIT]
#         total_n = tree.nodes[0][NodeData.VISIT]
#         depth = tree.nodes[child][NodeData.DEPTH]
        
#         if n == 0:
#             uct = float('inf')
#         else:
#             # exploitation = np.max(values)
#             exploitation = np.mean(values)
#             exploration = np.sqrt(np.log(total_n) / np.maximum(1., n))

#             # print(exploitation, c * exploration)
#             uct = exploitation + c / np.maximum(depth, 1) * exploration
#         ucts.append(uct)

#     best_node_list = np.argwhere(ucts == np.amax(ucts)).flatten().tolist()
#     best_node_idx = np.random.choice(best_node_list)
#     return best_node_idx

def find_idx_from_uct(tree, children, c):
    selected_values = [np.mean(tree.nodes[child][NodeData.VALUE_HISTORY]) for child in children]
    selected_visits = [tree.nodes[child][NodeData.VISIT] for child in children]
    total_visits = tree.nodes[0][NodeData.VISIT]
    
    selected_values = np.asarray(selected_values)
    selected_values[np.where(np.asarray(selected_visits) == 0)] = float('inf')

    ucts = selected_values + c * np.sqrt(np.log(total_visits) / np.maximum(1., selected_visits))
    best_node_idx = np.argmax(ucts)

    return best_node_idx

# TODO
def find_idx_from_bai_ucb(tree:nx.DiGraph, children, c):
    selected_values = [tree.nodes[child][NodeData.VALUE] for child in children]
    selected_visits = [tree.nodes[child][NodeData.VISIT] for child in children]
    depth = tree.nodes[children[0]][NodeData.DEPTH]
    total_visits = tree.nodes[0][NodeData.VISIT]
    best_node_idx = 0

    selected_values = np.asarray(selected_values)
    selected_values[np.where(np.asarray(selected_visits) == 0)] = float('inf')

    # print(selected_values)
    if len(selected_visits) == 1:
        best_node_idx = 0
    else:
        c = c / np.maximum(depth, 1)
        upper_bounds = selected_values + c * np.sqrt(1. / np.maximum(1., selected_visits))
        lower_bounds = selected_values - c * np.sqrt(1. / np.maximum(1., selected_visits))
        B_k = [np.max([upper_bounds[i] - lower_bounds[k] for i in range(len(selected_values)) if i is not k]) for k in range(len(selected_values))]
        b = np.argmin(B_k)
        u = np.argmax(upper_bounds)
        if selected_visits[b] > selected_visits[u]:
            best_node_idx = u
        else:
            best_node_idx = b

    return best_node_idx
    # best_node_idx = 0
    # upper_bounds = []
    # lower_bounds = []
    # selected_action_values = []
    # selected_visits = []
    
    # exploitation = -np.inf
    # for child in children:
    #     values = tree.nodes[child][NodeData.VALUE_HISTORY]
    #     visit = tree.nodes[child][NodeData.VISIT]
    #     depth = tree.nodes[child][NodeData.DEPTH]
    #     c = c / np.maximum(depth, 1)

    #     if values:
    #         exploitation = np.mean(values)
    #         # exploitation = np.max(values)
    #     selected_action_values.append(exploitation)
    #     selected_visits.append(visit)
    #     upper_bounds.append(exploitation + c * np.sqrt(1. / np.maximum(1., visit)))
    #     lower_bounds.append(exploitation - c * np.sqrt(1. / np.maximum(1., visit)))

    # if len(selected_visits) == 1:
    #     return best_node_idx
    
    # selected_action_values = np.asarray(selected_action_values)
    # # selected_action_values[np.isinf(selected_action_values)] = 0.

    # B_k = [np.max([upper_bounds[i] - lower_bounds[k] for i in range(len(selected_action_values)) if i is not k]) for k in range(len(selected_action_values))]
    # b = np.argmin(B_k)
    # u = np.argmax(upper_bounds)

    # if selected_visits[b] > selected_visits[u]:
    #     best_node_idx = u
    # else:
    #     best_node_idx = b

    # return best_node_idx

# TODO
def find_idx_from_bai_perturb(tree, children, c):
    best_node_idx = 0
    upper_bounds = []
    lower_bounds = []
    selected_action_values = []
    selected_visits = []
    
    exploitation = -np.inf
    for child in children:
        values = tree.nodes[child][NodeData.VALUE_HISTORY]
        visit = tree.nodes[child][NodeData.VISIT]
        depth = tree.nodes[child][NodeData.DEPTH]
        c = c / np.maximum(depth, 1)
        g = np.random.normal(size=(len(selected_visits)))
        
        if values:
            exploitation = np.mean(values)
            # exploitation = np.max(values)
        selected_action_values.append(exploitation)
        selected_visits.append(visit)
        upper_bounds.append(exploitation + c * np.sqrt(1. / np.maximum(1., visit)) * g)
        lower_bounds.append(exploitation - c * np.sqrt(1. / np.maximum(1., visit)) * g)

    if len(selected_visits) == 1:
        return best_node_idx
    
    selected_action_values = np.asarray(selected_action_values)
    # selected_action_values[np.isinf(selected_action_values)] = 0.

    B_k = [np.max([upper_bounds[i] - lower_bounds[k] for i in range(len(selected_action_values)) if i is not k]) for k in range(len(selected_action_values))]
    b = np.argmin(B_k)
    u = np.argmax(upper_bounds)

    if selected_visits[b] > selected_visits[u]:
        best_node_idx = u
    else:
        best_node_idx = b
    return best_node_idx