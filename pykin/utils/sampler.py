import numpy as np
import networkx as nx
import sys
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

def find_idx_from_uct(tree, children, c):
    # selected_values = [np.mean(tree.nodes[child][NodeData.VALUE_HISTORY]) for child in children]
    selected_values = [tree.nodes[child][NodeData.VALUE] for child in children]
    selected_visits = [tree.nodes[child][NodeData.VISIT] for child in children]
    total_visits = tree.nodes[0][NodeData.VISIT]
    
    selected_values = np.asarray(selected_values)
    # selected_values[np.isinf(selected_values)] = 0.
    selected_values[np.where(np.asarray(selected_visits) == 0)] = 0 #! 0
    
    ucts = selected_values + c * np.sqrt(1 / np.maximum(1., selected_visits))
    # print(ucts, selected_values , c * np.sqrt(total_visits / np.maximum(1., selected_visits)))
    best_node_idx = np.argmax(ucts)

    return best_node_idx

# TODO
def find_idx_from_bai_ucb(tree:nx.DiGraph, children, c):
    selected_values = [tree.nodes[child][NodeData.VALUE] for child in children]
    selected_visits = [tree.nodes[child][NodeData.VISIT] for child in children]
    best_node_idx = 0

    selected_values = np.asarray(selected_values)
    selected_values[np.isinf(selected_values)] = 0.
    # selected_values[np.where(np.asarray(selected_visits) == 0)] = sys.maxsize

    if len(selected_visits) == 1:
        best_node_idx = 0
    else:
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

# TODO
def find_idx_from_bai_perturb(tree, children, c):
    selected_values = [tree.nodes[child][NodeData.VALUE] for child in children]
    selected_visits = [tree.nodes[child][NodeData.VISIT] for child in children]
    best_node_idx = 0

    selected_values = np.asarray(selected_values)
    # selected_values[np.where(np.asarray(selected_visits) == 0)] = sys.maxsize
    selected_values[np.isinf(selected_values)] = 0.

    if len(selected_visits) == 1:
        best_node_idx = 0
    else:
        g = np.random.normal(size=(len(selected_visits)))
        upper_bounds = selected_values + c * np.sqrt(1. / np.maximum(1., selected_visits)) * g
        lower_bounds = selected_values - c * np.sqrt(1. / np.maximum(1., selected_visits)) * g
        B_k = np.array([np.max([upper_bounds[i] - lower_bounds[k] for i in range(len(selected_values)) if i is not k]) for k in range(len(selected_values))])
        
        # B_k[np.where(np.isnan(np.asarray(B_k)))] = float('inf')
        # upper_bounds[np.where(np.isnan(np.asarray(upper_bounds)))] = float('inf')
        b = np.argmin(B_k)
        u = np.argmax(upper_bounds)
        # print(B_k, upper_bounds)
        if selected_visits[b] > selected_visits[u]:
            best_node_idx = u
        else:
            best_node_idx = b
    return best_node_idx