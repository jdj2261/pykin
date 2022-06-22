import numpy as np
from pykin.search.node_data import NodeData

def find_best_idx_from_random(tree, children):
    # eps = self._config["eps"]
    # if eps > np.random.uniform():
    best_node_idx = np.random.choice(len([tree.nodes[child][NodeData.VALUE] for child in children]))
    return best_node_idx

def find_idx_from_greedy(tree, children):
    best_node_idx = np.argmax([tree.nodes[child][NodeData.VALUE] for child in children])
    return best_node_idx

def find_idx_from_uct(tree, children, c):
    ucts = []
    for child in children:
        values = tree.nodes[child][NodeData.VALUE_HISTORY]
        n = tree.nodes[child][NodeData.VISIT]
        total_n = tree.nodes[0][NodeData.VISIT]
        
        if n == 0:
            uct = float('inf')
        else:
            # u = np.max(values)
            u = np.mean(values)
            exploitation = u
            exploration = np.sqrt(np.log(total_n) / n)
            uct = exploitation + c * exploration
        ucts.append(uct)
    best_node_idx = np.argmax(ucts)
    return best_node_idx

def find_idx_from_bai_ucb(self, children):
    pass

# TODO
def find_idx_from_bai_perturb(self, children):
    pass