import numpy as np

selected_action_values = np.arange(1, 10)
test = [[i for i in range(len(selected_action_values)) if i is not k] for k in range(len(selected_action_values))]
print(test)


# B_k = [np.max([upper_bounds[i] - lower_bounds[k] for i in range(len(selected_action_values)) if i is not k]) for k in range(len(selected_action_values))]