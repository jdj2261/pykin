import numpy as np

def calc_jacobian(frames: list, transformations: dict, thetas: list) -> np.array:
    jsize = len(thetas)
    target_position = list(transformations.values())[-1].pos
    J = np.zeros((6, jsize))
    n = 0
    for frame in frames:
        if frame.joint.dtype == "revolute":
            n += 1
            w = np.dot(transformations[frame.link.name].matrix()[:3, :3], frame.joint.axis)
            v = np.cross(w, target_position - transformations[frame.link.name].pos)
            J[:, n - 1] = np.hstack((v, w))
        elif frame.joint.dtype == "prismatic":
            n += 1
            w = np.zeros(3)
            v = np.dot(transformations[frame.link.name].matrix()[:3, :3], frame.joint.axis)
            J[:, n - 1] = np.hstack((v, w))
    return J
