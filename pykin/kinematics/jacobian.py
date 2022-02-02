import numpy as np

def calc_jacobian(
    frames: list, 
    fk: dict, 
    jsize: int) -> np.array:
    """
    Args:
        frames (list): frames to compute jacobian
        fk (dict): forward kinematics
        thetas (int): size of joint space

    Returns:
        Jacobian (np.array(6, jsize)): return Jacobian
    """
    target_position = list(fk.values())[-1].pos
    J = np.zeros((6, jsize))
    n = 0
    for frame in frames:
        if frame.joint.dtype == "revolute":
            n += 1
            w = np.dot(fk[frame.link.name].h_mat[:3, :3], frame.joint.axis)
            v = np.cross(w, target_position - fk[frame.link.name].pos)
            J[:, n - 1] = np.hstack((v, w))
        elif frame.joint.dtype == "prismatic":
            n += 1
            w = np.zeros(3)
            v = np.dot(fk[frame.link.name].h_mat[:3, :3], frame.joint.axis)
            J[:, n - 1] = np.hstack((v, w))
    return J
