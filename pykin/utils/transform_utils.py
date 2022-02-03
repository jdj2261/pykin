import numpy as np
import math
from collections import Iterable

def vector_norm(data, axis=None, out=None):
    """
    Returns length, i.e. Euclidean norm, of ndarray along axis.
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    

def get_rot_mat_from_homogeneous(h_mat):
    """
    Returns rotation matrix from homogeneous matrix
    """
    return h_mat[:-1, :-1]


def get_pos_mat_from_homogeneous(h_mat):
    """
    Returns position matrix from homogeneous matrix
    """
    return h_mat[:-1,-1]


def get_pose_from_homogeneous(h_mat):
    """
    Returns (7,1) pose from homogeneous matrix
    """
    position = get_pos_mat_from_homogeneous(h_mat)
    orientation = get_quaternion_from_matrix(
        get_rot_mat_from_homogeneous(h_mat))
    return np.hstack((position, orientation))


def get_rpy_from_matrix(R):
    """
    Returns roll pitch, yaw from Rotation matrix
    """
    r = np.arctan2(R[2, 1], R[2, 2])
    p = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    y = np.arctan2(R[1, 0], R[0, 0])

    return np.asarray([r, p, y])


def get_rpy_from_quaternion(q, convention='wxyz'):
    """
    Returns roll pitch, yaw from quaternion
    """
    if convention == 'xyzw':
        x, y, z, w = q[0], q[1], q[2], q[3]  # (N,)
    elif convention == 'wxyz':
        w, x, y, z = q[0], q[1], q[2], q[3]  # (N,)
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x**2 + y**2))   # (N,)
    pitch = np.arcsin(2 * (w*y - z*x))                          # (N,)
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y**2 + z**2))    # (N,)
    rpy = np.asarray([roll, pitch, yaw]).T  # (N,3)
    return rpy

def get_matrix_from_rpy(rpy):
    """
    Returns rotation matrix from rpy
    """
    cr, cp, cy = [np.cos(i) for i in rpy]
    sr, sp, sy = [np.sin(i) for i in rpy]
    R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                  [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                  [-sp, cp*sr, cp*cr]])
    return R


def get_matrix_from_axis_angle(axis, angle):
    """
    Returns rotation matrix from axis angle
    """
    x, y, z = axis
    theta = angle
    c, s = np.cos(theta), np.sin(theta)
    v = 1 - c
    R = np.array([[x ** 2 * v + c, x * y * v - z * s, x * z * v + y * s],
                  [x * y * v + z * s, y ** 2 * v + c, y * z * v - x * s],
                  [x * z * v - y * s, y * z * v + x * s, z ** 2 * v + c]])
    return R


def get_matrix_from_quaternion(q, convention='wxyz'):
    """
    Returns rotation matrix from quaternion
    """
    if isinstance(q, Iterable):
        if convention == 'xyzw':
            x, y, z, w = q
        elif convention == 'wxyz':
            w, x, y, z = q
    else:
        raise TypeError
    R = np.array([[2 * (w**2 + x**2) - 1, 2 * (x*y - w*z), 2 * (x*z + w*y)],
                  [2 * (x*y + w*z), 2 * (w**2 + y**2) - 1, 2*(y*z - w*x)],
                  [2 * (x*z - w*y), 2 * (y*z + w*x), 2 * (w**2 + z**2) - 1]])
    return R


def get_quaternion_from_rpy(rpy, convention='wxyz'):
    """
    Returns quaternion from rpy
    """
    rpy = np.asarray(rpy)
    multiple_rpy = True
    if len(rpy.shape) < 2:
        multiple_rpy = False
        rpy = np.array([rpy])  # (1,3)

    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = np.cos(r/2.), np.sin(r/2.)
    cp, sp = np.cos(p/2.), np.sin(p/2.)
    cy, sy = np.cos(y/2.), np.sin(y/2.)

    w = cr * cp * cy + sr * sp * sy  # (N,)
    x = sr * cp * cy - cr * sp * sy  # (N,)
    y = cr * sp * cy + sr * cp * sy  # (N,)
    z = cr * cp * sy - sr * sp * cy  # (N,)

    if convention == 'xyzw':
        q = np.vstack([x, y, z, w]).T
    elif convention == 'wxyz':
        q = np.vstack([w, x, y, z]).T
    else:
        raise NotImplementedError(
            "Asking for a convention that has not been implemented")

    if not multiple_rpy:
        return q[0]
    return q


def get_quaternion_from_matrix(R, convention='wxyz'):
    """
    Returns quaternion from rotation matrix
    """
    w = 1./2 * np.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1)
    x, y, z = 1./2 * np.array([np.sign(R[2, 1] - R[1, 2]) * np.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1),
                               np.sign(R[0, 2] - R[2, 0]) *
                               np.sqrt(R[1, 1] - R[2, 2] - R[0, 0] + 1),
                               np.sign(R[1, 0] - R[0, 1]) * np.sqrt(R[2, 2] - R[0, 0] - R[1, 1] + 1)])

    if convention == 'xyzw':
        return np.array([x, y, z, w])
    elif convention == 'wxyz':
        return np.array([w, x, y, z])
    else:
        raise NotImplementedError(
            "Asking for a convention that has not been implemented")


def get_quaternion_from_axis_angle(axis, angle, convention='wxyz'):
    """
    Returns quaternion from axis angle
    """
    w = np.cos(angle / 2.)
    x, y , z = np.sin(angle / 2.) * axis
    if convention == 'xyzw':
        return np.array([x, y, z, w])
    elif convention == 'wxyz':
        return np.array([w, x, y, z])
    else:
        raise NotImplementedError("Asking for a convention that has not been implemented")


def get_quaternion_inverse(quaternion):
    """
    Returns quaternion inverse
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q / np.dot(q, q)


def get_quaternion_slerp(qA, qB, t):
    """
    referred https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp
    """
    if isinstance(qA, (np.ndarray, list)) and isinstance(qB, (np.ndarray, list)):
        qA = np.asarray(qA)
        qB = np.asarray(qB)
        if qA.shape != (4,):  # quaternion
            raise ValueError("Expecting the shape of the orientation to be (4,)")
        if qB.shape == (0,):
            qB = np.asarray([1.0, 0.0, 0.0, 0.0])
            
        a = qB * get_quaternion_inverse(qA)
        result = np.sign(a) * (np.abs(a)) ** t * qA
        return result


def get_linear_interpoation(postionA, postionB, step):
    postionA = np.asarray(postionA)
    postionB = np.asarray(postionB)
    return postionB * step + postionA * (1 - step)
    

def get_h_mat_from_quaternion(quaternion):
    """
    Returns homogeneous rotation matrix from quaternion.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def quaternion_multiply(quaternion1, quaternion0):
    """
    Returns multiplication of two quaternions.
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array(
        [
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ],
        dtype=np.float64,
    )

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

def get_quaternion_about_axis(angle, axis):
    """
    Returns quaternion for rotation about axis.
    """
    q = np.array([0.0, axis[0], axis[1], axis[2]])
    qlen = vector_norm(q)
    if qlen > _EPS:
        q *= math.sin(angle / 2.0) / qlen
    q[0] = math.cos(angle / 2.0)
    return q


def get_h_mat(position=np.zeros(3), orientation=np.array([1.0, 0.0, 0.0, 0.0])):
    """
    Returns homogeneous matrix from position and orientation
    """
    position = np.asarray(position)
    orientation = np.asarray(orientation)
    if orientation.shape == (3,):  # RPY Euler angles
        R = get_matrix_from_rpy(orientation)
    elif orientation.shape == (4,):  # quaternion in the form [x,y,z,w]
        R = get_matrix_from_quaternion(orientation)
    elif orientation.shape == (3, 3):  # Rotation matrix
        R = orientation

    H = np.vstack((np.hstack((R, position.reshape(-1, 1))),
                  np.array([[0, 0, 0, 1]])))
    return H


def get_inverse_homogeneous(matrix):
    """
    Returns homogeneous inverse
    """
    R = matrix[:3, :3].T
    p = -R.dot(matrix[:3, 3].reshape(-1,1))
    return np.vstack((np.hstack((R,p)),
                      np.array([[0, 0, 0, 1]])))


def get_identity_h_mat():
    """
    Returns identity matrix
    """
    return np.identity(4)


def homogeneous_to_pose(matrix):
    """
    Returns pose from h_mat
    """
    position = matrix[:3, -1]
    quaternion = get_quaternion_from_matrix(matrix[:3, :3])
    return np.concatenate((position, quaternion))


def pose_to_homogeneous(pose):
    """
    Returns h_mat from pose
    """
    pose = np.array(pose).flatten()
    position, orientation = pose[:3], pose[3:]
    return get_h_mat(position=position, orientation=orientation)


def get_quaternion(orientation, convention='wxyz'):
    """
    Returns quaternion from orientation
    """
    if isinstance(orientation, tuple) and len(orientation) == 2:
        angle, axis = orientation
        if isinstance(axis, (float, int)) and isinstance(angle, np.ndarray):
            angle, axis = axis, angle
        return get_quaternion_from_axis_angle(axis, angle, convention)
    if isinstance(orientation, (np.ndarray, list)):
        orientation = np.asarray(orientation)
        if orientation.shape == (3,):
            return get_quaternion_from_rpy(orientation, convention)
        if orientation.shape == (4,):
            if convention == 'wxyz':
                x, y, z, w = orientation
                return np.array([w, x, y, z])
            return orientation
        if orientation.shape == (3, 3):
            return get_quaternion_from_matrix(orientation, convention)
        else:
            raise ValueError("Expecting the shape of the orientation to be (3,), (3,3), or (4,), instead got: "
                             "{}".format(orientation.shape))
    else:
        raise TypeError("Expecting the given orientation to be a np.ndarray, quaternion, tuple or list, instead got: "
                        "{}".format(type(orientation)))


def get_rotation_matrix(orientation):
    """
    Returns rotation matrix from orientation
    """
    if isinstance(orientation, tuple) and len(orientation) == 2:
        angle, axis = orientation
        if isinstance(axis, (float, int)) and isinstance(angle, np.ndarray):
            angle, axis = axis, angle
        return get_matrix_from_axis_angle(axis, angle)
    if isinstance(orientation, (np.ndarray, list)):
        orientation = np.asarray(orientation)
        if orientation.shape == (3,):  # RPY Euler angles
            return get_matrix_from_rpy(orientation)
        if orientation.shape == (4,):  # quaternion
            return get_matrix_from_quaternion(orientation)
        if orientation.shape == (3, 3):  # rotation matrix
            return orientation
        else:
            raise ValueError("Expecting the shape of the orientation to be (3,), (3,3), or (4,), instead got: "
                         "{}".format(orientation.shape))
    else:
        raise TypeError("Expecting the given orientation to be a np.ndarray, quaternion, tuple or list, instead got: "
                    "{}".format(type(orientation)))


def compute_pose_error(target=np.eye(4), result=np.eye(4)):
    """
    Computes pose(homogeneous transform) error 

    Args:
        target (np.array): target homogeneous transform
        result (np.array): result homogeneous transform 

    Returns:
        error (np.array)
    """
    if  target.shape == (3,) and result.shape == (3,):
        error = np.linalg.norm(target - result)
        # target = get_h_mat(position = target)
        # result = get_h_mat(position = result)
        return error
    error = np.linalg.norm(
        np.dot(result, np.linalg.inv(target)) - np.mat(np.eye(4)))
    return error


def get_transform_to_visual(transform, visual_transform):
    return np.dot(transform, visual_transform)


def change_to_pose(pose_list):
    if isinstance(pose_list, list):
        result = np.asarray(pose_list)
        
    result = np.zeros(7)

    if len(pose_list) == 7:
        result = pose_list

    if len(pose_list) == 6:
        result[0] = pose_list[0]
        result[1] = pose_list[1]
        result[2] = pose_list[2]
        q = get_quaternion(pose_list[3:])
        result[3] = q[0]
        result[4] = q[1]
        result[5] = q[2]
        result[6] = q[3]

    return result

        
