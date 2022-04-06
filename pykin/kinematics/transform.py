import numpy as np
from pykin.utils import transform_utils as tf

def convert_transform(origin):
    """
    Args:
        origin (None or Transform): offset of object

    Returns:
        Transform: Returns Transform if origin is None
    """
    if origin is None:
        return Transform()
    else:
        return Transform(rot=origin.rot, pos=origin.pos)

class Transform:
    """
    This class calculates the rotation and translation of a 3D rigid body.

    Args:
        pos (sequence of float) : The translation parameter.
        rot (sequence of float) : The rotation parameter. Give in quaternions or roll pitch yaw.
    """
    def __init__(
        self, 
        pos=np.zeros(3),
        rot=np.array([1.0, 0.0, 0.0, 0.0]) 
    ):
        # Set rotation, position
        self.pos = self._to_pos(pos)
        self.rot = self._to_quaternion(rot)

    def __str__(self):
        return "Transform(pos={0}, rot={1})".format(self.pos, self.rot)

    def __repr__(self):
        return 'pykin.kinematics.transform.{}()'.format(type(self).__name__)

    def __mul__(self, other):
        rot = tf.quaternion_multiply(self.rot, other.rot)
        pos = self._to_rotation_vec(self.rot, other.pos) + self.pos
        return Transform(pos, rot)

    def inverse(self):
        """
        Returns:
            Transform : inverse transform
        """
        rot = tf.get_quaternion_inverse(self.rot)
        pos = -self._to_rotation_vec(rot, self.pos)
        return Transform(pos, rot)

    @property
    def pos(self):
        """
        Returns:
            np.array: position
        """
        return self._pos

    @pos.setter
    def pos(self, pos):
        self._pos = self._to_pos(pos)
        
    @property
    def rot(self):
        """
        Returns:
            np.array: rotation (quaternion)
        """
        return self._rot
    
    @rot.setter
    def rot(self, rot):
        self._rot = self._to_quaternion(rot)

    @property
    def pose(self):
        """
        Returns:
            np.array: pose
        """
        return np.hstack((self.pos, self.rot))

    @property
    def rotation_matrix(self):
        """
        Returns:
            np.array: rotation matrix
        """
        return tf.get_rotation_matrix(self.rot)

    @property
    def h_mat(self):
        """
        Returns:
            np.array: homogeneous matrix
        """
        mat = tf.get_h_mat_from_quaternion(self.rot)
        mat[:3, 3] = self.pos
        return mat

    @staticmethod
    def _to_rotation_vec(rot, vec):
        """
        Convert with quaternion and position to rotation vector

        Args:
            rot (np.array): rotation (quaternion)
            vec (np.array): position

        Returns:
            np.array: rotation vector
        """
        v4 = np.hstack([np.array([0.0]), vec])
        inv_rot = tf.get_quaternion_inverse(rot)
        ans = tf.quaternion_multiply(tf.quaternion_multiply(rot, v4), inv_rot)
        return ans[1:]

    @staticmethod
    def _to_quaternion(rot):
        """
        Convert to rotation (qauternion)

        Args:
            rot (sequence of float): rotation (quaternion)

        Returns:
            np.array: rotation (quaternion)
        """
        if len(rot) == 3:
            rot = tf.get_quaternion_from_rpy(rot, convention='wxyz')
        elif len(rot) == 4:
            rot = np.array(rot)
        else:
            raise ValueError("Size of rot must be 3 or 4.")
        return rot

    @staticmethod
    def _to_pos(pos):
        """
        Convert to pos vector

        Args:
            pos (sequence of float): position

        Returns:
            np.array: position
        """
        if not isinstance(pos, np.ndarray):
            if len(pos) == 3:
                pos = np.array(pos)
            else:
                raise ValueError("Size of pos must be 3.")
        assert pos.shape == (3,)
        
        return pos
