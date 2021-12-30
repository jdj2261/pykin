import numpy as np

from pykin.robots.robot import Robot
from pykin.utils.error_utils import NotFoundError

class SingleArm(Robot):
    """
    Initializes a single-armed robot simulation object.

    Args:
        fname (str): path to the urdf file.
        offset (Transform): robot init offset
    """
    def __init__(
        self,
        fname:str,
        offset=None
    ):
        super(SingleArm, self).__init__(fname, offset)
        self._base_name = ""
        self._eef_name  = ""
        self.desired_base_frame = ""
        self._set_joint_limits_upper_and_lower()

    def _set_joint_limits_upper_and_lower(self):
        """
        Set joint limits upper and lower
        """
        for joint, (limit_lower, limit_upper) in self.joint_limits.items():
            if "head" in joint:
                continue
            if self.joints[joint].dtype == "revolute":
                if limit_lower is None and limit_upper is None:
                    limit_lower = -np.pi
                    limit_upper = np.pi
                self.joint_limits_lower.append(limit_lower)
                self.joint_limits_upper.append(limit_upper)

    def check_limit_joint(self, q_in):
        """
        check q_in within joint limits
        If q_in is in joint limits, return True
        otherwise, return False

        Returns:
            bool(True or False)
        """
        return np.all([q_in >= self.joint_limits_lower, q_in <= self.joint_limits_upper])

    def setup_link_name(self, base_name="", eef_name=None):
        """
        Sets robot's desired frame

        Args:
            base_name (str): reference link name
            eef_name (str): end effector name
        """
        self._check_link_name(base_name, eef_name)
        self._base_name = base_name
        self._eef_name = eef_name
        self._set_desired_base_frame()
        self._set_desired_frame()

    def _check_link_name(self, base_name, eef_name):
        """
        Check link name

        Args:
            base_name (str): reference link name
            eef_name (str): end effector name
        """
        if base_name and not base_name in self.links.keys():
            raise NotFoundError(base_name)

        if eef_name is not None and eef_name not in self.links.keys():
            raise NotFoundError(eef_name)

    def _set_desired_base_frame(self):
        """
        Sets robot's desired base frame

        Args:
            arm (str): robot arm (right or left)
        """
        if self.base_name == "":
            self.desired_base_frame = self.root
        else:
            self.desired_base_frame = self.find_frame(
                self.base_name + "_frame")

    def _set_desired_frame(self):
        """
        Sets robot's desired frame

        Args:
            arm (str): robot arm (right or left)
        """
        self.desired_frames = super().generate_desired_frame_recursive(self.desired_base_frame, self.eef_name)
        self._revolute_joint_names = sorted(self.get_revolute_joint_names(self.desired_frames))

    def inverse_kin(self, current_joints, target_pose, method="LM", maxIter=1000):
        """
        Returns joint angles obtained by computing IK
        
        Args:
            current_joints (sequence of float): input joint angles
            target_pose (np.array): goal pose to achieve
            method (str): two methods to calculate IK (LM: Levenberg-marquardt, NR: Newton-raphson)
            maxIter (int): Maximum number of calculation iterations

        Returns:
            joints (np.array): target joint angles
        """
        self._set_desired_frame()

        joints = self.kin.inverse_kinematics(
            self.desired_frames,
            current_joints,
            target_pose,
            method,
            maxIter)
        return joints

    def get_eef_pose(self, transformations=None):
        """
        Get end effector's pose

        Args:
            transformations(OrderedDict)
        
        Returns:
            vals(dict)
        """
        if transformations is None:
            transformations = self.init_transformations

        return np.concatenate((transformations[self.eef_name].pos, transformations[self.eef_name].rot))

    def get_eef_h_mat(self, transformations=None):
        """
        Get end effector's homogeneous marix

        Args:
            transformations(OrderedDict)
        
        Returns:
            vals(dict)
        """
        if transformations is None:
            transformations = self.init_transformations

        return transformations[self.eef_name].h_mat

    def get_eef_pos(self, transformations=None):
        """
        Get end effector's position

        Args:
            transformations(OrderedDict)
        
        Returns:
            vals(dict)
        """
        if transformations is None:
            transformations = self.init_transformations

        return transformations[self.eef_name].pos

    def get_eef_ori(self, transformations=None):
        """
        Get end effector's orientation

        Args:
            transformations(OrderedDict)
        
        Returns:
            vals(dict)
        """
        if transformations is None:
            transformations = self.init_transformations

        return transformations[self.eef_name].rot

    def get_result_qpos(self, init_qpos, eef_pose, n_trials=5):
        cnt = 0
        is_limit_qpos = False
        while cnt <= n_trials:
            result_qpos = self.inverse_kin(np.random.randn(len(init_qpos)), eef_pose, "LM", 500)
            is_limit_qpos = self.check_limit_joint(result_qpos)
            if is_limit_qpos:
                break
            cnt += 1

        if not is_limit_qpos:
            return None
        return result_qpos

    @property
    def base_name(self):
        return self._base_name
        
    @property
    def eef_name(self):
        return self._eef_name

    @property
    def active_joint_names(self):
        return self._revolute_joint_names

    @property
    def arm_dof(self):
        return len([ joint for joint in self.get_revolute_joint_names() if "head" not in joint])
