
import numpy as np

from pykin.robots.robot import Robot
from pykin.utils.error_utils import NotFoundError
from pykin.utils.transform_utils import get_pose_from_homogeneous

class SingleArm(Robot):
    """
    Initializes a single-armed robot simulation object.

    Args:
        f_name (str): path to the urdf file.
        offset (Transform): robot init offset
    """
    def __init__(
        self,
        f_name:str,
        offset=None,
        has_gripper=False,
        gripper_name="panda_gripper"
    ):
        super(SingleArm, self).__init__(f_name, offset, has_gripper, gripper_name)
        self._base_name = ""
        self._eef_name  = ""
        self.desired_base_frame = ""
        self._set_joint_limits_upper_and_lower()
        self._init_qpos = np.zeros(self.arm_dof)
        
        self.info = self._init_robot_info()

        if has_gripper:
            self.gripper.info = super()._init_gripper_info()
        
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

    def get_result_qpos(self, init_qpos, eef_pose, method="LM", max_iter=100):
        is_limit_qpos = False
        result_qpos = self.inverse_kin(init_qpos, eef_pose, method=method, max_iter=max_iter)
        is_limit_qpos = self.check_limit_joint(result_qpos)
        limit_cnt = 0

        if is_limit_qpos:
            return result_qpos

        while not is_limit_qpos:
            limit_cnt += 1
            if limit_cnt > 3:
                break
            result_qpos = self.inverse_kin(np.random.randn(len(init_qpos)), eef_pose, method=method, max_iter=max_iter)
            is_limit_qpos = self.check_limit_joint(result_qpos)
        return result_qpos

    def get_info(self, geom="all"):
        if geom == "all":
            return self.info

        if geom == "collision":
            return self.info["collision"]
        
        if geom == "visual":
            return self.info["visual"]

    def get_gripper_init_pose(self):
        return self.init_fk["right_gripper"].h_mat

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

    def inverse_kin(self, current_joints, target_pose, method="LM", max_iter=100):
        """
        Returns joint angles obtained by computing IK
        
        Args:
            current_joints (sequence of float): input joint angles
            target_pose (np.array): goal pose to achieve
            method (str): two methods to calculate IK (LM: Levenberg-marquardt, NR: Newton-raphson)
            max_iter (int): Maximum number of calculation iterations

        Returns:
            joints (np.array): target joint angles
        """
        target_pose = np.asarray(target_pose)
        
        if target_pose.shape == (4,4):
            target_pose = get_pose_from_homogeneous(target_pose)

        joints = self.kin.inverse_kinematics(
            self.desired_frames,
            current_joints,
            target_pose,
            method,
            max_iter)
        return joints

    def compute_eef_pose(self, fk=None):
        """
        Get end effector's pose

        Args:
            fk(OrderedDict)
        
        Returns:
            vals(dict)
        """
        if fk is None:
            fk = self.init_fk

        return np.concatenate((fk[self.eef_name].pos, fk[self.eef_name].rot))

    def compute_eef_h_mat(self, fk=None):
        """
        Get end effector's homogeneous marix

        Args:
            fk(OrderedDict)
        
        Returns:
            vals(dict)
        """
        if fk is None:
            fk = self.init_fk

        return fk[self.eef_name].h_mat

    @property
    def base_name(self):
        return self._base_name
        
    @property
    def eef_name(self):
        return self._eef_name

    @eef_name.setter
    def eef_name(self, eef_name):
        self._eef_name = eef_name

    @property
    def active_joint_names(self):
        return self._revolute_joint_names

    @property
    def arm_dof(self):
        return len([ joint for joint in self.get_revolute_joint_names() if "head" not in joint])

    @property
    def init_qpos(self):
        return self._init_qpos
    
    @init_qpos.setter
    def init_qpos(self, init_qpos):
        self._init_qpos = init_qpos
