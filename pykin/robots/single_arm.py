import numpy as np

from pykin.robots.robot import Robot
from pykin.utils.error_utils import NotFoundError

class SingleArm(Robot):
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
        for joint, (limit_lower, limit_upper) in self.joint_limits.items():
            if "head" in joint:
                continue
            if self.joints[joint].dtype == "revolute":
                if limit_lower is None and limit_upper is None:
                    limit_lower = -np.pi
                    limit_upper = np.pi
                self.joint_limits_lower.append(limit_lower)
                self.joint_limits_upper.append(limit_upper)

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
        self._set_desired_frame()

    def _check_link_name(self, base_name, eef_pose):
        if base_name and not base_name in self.links.keys():
            print(self.links.keys())
            raise NotFoundError(base_name)

        if eef_pose is not None and eef_pose not in self.links.keys():
            print(self.links.keys())
            raise NotFoundError(eef_pose)

    def _set_desired_frame(self):
        self._set_desired_base_frame()
        self.desired_frames = self.generate_desired_frame_recursive(
            self.desired_base_frame, self.eef_name)
        self.frames = self.generate_desired_frame_recursive(self.desired_base_frame, self.eef_name)
        self._revolute_joint_names = sorted(self.get_revolute_joint_names(self.frames))

    def _set_desired_base_frame(self):
        if self.base_name == "":
            self.desired_base_frame = self.root
        else:
            self.desired_base_frame = self.find_frame(self.base_name + "_frame")

    def inverse_kin(self, current_joints, target_pose, method="LM", maxIter=1000):
        self._set_desired_frame()
        joints = self.kin.inverse_kinematics(
            self.frames,
            current_joints,
            target_pose,
            method,
            maxIter=1000)
        return joints

    def compute_eef_pose(self, transformations):
        return np.concatenate((transformations[self.eef_name].pos, transformations[self.eef_name].rot))

    @property
    def base_name(self):
        return self._base_name
        
    @property
    def eef_name(self):
        return self._eef_name

    @property
    def active_joint_names(self):
        return self._revolute_joint_names