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
        self._active_joint_names = self.get_actuated_joint_names()


    def setup_link_name(self, base_name="", eef_name=None):
        """
        Sets robot's desired frame

        Args:
            base_name (str): reference link name
            eef_name (str): end effector name
        """
        self._check_link_name(base_name, eef_name)
        self.base_name = base_name
        self.eef_name = eef_name
        self._set_desired_frame()

    def _check_link_name(self, base_name, eef_pose):
        if base_name and not base_name in self.links.keys():
            print(self.links.keys())
            raise NotFoundError(base_name)

        if eef_pose is not None and eef_pose not in self.links.keys():
            print(self.links.keys())
            raise NotFoundError(eef_pose)

    def _set_desired_base_frame(self):
        if self.base_name == "":
            self.desired_base_frame = self.root
        else:
            self.desired_base_frame = self.find_frame(self.base_name + "_frame")

    def _set_desired_frame(self):
        self._set_desired_base_frame()
        self.frames = self.generate_desired_frame_recursive(self.desired_base_frame, self.eef_name)
        self._active_joint_names = sorted(self.get_actuated_joint_names(self.frames))

    def _remove_desired_frames(self):
        """
        Resets robot's desired frame
        """
        self.frames = self.root
        self._active_joint_names = self.get_actuated_joint_names()

    def forward_kin(self, thetas):
        self._remove_desired_frames()
        transformation = self.kin.forward_kinematics(self.frames, thetas)
        return transformation

    def inverse_kin(self, current_joints, target_pose, method="LM", maxIter=1000):
        self._set_desired_frame()
        joints = self.kin.inverse_kinematics(
            self.frames,
            current_joints,
            target_pose,
            method,
            maxIter=1000)
        return joints

    @property
    def base_name(self):
        return self._base_name

    @base_name.setter
    def base_name(self, name):
        self._base_name = name

    @property
    def eef_name(self):
        return self._eef_name

    @eef_name.setter
    def eef_name(self, name):
        self._eef_name = name

    @property
    def eef_pos(self):
        return self.kin._transformations[self.eef_name].pos

    @property
    def eef_rot(self):
        return self.kin._transformations[self.eef_name].rot

    @property
    def eef_pose(self):
        return np.concatenate((self.eef_pos, self.eef_rot))

    @property
    def active_joint_names(self):
        return self._active_joint_names