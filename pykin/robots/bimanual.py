import numpy as np
from pykin.robots.robot import Robot

class Bimanual(Robot):
    def __init__(
        self,
        fname: str,
        offset=None
    ):
        super(Bimanual, self).__init__(fname, offset)

        self.setup_input2dict()
        self._set_joint_limits_upper_and_lower()

    def setup_link_name(self, base_name="", eef_name=None):
        """
        Sets robot's desired frame

        Args:
            base_name (str): reference link name
            eef_name (str): end effector name
        """
        if "right" in eef_name:
            self._base_name["right"] = base_name
            self._eef_name["right"] = eef_name
            self._set_desired_frame("right")
            
        if "left" in eef_name:
            self._base_name["left"] = base_name
            self._eef_name["left"] = eef_name
            self._set_desired_frame("left")

    def setup_input2dict(self):
        self._base_name = self._input2dict("")
        self._eef_name  = self._input2dict("")
        self.desired_base_frame = self._input2dict(None)
        self.desired_frames = self._input2dict(None)
        self._frames = self._input2dict(None)
        self._revolute_joint_names = self._input2dict(None)
        self._target_pose = self._input2dict(None)
        self.joint_limits_lower = self._input2dict(None)
        self.joint_limits_upper = self._input2dict(None)

    def _set_joint_limits_upper_and_lower(self):
        limits_lower = []
        limits_upper = []

        for joint, (limit_lower, limit_upper) in self.joint_limits.items():
            limits_lower.append((joint, limit_lower))
            limits_upper.append((joint, limit_upper))

        for arm in self.arms:
            self.joint_limits_lower[arm] = [limit_lower for joint, limit_lower in limits_lower if arm in joint]
            self.joint_limits_upper[arm] = [limit_upper for joint, limit_upper in limits_upper if arm in joint]
            
    def joints_in_limits(self, q, arm):
        lower_lim = self._input2dict(None)
        upper_lim = self._input2dict(None)
        result = self._input2dict(None)
        
        lower_lim[arm] = self.joint_limits_lower[arm]
        upper_lim[arm] = self.joint_limits_upper[arm]
        result[arm] = np.all([q >= lower_lim[arm], q <= upper_lim[arm]], 0)
        
        return result

    def _set_desired_frame(self, arm):
        self._set_desired_base_frame(arm)
        self.desired_frames[arm] = self.generate_desired_frame_recursive(
            self.desired_base_frame[arm], 
            self.eef_name[arm])
        
        self._frames[arm] = self.desired_frames[arm]
        self._revolute_joint_names[arm] = self.get_revolute_joint_names(self._frames[arm])
        self._target_pose[arm] = np.zeros(len(self._revolute_joint_names[arm]))

    def _set_desired_base_frame(self, arm):
        if self.base_name[arm] == "":
            self.desired_base_frame[arm] = self.root
        else:
            self.desired_base_frame[arm] = self.find_frame(self.base_name[arm] + "_frame")

    def _remove_desired_frames(self):
        """
        Resets robot's desired frame
        """
        self._frames = self.root
        self._revolute_joint_names = self.get_revolute_joint_names()

    def forward_kin(self, thetas, desired_frames=None): 

        if desired_frames is not None:
            self._frames = desired_frames
        else:
            self._remove_desired_frames()

        transformation = self.kin.forward_kinematics(self._frames, thetas)
        return transformation

    def inverse_kin(self, current_joints, target_pose, method="LM", maxIter=1000):
        if not isinstance(target_pose, dict):
            raise TypeError("Be sure to input the target pose in dictionary form.")

        joints = {}
        self._frames = self._input2dict(None)
        self._revolute_joint_names = self._input2dict(None)
        for arm in self.arms:
            if self.eef_name[arm]:
                self._set_desired_frame(arm)
                self._target_pose[arm] = target_pose[arm].flatten()
                joints[arm] = self.kin.inverse_kinematics(
                    self._frames[arm],
                    current_joints,
                    self._target_pose[arm],
                    method,
                    maxIter)
        return joints

    def _input2dict(self, inp):
        """
        Helper function that converts an input that is either a single value or a list into a dict with keys for
        each arm: "right", "left"

        Args:
            inp (str or list or None): Input value to be converted to dict

            :Note: If inp is a list, then assumes format is [right, left]

        Returns:
            dict: Inputs mapped for each robot arm
        """
        # First, convert to list if necessary
        if not isinstance(inp, list):
            inp = [inp for _ in range(2)]
        # Now, convert list to dict and return
        return {key: value for key, value in zip(self.arms, inp)}

    @property
    def arms(self):
        """
        Returns name of arms used as naming convention throughout this module

        Returns:
            2-tuple: ('right', 'left')
        """
        return "right", "left"

    @property
    def base_name(self):
        return self._base_name

    @property
    def eef_name(self):
        return self._eef_name

    @property
    def eef_pos(self):
        vals = {}
        for arm in self.arms:
            if self.eef_name[arm]:
                vals[arm] = self.kin._transformations[self.eef_name[arm]].pos
        return vals

    @property
    def eef_rot(self):
        vals = {}
        for arm in self.arms:
            if self.eef_name[arm]:
                vals[arm] = self.kin._transformations[self.eef_name[arm]].rot
        return vals

    @property
    def eef_pose(self):
        vals = {}
        for arm in self.arms:
            if self.eef_name[arm]:
                vals[arm] = np.concatenate((self.eef_pos[arm], self.eef_rot[arm]))
        return vals

    @property
    def frame(self):
        return self._frames

    @property
    def active_joint_names(self):
        return self._revolute_joint_names
        