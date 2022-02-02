import numpy as np
from pykin.robots.robot import Robot
from pykin.utils.error_utils import NotFoundError

class Bimanual(Robot):
    """
    Initializes a bimanual robot simulation object.

    Args:
        fname (str): path to the urdf file.
        offset (Transform): robot init offset
    """
    def __init__(
        self,
        fname: str,
        offset=None
    ):
        super(Bimanual, self).__init__(fname, offset)
        self._setup_input2dict()
        self._set_joint_limits_upper_and_lower()

    def _setup_input2dict(self):
        """
        Setup dictionary name
        """
        self._base_name = self._input2dict("")
        self._eef_name = {}
        self.desired_base_frame = self._input2dict(None)
        self.desired_frames = self._input2dict(None)
        self._frames = self._input2dict(None)
        self._revolute_joint_names = self._input2dict(None)
        self._target_pose = self._input2dict(None)
        self.joint_limits_lower = self._input2dict(None)
        self.joint_limits_upper = self._input2dict(None)

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
        return {key: value for key, value in zip(self._arms, inp)}

    def _set_joint_limits_upper_and_lower(self):
        """
        Set joint limits upper and lower
        """
        limits_lower = []
        limits_upper = []

        for joint, (limit_lower, limit_upper) in self.joint_limits.items():
            limits_lower.append((joint, limit_lower))
            limits_upper.append((joint, limit_upper))

        for arm in self._arms:
            self.joint_limits_lower[arm] = [
                limit_lower for joint, limit_lower in limits_lower if arm in joint]
            self.joint_limits_upper[arm] = [
                limit_upper for joint, limit_upper in limits_upper if arm in joint]

    def setup_link_name(self, base_name="", eef_name=None):
        """
        Sets robot's link name

        Args:
            base_name (str): reference link name
            eef_name (str): end effector name
        """
        if "right" in eef_name:
            self._base_name["right"] = base_name
            self._eef_name["right"] = eef_name
            self._set_desired_base_frame("right")
            self._set_desired_frame("right")
            
        if "left" in eef_name:
            self._base_name["left"] = base_name
            self._eef_name["left"] = eef_name
            self._set_desired_base_frame("left")
            self._set_desired_frame("left")

    def _set_desired_base_frame(self, arm):
        """
        Sets robot's desired base frame

        Args:
            arm (str): robot arm (right or left)
        """
        if self.base_name[arm] == "":
            self.desired_base_frame[arm] = self.root
        else:
            self.desired_base_frame[arm] = super().find_frame(self.base_name[arm] + "_frame")

    def _set_desired_frame(self, arm):
        """
        Sets robot's desired frame

        Args:
            arm (str): robot arm (right or left)
        """
        self.desired_frames[arm] = super().generate_desired_frame_recursive(
            self.desired_base_frame[arm],
            self.eef_name[arm])

        self._frames[arm] = self.desired_frames[arm]
        self._revolute_joint_names[arm] = super().get_revolute_joint_names(self._frames[arm])
        self._target_pose[arm] = np.zeros(len(self._revolute_joint_names[arm]))

    def inverse_kin(self, current_joints, target_pose, method="LM", max_iter=1000):
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
        if not isinstance(target_pose, dict):
            raise TypeError("Be sure to input the target pose in dictionary form.")

        joints = {}
        self._frames = self._input2dict(None)
        self._revolute_joint_names = self._input2dict(None)
        for arm in target_pose.keys():
            if self.eef_name[arm]:
                self._set_desired_frame(arm)
                self._target_pose[arm] = self._convert_target_pose_type_to_npy(target_pose[arm])

                joints[arm] = self.kin.inverse_kinematics(
                    self._frames[arm],
                    current_joints,
                    self._target_pose[arm],
                    method,
                    max_iter)
        return joints

    def _convert_target_pose_type_to_npy(self, value):
        """
        convert input type to numpy array

        Args:
            value(list or tupe)
        
        Returns:
            np.array
        """
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        return value.flatten()

    def get_eef_pose(self, fk):
        """
        Compute end effector's pose

        Args:
            fk(OrderedDict)
        
        Returns:
            vals(dict)
        """
        vals = {}
        for arm in self.arm_type:
            if self.eef_name[arm]:
                vals[arm] = np.concatenate((fk[self.eef_name[arm]].pos, fk[self.eef_name[arm]].rot))
        return vals

    @property
    def _arms(self):
        """
        Returns name of arms used as naming convention throughout this module

        Returns:
            2-tuple: ('right', 'left')
        """
        return ("right", "left")

    @property
    def arm_type(self):
        """
        Return arm type 
        If number of eef_name is two, return tuple type("right", "left)
        otherwise, return list type(["right] or ["left"])

        Returns:
            arm types (tuple or list)
        """
        if len(self._eef_name.keys()) == 2:
            return self._arms
        elif "right" in self.eef_name.keys():
            return ["right"]
        elif "left" in self.eef_name.keys():
            return ["left"]
        else:
            raise NotFoundError("Can not find robot's arm type")
        
    @property
    def base_name(self):
        return self._base_name

    @property
    def eef_name(self):
        return self._eef_name

    @property
    def active_joint_names(self):
        return self._revolute_joint_names
        
