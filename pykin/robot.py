
import sys, os
import numpy as np
from pprint import pprint
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../" )
sys.path.append(pykin_path)

from pykin.kinematics import jacobian as jac
from pykin.kinematics.kinematics import Kinematics
from pykin.kinematics.transform import Transform
from pykin.urdf.urdf_parser import URDFParser
from pykin.utils import plot as plt
from pykin.utils.shell_color import ShellColors as scolors
from pykin.utils.logs import logging_time

class Robot:
    def __init__(self, filepath=None, offset=Transform(), joint_safety=False):
        if filepath is None:
            filepath = pykin_path + "/asset/urdf/baxter.urdf"
        self._offset = offset
        self.tree = None
        self.desired_frame = None
        self._load_urdf(filepath)
        self.joint_safety = joint_safety
        if self.joint_safety and self.desired_frame is None:
            self._set_joint_limit()

    def __repr__(self):
        return f"""ROBOT : {__class__.__name__} 
        {self.links} 
        {self.joints}"""

    def _load_urdf(self, filepath):
        parser = URDFParser(filepath)
        self.tree = parser.tree
        self.tree.offset = self.offset
        self.__kinematics = Kinematics(self.tree)

    @property
    def links(self):
        links = []
        for link in self.tree.links.values():
            links.append(link)
        return links

    @property
    def joints(self):
        joints = []
        for joint in self.tree.joints.values():
            joints.append(joint)
        return joints

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

    @property
    def num_dofs(self):
        return self.tree.num_dofs

    @property
    def num_links(self):
        return self.tree.num_links

    @property
    def num_joints(self):
        return self.tree.num_joints

    @property
    def num_active_joints(self):
        return self.tree.num_actuated_joints

    @property
    def get_active_joint_names(self):
        return self.tree.get_joint_parameter_names

    def show_robot_info(self):
        print("*" * 20)
        print(f"robot information: \n{self}")
        print(f"robot's dof : {self.num_dofs}")
        print(f"active joint's info: \n{self.get_active_joint_names}")
        print("*" * 20)

    def set_desired_tree(self, root_link="", end_link=""):
        self.desired_frame = self.tree._set_desired_tree(root_link, end_link)
        if self.joint_safety:
            self._set_joint_limit()
        return self.desired_frame

    def _set_joint_limit(self):
        self.joint_limits_lower = []
        self.joint_limits_upper = []

        def replace_none(x, v):
            if x is None:
                return v
            return x

        if self.desired_frame is None:
            for f in self.tree.joints.values():
                for joint_name in self.get_active_joint_names:
                    if f.name == joint_name:
                        self.joint_limits_lower.append(f.limit[0])
                        self.joint_limits_upper.append(f.limit[1])
        else:
            for f in self.desired_frame:
                for joint_name in self.get_active_joint_names:
                    if f.joint.name == joint_name:
                        self.joint_limits_lower.append(f.joint.limit[0])
                        self.joint_limits_upper.append(f.joint.limit[1])

        self.joint_limits_lower = np.array([replace_none(jl, -np.inf)
                                            for jl in self.joint_limits_lower])
        self.joint_limits_upper = np.array([replace_none(jl, np.inf)
                                            for jl in self.joint_limits_upper])

    def get_joint_limit(self, joints):
        return np.clip(joints, self.joint_limits_lower, self.joint_limits_upper)

    def forward_kinematics(self, theta, desired_tree=None):
        self.transformations = self.__kinematics.forward_kinematics(
            theta, offset=self._offset, desired_tree=self.desired_frame
        )
        return self.transformations

    @logging_time
    def inverse_kinematics(
        self, current_joints, target_pose, method="LM", desired_tree=None, maxIter=1000
    ):

        if method == "analytical":
            joints = self.__kinematics.analytical_inverse_kinematics(target_pose)
        if method == "NR":
            joints = self.__kinematics.numerical_inverse_kinematics_NR(
                current_joints, target_pose, 
                desired_tree=self.desired_frame, 
                lower = self.joint_limits_lower, 
                upper=self.joint_limits_upper,
                maxIter=maxIter
            )
        else:
            joints = self.__kinematics.numerical_inverse_kinematics_LM(
                current_joints, target_pose, 
                desired_tree=self.desired_frame, 
                lower=self.joint_limits_lower, 
                upper=self.joint_limits_upper,
                maxIter=maxIter
            )
        return joints

    def jacobian(self, fk, th):
        return jac.calc_jacobian(self.desired_frame, fk, th)
