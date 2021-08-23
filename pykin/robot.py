
import sys, os
import numpy as np
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../" )
sys.path.append(pykin_path)

from pykin.kinematics import jacobian as jac
from pykin.kinematics.kinematics import Kinematics
from pykin.kinematics.transform import Transform
from pykin.geometry.geometry import Geometry
from pykin.urdf.urdf_parser import URDFParser
from pykin.utils.logs import logging_time


class Robot:
    """
    Initializes a robot object, as defined by a single corresponding robot URDF

    Args:
        file_path (str): 
    """
    def __init__(
        self, 
        file_path=None, 
        offset=Transform(), 
        joint_safety=False,
    ):
        if file_path is None:
            file_path = pykin_path + "/asset/urdf/baxter.urdf"
        self._offset = offset
        self.joint_safety = joint_safety
        
        self.robot_tree = None
        self.desired_robot_tree = None
        self._load_urdf(file_path)

        self.joint_limits_lower = None
        self.joint_limits_upper = None
        if self.joint_safety:
            self._set_joint_limit()

    def __repr__(self):
        return f"""ROBOT : {__class__.__name__} 
        {self.robot_link} 
        {self.robot_joint}"""

    def _load_urdf(self, file_path):
        parser = URDFParser(file_path)
        self.robot_tree = parser.tree
        self.robot_tree.offset = self.offset

        # TODO
        self.__kinematics = Kinematics(self.robot_tree)


    @property
    def robot_link(self):
        links = []
        for link in self.robot_tree.links.values():
            links.append(link)
        return links

    @property
    def robot_joint(self):
        joints = []
        for joint in self.robot_tree.joints.values():
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
        return self.robot_tree.num_dofs

    @property
    def num_links(self):
        return self.robot_tree.num_links

    @property
    def num_joints(self):
        return self.robot_tree.num_joints

    @property
    def num_active_joints(self):
        return self.robot_tree.num_actuated_joints

    @property
    def get_active_joint_names(self):
        return self.robot_tree.get_joint_parameter_names

    def show_robot_info(self):
        print("*" * 20)
        print(f"robot information: \n{self}")
        print(f"robot's dof : {self.num_dofs}")
        print(f"active joint's info: \n{self.get_active_joint_names}")
        print("*" * 20)

    def set_desired_robot_tree(self, root_link="", end_link=""):
        self.desired_robot_tree = self.robot_tree.set_desired_robot_tree(root_link, end_link)
        if self.joint_safety:
            self._set_joint_limit()
        return self.desired_robot_tree

    def _set_joint_limit(self):
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        
        def replace_none(x, v):
            if x is None:
                return v
            return x

        if self.desired_robot_tree is None:
            for f in self.robot_tree.joints.values():
                for joint_name in self.get_active_joint_names:
                    if f.name == joint_name:
                        self.joint_limits_lower.append(f.limit[0])
                        self.joint_limits_upper.append(f.limit[1])
        else:
            for f in self.desired_robot_tree:
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

    def forward_kinematics(self, theta):
        self.transformations = self.__kinematics.forward_kinematics(
            theta, offset=self._offset, desired_tree=self.desired_robot_tree
        )
        return self.transformations

    @logging_time
    def inverse_kinematics(self, current_joints, target_pose, method="LM", maxIter=1000):

        if method == "analytical":
            joints = self.__kinematics.analytical_inverse_kinematics(target_pose)
        if method == "NR":
            joints = self.__kinematics.numerical_inverse_kinematics_NR(
                current_joints, target_pose, 
                desired_tree=self.desired_robot_tree, 
                lower = self.joint_limits_lower, 
                upper=self.joint_limits_upper,
                maxIter=maxIter
            )
        if method == "LM":
            joints = self.__kinematics.numerical_inverse_kinematics_LM(
                current_joints, target_pose, 
                desired_tree=self.desired_robot_tree, 
                lower=self.joint_limits_lower, 
                upper=self.joint_limits_upper,
                maxIter=maxIter
            )
        return joints

    def jacobian(self, fk, th):
        return jac.calc_jacobian(self.desired_robot_tree, fk, th)

    def compute_pose_error(self, target, result):
        error = np.linalg.norm(np.dot(result, np.linalg.inv(target)) - np.mat(np.eye(4)))
        return error
        
    @logging_time
    def set_geomtry(self, fk, visible=False):
        self.geo = Geometry(robot=self, fk=fk)
        self.geo.collision_check(visible=visible)
