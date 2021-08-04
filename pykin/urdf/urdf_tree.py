import numpy as np
from collections import OrderedDict

from pykin.kinematics import transform as tf
from pykin.kinematics import transformation


class URDFTree:
    def __init__(self, name=None, offset=tf.Transform(), root=None):
        self.name = name
        self.offset = offset
        self.links = OrderedDict()
        self.joints = OrderedDict()
        self.root = root
        self.desired_root = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

    @property
    def num_dofs(self):
        return sum([joint.num_dof for joint in self.joints.values()])

    @property
    def num_links(self):
        return len(self.links)

    @property
    def num_joints(self):
        return len(self.joints)

    @property
    def num_fixed_joints(self):
        return sum([1 for joint in self.joints.values() if joint.num_dof == 0])

    @property
    def num_actuated_joints(self):
        return sum([1 for joint in self.joints.values() if joint.num_dof != 0])

    def find_frame(self, name):
        if self.root.name == name:
            return self.root
        return self._find_frame_recursive(name, self.root)

    @staticmethod
    def _find_frame_recursive(name, frame):
        for child in frame.children:
            if child.name == name:
                return child
            ret = URDFTree._find_frame_recursive(name, child)
            if not ret is None:
                return ret
        return None

    def find_link(self, name):
        if self.root.link.name == name:
            return self.root.link
        return self._find_link_recursive(name, self.root)

    @staticmethod
    def _find_link_recursive(name, frame):
        for child in frame.children:
            if child.link.name == name:
                return child.link
            ret = URDFTree._find_link_recursive(name, child)
            if not ret is None:
                return ret
        return None

    @property
    def get_joint_parameter_names(self):
        joint_names = []
        joint_names = self._get_joint_parameter_names(joint_names, self.root)
        return joint_names

    def _get_joint_parameter_names(self, joint_names, frame):
        if frame.joint.num_dof != 0:
            joint_names.append(frame.joint.name)
        for child in frame.children:
            self._get_joint_parameter_names(joint_names, child)
        return joint_names

    def _set_desired_tree(self, root_link_name="", end_link_name=""):
        if root_link_name == "":
            self.desired_root = self.root
        else:
            self.desired_root = self.find_frame(root_link_name + "_frame")
        self.desired_frame = self._generate_desired_tree_recursive(
            self.desired_root, end_link_name
        )
        if self.desired_root is None:
            raise ValueError("Invalid end frame name %s." % end_link_name)
        self.desired_frame = [self.desired_root] + self.desired_frame
        return self.desired_frame

    @staticmethod
    def _generate_desired_tree_recursive(root_frame, end_link_name):
        for child in root_frame.children:
            if child.link.name == end_link_name:
                return [child]
            else:
                frames = URDFTree._generate_desired_tree_recursive(child, end_link_name)
                if frames is not None:
                    return [child] + frames

    def get_desired_joint_parameter_names(self):
        names = []
        for f in self.desired_frame:
            if f.joint.dtype == "fixed":
                continue
            names.append(f.joint.name)
        return names
