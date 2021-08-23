from collections import OrderedDict

class RobotModel:
    def __init__(self, fname=None):
        self.name = fname
        self.root = None
        self._links = OrderedDict()
        self._joints = OrderedDict()

    def find_frame(self, frame_name):
        raise NotImplementedError

    def find_link(self, link_name):
        raise NotImplementedError

    def find_joint(self, joint_name):
        raise NotImplementedError

    @property
    def links(self):
        return self._links

    @property
    def joints(self):
        return self._joints

    @property
    def get_actuated_joint_names(self):
        raise NotImplementedError

    @property
    def dof(self):
        raise NotImplementedError

    @property
    def num_links(self):
        raise NotImplementedError

    @property
    def num_joints(self):
        raise NotImplementedError

    @property
    def num_fixed_joints(self):
        raise NotImplementedError

    @property
    def num_actuated_joints(self):
        raise NotImplementedError