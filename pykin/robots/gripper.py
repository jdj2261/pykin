import numpy as np
from collections import OrderedDict

from pykin.utils.mesh_utils import get_absolute_transform

class Gripper:
    def __init__(
        self, 
        name, 
        element_names,
        max_width,
        max_depth,
        tcp_position,
    ):
        self.name = name
        self.element_names = element_names
        self.max_width = max_width
        self.max_depth = max_depth
        self.tcp_position = tcp_position
        
        self.info = OrderedDict()
        
        self.is_attached = False
        self.attached_obj_name = None
        
        self.grasp_pose = None
        self.release_pose = None
    
        self.transform_bet_gripper_n_obj = None
        self.pick_obj_pose = None
        self.place_obj_pose = None

    def get_gripper_pose(self):
        return self.info["right_hand"][3]

    def set_gripper_pose(self, eef_pose=np.eye(4)):
        tcp_pose = self.compute_tcp_pose_from_eef_pose(eef_pose)
        T = get_absolute_transform(self.info[self.element_names[-1]][3], tcp_pose)
        for link, info in self.info.items():
            self.info[link][3] = np.dot(T, info[3])
            
    def get_gripper_tcp_pose(self):
        return self.info["tcp"][3]

    def set_gripper_tcp_pose(self, tcp_pose=np.eye(4)):
        T = get_absolute_transform(self.info[self.element_names[-1]][3], tcp_pose)
        for link, info in self.info.items():
            self.info[link][3] = np.dot(T, info[3])

    def compute_eef_pose_from_tcp_pose(self, tcp_pose=np.eye(4)):
        eef_pose = np.eye(4)
        eef_pose[:3, :3] = tcp_pose[:3, :3]
        eef_pose[:3, 3] = tcp_pose[:3, 3] - np.dot(abs(self.tcp_position[-1]), tcp_pose[:3, 2])
        return eef_pose

    def compute_tcp_pose_from_eef_pose(self, eef_pose=np.eye(4)):
        tcp_pose = np.eye(4)
        tcp_pose[:3, :3] = eef_pose[:3, :3]
        tcp_pose[:3, 3] = eef_pose[:3, 3] + np.dot(abs(self.tcp_position[-1]), eef_pose[:3, 2])
        return tcp_pose

    def get_gripper_fk(self):
        fk = {}
        for link, info in self.info.items():
            fk[link] = info[3]
        return fk
        
class PandaGripper(Gripper):
    def __init__(self):
        gripper_name="panda_gripper"
        element_names=["right_hand", "right_gripper", "leftfinger", "rightfinger", "tcp"]
        max_width=0.08
        max_depth=0.035
        tcp_position=np.array([0, 0, 0.097])
        super(PandaGripper, self).__init__(gripper_name, element_names, max_width, max_depth, tcp_position)

class Robotiq140Gripper(Gripper):
    def __init__(self):
        gripper_name="robotiq140_gripper"
        element_names=["right_hand", "right_gripper", "left_outer_knuckle", "right_outer_knuckle", \
                       "left_outer_finger", "right_outer_finger", "left_inner_knuckle", "right_inner_knuckle", \
                       "left_inner_finger", "right_inner_finger", "right_inner_finger_pad","left_inner_finger_pad", "collision_pad", "tcp"]
        max_width=0.140
        max_depth=0.2
        tcp_position=np.array([0, 0, 0.2075])
        super(Robotiq140Gripper, self).__init__(gripper_name, element_names, max_width, max_depth, tcp_position)