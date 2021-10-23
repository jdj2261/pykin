import sys, os
import numpy as np
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../" )
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.kinematics.kinematics import Kinematics
from pykin.models.urdf_model import URDFModel

class Robot(URDFModel):
    """
    Initializes a robot object, as defined by a single corresponding robot URDF

    Args:
        fname (str): path to the urdf file.
        offset (Transform): robot init offset
    """
    def __init__(
        self, 
        fname=None, 
        offset=None, 
    ):
        if fname is None:
            fname = pykin_path + "/asset/urdf/baxter/baxter.urdf"
        self._offset = offset
        if offset is None:
            self._offset = Transform()
            
        super(Robot, self).__init__(fname)

        self.joint_limits_lower = []
        self.joint_limits_upper = []

        self._setup_kinematics()
        self._setup_init_transform()

        self.joint_limits = self._get_limited_joint_names()

    def __str__(self):
        return f"""ROBOT : {self.robot_name} 
        {self.links} 
        {self.joints}"""

    def __repr__(self):
        return 'pykin.robot.{}()'.format(type(self).__name__)

    def show_robot_info(self):
        """
        Shows robot's info 
        """
        print("*" * 100)
        print(f"Robot Information:")

        for link in self.links.values():
            print(link)
        for joint in self.joints.values():
            print(joint)
            
        print(f"robot's dof : {self.dof}")
        print(f"active joint names: \n{self.get_actuated_joint_names()}")
        print(f"revolute joint names: \n{self.get_revolute_joint_names()}")
        print("*" * 100)

    def compute_pose_error(self, target_HT=np.eye(4), result_HT=np.eye(4)):
        """
        Computes pose(homogeneous transform) error 

        Args:
            target_HT (np.array): target homogeneous transform
            result_HT (np.array): result homogeneous transform 

        Returns:
            error (np.array)
        """
        error = np.round(np.linalg.norm(
            np.dot(result_HT, np.linalg.inv(target_HT)) - np.mat(np.eye(4))), 6)
        return error
        
    def _setup_kinematics(self):
        """
        Setup Kinematics
        """
        self.kin = Kinematics(robot_name=self.robot_name,
                              offset=self.offset,
                              active_joint_names=self.get_revolute_joint_names(),
                              base_name="", 
                              eef_name=None
                              )
    
    def _setup_init_transform(self):
        """
        Initializes robot's transformation
        """
        thetas = np.zeros(len(self.get_revolute_joint_names()))
        transformations = self.kin.forward_kinematics(self.root, thetas)
        self.init_transformations = transformations

    def _get_limited_joint_names(self):
        result = {}
        for joint, value in self.joints.items():
            for active_joint in self.get_revolute_joint_names():
                if joint == active_joint:
                    result.update({joint : (value.limit[0], value.limit[1])})
        return result

    def setup_link_name(self, base_name, eef_name):
        """
        Sets robot's link name

        Args:
            base_name (str): reference link name
            eef_name (str): end effector name
        """
        raise NotImplementedError

    def forward_kin(self, thetas, desired_frames=None):
        if desired_frames is not None:
            self._frames = desired_frames
        else:
            self._convert_desired_frames_to_all_frames()
        transformation = self.kin.forward_kinematics(self._frames, thetas)
        return transformation

    def _convert_desired_frames_to_all_frames(self):
        """
        Resets robot's desired frame
        """
        self._frames = self.root
        self._revolute_joint_names = self.get_revolute_joint_names()

    def inverse_kin(self, current_joints, target_pose, method, maxIter):
        raise NotImplementedError

    def _set_joint_limits_upper_and_lower(self):
        raise NotImplementedError

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

    @property
    def base_name(self):
        raise NotImplementedError
    
    @property
    def eef_name(self):
        raise NotImplementedError

    @property
    def frame(self):
        raise NotImplementedError

    @property
    def active_joint_names(self):
        raise NotImplementedError

