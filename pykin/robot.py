import sys, os
import numpy as np
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../" )
sys.path.append(pykin_path)

from pykin.kinematics.kinematics import Kinematics
from pykin.kinematics.transform import Transform
from pykin.models.urdf_model import URDFModel

class Robot(URDFModel):
    """
    Initializes a robot object, as defined by a single corresponding robot URDF

    Args:
        fname (str): path to the urdf file.
    """
    def __init__(
        self, 
        fname=None, 
        offset=Transform(), 
    ):
        if fname is None:
            fname = pykin_path + "/asset/urdf/baxter/baxter.urdf"

        self._offset = offset
        super(Robot, self).__init__(fname)

        # self.transformations = None
        self.fcl_utils = None
        self.kin = None
        self._setup_kinematics()

    def __str__(self):
        return f"""ROBOT : {self.robot_name} 
        {self.links} 
        {self.joints}"""

    def __repr__(self):
        return 'pykin.robot.{}()'.format(type(self).__name__)

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

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
        print("*" * 100)

    def compute_pose_error(self, target=np.eye(4), result=np.eye(4)):
        """
        Computes pose error

        Args:
            target (np.array): target pose
            result (np.array): result pose 

        Returns:
            error (np.array)
        """
        error = np.linalg.norm(np.dot(result, np.linalg.inv(target)) - np.mat(np.eye(4)))
        return error

    def _setup_kinematics(self):
        """
        Sets instance of Kinematics
        """
        self.kin = Kinematics(robot_name=self.robot_name,
                              offset=self.offset,
                              active_joint_names=self.get_actuated_joint_names(),
                              base_name="", 
                              eef_name=None,
                              frames=self.root
                              )
        self._init_transform()
        
    def _init_transform(self):
        """
        Initializes robot's transformation
        """
        thetas = np.zeros(self.dof)
        self.kin.forward_kinematics(thetas)

    def set_desired_frame(self, base_name="", eef_name=None):
        """
        Sets robot's desired frame

        Args:
            base_name (str): reference link name
            eef_name (str): end effector name
        """
        self.kin.base_name = base_name
        self.kin.eef_name = eef_name

        if base_name == "":
            desired_base_frame = self.root
        else:
            desired_base_frame = self.find_frame(base_name + "_frame")

        self.desired_frames = self.generate_desired_frame_recursive(desired_base_frame, eef_name)
        self.kin.frames = self.desired_frames
        self.kin.active_joint_names = self.get_actuated_joint_names(self.kin.frames)

    def reset_desired_frames(self):
        """
        Resets robot's desired frame
        """
        self.kin.frames = self.root
        self.kin.active_joint_names = self.get_actuated_joint_names()

    @property
    def transformations(self):
        return self.kin._transformations

    @transformations.setter
    def transformations(self, transformations):
        self.transformations = transformations

    @property
    def active_joint_names(self):
        return self.kin._active_joint_names