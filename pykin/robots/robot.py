import sys, os
import numpy as np

import signal
def handler(signum, frame):
    exit()
# Set the signal handler
signal.signal(signal.SIGINT, handler)

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../" )
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.kinematics.kinematics import Kinematics
from pykin.models.urdf_model import URDFModel
from pykin.utils.transform_utils import compute_pose_error

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
        self._setup_init_fk()

        self.joint_limits = self._get_limited_joints()

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
        print(f"active joint names: \n{self.get_all_active_joint_names()}")
        print(f"revolute joint names: \n{self.get_revolute_joint_names()}")
        print("*" * 100)

    def _setup_kinematics(self):
        """
        Setup Kinematics
        """
        self.kin = Kinematics(robot_name=self.robot_name,
                              offset=self.offset,
                              active_joint_names=super().get_revolute_joint_names(),
                              base_name="", 
                              eef_name=None
                              )
    
    def _setup_init_fk(self):
        """
        Initializes robot's forward kinematics
        """
        thetas = np.zeros(len(super().get_revolute_joint_names()))
        fk = self.kin.forward_kinematics(self.root, thetas)
        self.init_fk = fk

    def _get_limited_joints(self):
        """
        Get limit joint

        Returns:
            result (dict): joint_name: (limit joint lower, limit joint upper)
        """
        result = {}
        for joint, value in self.joints.items():
            for active_joint in super().get_revolute_joint_names():
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

    def forward_kin(self, thetas):
        """
        Sets robot's link name

        Args:
            thetas (sequence of float): input joint angles

        Returns:
            fk (OrderedDict): transformations
        """
        self._frames = self.root
        fk = self.kin.forward_kinematics(self._frames, thetas)
        return fk

    def inverse_kin(self, current_joints, target_pose, method, max_iter):
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
        raise NotImplementedError

    def _set_joint_limits_upper_and_lower(self):
        """
        Set joint limits upper and lower
        """
        raise NotImplementedError

    def get_pose_error(self, target=np.eye(4), result=np.eye(4)):
        """
        Get pose(homogeneous transform) error 

        Args:
            target (np.array): target homogeneous transform
            result (np.array): result homogeneous transform 

        Returns:
            error (np.array)
        """
        return compute_pose_error(target, result)

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

    @property
    def init_qpos(self):
        raise NotImplementedError
