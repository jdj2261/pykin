import sys, os
import numpy as np
import signal
import trimesh

def handler(signum, frame):
    exit()
# Set the signal handler
signal.signal(signal.SIGINT, handler)

from pykin.robots.gripper import PandaGripper, Robotiq140Gripper
from pykin.kinematics.transform import Transform
from pykin.kinematics.kinematics import Kinematics
from pykin.models.urdf_model import URDFModel
from pykin.utils.transform_utils import compute_pose_error

class Robot(URDFModel):
    """
    Initializes a robot object, as defined by a single corresponding robot URDF

    Args:
        f_name (str): path to the urdf file.
        offset (Transform): robot init offset
    """
    def __init__(
        self,
        f_name,
        offset, 
        has_gripper,
        gripper_name
    ):
        super(Robot, self).__init__(f_name)

        self._offset = offset
        if offset is None:
            self._offset = Transform()
        self.has_gripper = has_gripper
        self.gripper_name = gripper_name
            
        self.urdf_name = os.path.abspath(self.file_path)
        self.mesh_path = os.path.abspath(self.file_path + "/../") + '/'
        self.info = {}
        self.gripper = None

        if has_gripper:
            if "panda" in self.gripper_name:
                self.gripper = PandaGripper()
            if "robotiq140" in self.gripper_name:
                self.gripper = Robotiq140Gripper()

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

    def set_transform(self, thetas):
        fk = self.forward_kin(thetas)
        for link, transform in fk.items():

            collision_h_mat = np.dot(transform.h_mat, self.links[link].collision.offset.h_mat)
            visual_h_mat = np.dot(transform.h_mat, self.links[link].visual.offset.h_mat)

            self.info["collision"][link][3] = collision_h_mat
            self.info["visual"][link][3] = visual_h_mat
            
            if self.has_gripper:
                if link in self.gripper.element_names:
                    self.gripper.info[link][3] = collision_h_mat

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

    def _init_robot_info(self):
        robot_info = {}
        robot_info["collision"] = {}
        robot_info["visual"] = {}

        for link, transform in self.init_fk.items():
            col_gparam = []
            col_gtype = self.links[link].collision.gtype
            
            vis_gparam = []
            vis_gtype = self.links[link].visual.gtype

            if col_gtype == "mesh":
                mesh_path = self.mesh_path + self.links[link].collision.gparam.get('filename')
                mesh = trimesh.load_mesh(mesh_path)
                col_gparam.append(mesh)
            if col_gtype == "box":
                col_gparam.append(self.links[link].collision.gparam.get('size'))
            if col_gtype == "cylinder":
                length = float(self.links[link].collision.gparam.get('length'))
                radius = float(self.links[link].collision.gparam.get('radius'))
                col_gparam.append((length, radius))
            if col_gtype == "sphere":
                col_gparam.append(float(self.links[link].collision.gparam.get('radius')))
            col_h_mat = np.dot(transform.h_mat, self.links[link].collision.offset.h_mat)
            robot_info["collision"][link] = [link, col_gtype, col_gparam, col_h_mat]
            
            if vis_gtype == "mesh":
                for file_name in self.links[link].visual.gparam.get('filename'):
                    mesh_path = self.mesh_path + file_name
                    mesh = trimesh.load_mesh(mesh_path)
                    mesh.apply_scale(self.links.get(link).visual.gparam.get('scale'))
                    vis_gparam.append(mesh)
            if vis_gtype == "box":
                vis_gparam.append(self.links[link].visual.gparam.get('size'))
            if vis_gtype == "cylinder":
                length = float(self.links[link].visual.gparam.get('length'))
                radius = float(self.links[link].visual.gparam.get('radius'))
                vis_gparam.append((length, radius))
            if vis_gtype == "sphere":
                vis_gparam.append(float(self.links[link].visual.gparam.get('radius')))
            vis_h_mat = np.dot(transform.h_mat, self.links[link].visual.offset.h_mat)
            robot_info["visual"][link] = [link, vis_gtype, vis_gparam, vis_h_mat]

        return robot_info

    def _init_gripper_info(self):
        gripper_info = {}
        for link, transform in self.init_fk.items():
            col_gparam = []
            if link in self.gripper.element_names:
                gtype = self.links[link].collision.gtype
                mesh = None
                if gtype == "mesh":
                    mesh_path = self.mesh_path + self.links[link].collision.gparam.get('filename')
                    mesh = trimesh.load_mesh(mesh_path)
                    col_gparam.append(mesh)
                if gtype == "box":
                    col_gparam.append(self.links[link].collision.gparam.get('size'))
                if gtype == "cylinder":
                    length = float(self.links[link].collision.gparam.get('length'))
                    radius = float(self.links[link].collision.gparam.get('radius'))
                    col_gparam.append((length, radius))
                if gtype == "sphere":
                    col_gparam.append(float(self.links[link].collision.gparam.get('radius')))
                h_mat = np.dot(transform.h_mat, self.links[link].collision.offset.h_mat)
                gripper_info[link] = [link, gtype, col_gparam, h_mat]
        return gripper_info

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
