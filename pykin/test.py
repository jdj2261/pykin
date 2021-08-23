import sys, os
import numpy as np
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../" )
sys.path.append(pykin_path)
from pykin.kinematics import jacobian as jac
from pykin.kinematics.kinematics import Kinematics
from pykin.kinematics.transform import Transform
from pykin.geometry.geometry import Geometry
from pykin.models.urdf_model import URDFModel
from pykin.utils.kin_utils import logging_time
import pykin.utils.plot_utils as plt

class Robot(URDFModel):
    """
    Initializes a robot object, as defined by a single corresponding robot URDF

    Args:
        fname (str): 
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

        self.setup_kinematics()

    def __repr__(self):
        return f"""ROBOT : {self.robot_name} 
        {self.links} 
        {self.joints}"""

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

    def show_robot_info(self):
        print("*" * 100)
        print(f"Robot Information: \n{self}")
        print(f"robot's dof : {self.dof}")
        print(f"active joint names: \n{self.get_actuated_joint_names()}")
        print("*" * 100)

    def setup_kinematics(self):
        self.kin = Kinematics(robot_name=self.robot_name,
                              offset=self.offset,
                              active_joint_names=self.get_actuated_joint_names(),
                              base_name="", 
                              eef_name=None,
                              frames=self.root)

    def set_desired_frame(self, base_name="", eef_name=None):
        self.kin.base_name = base_name
        self.kin.eef_name = eef_name

        if base_name == "":
            desired_base_frame = self.root
        else:
            desired_base_frame = self.find_frame(base_name + "_frame")

        self.desired_frames = self.generate_desired_frame_recursive(desired_base_frame, eef_name)
        self.kin.frames = self.desired_frames
        self.kin.active_joint_names = self.get_actuated_joint_names(self.kin.frames)

    @property
    def transformations(self):
        return self.kin.transformations

if __name__ == "__main__":
    robot = Robot(fname="../asset/urdf/baxter/baxter.urdf")

    # robot.set_desired_frame(base_name="base", eef_name="left_wrist")
    # left_arm_thetas = [0, 0, np.pi, 0, 0, 0, 0]

    # robot.kin.frames = robot.root
    # robot.kin.active_joint_names = robot.get_actuated_joint_names()

    left_arm_thetas = np.zeros(15)
    transformations = robot.kin.forward_kinematics(left_arm_thetas)

    for link in robot.links.values():
        print(link.name, link.dtype)
    # for link, transformation in transformations.items():
    #     print(link, transformation)
    _, ax = plt.init_3d_figure()
    plt.plot_robot(robot, 
                   ax, 
                   name="baxter",
                   visible_collision=True, visible_mesh=True, mesh_path='../asset/urdf/baxter/')
    ax.legend()
    plt.show_figure()

    # robot.kin.forward_kinematics(left_arm_thetas)
    # robot.show_robot_info()
    # print(robot.find_frame("left_wrist_frame"))
    # print(robot.get_urdf())