import sys, os
import numpy as np
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../" )
sys.path.append(pykin_path)
from pykin.kinematics.kinematics import Kinematics
from pykin.kinematics.transform import Transform
from pykin.models.urdf_model import URDFModel
from pykin.utils.fcl_utils import FclUtils, convert_fcl_objects
import pykin.utils.plot_utils as plt
import matplotlib.animation as animation

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

        # self.transformations = None
        self.fcl_utils = None
        self.kin = None
        self.setup_collision_check()
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
        print(f"active joint names: \n{self._get_actuated_joint_names()}")
        print("*" * 100)

    def setup_collision_check(self):
        fcl_objects = convert_fcl_objects(self.links)
        self.fcl_utils = FclUtils(fcl_objects=fcl_objects)

    def setup_kinematics(self):
        self.kin = Kinematics(robot_name=self.robot_name,
                              offset=self.offset,
                              active_joint_names=self._get_actuated_joint_names(),
                              base_name="", 
                              eef_name=None,
                              frames=self.root,
                              fcl_utils=self.fcl_utils)

    def set_desired_frame(self, base_name="", eef_name=None):
        self.kin.base_name = base_name
        self.kin.eef_name = eef_name

        if base_name == "":
            desired_base_frame = self.root
        else:
            desired_base_frame = self.find_frame(base_name + "_frame")

        self.desired_frames = self.generate_desired_frame_recursive(desired_base_frame, eef_name)
        self.kin.frames = self.desired_frames
        self.kin.active_joint_names = self._get_actuated_joint_names(self.kin.frames)

    def reset_desired_frames(self):
        self.kin.frames = self.root
        self.kin.active_joint_names = self._get_actuated_joint_names()

    @property
    def transformations(self):
        return self.kin._transformations

    @property
    def active_joint_names(self):
        return self.kin._active_joint_names

if __name__ == "__main__":
    robot = Robot(fname="../asset/urdf/baxter/baxter.urdf")

    head_thetas = [0]
    left_thetas = np.array([0, 0, 0, 0, 0, 0, 0])
    right_thetas = np.array([0, 0, 0, 0, 0, 0, 0])

    # init_thetas = np.random.randn(7)
    # baxter_thetas = np.concatenate((head_thetas, left_thetas, right_thetas))
    # transformations = robot.kin.forward_kinematics(baxter_thetas)q
    
    # target_l_pose = np.hstack((transformations["left_wrist"].pos, transformations["left_wrist"].rot))
    # target_r_pose = np.hstack((transformations["right_wrist"].pos, transformations["right_wrist"].rot))

    robot.set_desired_frame(base_name="base", eef_name="left_wrist")
    left_arm_thetas = [np.pi, 0, 0, 0, 0, 0, 0]
    init_left_thetas = np.random.randn(7)
    left_transformations = robot.kin.forward_kinematics(left_arm_thetas)
    target_l_pose = np.concatenate((left_transformations["left_wrist"].pos, left_transformations["left_wrist"].rot))
    ik_left_result, trajectory_joints_l = robot.kin.inverse_kinematics(init_left_thetas, 
                                                                     target_l_pose, 
                                                                     method="LM", 
                                                                     maxIter=100)
    
    robot.set_desired_frame(base_name="base", eef_name="right_wrist")
    right_arm_thetas = [np.pi, 0, 0, 0, 0, 0, 0]
    init_right_thetas = np.random.randn(7)
    right_transformations = robot.kin.forward_kinematics(right_arm_thetas)
    target_r_pose = np.concatenate((right_transformations["right_wrist"].pos, right_transformations["right_wrist"].rot))
    ik_right_result, trajectory_joints_r = robot.kin.inverse_kinematics(init_right_thetas, 
                                                                        target_r_pose, 
                                                                        method="LM", 
                                                                        maxIter=100)

    trajectory_joints = list(zip(trajectory_joints_l, trajectory_joints_r))

    trajectory_pos = []
    robot.reset_desired_frames()
    for left_joint, right_joint in trajectory_joints:
        current_joint = np.concatenate((head_thetas, left_joint, right_joint))
        test_pose = robot.kin.forward_kinematics(current_joint)
        trajectory_pos.append(test_pose)
    # trajectory_pos = np.array(trajectory_pos)
    # target_pose = left_transformations["left_wrist"].matrix()
    # result_pose = trajectory_pos[-1]["left_wrist"].matrix()

    # print(f"Desired Pose: {target_pose}")
    # print(f"Current Pose: {result_pose}")
    # # print(f"Current Pose: {}")
    plt.plot_anmation(robot, trajectory_pos, interval=10)
