
import sys, os
import numpy as np
from pprint import pprint
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../" )
sys.path.append(pykin_path)

from pykin.kinematics import jacobian as jac
from pykin.kinematics.kinematics import Kinematics
from pykin.kinematics.transform import Transform
from pykin.urdf.urdf_parser import URDFParser
from pykin.utils import plot as plt
from pykin.utils.shell_color import ShellColors as scolors


class Robot:
    def __init__(self, filepath=None, offset=Transform()):
        if filepath is None:
            filepath = pykin_path + "/asset/urdf/baxter.urdf"
        self._offset = offset
        self.tree = None
        self.desired_frame = None
        self._load_urdf(filepath)

    def __repr__(self):
        return f"""ROBOT : {__class__.__name__} 
        {self.links} 
        {self.joints}"""

    def _load_urdf(self, filepath):
        parser = URDFParser(filepath)
        self.tree = parser.tree
        self.tree.offset = self.offset
        self.__kinematics = Kinematics(self.tree)

    @property
    def links(self):
        links = []
        for link in self.tree.links.values():
            links.append(link)
        return links

    @property
    def joints(self):
        joints = []
        for joint in self.tree.joints.values():
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
        return self.tree.num_dofs

    @property
    def num_links(self):
        return self.tree.num_links

    @property
    def num_joints(self):
        return self.tree.num_joints

    @property
    def num_active_joints(self):
        return self.tree.num_actuated_joints

    @property
    def get_active_joint_names(self):
        return self.tree.get_joint_parameter_names

    def show_robot_info(self):
        print("*" * 20)
        print(f"robot information: \n{self}")
        print(f"robot's dof : {self.num_dofs}")
        print(f"active joint's info: \n{self.get_active_joint_names}")
        print("*" * 20)

    def set_desired_tree(self, root_link="", end_link=""):
        self.desired_frame = self.tree._set_desired_tree(root_link, end_link)
        return self.desired_frame

    def forward_kinematics(self, theta, desired_tree=None):
        self.transformations = self.__kinematics.forward_kinematics(
            theta, offset=self._offset, desired_tree=self.desired_frame
        )
        return self.transformations

    def inverse_kinematics(
        self, current_joints, target_pose, method="numerical", desired_tree=None
    ):
        if method == "analytical":
            return self.__kinematics.analytical_inverse_kinematics(target_pose)
        if method == "numerical":
            return self.__kinematics.numerical_inverse_kinematics(
                current_joints, target_pose, desired_tree=self.desired_frame
            )

    def jacobian(self, fk, th):
        return jac.calc_jacobian(self.desired_frame, fk, th)


if __name__ == "__main__":
    file_path = str(pykin_path) + "/asset/urdf/baxter.urdf"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    robot = Robot(file_path)
    robot.show_robot_info()
    # print(robot)
    ##################################################################
    ## Forward Kinematics
    head_thetas = [0.0]
    right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
    left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]

    # # left_arm_thetas = [0, -np.pi/6, np.pi, -np.pi, 0, -np.pi/6, 0]

    # for jacobian
    robot.set_desired_tree("base", "left_gripper")
    print(robot.tree.get_desired_joint_parameter_names())

    fk = robot.forward_kinematics(left_arm_thetas)
    position, orientation = fk["left_gripper"].pos, fk["left_gripper"].rot
    pose = np.append(position, orientation)
    print(fk)
    # J = robot.jacobian(fk, left_arm_thetas)
    # print(J)
    # pose = [0.06402554 , 1.18271238, 0.320976,   0.49999954, - 0.50000046 , 0.49999954 ,0.50000046]
    # pose = [0.06402611,  0.87651951, - 0.26023899 , 0.2705978, - 0.65328208  ,0.65328088 ,0.2705983]
    pose = [
        0.06402868,
        -0.52665762,
        0.478976,
        -0.49999954,
        -0.50000046,
        0.49999954,
        -0.50000046,
    ]
    result = robot.inverse_kinematics(left_arm_thetas, pose, method="numerical")

    thethas = head_thetas + right_arm_thetas + result
    print(result)
    robot.desired_frame = None
    _, ax = plt.init_3d_figure()
    plt.plot_robot(robot, thethas, ax, "baxter")
    ax.legend()
    plt.show_figure()
    ####################
    ########################################################################

    # thetas = head_thetas + right_arm_thetas + left_arm_thetas
    # transformations = robot.forward_kinematics(thetas)

    # position, orientation = transformations["left_gripper"].pos, transformations["left_gripper"].rot
    # pose = np.append(position, orientation)

    # ## Inverse Kinematics
    # print(f"target pose : {pose}")

    # _, ax = plt.init_3d_figure()
    # plt.plot_robot(robot, thetas, ax, "baxter")
    # ax.legend()
    # plt.show_figure()

    # result1, result2 = robot.inverse_kinematics(pose, method="analytical")
    # thethas = head_thetas + right_arm_thetas + result1
    # print(result1)
    # _, ax = plt.init_3d_figure()
    # plt.plot_robot(robot, thethas, ax, "baxter")
    # ax.legend()
    # plt.show_figure()

    # print(result2)
    # thethas = head_thetas + right_arm_thetas + result2
    # _, ax = plt.init_3d_figure()
    # plt.plot_robot(robot, thethas, ax, "baxter")
    # ax.legend()
    # plt.show_figure()
    ######################################################

    ######################################################
    head_thetas = [0.0]
    right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
    left_arm_thetas = [np.pi / 4, np.pi, 0, 0, 0, 0, 0]
    thetas = head_thetas + right_arm_thetas + left_arm_thetas
    transformations = robot.forward_kinematics(thetas)
    position, orientation = (
        transformations["left_wrist"].pos,
        transformations["left_wrist"].rot,
    )
    pose = np.append(position, orientation)
    print(pose)
    _, ax = plt.init_3d_figure()
    plt.plot_robot(robot, thetas, ax, "baxter")
    ax.legend()
    plt.show_figure()
    ####################################################

    # # # To calculate right arm transformations
    # head_thetas = [0.0]
    # right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
    # left_arm_thetas = [0, 0, 0, 0, 0, 0, 0]

    # thetas = head_thetas + right_arm_thetas + left_arm_thetas
    # robot.forward_kinematics(thetas)
    # active_joints_transformation = robot.get_active_joint_transform()
    # right_links = list(active_joints_transformation)[1:8]

    # right_joint_transforms = []
    # for link, (joint, transform) in active_joints_transformation.items():
    #     for right_link in right_links:
    #         if link == right_link:
    #             right_joint_transforms.append(transform)

    # # print(right_joint_transforms)
    # homogeneous_matrixes = []
    # for right_joint_transform in right_joint_transforms:
    #     T = tf.get_homogeneous_matrix(right_joint_transform.pos, right_joint_transform.rot)
    #     homogeneous_matrixes.append(np.round(T, decimals=5))
    # print(homogeneous_matrixes)
    # _, ax = plt.init_3d_figure()
    # plt.plot_right_arm(robot, homogeneous_matrixes, ax, "test")
    # plt.show_figure()
    # ####################################################

    # ####################################################
    ## iiwa14
    # thetas = [0, 0, 0, 0, 0, 0, 0]

    # transformations = robot.forward_kinematics(thetas)

    # _, ax = plt.init_3d_figure()
    # plt.plot_robot(robot, thetas, ax, "test")
    # ax.legend()
    # plt.show_figure()
    # ####################################################

    # print(robot.get_active_joint_names)
    # print(robot.num_dofs)
    # th = np.zeros(robot.num_dofs)
