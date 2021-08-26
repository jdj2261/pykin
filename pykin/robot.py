import sys, os
import numpy as np
from itertools import zip_longest
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../" )
sys.path.append(pykin_path)

from pykin.kinematics.kinematics import Kinematics
from pykin.kinematics.transform import Transform
from pykin.models.urdf_model import URDFModel

# from pykin.utils.fcl_utils import FclManager
# from pykin.utils.kin_utils import get_robot_geom
# import pykin.utils.plot_utils as plt

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
        self._setup_kinematics()

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

    def compute_pose_error(self, target=np.eye(4), result=np.eye(4)):
        error = np.linalg.norm(np.dot(result, np.linalg.inv(target)) - np.mat(np.eye(4)))
        return error

    def _setup_kinematics(self):
        self.kin = Kinematics(robot_name=self.robot_name,
                              offset=self.offset,
                              active_joint_names=self._get_actuated_joint_names(),
                              base_name="", 
                              eef_name=None,
                              frames=self.root
                              )
        self._init_transform()
        
    def _init_transform(self):
        thetas = np.zeros(self.dof)
        self.kin.forward_kinematics(thetas)

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

    @transformations.setter
    def transformations(self, transformations):
        self.transformations = transformation

    @property
    def active_joint_names(self):
        return self.kin._active_joint_names

if __name__ == "__main__":
    robot = Robot(fname="../asset/urdf/baxter/baxter.urdf")

    head_thetas = [0]
    left_thetas = np.array([0, -np.pi/2, 0, -np.pi/2, 0, 0, 0])
    right_thetas = np.array([0, 0, 0, 0, 0, 0, 0])

    init_thetas = np.random.randn(7)
    baxter_thetas = np.concatenate((head_thetas, left_thetas, right_thetas))
    transformations = robot.kin.forward_kinematics(baxter_thetas)

    # fcl_manager = FclManager()
    # for link, transformation in transformations.items():
    #     name, gtype, gparam = get_robot_geom(robot.links[link])
    #     transform = transformation.matrix()
    #     fcl_manager.add_object(name, gtype, gparam, transform)
    
    # result, datas, t = fcl_manager.collision_check(
    #     return_names=True, return_data=True)
    # print(result, datas, t)

    # fig, ax = plt.init_3d_figure()
    # plt.plot_robot(robot, transformations, ax, name="baxter", visible_collision=True)
    # plt.show_figure()

    target_l_pose = np.hstack((transformations["left_wrist"].pos, transformations["left_wrist"].rot))
    target_r_pose = np.hstack((transformations["right_wrist"].pos, transformations["right_wrist"].rot))

    robot.set_desired_frame(base_name="base", eef_name="left_wrist")
    left_arm_thetas = [0, np.pi/2, 0, -np.pi/2, 0, 0, 0]
    init_left_thetas = np.random.randn(7)
    left_transformations = robot.kin.forward_kinematics(left_arm_thetas)


    target_l_pose = np.concatenate((left_transformations["left_wrist"].pos, left_transformations["left_wrist"].rot))
    ik_left_result, trajectory_joints_l = robot.kin.inverse_kinematics(init_left_thetas, 
                                                                     target_l_pose, 
                                                                     method="LM", 
                                                                     maxIter=50)

    robot.set_desired_frame(base_name="base", eef_name="right_wrist")
    right_arm_thetas = [0, 0, 0, 0, 0, 0, 0]
    init_right_thetas = np.random.randn(7)
    right_transformations = robot.kin.forward_kinematics(right_arm_thetas)
    target_r_pose = np.concatenate((right_transformations["right_wrist"].pos, right_transformations["right_wrist"].rot))
    ik_right_result, trajectory_joints_r = robot.kin.inverse_kinematics(init_right_thetas, 
                                                                        target_r_pose, 
                                                                        method="LM", 
                                                                        maxIter=50)

    trajectory_joints = list(zip_longest(trajectory_joints_l, trajectory_joints_r))

    fcl_manager = FclManager()

    trajectory_pos = []
    results = []
    robot.reset_desired_frames()
    print(trajectory_joints[0])
    for i, (left_joint, right_joint) in enumerate(trajectory_joints):

        if left_joint is None:
            left_joint = last_left_joint
        if right_joint is None:
            right_joint = last_right_joint
        last_left_joint = left_joint
        last_right_joint = right_joint

        current_joint = np.concatenate((head_thetas, left_joint, right_joint))
        transformations = robot.kin.forward_kinematics(current_joint)
        for link, transformation in transformations.items():
            name, gtype, gparam = get_robot_geom(robot.links[link])
            transform = transformation.matrix()
            fcl_manager.add_object(name, gtype, gparam, transform)
        result, names = fcl_manager.collision_check(return_names=True, return_data=False)
        results.append(result)
        trajectory_pos.append(transformations)
        fcl_manager.remove_all_object()

    plt.plot_anmation(robot, results, trajectory_pos, interval=1, repeat=False)


    # trajectory_pos = np.array(trajectory_pos)
    # target_pose = left_transformations["left_wrist"].matrix()
    # result_pose = trajectory_pos[-1]["left_wrist"].matrix()

    # print(f"Desired Pose: {target_pose}")
    # print(f"Current Pose: {result_pose}")
    # # print(f"Current Pose: {}")
