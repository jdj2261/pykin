from abc import abstractclassmethod, ABCMeta
from dataclasses import dataclass
from copy import deepcopy

from pykin.scene.scene_manager import SceneManager
from pykin.utils.mesh_utils import surface_sampling
from pykin.planners.cartesian_planner import CartesianPlanner
from pykin.planners.rrt_star_planner import RRTStarPlanner

@dataclass
class ActionInfo:
    TYPE = "type"
    PICK_OBJ_NAME = "pick_obj_name"
    HELD_OBJ_NAME = "held_obj_name"
    PLACE_OBJ_NAME = "place_obj_name"
    GRASP_POSES = "grasp_poses"
    TCP_POSES = "tcp_poses"
    RELEASE_POSES = "release_poses"
    LEVEL = "level"


@dataclass
class MoveData:
    """
    Grasp Status Enum class
    """
    MOVE_pre_grasp = "pre_grasp"
    MOVE_grasp = "grasp"
    MOVE_post_grasp = "post_grasp"
    MOVE_default_grasp = "default_grasp"
    
    MOVE_pre_release = "pre_release"
    MOVE_release = "release"
    MOVE_post_release = "post_release"
    MOVE_default_release = "default_release"

class ActivityBase(metaclass=ABCMeta):
    """
    Activity Base class

    Args:
        robot (SingleArm or Bimanual): manipulator type
        robot_col_mngr (CollisionManager): robot's CollisionManager
        object_mngr (ObjectManager): object's Manager
    """
    def __init__(
        self,
        scene_mngr:SceneManager,
        retreat_distance=0.1
    ):
        self.scene_mngr = scene_mngr.deepcopy_scene(scene_mngr)
        self.retreat_distance = retreat_distance
        self.info = ActionInfo
        self.move_data = MoveData

        self.cartesian_planner = CartesianPlanner()
        self.rrt_planner = RRTStarPlanner(delta_distance=0.05, epsilon=0.2, gamma_RRT_star=2)

    def __repr__(self) -> str:
        return 'pykin.action.activity.{}()'.format(type(self).__name__)

    @abstractclassmethod
    def get_possible_actions_level_1(self):
        raise NotImplementedError

    @abstractclassmethod
    def get_action_level_1_for_single_object(self):
        raise NotImplementedError

    @abstractclassmethod
    def get_possible_ik_solve_level_2(self):
        raise NotImplementedError

    @abstractclassmethod
    def get_possible_joint_path_level_3(self):
        raise NotImplementedError

    @abstractclassmethod
    def get_possible_transitions(self):
        raise NotImplementedError

    def get_surface_points_from_mesh(self, mesh, n_sampling=100, weights=None):
        contact_points, _, normals = surface_sampling(mesh, n_sampling, weights)
        return contact_points, normals

    def _collide(self, is_only_gripper:bool)->bool:
        collide = False
        if is_only_gripper:
            collide = self.scene_mngr.collide_objs_and_gripper()
        else:
            collide = self.scene_mngr.collide_objs_and_robot()
        return collide

    def _solve_ik(self, pose1, pose2, eps=1e-3):
        pose_error = self.scene_mngr.scene.robot.get_pose_error(pose1, pose2)
        if pose_error < eps:
            return True
        return False
    
    def deepcopy_scene(self, scene=None):
        if scene is None:
            scene = self.scene_mngr.scene
        self.scene_mngr.scene = deepcopy(scene)

    def get_cartesian_path(self, cur_q, goal_pose, n_step=500, collision_check=False):
        self.cartesian_planner._n_step = n_step
        self.cartesian_planner.run(self.scene_mngr, cur_q, goal_pose, resolution=0.1, collision_check=collision_check)
        return self.cartesian_planner.get_joint_path()

    def get_rrt_star_path(self, cur_q, goal_pose, max_iter=500, n_step=10):
        self.rrt_planner.run(self.scene_mngr, cur_q, goal_pose, max_iter)
        return self.rrt_planner.get_joint_path(n_step=n_step)

    def show(self):
        self.scene_mngr.show()