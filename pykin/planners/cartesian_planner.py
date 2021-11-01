import numpy as np

from pykin.planners.planner import Planner
from pykin.utils.error_utils import OriValueError, NotFoundError
import pykin.utils.transform_utils as t_utils
import pykin.utils.kin_utils as k_utils
import pykin.kinematics.jacobian as jac

class CartesianPlanner(Planner):
    """
    path planner in Cartesian space

    Args:
        robot(SingleArm or Bimanual): The manipulator robot type is SingleArm or Bimanual
        obstacles(Obstacle): The obstacles
    """
    def __init__(
        self,
        robot,
        obstacles,
    ):
        super(CartesianPlanner, self).__init__(robot, obstacles)
        self.cur_pose = None
        self.cur_pos = None
        self.cur_ori = None
        self.tar_pose = None
        self.eef_name = self.robot.eef_name

    def __repr__(self):
        return 'pykin.planners.cartesian_planner.{}()'.format(type(self).__name__)
        
    def setup_init_joint(
        self, 
        current_q, 
        init_transformation=None
    ):
        """
        Setup init  joints and goal joints

        Args:
            current_q(np.array or Iterable of floats): current joint angle
            init_transformation: Initial transformations
        """
        if self._check_q_datas(current_q):
            self.cur_q = current_q

        self.init_transformation = init_transformation
        if init_transformation is None:
            self.init_transformation = self.robot.init_transformations
        
        # self._setup_q_limits()
        # self._setup_eef_name()
        # self._setup_fcl_manager(init_transformation)
        # self._check_init_collision()

    @staticmethod
    def _check_q_datas(current_q):
        """
        Check input current joints

        Args:
            current_q(np.array or Iterable of floats): current joint angle
            
        Return:
            True(bool): Return True if everything is confirmed
        """
        if not isinstance(current_q, (np.ndarray)):
            current_q = np.array(current_q)

        if current_q.size == 0:
            raise NotFoundError("Make sure set current or goal joints..")
        return True

    def get_path_in_joinst_space(
        self, 
        waypoints,
        resolution=1, 
        damping=0.5,
        epsilon=1e-12
    ):
        paths, target_poses = self._compute_path_and_target_pose(waypoints, resolution, damping, epsilon)
        return paths, target_poses

    def _compute_path_and_target_pose(self, waypoints, resolution, damping, epsilon):
        cur_T = t_utils.get_homogeneous_matrix(self.cur_pos, self.cur_ori)
        cur_fk = self.init_transformation
        current_joints = self.cur_q
        dof = len(self.cur_q)
        paths = [self.cur_q]
        target_poses = [self.cur_pos]
        for step, (pos, ori) in enumerate(waypoints):
            tar_T = t_utils.get_homogeneous_matrix(pos, ori)
            err_pose = k_utils.calc_pose_error(tar_T, cur_T, epsilon)
            J = jac.calc_jacobian(self.robot.desired_frames, cur_fk, dof)
            Jh = np.dot(np.linalg.inv(np.dot(J.T, J) + damping*np.identity(dof)), J.T)

            dq = damping * np.dot(Jh, err_pose)
            current_joints = np.array([(current_joints[i] + dq[i]) for i in range(dof)]).reshape(dof,)

            if step % (1/resolution) == 0 or step == len(waypoints)-1:
                paths.append(current_joints)
                target_poses.append(pos)
            cur_fk = self.robot.kin.forward_kinematics(self.robot.desired_frames, current_joints)
            cur_T = cur_fk[self.robot.eef_name].homogeneous_matrix

        return paths, target_poses
    def get_path_in_cartesian_space(
        self, 
        current_pose,
        goal_pose,
        n_step=100,
        method="linear"):

        self.cur_pose = self._change_pose_type(current_pose)
        self.cur_pos = self.cur_pose[:3]
        self.cur_ori = self.cur_pose[3:]

        self.tar_pose = self._change_pose_type(goal_pose)

        if method == "linear":
            paths = [path for path in self._get_linear_path(self.cur_pose, self.tar_pose, n_step)]

        if method == "cubic":
            pass

        if method == "circular":
            pass
        
        return paths

    def _change_pose_type(self, pose):
        ret = np.zeros(7)
        ret[:3] = pose[:3]
        
        if isinstance(pose, (list, tuple)):
            pose = np.asarray(pose)
        ori = pose[3:]

        if ori.shape == (3,):
            ori = t_utils.get_quaternion_from_rpy(ori)
            ret[3:] = ori
        elif ori.shape == (4,):
            ret[3:] = ori
        else:
            raise OriValueError(ori.shape)

        return ret

    def _get_linear_path(self, cur_pose, tar_pose, n_step):
        for step in range(1, n_step + 1):
            delta_t = step / n_step
            pos = t_utils.get_linear_interpoation(cur_pose[:3], tar_pose[:3], delta_t)
            ori = t_utils.get_quaternion_slerp(cur_pose[3:], tar_pose[3:], delta_t)

            yield (pos, ori)

    def _get_cubic_path(self):
        pass

    def _get_cicular_path(self):
        pass

