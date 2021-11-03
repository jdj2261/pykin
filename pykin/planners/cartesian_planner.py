import numpy as np

from pykin.planners.planner import Planner
from pykin.utils.error_utils import OriValueError, CollisionError, LimitJointError
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
        current_pose,
        goal_pose,
        n_step=500,
        arm_type=None,
        waypoint_type="Linear"
    ):
        super(CartesianPlanner, self).__init__(robot, obstacles)
        self.cur_pose = super()._change_types(current_pose)
        self.goal_pose = super()._change_types(goal_pose)
        self.n_step = n_step
        self.waypoint_type = waypoint_type
        self.eef_name = self.robot.eef_name
        self.arm = None

        # [TODO]
        self.dof = len(self.robot.get_revolute_joint_names(arm_type=arm_type))

        super()._setup_q_limits()
        super()._setup_eef_name()
        
        self.waypoints = self.genearte_waypoints()

    def __repr__(self):
        return 'pykin.planners.cartesian_planner.{}()'.format(type(self).__name__)
        
    def get_path_in_joinst_space(
        self, 
        waypoints=None,
        resolution=1, 
        damping=0.5,
        epsilon=1e-12,

    ):
        if waypoints is None:
            waypoints = self.waypoints
        paths, target_posistions = self._compute_path_and_target_pose(waypoints, resolution, damping, epsilon)

        # TODO
        # paths = paths + [self.goal_q]

        return paths, target_posistions

    def _compute_path_and_target_pose(
        self, 
        waypoints, 
        resolution, 
        damping, 
        epsilon
    ):
        current_joints = self.robot.inverse_kin(np.random.randn(self.dof), self.cur_pose)
        print(current_joints)
        if not self._check_q_in_limits(current_joints):
            raise LimitJointError(current_joints, self.q_limits_lower, self.q_limits_upper)

        cur_fk = self.robot.kin.forward_kinematics(self.robot.desired_frames, current_joints)

        super()._setup_fcl_manager(cur_fk)
        super()._check_init_collision()

        current_transform = cur_fk[self.eef_name].homogeneous_matrix
        eef_position = cur_fk[self.eef_name].pos

        paths = [current_joints]
        target_posistions = [eef_position]

        for step, (pos, ori) in enumerate(waypoints):

            target_transform = t_utils.get_homogeneous_matrix(pos, ori)
            err_pose = k_utils.calc_pose_error(target_transform, current_transform, epsilon) 
            J = jac.calc_jacobian(self.robot.desired_frames, cur_fk, self.dof)
            Jh = np.dot(np.linalg.pinv(np.dot(J.T, J) + damping * np.identity(self.dof)), J.T)

            dq = damping * np.dot(Jh, err_pose)
            current_joints = np.array([(current_joints[i] + dq[i]) for i in range(self.dof)]).reshape(self.dof,)

            # # is_collision, name = self.collision_free(current_joints, visible_name=True)
            # # if is_collision:
            # #     print(name)
            
            # if not self._check_q_in_limits(current_joints):
            #     print("Can not move robot's arm")
            #     current_joints = [current_joints[i] - dq[i] for i in range(dof)]
            #     cur_fk = self.robot.kin.forward_kinematics(self.robot.desired_frames, current_joints)
            #     cur_pose = cur_fk[self.robot.eef_name].homogeneous_matrix
            #     continue

            cur_fk = self.robot.kin.forward_kinematics(self.robot.desired_frames, current_joints)
            current_transform = cur_fk[self.robot.eef_name].homogeneous_matrix

            if step % (1/resolution) == 0 or step == len(waypoints)-1:
                paths.append(current_joints)
                target_posistions.append(pos)
                
        return paths, target_posistions

    def genearte_waypoints(self):
        if self.waypoint_type == "Linear":
            waypoints = [path for path in self._get_linear_path()]
        if self.waypoint_type == "Cubic":
            pass
        if self.waypoint_type == "Circular":
            pass
        return waypoints

    def get_waypoints(self):
        return self.waypoints

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

    def _get_linear_path(self):
        for step in range(1, self.n_step + 1):
            delta_t = step / self.n_step
            pos = t_utils.get_linear_interpoation(self.cur_pose[:3], self.goal_pose[:3], delta_t)
            ori = t_utils.get_quaternion_slerp(self.cur_pose[3:], self.goal_pose[3:], delta_t)

            yield (pos, ori)

    def _get_cubic_path(self):
        pass

    def _get_cicular_path(self):
        pass

