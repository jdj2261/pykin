from pykin.planners.planner import Planner

class JointPlanner(Planner):
    """
    Base Joint space Planner class 

    Args:
        robot (SingleArm or Bimanual): The manipulator robot type is SingleArm or Bimanual
        obstacles (dictionary) : The obstacles
    """
    def __init__(
        self,
        robot,
        collision_manager,
        dimension
    ):
        super(JointPlanner, self).__init__(robot, collision_manager,dimension)

    def __repr__(self):
        return 'pykin.planners.joint_planner.{}()'.format(type(self).__name__)
        
    def setup_start_goal_joint(
        self,
        current_q,
        goal_q,
        arm=None,
        transformation=None
    ):
        """
        Setup start joints and goal joints

        Args:
            current_q(np.array or Iterable of floats): current joint angle
            goal_q(np.array or Iterable of floats): target joint angle obtained through Inverse Kinematics 
            arm(str): If Manipulator types is bimanul, must put the arm type.
            init_transformation: Initial transformations
        """
        raise NotImplementedError

    def get_path_in_joinst_space(self):
        """
        write planner algorithm you want 
        """
        raise NotImplementedError
