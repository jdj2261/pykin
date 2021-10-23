from abc import ABC, abstractclassmethod

class Planner(ABC):
    """
    Base Planner class 

    Args:
        robot (SingleArm or Bimanual): The manipulator robot type is SingleArm or Bimanual
        obstacles (dictionary) : The obstacles
    """
    def __init__(
        self,
        robot,
        obstacles
    ):
        self.robot = robot
        self.obstacles = obstacles
    
    @abstractclassmethod
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

    @abstractclassmethod
    def generate_path(self):
        """
        write planner algorithm you want 
        """
        raise NotImplementedError
