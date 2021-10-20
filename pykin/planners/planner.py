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
    def generate_path(self):
        """
        Write path algorithm you want 
        """
        raise NotImplementedError
