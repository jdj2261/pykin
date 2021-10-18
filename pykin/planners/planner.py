from abc import ABC, abstractclassmethod

class Planner(ABC):
    def __init__(
        self,
        robot,
        obstacles
    ):
        self.robot = robot
        self.obstacles = obstacles
    
    @abstractclassmethod
    def generate_path(self):
        raise NotImplementedError
