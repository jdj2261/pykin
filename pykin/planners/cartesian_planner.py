class CartesianPlanner:
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
        waypoints=None,
        eef_step=None
    ):
        self.robot = robot
        self.obstacles = obstacles
        self.waypoints = waypoints

    def __call__(self):
        self.generate_path()

    def __repr__(self):
        return 'pykin.planners.cartesian_planner.{}()'.format(type(self).__name__)
        
    def generate_path(self):
        pass




