import sys, os
import numpy as np
from collections import defaultdict

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)
from pykin.kinematics.transform import Transform

from pykin.utils.error_utils import NotFoundError

class Obstacle():
    """
    Obstacle class 
    Obstacles are noe of three types(sphere, box, cylinder)

    """
    obstacle_types = ["mesh", "sphere", "box", "cylinder"]
    def __init__(self):
        self._obstacles = defaultdict(tuple)
        self._gtype = None

    def __call__(self, *args, **kwards):
        self.add_obstacles(*args, **kwards)

    def __repr__(self):
        return f"{self._obstacles}"

    def __iter__(self):
        items = list(self._obstacles.items())
        items.sort(key=lambda x : (x[1][0], x[0]))
        for key, value in items:
            yield (key, value)
            
    def add_obstacles(
        self, 
        name=None, 
        gtype=None, 
        gparam=None, 
        transform=Transform()):
        """
        Add obstacles

        Args:
            name (str): An identifier for the object
            gtype (str): object type (cylinder, sphere, box)
            gparam (float or tuple): object parameter (radius, length, size)
            transform (np.array): Homogeneous transform matrix for the object
        """
        obs_name = self._convert_name(name)
        self._check_gtype(gtype)
        self._check_gparam(gtype, gparam)
        self.obstacles[obs_name] = (gtype, gparam, transform)

    @staticmethod
    def _convert_name(name):
        """
        convert input name to obstacle name

        Args:
            nam (str): An identifier for the object

        Returns:
            name(str) : obstacles_ + name
        """
        if name and "obstacle" not in name:
            name = "obstacle_" + name
        return name
    
    @staticmethod
    def _check_gtype(gtype):
        """
        check obstacle's geom type
        """
        if gtype not in Obstacle.obstacle_types:
            raise NotFoundError(f"'{gtype}' is not in {Obstacle.obstacle_types}")
    
    @staticmethod
    def _check_gparam(gtype, gparam):
        """
        check obstacle's geom param 
        """
        if not isinstance(gparam, (tuple, list, np.ndarray)):
            gparam = [gparam]
        if gtype == "sphere":
            assert len(gparam) == 1, f"{gtype}'s parameter need only 'radius'"
        if gtype == "box":
            assert len(gparam) == 3, f"{gtype}'s parameter need box 'size(x, y, z)'"
        if gtype == "cylinder":
            assert len(gparam) == 2, f"{gtype}'s parameter need 'radius' and 'length'"

    @property
    def obstacles(self):
        return self._obstacles
    
