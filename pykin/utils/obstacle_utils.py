import sys, os
import numpy as np
from collections import defaultdict

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../" )
sys.path.append(pykin_path)
from pykin.kinematics.transform import Transform
from pykin.utils.error_utils import NotFoundError

class Obstacle():
    obstacle_types = ["sphere", "box", "cylinder"]
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
        gpose=None):
        
        obs_name = self._convert_name(name)
        self._check_gtype(gtype)
        self._check_gparam(gtype, gparam)
        self.obstacles[obs_name] = (gtype, gparam, gpose)

    @staticmethod
    def _convert_name(name):
        if name and "obstacle" not in name:
            name = "obstacle_" + name
        return name
    
    @staticmethod
    def _check_gtype(gtype):
        if gtype not in Obstacle.obstacle_types:
            raise NotFoundError(f"'{gtype}' is not in {Obstacle.obstacle_types}")
    
    @staticmethod
    def _check_gparam(gtype, gparam):
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

if __name__ == "__main__":
    obs = Obstacle()
    obs(
        name="sphere_1", 
        gtype="sphere",
        gparam=0.3,
        gpose=(0.3,0.7,0.2))

    obs(
        name="sphere_2", 
        gtype="sphere",
        gparam=0.3,
        gpose=(0.3,0.7,0.2))

    obs(
        name="box_1", 
        gtype="box",
        gparam=(0.2, 0.2, 0.2),
        gpose=(0.3,0.7,0.2))

    obs(
        name="box_2", 
        gtype="box",
        gparam=(0.2, 0.2, 0.2),
        gpose=(0.3,0.7,0.2))

    obs(
        name="box_2", 
        gtype="box",
        gparam=(0.2, 0.2, 0.2),
        gpose=(0.3,0.7,0.2))


    for key,values in obs:
        print(key)

    