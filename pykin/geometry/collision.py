
import os, sys
import numpy as np
from collections import OrderedDict
import fcl
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../")
sys.path.append(pykin_path)
from pykin.kinematics import transform as tf
from pykin.utils.shell_color import ShellColors as scolors
from pykin.utils import plot as plt
from pykin.kinematics import transformation as tf

class Collision:
    
    def __init__(self, robot=None, obj=None, fk:dict=None):
        self.robot = robot
        self._obj = obj
        self.objects = []
        self.link_type = OrderedDict()

        if obj is not None:
            self.objects.append(obj)

        if robot is not None:
            self.get_link_type()

        if fk is not None:
            self.fk = fk

    def __repr__(self):
        if self.robot is not None:
            return f"""Robot Collision Info:
            {list(self.link_type.values())}"""
        else:
            return f"""Object Collision Info: {self._obj}"""

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, obj):
        self._obj = obj

    def get_objects(self, gtype, A2B=None):
        if self._obj.keys():
            if len(self.objects) != 0:
                objects = [obj[1] for obj in self.objects]
                assert objects[0].keys() != self._obj.keys(
                ), f"Duplicate name. please check again"
            
            self.objects.append((gtype, self._obj, A2B))
 
    def get_link_type(self):
        for link in self.fk.keys():
            link_info = self.robot.tree.links[link]
            self.link_type[link_info.name] = link_info


    # def pairwise_collsion_check(self):
    #     pass

    # def pairwise_distance_check(self):
    #     pass

    # def continous_collision_check(self):
    #     pass
