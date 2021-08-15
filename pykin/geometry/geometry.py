
import os, sys
import numpy as np
from collections import OrderedDict
import fcl
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../")
sys.path.append(pykin_path)
from pykin.geometry.collision import Collision
from pykin.robot import Robot
from pykin.kinematics import transform as tf
from pykin.utils.shell_color import ShellColors as scolors
from pykin.utils import plot as plt

class Box():
    def __init__(self, robot=None, obj:dict=None, fk:dict=None):
        self.robot = robot
        self.obj = obj

        if robot is not None:
            self.link_type = OrderedDict()

    def __repr__(self):
        if self.robot is not None:
            return f"""Robot Collision Info: {list(self.link_type.values())}"""
        else:
            return f"""Box Info: {self.obj}"""

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, obj):
        self._obj = obj
        if obj is not None:
            self.name = obj.keys()
            self.size = obj.values()

class Geometry(Collision):
    def __init__(self, gtype, robot=None, obj: dict = None, fk: dict = None):
        self.robot = robot
        self.obj = obj
        if gtype == 'box':
            self.box = Box(robot=None, obj=None, fk=None)
            self.obj = self.box.obj
        super(Geometry, self).__init__(robot, obj, fk)

    def __repr__(self):
        if self.robot is not None:
            return f"""Robot Collision Info: {list(self.link_type.values())}"""
        else:
            return f"""Geometry Info: {self.obj}"""

    def add_objects(self, gtype, obj):
        if gtype == 'box':
            self.box = Box(robot=None, obj=obj, fk=None)
            self.obj = self.box.obj
        if gtype == 'cylinder':
            pass
        if gtype == 'sphere':
            pass
        if gtype == 'mesh':
            pass
        self.get_objects()

if __name__ == "__main__":
    geo = Geometry('box')
    box_size = (1.0, 2.0, 3.0)
    box1 = {'box1': box_size}
    box2 = {'box2': box_size}

    geo.add_objects('box', box1)
    geo.add_objects('box', box2)
    
    print(geo.objects)


