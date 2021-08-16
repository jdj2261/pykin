import os, sys
import numpy as np
from collections import OrderedDict
import fcl
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../")
sys.path.append(pykin_path)
from pykin.geometry.collision import Collision
from pykin.kinematics import transform as tf
from pykin.utils.shell_color import ShellColors as scolors
from pykin.utils import plot as plt

class Box:
    boxes = []
    def __init__(self, robot=None, box:dict=None, fk:dict=None):
        self.robot = robot
        self.box = box
        if robot is not None:
            self.link_type = OrderedDict()

    def __repr__(self):
        if self.robot is not None:
            return f"""Robot's Box Info: {list(self.link_type.values())}"""
        else:
            return f"""Box Info: {self.boxes}"""

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, box):
        self._box = box
        if box is not None:
            self.boxes.append(box)
            self.name = [list(box.keys()) for box in self.boxes]
            self.size = [list(box.values()) for box in self.boxes]
            

class Cylinder:
    cylinders = []
    def __init__(self, robot=None, cyl: dict = None, fk: dict = None):
        self.robot = robot
        self.cyl = cyl
        if robot is not None:
            self.link_type = OrderedDict()

    def __repr__(self):
        if self.robot is not None:
            return f"""Robot's Cylinder Info: {list(self.link_type.values())}"""
        else:
            return f"""Cylinder Info: {self.cylinders}"""

    @property
    def cyl(self):
        return self._cyl

    @cyl.setter
    def cyl(self, cyl):
        self._cyl = cyl
        if cyl is not None:
            self.cylinders.append(cyl)
            self.name = [list(cylinder.keys()) for cylinder in self.cylinders]
            info = [list(cylinder.values()) for cylinder in self.cylinders]
            self.radius = [x[0][0] for x in info]
            self.length = [x[0][1] for x in info]


class Sphere:
    spheres = []
    def __init__(self, robot=None, sphere: dict = None, fk: dict = None):
        self.robot = robot
        self.sphere = sphere

        if robot is not None:
            self.link_type = OrderedDict()

    def __repr__(self):
        if self.robot is not None:
            return f"""Robot's Sphere Info: {list(self.link_type.values())}"""
        else:
            return f"""Sphere Info: {self.sphere}"""

    @property
    def sphere(self):
        return self._sphere

    @sphere.setter
    def sphere(self, sphere):
        self._sphere = sphere
        if sphere is not None:
            self.spheres.append(sphere)
            self.name = [list(sphere.keys()) for sphere in self.spheres]
            self.size = [list(sphere.values()) for sphere in self.spheres]


class Mesh:
    def __init__(self, robot=None, box: dict = None, fk: dict = None):
        pass


class Geometry(Collision):
    def __init__(self, gtype=None, robot=None, obj: dict = None, fk: dict = None):
        self.gtype = gtype
        self.robot = robot
        self._obj = obj

        super(Geometry, self).__init__(robot, obj, fk)

    def __repr__(self):
        if self.robot is not None:
            return f"""Robot Collision Info: {list(self.link_type.values())}"""
        else:
            return f"""Geometry Info: {self._obj}"""

    def add_objects(self, gtype, obj):
        if gtype == 'box':
            self.box = Box(robot=None, box=obj, fk=None)
            self._obj = self.box.box
            self.get_objects(gtype)

        if gtype == 'cylinder':
            self.cyl = Cylinder(robot=None, cyl=obj, fk=None)
            self._obj = self.cyl.cyl
            self.get_objects(gtype)

        if gtype == 'sphere':
            self.sphere = Sphere(robot=None, sphere=obj, fk=None)
            self._obj = self.sphere.sphere
            self.get_objects(gtype)

        if gtype == 'mesh':
            pass

        

if __name__ == "__main__":
    geo = Geometry()
    box_size1 = (0.1, 0.2, 0.3)
    box_size2= (0.3, 0.6, 0.9)
    box1 = {'box1': {'size': box_size1}}
    box2 = {'box2': {'size': box_size2}}
    cyl1 = {'cylinder1': ({'radius': 0.2}, {'length' : 0.4})}
    cyl2 = {'cylinder2': ({'radius': 0.6}, {'length' : 0.4})}
    sphere1 = {'sphere1': {'radius': 0.2}}

    geo.add_objects('box', box1)
    geo.add_objects('box', box2)
    # geo.add_objects('cylinder', cyl1)
    # geo.add_objects('cylinder', cyl2)
    geo.add_objects('sphere', sphere1)
    print(geo.objects)

    geo.plot()
    plt.show_figure()





