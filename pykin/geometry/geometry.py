import os, sys
import numpy as np
from collections import OrderedDict
import fcl
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../")
sys.path.append(pykin_path)
from pykin.geometry.collision import Collision
import pykin.kinematics.transformation as tf
from pykin.utils.shell_color import ShellColors as scolors
from pykin.utils import plot as plt

class Box:
    boxes = []
    def __init__(self, box=None):
        self.box = box
        
    def __repr__(self):
        return f"Box Info: {self.boxes}"

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
    def __init__(self, cyl: dict = None):
        self.cyl = cyl

    def __repr__(self):
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
    def __init__(self, sphere: dict = None):
        self.sphere = sphere

    def __repr__(self):
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
    def __init__(self, mesh: dict = None):
        pass


class Geometry(Collision):
    def __init__(self, gtype=None, robot=None, obj: dict = None, fk: dict = None):
        self.gtype = gtype
        self.robot = robot
        self._obj = obj
        self.fk = fk
        # print(self.fk)
        
        super(Geometry, self).__init__(robot, obj, fk)

    def __repr__(self):
        if self.robot is not None:
            return f"""Robot Collision Info: {list(self.link_type.values())}"""
        else:
            return f"""Geometry Info: {self._obj}"""

    def add_objects(self, gtype, obj, A2B=np.eye(4)):
        if gtype == 'box':
            self.box = Box(box=obj)
            self._obj = self.box.box
            self.get_objects(gtype, A2B)
        if gtype == 'cylinder':
            self.cyl = Cylinder(cyl=obj)
            self._obj = self.cyl.cyl
            self.get_objects(gtype, A2B)
        if gtype == 'sphere':
            self.sphere = Sphere(sphere=obj)
            self._obj = self.sphere.sphere
            self.get_objects(gtype, A2B)
        if gtype == 'mesh':
            pass

    def plot_objects(self, ax=None):
        for obj in self.objects:
            obj_type = obj[0]
            obj_info = obj[1]
            obj_pose = obj[2]
            if obj_type == 'cylinder':
                radius = float(list(obj_info.values())[0][0].get('radius'))
                length = float(list(obj_info.values())[0][1].get('length'))
                plt.plot_cylinder(ax=ax, A2B=obj_pose, radius=radius,
                                  length=length, alpha=0.5)
            if obj_type == 'sphere':
                radius = float(list(obj_info.values())[0].get('radius'))
                plt.plot_sphere(ax=ax, p=obj_pose[:3,-1], radius=radius, alpha=0.5)

            if obj_type == 'box':
                size = list(obj_info.values())[0].get('size')
                plt.plot_box(ax=ax, A2B=obj_pose, size=size, alpha=0.5)
                                  
if __name__ == "__main__":
    geo = Geometry()
    box_size1 = (0.1, 0.2, 0.3)
    box_size2= (0.1, 0.2, 0.3)
    box1 = {'box1': {'size': box_size1}}
    box2 = {'box2': {'size': box_size2}}
    cyl1 = {'cylinder1': ({'radius': 0.2}, {'length' : 0.4})}
    cyl2 = {'cylinder2': ({'radius': 0.6}, {'length' : 0.4})}
    sphere1 = {'sphere1': {'radius': 0.2}}
    
    rot = np.eye(3)
    pos1 = np.array([0.3, 0, 0])
    pos2 = np.array([0.4, 0, 0])
    A2B1 = tf.get_homogeneous_matrix(pos1, rot)
    A2B2 = tf.get_homogeneous_matrix(pos2, rot)

    geo.add_objects('box', box1, A2B1)
    geo.add_objects('box', box2, A2B2)
    # geo.add_objects('cylinder', cyl1, A2B)
    # geo.add_objects('cylinder', cyl2)
    # geo.add_objects('sphere', sphere1)
    _, ax = plt.init_3d_figure("Target")
    plt.plot_basis(ax=ax, arm_length=1)
    geo.plot_objects(ax)
    for obj in geo.objects:
        print(obj)
    ax.legend()
    plt.show_figure()




