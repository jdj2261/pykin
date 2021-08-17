import os, sys
import numpy as np
from collections import OrderedDict
import numpy as np
import fcl
pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../")
sys.path.append(pykin_path)
from pykin.geometry.collision import Collision
from pykin.kinematics.transform import Transform
from pykin.kinematics import transformation as tf
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
        super(Geometry, self).__init__(robot, obj, fk)

        if robot is not None:
            self.get_link_type()
        if fk is not None:
            self.fk = fk

    def __repr__(self):
        if self.robot is not None:
            return f"""Robot Collision Info: {list(self.link_type.values())}"""
        else:
            return f"""Geometry Info: {self._obj}"""

    def get_link_type(self):
        for link in self.fk.keys():
            link_info = self.robot.tree.links[link]
            self.link_type[link_info.name] = link_info
        self.seperate_link_type()

    def seperate_link_type(self):
        for info in self.link_type.values():
            if info.dtype == 'cylinder':
                radius = float(info.radius)
                length = float(info.length)
                cylinder = {info.name: ({'radius': radius},
                                        {'length': length})}
                A2B = self.fk[info.name]
                self.add_objects(info.dtype, cylinder,
                                offset=A2B, color=info.color)
                                
            if info.dtype == 'box':
                box = {info.name: {'size': info.size}}
                A2B = self.fk[info.name]
                self.add_objects(info.dtype, box, 
                                offset=A2B, color=info.color)

            if info.dtype == 'sphere':
                sphere = {info.name: {'radius': float(info.radius)}}
                A2B = self.fk[info.name]
                self.add_objects(info.dtype, sphere,
                                offset=A2B, color=info.color)

    def add_objects(self, gtype, obj, offset=Transform(), color='k'):
        if gtype == 'box':
            self.box = Box(box=obj)
            self._obj = self.box.box
            self.append_objects(gtype, offset, color=color)
        if gtype == 'cylinder':
            self.cyl = Cylinder(cyl=obj)
            self._obj = self.cyl.cyl
            self.append_objects(gtype, offset, color=color)
        if gtype == 'sphere':
            self.sphere = Sphere(sphere=obj)
            self._obj = self.sphere.sphere
            self.append_objects(gtype, offset, color=color)
        if gtype == 'mesh':
            pass


if __name__ == "__main__":
    geo = Geometry()
    box_size1 = (0.1, 0.2, 0.3)
    box_size2= (0.1, 0.2, 0.3)

    box1 = {'box1': {'size': box_size1}}
    box2 = {'box2': {'size': box_size2}}
    
    fcl_box1 = fcl.Box(*box_size1)
    fcl_box2 = fcl.Box(*box_size2)

    req = fcl.CollisionRequest(enable_contact=True) 
    res = fcl.CollisionResult()

    n_contacts = fcl.collide(fcl.CollisionObject(fcl_box1, fcl.Transform(np.array([0.3, 0, 0]))),
                             fcl.CollisionObject(
                                fcl_box2, fcl.Transform(np.array([0.35, 0, 0]))),
                             req, res)
    geo.print_collision_result('Box', 'Box', res)

    req = fcl.DistanceRequest(enable_nearest_points=True)
    res = fcl.DistanceResult()

    dist = fcl.distance(fcl.CollisionObject(fcl_box1, fcl.Transform(np.array([0.3, 0, 0]))),
                        fcl.CollisionObject(
                            fcl_box1, fcl.Transform(np.array([0.35, 0, 0]))),
                        req, res)

    geo.print_distance_result('Box', 'Box', res)

    # cyl1 = {'cylinder1': ({'radius': 0.2}, {'length' : 0.4})}
    # cyl2 = {'cylinder2': ({'radius': 0.6}, {'length' : 0.4})}
    # sphere1 = {'sphere1': {'radius': 0.2}}
    
    box1_H = Transform(rot=np.array([0, 0, 0]), pos=np.array([0.1, 0.0, 0.0]))
    box2_H = Transform(rot=np.array([0, 0, 0]), pos=np.array([0.2, 0.0, 0.0]))

    geo.add_objects('box', box1, box1_H, color='red')
    geo.add_objects('box', box2, box2_H, color='blue')
    # geo.add_objects('cylinder', cyl1, A2B)
    # geo.add_objects('cylinder', cyl2)
    # geo.add_objects('sphere', sphere1)

    geo.collision_check(visible=False)
    for obj in geo.objects:
        print(obj)
    
    plt.show_figure()




