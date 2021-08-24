import os, sys
import numpy as np
# from pykin.geometry.geometry import Collision
from pykin.kinematics.transform import Transform
# import pykin.utils.plot_utils as plt

class Box:
    def __init__(self, size):
        self.size = size

    def __repr__(self):
        return f"""Box(size={self.size})"""


class Cylinder:
    def __init__(self, radius, length):
        self.radius = radius
        self.length = length

    def __repr__(self):
        return f"""Cylinder(radius={self.radius} 
                            length={self.length})"""


class Sphere:
    def __init__(self, radius):
        self.radius = radius

    def __repr__(self):
        return f"""Sphere(radius={self.radius}"""


class Mesh:
    def __init__(self):
        pass


class Visual:
    TYPES = ['box', 'cylinder', 'sphere', 'capsule', 'mesh']
    def __init__(self, offset=Transform(), 
                 geom_type=None, geom_param=None):
        self.offset = offset
        self.gtype = geom_type
        self.gparam = geom_param

    def __repr__(self):
        return f"""Visual(offset={self.offset},
                           geom_type={self.gtype}, 
                           geom_param={self.gparam})"""

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = Transform(offset.rot, offset.pos)


class Collision:
    TYPES = ['box', 'cylinder', 'sphere', 'mesh']
    def __init__(self, offset=Transform(), 
                 geom_type=None, geom_param=None):
        self.offset = offset
        self.gtype = geom_type
        self.gparam = geom_param

    def __repr__(self):
        return f"""Collision(offset={self.offset},
                              geom_type={self.gtype}, 
                              geom_param={self.gparam})"""

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = Transform(offset.rot, offset.pos)


class Geometry(Collision):
    def __init__(self, gtype=None, obj: dict = None):
        self.gtype = gtype
        self._obj = obj
        super(Geometry, self).__init__(obj)
  
    def __repr__(self):
        if self.robot is not None:
            return f"""Robot Geometry Info: {list(self.links.values())}"""
        else:
            return f"""Geometry Info: {self.objects}"""

    def get_links(self):
        for link in self.fk.keys():
            link_info = self.robot.robot_tree.links[link]
            self.links[link_info.name] = link_info
        self.get_objects()

    def get_objects(self):
        for info in self.links.values():
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
    import fcl
    geo = Geometry(robot=None, fk=None)
    box_size1 = (0.1, 0.2, 0.3)
    box_size2= (0.1, 0.2, 0.3)

    box1 = {'box1': {'size': box_size1}}
    box2 = {'box2': {'size': box_size2}}
    
    fcl_box1 = fcl.Box(*box_size1)
    fcl_box2 = fcl.Box(*box_size2)

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

    geo.collision_check(visible=True)
    for obj in geo.objects:
        print(obj)
    print(geo)
    plt.show_figure()




