import fcl
from pykin.kinematics.transform import Transform

def convert_fcl_objects(robot_links):
    objects = []
    for link in robot_links.values():
        if link.collision.gtype == 'cylinder':
            convert_fcl_cylinder(link, objects)
        if link.collision.gtype == 'sphere':
            convert_fcl_sphere(link, objects)
        if link.collision.gtype == 'box':
            convert_fcl_box(link, objects)
    return objects


def convert_fcl_cylinder(link, objects):
    radius = float(link.collision.gparam.get('radius'))
    length = float(link.collision.gparam.get('length'))
    geom = fcl.Cylinder(radius, length)
    objects.append((link.name, geom))


def convert_fcl_sphere(link, objects):
    radius = float(link.collision.gparam.get('radius'))
    geom = fcl.Sphere(radius)
    objects.append((link.name, geom))


def convert_fcl_box(link, objects):
    size = list(link.collision.gparam.get('size'))
    geom = fcl.Box(size)
    objects.append((link.name, geom))


class FclUtils:
    def __init__(self, fcl_objects=None):
        if fcl_objects is not None:
            self.fcl_objects = fcl_objects
            self.geoms = [object[1] for object in fcl_objects]
            self.names = [object[0] for object in fcl_objects]
        self.objs = []

    def add_object(self, obj_name:str, obj_geom:fcl):
        self.fcl_objects.append((obj_name, obj_geom))

    def collision_check(self, transformation):
        pass
        # for name in self.names:
        #     print(name)

        # return True
