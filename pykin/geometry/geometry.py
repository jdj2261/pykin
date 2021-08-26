from pykin.kinematics.transform import Transform


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