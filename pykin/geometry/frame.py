import numpy as np
import pykin.kinematics.transformation as tf
from pykin.kinematics.transform import Transform
from pykin.utils.shell_color import ShellColors as scolors

class Link:
    def __init__(self, name=None, offset=Transform(), dtype=None, radius=0, length=0, size=None):
        self.name = name
        self.offset = offset
        self.dtype = dtype
        self.radius = radius
        self.length = length
        self.size = np.array(size)
        self.color = None
        self.mesh = None

    def __repr__(self):
        radius = "radius= " + self.radius + ", " if self.dtype in ['cylinder', 'sphere'] else ""
        length = "length= " + self.length if self.dtype in ['cylinder'] else ""
        size = "size= "     + str(self.size) if self.dtype in ['box'] else ""
        return f"""
        {scolors.OKBLUE}Link{scolors.ENDC}( name= {scolors.HEADER}{self.name}{scolors.ENDC}
            offset= {scolors.HEADER}{self.offset}{scolors.ENDC}
            dtype= {scolors.HEADER}{self.dtype}{scolors.ENDC} 
            {radius} {length} {size}"""

class Joint:
    TYPES = ['fixed', 'revolute', 'prismatic']

    def __init__(self, name=None, offset=Transform(),
                 dtype='fixed', axis=None, limit=[None, None], parent=None, child=None):
        self.name = name
        self.offset = offset
        self.parent = parent
        self.child = child
        self.num_dof = 0
        self.dtype = dtype
        self.axis = np.array(axis)
        self.limit = limit
 
    def __repr__(self):
        return f"""
        {scolors.OKGREEN}Joint{scolors.ENDC}( name= {scolors.HEADER}{self.name}{scolors.ENDC} 
            offset= {scolors.HEADER}{self.offset}{scolors.ENDC}
            dtype= {scolors.HEADER}'{self.dtype}'{scolors.ENDC}
            axis= {scolors.HEADER}{self.axis}{scolors.ENDC}
            limit= {scolors.HEADER}{self.limit}{scolors.ENDC}"""

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        if dtype is not None:
            dtype = dtype.lower().strip()
            if dtype in {'fixed'}:
                dtype = 'fixed'
                self.num_dof = 0
            elif dtype in {'revolute'}:
                dtype = 'revolute'
                self.num_dof = 1
            elif dtype in {'prismatic'}:
                dtype = 'prismatic'
                self.num_dof = 1
        self._dtype = dtype

    @property
    def num_dof(self):
        return self._num_dof

    @num_dof.setter
    def num_dof(self, dof):
        self._num_dof = int(dof)


class Frame:
    def __init__(self, name=None, link=Link(),
                 joint=Joint(), children=[]):
        self.name = 'None' if name is None else name
        self.link = link
        self.joint = joint
        self.children = children

    def __repr__(self, level=0):
        ret = "  " * level + self.name + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

    def add_child(self, child):
        self.children.append(child)

    def is_end(self):
        return (len(self.children) == 0)

    def get_transform(self, theta):
        if self.joint.dtype == 'revolute':
            t = Transform(
                tf.get_quaternion_about_axis(theta, self.joint.axis))
        elif self.joint.dtype == 'prismatic':
            t = Transform(pos=theta * self.joint.axis)
        elif self.joint.dtype == 'fixed':
            t = Transform()
        else:
            raise ValueError("Unsupported joint type %s." %
                             self.joint.dtype)
        return self.joint.offset * t
