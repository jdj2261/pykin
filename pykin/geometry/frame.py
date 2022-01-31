import numpy as np

from pykin.kinematics.transform import Transform
from pykin.geometry.geometry import Visual, Collision
from pykin.utils import transform_utils as tf
from pykin.utils.kin_utils import ShellColors as scolors


class Link:
    """class of Link

    Args:
        name (str): link name
        offset (pykin.kinematics.transform.Transform): link offset described in the urdf file
        visual (pykin.geometry.geometry.Visual): link visual described in the urdf file
        collision (pykin.geometry.geometry.Collision): link collision described in the urdf file
    """
    def __init__(
        self, 
        name=None, 
        offset=Transform(), 
        visual=Visual(), 
        collision=Collision()
    ):
        self.name = name
        self.offset = offset
        self.visual = visual
        self.collision = collision

    def __str__(self):
        return f"""
        {scolors.OKBLUE}Link{scolors.ENDC}( name= {scolors.HEADER}{self.name}{scolors.ENDC}
            offset= {scolors.HEADER}{self.offset}{scolors.ENDC}
            visual= {scolors.HEADER}{self.visual}{scolors.ENDC} 
            collision= {scolors.HEADER}{self.collision}{scolors.ENDC}"""
    
    def __repr__(self):
        return 'pykin.geometry.frame.{}()'.format(type(self).__name__)
        
class Joint:
    """
    class of Joint

    Args:
        name (str): join name
        offset (pykin.kinematics.transform.Transform): joint offset described in the urdf file
        dtype (str): joint type (fixed, revolute, prismatic) described in the urdf file
        axis (np.array): joint axis described in the urdf file
        limit (list): joint limit described in the urdf file
        parent (Link): joint parent link described in the urdf file
        child (Link): joint child link described in the urdf file
    """
    TYPES = ['fixed', 'revolute', 'prismatic']

    def __init__(
        self,
        name=None, 
        offset=Transform(),
        dtype='fixed', 
        axis=None, 
        limit=[None, None], 
        parent=None, 
        child=None
    ):
        self.name = name
        self.offset = offset
        self.num_dof = 0
        self.dtype = dtype
        self.axis = np.array(axis)
        self.limit = limit
        self.parent = parent
        self.child = child

    def __str__(self):
        return f"""
        {scolors.OKGREEN}Joint{scolors.ENDC}( name= {scolors.HEADER}{self.name}{scolors.ENDC} 
            offset= {scolors.HEADER}{self.offset}{scolors.ENDC}
            dtype= {scolors.HEADER}'{self.dtype}'{scolors.ENDC}
            axis= {scolors.HEADER}{self.axis}{scolors.ENDC}
            limit= {scolors.HEADER}{self.limit}{scolors.ENDC}"""

    def __repr__(self):
        return 'pykin.geometry.frame.{}()'.format(type(self).__name__)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        """
        Sets dof 0 if dtype is fixed else 1
        """
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
        """
        Number of dof
        """
        self._num_dof = int(dof)


class Frame:
    """class of Frame

    Args:
        name (str): frame name
        link (Link): Link frame
        joint (Joint): Joint frame
        children (list): all child frame
    """
    def __init__(
        self, 
        name=None, 
        link=Link(),
        joint=Joint(), 
        children=[]
    ):
        self.name = 'None' if name is None else name
        self.link = link
        self.joint = joint
        self.children = children

    def __str__(self, level=0):
        ret = "  " * level + self.name + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return 'pykin.geometry.frame.{}()'.format(type(self).__name__)

    def get_transform(self, theta):
        """
        Args:
            theta (list): Angle to convert

        Returns:
            Transform: Compute transform by multiplying current joint offset and transfrom obtained from input angle
        """
        if self.joint.dtype == 'revolute':
            t = Transform(rot=tf.get_quaternion_about_axis(theta, self.joint.axis))
        elif self.joint.dtype == 'prismatic':
            t = Transform(pos=theta * self.joint.axis)
        elif self.joint.dtype == 'fixed':
            t = Transform()
        else:
            raise ValueError("Unsupported joint type %s." %self.joint.dtype)
        return self.joint.offset * t
