import numpy as np

from dataclasses import dataclass
from collections import OrderedDict
from pykin.utils.error_utils import NotFoundError
import pykin.utils.plot_utils as plt

object_types = ["mesh", "sphere", "box", "cylinder"]

@dataclass
class ObjectData:
    NAME = "name"
    G_TYPE = "gtype"
    G_PARAM = "gparam"
    POSE = "pose"

class ObjectManager(ObjectData):
    """
    ObjectManager class 
    """
    
    def __init__(self):
        self._objects = OrderedDict()
        self.grasp_objects = OrderedDict()
        self.support_objects = OrderedDict()
        self._gtype = None

    def __call__(self, *args, **kwards):
        self.add_object(*args, **kwards)

    def __repr__(self):
        return f"{self._objects}"

    def __iter__(self):
        items = list(self._objects.items())
        items.sort(key=lambda x : (x[1][0], x[0]))
        for key, value in items:
            yield (key, value)
            
    def add_object(
        self, 
        name=None, 
        gtype=None, 
        gparam=None, 
        h_mat=np.eye(4),
        color='k',
        for_grasp=False,
        for_support=False,
        ):
        """
        Add object

        Args:
            name (str): An identifier for the object
            gtype (str): object type (cylinder, sphere, box)
            gparam (float or tuple): object parameter (radius, length, size)
            h_mat (np.array): Homogeneous transform matrix for the object
            color (np.array or str) : object color
        """
        obs_name = self._convert_name(name)
        self._check_gtype(gtype)
        self._check_gparam(gtype, gparam)
        self._objects[obs_name] = [gtype, gparam, h_mat, color]

        if for_grasp:
            self.grasp_objects[obs_name] = [gtype, gparam, h_mat, color]
        
        if for_support:
            self.support_objects[obs_name] = [gtype, gparam, h_mat, color]

    def remove_object(self, name):
        name = self._convert_name(name)
        if name in list(self._objects.keys()):
            self._objects.pop(name, None)

    def set_transform(self, name, h_mat=np.eye(4)):
        self._objects[self._convert_name(name)][2] = h_mat

    def get_info(self, name):
        name = self._convert_name(name)

        if name not in list(self._objects.keys()):
            raise NotFoundError(f"'{name}' is not in {self._objects.keys()}")

        info = {}
        info[ObjectData.NAME] = name
        info[ObjectData.G_TYPE] = self._objects[name][0]
        info[ObjectData.G_PARAM] = self._objects[name][1]
        info[ObjectData.POSE] = self._objects[name][2]

        return info

    def visualize(
        self,
        ax, 
        alpha=1.0,
    ):
        for info in self._objects.values():
            plt.plot_mesh(
                ax=ax, 
                mesh=info[1], 
                h_mat=info[2], 
                color=info[3],
                alpha=alpha,
            )
        plt.plot_basis(ax)
        
    @staticmethod
    def _convert_name(name):
        """
        convert input name to object name

        Args:
            nam (str): An identifier for the object

        Returns:
            name(str) : objects_ + name
        """
        if name and "object" not in name:
            name = "object_" + name
        return name
    
    @staticmethod
    def _check_gtype(gtype):
        """
        check object's geom type
        """
        if gtype not in object_types:
            raise NotFoundError(f"'{gtype}' is not in {object.object_types}")
    
    @staticmethod
    def _check_gparam(gtype, gparam):
        """
        check object's geom param 
        """
        if not isinstance(gparam, (tuple, list, np.ndarray)):
            gparam = [gparam]
        if gtype == "sphere":
            assert len(gparam) == 1, f"{gtype}'s parameter need only 'radius'"
        if gtype == "box":
            assert len(gparam) == 3, f"{gtype}'s parameter need box 'size(x, y, z)'"
        if gtype == "cylinder":
            assert len(gparam) == 2, f"{gtype}'s parameter need 'radius' and 'length'"

    @property
    def objects(self):
        return self._objects
    
