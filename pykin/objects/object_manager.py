import numpy as np
from collections import OrderedDict


import pykin.utils.plot_utils as plt
from pykin.utils.error_utils import NotFoundError

from pykin.objects.object_info import ObjectInfo
from pykin.objects.object_info import ObjectData

object_types = ("mesh", "sphere", "box", "cylinder")

class ObjectManager(ObjectInfo):
    """
    ObjectManager class 
    """
    
    def __init__(self):
        self._objects = OrderedDict()
        self.grasp_objects = OrderedDict()
        self.support_objects = OrderedDict()
        self._gtype = None
        self.logical_states = OrderedDict()
        
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
        obj_info=None
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
        if isinstance(obj_info, ObjectInfo):
            self._objects[obj_info.name] = obj_info
            
            if for_grasp:
                self.grasp_objects[obj_info.name] = obj_info
            if for_support:
                self.support_objects[obj_info.name] = obj_info

        else:
            self._objects[name] = ObjectInfo(name, gtype, gparam, h_mat, color)
            if for_grasp:
                self.grasp_objects[name] = ObjectInfo(name, gtype, gparam, h_mat, color, for_grasp=for_grasp)
            
            if for_support:
                self.support_objects[name] = ObjectInfo(name, gtype, gparam, h_mat, color, for_support=for_support)

    def remove_object(self, name):
        if name in list(self._objects.keys()):
            self._objects.pop(name, None)

    def set_transform(self, name, h_mat=np.eye(4)):
        self._objects[name].pose = h_mat

    def get_info(self, name):
        if name not in list(self._objects.keys()):
            raise NotFoundError(f"'{name}' is not in {self._objects.keys()}")

        info = {}
        info[ObjectData.NAME] = name
        info[ObjectData.G_TYPE] = self._objects[name].gtype
        info[ObjectData.G_PARAM] = self._objects[name].gparam
        info[ObjectData.POSE] = self._objects[name].pose

        return info

    def visualize_all_objects(
        self,
        ax, 
        alpha=1.0,
    ):
        for info in self._objects.values():
            plt.plot_mesh(
                ax=ax, 
                mesh=info.gparam, 
                h_mat=info.pose, 
                color=info.color,
                alpha=alpha,
            )
        plt.plot_basis(ax)
        
    @property
    def objects(self):
        return self._objects