import numpy as np
from collections import OrderedDict
from pykin.collision.collision_manager import CollisionManager
from pykin.objects.gripper import GripperManager

import pykin.utils.plot_utils as plt

from pykin.utils.error_utils import NotFoundError
from pykin.objects.object_info import ObjectInfo
from pykin.objects.object_info import ObjectData

object_types = ("mesh", "sphere", "box", "cylinder")

class ObjectManager:
    """
    ObjectManager class 
    """
    
    def __init__(self):
        self._objects = OrderedDict()
        self.grasp_objects = OrderedDict()
        self.support_objects = OrderedDict()
        self.obj_col_manager = CollisionManager()
        self.gripper_manager = None
        
    def __call__(self, *args, **kwards):
        self.add_object(*args, **kwards)

    def __repr__(self):
        return 'pykin.objects.object_manager.{}()'.format(type(self).__name__)

    def __iter__(self):
        items = list(self._objects.items())
        items.sort(key=lambda x : (x[0]))
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
            self.obj_col_manager.add_object(
                obj_info.name, 
                obj_info.gtype, 
                obj_info.gparam, 
                obj_info.h_mat)
            
            if for_grasp:
                self.grasp_objects[obj_info.name] = obj_info
            if for_support:
                self.support_objects[obj_info.name] = obj_info

        else:
            self._objects[name] = ObjectInfo(name, gtype, gparam, h_mat, color)
            self.obj_col_manager.add_object(name, gtype, gparam, h_mat)
            if for_grasp:
                self.grasp_objects[name] = ObjectInfo(name, gtype, gparam, h_mat, color)
            
            if for_support:
                self.support_objects[name] = ObjectInfo(name, gtype, gparam, h_mat, color)

    def add_gripper(self, gripper_mngr:GripperManager):
        self.gripper_manager = gripper_mngr

    def remove_object(self, name):
        if name in list(self._objects.keys()):
            self._objects.pop(name, None)
            self.obj_col_manager.remove_object(name)

    def set_transform(self, name, h_mat=np.eye(4)):
        self._objects[name].h_mat = h_mat
        self.obj_col_manager.set_transform(name, h_mat)

    def get_info(self, name):
        if name not in list(self._objects.keys()):
            raise NotFoundError(f"'{name}' is not in {self._objects.keys()}")

        info = {}
        info[ObjectData.NAME] = name
        info[ObjectData.G_TYPE] = self._objects[name].gtype
        info[ObjectData.G_PARAM] = self._objects[name].gparam
        info[ObjectData.POSE] = self._objects[name].h_mat

        return info

    def get_logical_all_states(self):
        logical_states = []
        for name, info in self.objects.items():
            logical_states.append((name, info.logical_state))
        if self.gripper_manager is not None:
            logical_states.append((self.gripper_manager.name, self.gripper_manager.logical_state))
        return logical_states

    def get_logical_on_states(self):
        pass

    def get_logical_static_states(self):
        pass

    def get_logical_support_states(self):
        pass

    def check_collision(self):
        pass

    def visualize_all_objects(
        self,
        ax, 
        alpha=1.0,
    ):
        plt.plot_basis(ax)
        for info in self._objects.values():
            plt.plot_mesh(
                ax=ax, 
                mesh=info.gparam, 
                h_mat=info.h_mat, 
                color=info.color,
                alpha=alpha,
            )
        if self.gripper_manager:
            for link, info in self.gripper_manager.gripper.items():
                mesh = info[2]
                h_mat = info[3]
                if info[1] == "mesh":
                    # mesh_color = self.gripper_manager.robot.links[link].collision.gparam.get('color')
                    # mesh_color = np.array([color for color in mesh_color.values()]).flatten()               
                    plt.plot_mesh(ax, mesh, h_mat, color=self.gripper_manager.color, alpha=alpha)

    @property
    def objects(self):
        return self._objects