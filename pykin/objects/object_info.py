import numpy as np

from pykin.utils.error_utils import NotFoundError
from pykin.objects.object_data import ObjectData
from pykin.utils.transform_utils import get_pose_from_homogeneous
from pykin.utils.kin_utils import ShellColors as scolors

object_types = ("mesh", "sphere", "box", "cylinder")
class ObjectInfo(ObjectData):
    def __init__(
        self, 
        name, 
        gtype, 
        gparam, 
        h_mat=np.eye(4), 
        color='k', 
        logical_state={}
    ):
        self.name = name
        self.gtype = gtype
        self.gparam = gparam
        self.h_mat = h_mat
        self.color = color
        self.logical_state = logical_state

        self._check_gtype(gtype)
        self._check_gparam(gtype, gparam)

    def __repr__(self):
        pose = get_pose_from_homogeneous(self.h_mat)
        pos = pose[:3]
        return f"""{scolors.HEADER}ObjectInfo{scolors.ENDC}(name={self.name}, pos={pos})"""
    
    @staticmethod
    def _check_gtype(gtype):
        """
        check object's geom type
        """
        if gtype not in object_types:
            raise NotFoundError(f"'{gtype}' is not in {object_types}")
    
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