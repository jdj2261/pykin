import numpy as np

from pykin.utils.error_utils import NotFoundError
from pykin.utils.kin_utils import ShellColors as scolors
from pykin.utils.transform_utils import get_pose_from_homogeneous

object_types = ("mesh", "sphere", "box", "cylinder")

class Object:
    def __init__(
        self, 
        name, 
        gtype, 
        gparam, 
        h_mat=np.eye(4), 
        color='k'
    ):
        self.name = name
        self.gtype = gtype
        self.gparam = gparam
        self.h_mat = h_mat
        self.color = np.asarray(color, dtype=np.float32)
        
        self._check_gtype(gtype)
        self._check_gparam(gtype, gparam)

    def __repr__(self):
        pose = get_pose_from_homogeneous(self.h_mat)
        pos = pose[:3]
        return f"""{scolors.HEADER}Object{scolors.ENDC}(name={self.name}, pos={pos})"""

    def __eq__(self, other):
        if (self.name == other.name):
            return True
        else:
            return False

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