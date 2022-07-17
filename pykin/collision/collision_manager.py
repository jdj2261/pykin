import numpy as np
import collections 
import itertools
import copy
from copy import deepcopy

try:
    # pip install python-fcl
    # pip install trimesh[easy]
    import fcl
    # import trimesh
except BaseException:
    fcl = None
    # trimesh = None

from pykin.utils.transform_utils import get_h_mat
from pykin.utils.log_utils import create_logger
from pykin.utils.kin_utils import ShellColors as sc

logger = create_logger('Collision Manager', "debug")

class CollisionManager:
    """
    A rigid body collision manager.

    Args:
        mesh_path (str): absolute path of mesh
    """

    def __init__(self, is_robot=False):
        if fcl is None:
            raise ValueError('No FCL Available! Please install the python-fcl library')
        
        self._objs = {}
        self._names = collections.defaultdict(lambda: None)
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()
        self.filtered_link_names = set()
        
        if is_robot:
            self.is_robot = is_robot

    # def __deepcopy__(self, memo=None):
    #     return deepcopy(self)

    def __repr__(self):
        return 'pykin.collision.collision_manager.{}()'.format(type(self).__name__)

    def setup_robot_collision(self, robot, geom="collision"):
        """
        Setup robots' collision

        Args:
            robot (SingleArm or Bimanual): pykin robot
            fk (OrderedDict): result(forward kinematics) of computing robots' forward kinematics
            geom (str): robot's geometry type name ("visual" or "collision")
        """
        if not self.is_robot:
            raise ValueError('Check argument!! Is is_robot True?')
        self.geom = geom
        self._filter_contact_names(robot, geom)

    def setup_gripper_collision(self, robot, fk=None, geom="collision"):
        if fk is None:
            fk = robot.init_fk

        info = robot.gripper.info
        for name, transform in fk.items():
            if name in list(info.keys()):
                if geom == "collision":
                    h_mat = np.dot(transform.h_mat, robot.links[name].collision.offset.h_mat)
                    for param in info[name][2]:
                        self.add_object(name, info[name][1], param, h_mat)
                else:
                    h_mat = np.dot(transform.h_mat, robot.links[name].visual.offset.h_mat)
                    for param in info[name][2]:
                        self.add_object(name, info[name][1], param, h_mat)

    def _filter_contact_names(self, robot, geom):      
        """
        Filter contact names in the beginning

        Args:
            robot (SingleArm or Binmanul): pykin robot
            fk (OrderedDict): result(forward kinematics) of computing robots' forward kinematics
            geom (str): robot's geometry type name ("visual" or "collision")
        """
        for link, info in robot.info[geom].items():
            for param in info[2]:
                self.add_object(info[0], info[1], param, info[3])

        _, names = self.in_collision_internal(return_names=True)
        self.filtered_link_names = copy.deepcopy(names)

    def add_object(self, 
                   name, 
                   gtype=None,
                   gparam=None,
                   h_mat=None):
        """
        Add an object to the collision manager.
        If an object with the given name is already in the manager, replace it.

        Args:
            name (str): An identifier for the object
            gtype (str): object type (cylinder, sphere, box)
            gparam (float or tuple): object parameter (radius, length, size)
            h_mat (np.array): Homogeneous transform matrix for the object
        """
        if gtype is None:
            return

        if h_mat is None:
            h_mat = np.eye(4)
        h_mat = np.asanyarray(h_mat, dtype=np.float32)
        if h_mat.shape != (4, 4):
            if h_mat.shape == (3,):
                h_mat = get_h_mat(position=h_mat)
            else:
                raise ValueError('transform must be (4,4)!')

        if gtype == "mesh":
            geom = self._get_BVH(gparam)
        else:
            geom = self._get_geom(gtype, gparam)
        
        t = fcl.Transform(h_mat[:3, :3], h_mat[:3, 3])
        o = fcl.CollisionObject(geom, t)

        # Add collision object to set
        if name in self._objs:
            self._manager.unregisterObject(self._objs[name])
        self._objs[name] = {'obj': o,
                            'geom': geom}
        # store the name of the geometry
        self._names[id(geom)] = name

        self._manager.registerObject(o)
        self._manager.update()

    def set_transform(self, name=None, h_mat=np.eye(4)):
        """
        Set the transform for one of the manager's objects.
        This replaces the prior transform.
        
        Args:
            name (str): An identifier for the object already in the manager
            h_mat (np.array): A new homogeneous transform matrix for the object
        """
        if name is None:
            return
            
        if name in self._objs:
            o = self._objs[name]['obj']
            o.setRotation(h_mat[:3, :3])
            o.setTranslation(h_mat[:3, 3])
            self._manager.update(o)
        else:
            raise ValueError('{} not in collision manager!'.format(name))

    def remove_object(self, name):
        """
        Delete an object from the collision manager.
        
        Args:
            name (str): The identifier for the object
        """
        if name in self._objs:
            self._manager.unregisterObject(self._objs[name]['obj'])
            self._manager.update(self._objs[name]['obj'])
            # remove objects from _objs
            geom_id = id(self._objs.pop(name)['geom'])
            # remove names
            self._names.pop(geom_id)
        else:
            logger.warn('{} not in collision manager!'.format(name))

    def reset_all_object(self):
        """
        Reset all object from the collision manager.
        """
        self._objs = {}
        self._names = collections.defaultdict(lambda: None)
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()

    def in_collision_internal(self, return_names=False):
        """
        Check if any pair of objects in the manager collide with one another.

        Args:
            return_names (bool): If true, a set is returned containing the names 
                                 of all pairs of objects in collision.
        
        Returns:
            is_collision (bool): True if a collision occurred between any pair of objects and False otherwise
            names (set of 2-tup): The set of pairwise collisions. Each tuple
                                  contains two names in alphabetical order indicating
                                  that the two corresponding objects are in collision.
        """
        cdata = fcl.CollisionData()
        if return_names:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(
                    num_max_contacts=100000, enable_contact=True))

        self._manager.collide(cdata, fcl.defaultCollisionCallback)

        result = cdata.result.is_collision

        objs_in_collision = set()
        for contact in cdata.result.contacts:
            names = (self._extract_name(contact.o1),
                        self._extract_name(contact.o2))
            names = tuple(sorted(names))
            if (names[0], names[1]) in self.filtered_link_names:
                continue
            if return_names:
                objs_in_collision.add(names)

        if not objs_in_collision:
            result = False
            objs_in_collision = "No object collided.."

        if return_names:
            return result, objs_in_collision
        else:
            return result

    def in_collision_other(self, other_manager=None, return_names=False): 
        """
        Check if any object from this manager collides with any object
        from another manager.

        Args:
            other_manager (CollisionManager): Another collision manager object
            return_names (bool): If true, a set is returned containing the names 
                                 of all pairs of objects in collision.
        Returns:
            is_collision (bool): True if a collision occurred between any pair of objects and False otherwise
            names (set of 2-tup): The set of pairwise collisions. Each tuple
                                  contains two names (first from this manager,
                                  second from the other_manager) indicating
                                  that the two corresponding objects are in collision.
        """
        
        if other_manager is None:
            if return_names:
                return None, None
            return None
            
        cdata = fcl.CollisionData()
        if return_names:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(
                num_max_contacts=100000, enable_contact=False))

        self._manager.collide(other_manager._manager,
                              cdata,
                              fcl.defaultCollisionCallback)
        
        result = cdata.result.is_collision

        objs_in_collision = set()

        for contact in cdata.result.contacts:
            coll_names = (self._extract_name(contact.o1), other_manager._extract_name(contact.o2))
            if (coll_names[0], coll_names[1]) in self.filtered_link_names:
                continue
            if coll_names[0] is None:
                coll_names = (self._extract_name(contact.o2), other_manager._extract_name(contact.o1))

            if return_names:
                objs_in_collision.add(coll_names)

        if return_names:
            if not objs_in_collision:
                result = False
                objs_in_collision = "No object collided.."

        if return_names:
            return result, objs_in_collision
        else:
            return result

    def get_distances_internal(self):
        req = fcl.DistanceRequest()
        res = fcl.DistanceResult()

        result = collections.defaultdict(float)
        for (o1, o2) in list(itertools.permutations(self._objs, 2)):
            if (o1, o2) in self.filtered_link_names:
                continue
            distance = np.round(fcl.distance(self._objs[o1]['obj'],self._objs[o2]['obj'], req, res), 6)
            result[(o1, o2)] = distance
        
        return result

    def get_distances_other(self, other_manager):
        """
        Get the minimum distance between any pair of objects, one in each manager.
        
        Args:
            other_manager (CollisionManager): Another collision manager object

        Returns:
            distance (float): The min distance between a pair of objects, one from each manager.
        """
        
        def _mix_objects():
            for o1 in self._objs:
                for o2 in other_manager._objs:
                    yield (o1, o2)

        req = fcl.DistanceRequest()
        res = fcl.DistanceResult()
        
        result = collections.defaultdict(float)
        for (o1, o2) in _mix_objects():
            distance = np.round(fcl.distance(self._objs[o1]['obj'], other_manager._objs[o2]['obj'], req, res), 6)
            result[(o1, o2)] = distance
        
        return result

    def get_collision_info(self):
        """
        Get CollisionManager info (name, transform)

        Returns:
            col_info (dict): Collision Info
        """
        col_info = {}
        for name, info in self._objs.items():
            T = get_h_mat(position=info["obj"].getTranslation(), orientation=info["obj"].getRotation())
            col_info[name] = T
        return col_info

    def show_collision_info(self, name="Robot"):
        """
        Show CollisionManager info (name, transform)

        Args:
            name (str)
        """
        print(f"*"*20 + f" {sc.OKGREEN}{name} Collision Info{sc.ENDC} "+ f"*"*20)
        for name, info in self.get_collision_info().items():
            print(name, info[:3, 3])
        print(f"*"*63 + "\n")

    def _get_BVH(self, mesh):
        """
        Get a BVH for a mesh.

        Args:
            mesh (Trimesh): Mesh to create BVH for

        Returns:
            bvh (fcl.BVHModel): BVH object of source mesh
        """
        bvh = self.mesh_to_BVH(mesh)
        return bvh

    @staticmethod
    def mesh_to_BVH(mesh):
        """
        Create a BVHModel object from a Trimesh object

        Args:
            mesh (Trimesh): Input geometry

        Returns:
            bvh (fcl.BVHModel): BVH of input geometry
        """
        bvh = fcl.BVHModel()
        bvh.beginModel(num_tris_=len(mesh.faces),
                    num_vertices_=len(mesh.vertices))
        bvh.addSubModel(verts=mesh.vertices,
                        triangles=mesh.faces)
        bvh.endModel()
        return bvh

    @staticmethod
    def _get_geom(gtype, gparam):
        """
        Get fcl geometry from robot's geometry type or params
        
        Args:
            geom (CollisionObject): Input model
        
        Returns:
            names (hashable): Name of input geometry
        """
        geom = None
        if gtype == "cylinder":
            length = gparam[0]
            radius = gparam[1]
            geom = fcl.Cylinder(radius, length)
        elif gtype == "sphere":
            radius = float(gparam)
            geom = fcl.Sphere(radius)
        elif gtype == "box":
            size = gparam
            geom = fcl.Box(*size)
        elif gtype == "mesh":
            pass
        return geom

    def _extract_name(self, geom):
        """
        Retrieve the name of an object from the manager by its
        CollisionObject, or return None if not found.
        
        Args:
            geom (CollisionObject): Input model
        
        Returns:
            names (hashable): Name of input geometry
        """
        return self._names[id(geom)]