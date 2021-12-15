import numpy as np
import collections 
import itertools
import copy

try:
    import fcl
    import trimesh
except BaseException:
    fcl = None
    trimesh = None

from pykin.utils.error_utils import CollisionError, NotFoundError
from pykin.utils.transform_utils import get_h_mat
from pykin.utils.log_utils import create_logger
from pykin.collision.contact_data import ContactData
logger = create_logger('Collision Manager', "debug",)

class CollisionManager:
    """
    A rigid body collision manager.
    """

    def __init__(self, mesh_path=None):
        self.mesh_path = mesh_path
        self._objs = {}
        self._names = collections.defaultdict(lambda: None)
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()
        self._filter_names = set()

    def __repr__(self):
        return 'pykin.collision.collision_manager.{}()'.format(type(self).__name__)

    def filter_contact_names(self, robot, fk=None, geom="visual"):

        if fk is None:
            fk = robot.init_transformations
            
        is_collision = False

        def _get_collision_datas():
            nonlocal is_collision
            result = []
            for (name1, name2) in self._filter_names:
                index_name1 = list(self._objs.keys()).index(name1)
                index_name2 = list(self._objs.keys()).index(name2)

                if abs(index_name1-index_name2) > 1:
                    for joint in robot.joints.values():
                        if name1 == joint.parent:
                            if joint.dtype == "revolute":
                                print(name1, name2)
                                print(index_name1, index_name2)
                                is_collision = True
                                result.append((name1, name2))
            return result


        for link, transformation in fk.items():
            if geom == "visual":
                if robot.links[link].visual.gtype == "mesh":
                    mesh_name = robot.links[link].visual.gparam.get('filename')
                    file_name = self.mesh_path + mesh_name
                    mesh = trimesh.load_mesh(file_name)
                    A2B = np.dot(transformation.h_mat, robot.links[link].visual.offset.h_mat)
                    self.add_object(robot.links[link].name, "mesh", mesh, A2B)

            if geom == "collision":
                if robot.links[link].collision.gtype == "mesh":
                    mesh_name = robot.links[link].collision.gparam.get('filename')
                    file_name = self.mesh_path + mesh_name
                    mesh = trimesh.load_mesh(file_name)
                    A2B = np.dot(transformation.h_mat, robot.links[link].collision.offset.h_mat)
                    self.add_object(robot.links[link].name, "mesh", mesh, A2B)

        _, names = self.in_collision_internal(return_names=True)
        self._filter_names = copy.deepcopy(names)

        if robot.robot_name == "ur5e":
            return

        collision_datas = _get_collision_datas()
        if is_collision:
            for name1, name2 in collision_datas:
                logger.error(f"{name1} and {name2} is Collision..")
            raise CollisionError(f"Conflict confirmed. Check the joint settings again") 
    
    def add_object(self, 
                   name, 
                   gtype=None,
                   gparam=None,
                   transform=None):
        """
        Add an object to the collision manager.
        If an object with the given name is already in the manager, replace it.

        Args:
            name (str): An identifier for the object
            gtype (str): object type (cylinder, sphere, box)
            gparam (float or tuple): object parameter (radius, length, size)
            transform (np.array): Homogeneous transform matrix for the object
        """
        if gtype is None:
            return

        if transform is None:
            transform = np.eye(4)
        transform = np.asanyarray(transform, dtype=np.float32)
        if transform.shape != (4, 4):
            if transform.shape == (3,):
                transform = get_h_mat(position=transform)
            else:
                raise ValueError('transform must be (4,4)!')

        if gtype == "mesh":
            geom = self._get_BVH(gparam)
        else:
            geom = self._get_geom(gtype, gparam)
        
        t = fcl.Transform(transform[:3, :3], transform[:3, 3])
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

    def set_transform(self, name=None, transform=np.eye(4)):
        """
        Set the transform for one of the manager's objects.
        This replaces the prior transform.
        
        Args:
            name (str): An identifier for the object already in the manager
            transform (np.array): A new homogeneous transform matrix for the object
        """
        if name is None:
            return
            
        if name in self._objs:
            o = self._objs[name]['obj']
            o.setRotation(transform[:3, :3])
            o.setTranslation(transform[:3, 3])
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
            print(f"{name} object is removed")
        else:
            raise ValueError('{} not in collision manager!'.format(name))

    def reset_all_object(self):
        """
        Reset all object from the collision manager.
        """
        self._objs = {}
        self._names = collections.defaultdict(lambda: None)
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()

    def in_collision_internal(self, return_names=False, return_data=False):
        cdata = fcl.CollisionData()
        if return_names or return_data:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(
                num_max_contacts=100000, enable_contact=False))

        self._manager.collide(cdata, fcl.defaultCollisionCallback)

        result = cdata.result.is_collision

        objs_in_collision = set()
        contact_data = []

        if return_names or return_data:
            for contact in cdata.result.contacts:
                coll_names = (self._extract_name(contact.o1),self._extract_name(contact.o2))
                coll_names = tuple(sorted(coll_names))

                if (coll_names[0], coll_names[1]) in self._filter_names:
                    continue

                if ("obstacle" in coll_names[0] and "obstacle" in coll_names[1]):
                    continue

                if return_names:
                    objs_in_collision.add(coll_names)
                if return_data:
                    contact_data.append(ContactData(coll_names, contact))

        if not objs_in_collision:
            result = False
            objs_in_collision = "No object collided.."

        return self._get_returns(return_names, return_data, result, objs_in_collision, contact_data)

    def in_collision_other(self, other_manager=None, return_names=False, return_data=False):
        if other_manager is None:
            return 
            
        cdata = fcl.CollisionData()
        if return_names or return_data:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(
                num_max_contacts=100000, enable_contact=False))

        self._manager.collide(other_manager._manager, cdata, fcl.defaultCollisionCallback)

        result = cdata.result.is_collision

        objs_in_collision = set()
        contact_data = []

        if return_names or return_data:
            for contact in cdata.result.contacts:
                reverse = False
                coll_names = (self._extract_name(contact.o1), other_manager._extract_name(contact.o2))

                if (coll_names[0], coll_names[1]) in self._filter_names:
                    continue

                if ("obstacle" in coll_names[0] and "obstacle" in coll_names[1]):
                    continue

                if coll_names[0] is None:
                    coll_names = (self._extract_name(contact.o2), other_manager._extract_name(contact.o1))
                    reverse = True

                if return_names:
                    objs_in_collision.add(coll_names)
                if return_data:
                    if reverse:
                        coll_names = reversed(coll_names)
                    contact_data.append(ContactData(coll_names, contact))

            if not objs_in_collision:
                result = False
                objs_in_collision = "No object collided.."

        return self._get_returns(return_names, return_data, result, objs_in_collision, contact_data)

    def get_distances_internal(self):
        req = fcl.DistanceRequest()
        res = fcl.DistanceResult()

        result = collections.defaultdict(float)
        for (o1, o2) in list(itertools.combinations(self._objs, 2)):
            if (o1, o2) in self._filter_names:
                continue
            distance = np.round(fcl.distance(self._objs[o1]['obj'],self._objs[o2]['obj'], req, res), 6)
            result[(o1, o2)] = distance
        
        return result

    def get_distances_other(self, other_manager):
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

    @staticmethod
    def _get_returns(return_names, return_data, *args):
        if return_names and return_data:
            return args[0], args[1], args[2]
        elif return_names:
            return args[0], args[1]
        elif return_data:
            return args[0], args[2]
        else:
            return args[0]

    def _get_BVH(self, mesh):
        """
        Get a BVH for a mesh.

        Parameters
        -------------
        mesh : Trimesh
          Mesh to create BVH for

        Returns
        --------------
        bvh : fcl.BVHModel
          BVH object of source mesh
        """
        bvh = self.mesh_to_BVH(mesh)
        return bvh

    @staticmethod
    def mesh_to_BVH(mesh):
        """
        Create a BVHModel object from a Trimesh object

        Parameters
        -----------
        mesh : Trimesh
        Input geometry

        Returns
        ------------
        bvh : fcl.BVHModel
        BVH of input geometry
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
            radius = gparam[0]
            length = gparam[1]
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


