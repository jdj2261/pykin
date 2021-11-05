import numpy as np
import collections

try:
    import fcl
except BaseException:
    fcl = None

from pykin.collision.contact_data import ContactData
from pykin.collision.distance_data import DistanceData

class CollisionManager:
    """
    A rigid body collision manager.
    """

    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self._objs = {}
        self._names = collections.defaultdict(lambda: None)
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()

    def __repr__(self):
        return 'pykin.collision.collision_manager.{}()'.format(type(self).__name__)

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

    def collision_check(self, return_names=False, return_data=False):
        """
        Check if any pair of objects in the manager collide with one another.
        
        Args:
            return_names (bool): If true, a set is returned containing the names of all pairs of objects in collision.
            return_data (bool): If true, a list of ContactData is returned as well
        
        Returns:
            is_collision (bool): True if a collision occurred between any pair of objects and False otherwise
            names (set of 2-tup): The set of pairwise collisions. 
            contacts (list of ContactData): All contacts detected
        """
        cdata = fcl.CollisionData()
        if return_names or return_data:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(
                num_max_contacts=100000, enable_contact=True))

        self._manager.collide(cdata, fcl.defaultCollisionCallback)

        result = cdata.result.is_collision

        objs_in_collision = set()
        contact_data = []
        
        if return_names or return_data:
            for contact in cdata.result.contacts:
                coll_names = (self._extract_name(contact.o1),self._extract_name(contact.o2))
                coll_names = tuple(sorted(coll_names))

                if ("obstacle" in coll_names[0] and "obstacle" in coll_names[1]):
                    continue

                if return_names:
                    objs_in_collision.add(coll_names)
                if return_data:
                    contact_data.append(ContactData(coll_names, contact))

        if not objs_in_collision:
            result = False
            objs_in_collision = "No object collided.."

        if return_names and return_data:
            return result, objs_in_collision, contact_data
        elif return_names:
            return result, objs_in_collision
        elif return_data:
            return result, contact_data
        else:
            return result

    def get_min_distance(self, return_names=False, return_data=False):
        ddata = fcl.DistanceData()
        if return_data:
            ddata = fcl.DistanceData(
                fcl.DistanceRequest(enable_nearest_points=True),
                fcl.DistanceResult()
            )

        self._manager.distance(ddata, fcl.defaultDistanceCallback)

        distance = ddata.result.min_distance

        names, data = None, None
        if return_names or return_data:
            names = (self._extract_name(ddata.result.o1),
                     self._extract_name(ddata.result.o2))
            data = DistanceData(names, ddata.result)
            names = tuple(sorted(names))

        if return_names and return_data:
            return distance, names, data
        elif return_names:
            return distance, names
        elif return_data:
            return distance, data
        else:
            return distance

    def get_distance(self):
        pass

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


