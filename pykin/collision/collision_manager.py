import numpy as np
import collections 
import itertools
import copy

try:
    # pip install python-fcl
    # pip install trimesh[easy]
    import fcl
    import trimesh
except BaseException:
    fcl = None
    trimesh = None

from pykin.collision.contact_data import ContactData
from pykin.utils.transform_utils import get_h_mat
from pykin.utils.log_utils import create_logger

logger = create_logger('Collision Manager', "debug")

class CollisionManager:
    """
    A rigid body collision manager.

    Args:
        mesh_path (str): absolute path of mesh
    """

    def __init__(self, mesh_path=None):
        if fcl is None:
            raise ValueError('No FCL Available! Please install the python-fcl library')
        
        self.mesh_path = mesh_path
        self._objs = {}
        self._names = collections.defaultdict(lambda: None)
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()
        self._filter_names = set()
        self.geom = "visual"
        self.objects = None

    def __repr__(self):
        return 'pykin.collision.collision_manager.{}()'.format(type(self).__name__)

    def setup_robot_collision(self, robot, fk=None, geom="visual"):
        """
        Setup robots' collision

        Args:
            robot (SingleArm or Bimanual): pykin robot
            fk (OrderedDict): result(forward kinematics) of computing robots' forward kinematics
            geom (str): robot's geometry type name ("visual" or "collision")
        """
        if fk is None:
            fk = robot.init_fk
        self.geom = geom
        self._filter_contact_names(robot, fk, geom)

    def setup_object_collision(self, objects):
        """
        Setup object' collision

        Args:
            objects (defaultdict): pykin objects
        """
        self.objects = objects
        for name, info in objects:
            self.add_object(name, info[0], info[1], info[2])

    def _filter_contact_names(self, robot, fk, geom):      
        """
        Filter contact names in the beginning

        Args:
            robot (SingleArm or Binmanul): pykin robot
            fk (OrderedDict): result(forward kinematics) of computing robots' forward kinematics
            geom (str): robot's geometry type name ("visual" or "collision")
        """
        for link, transformation in fk.items():
            if geom == "visual":
                robot_gtype = robot.links[link].visual.gtype
                h_mat = np.dot(transformation.h_mat, robot.links[link].visual.offset.h_mat)
                
                if robot_gtype is None:
                    continue

                if robot_gtype == "mesh":
                    mesh_name = self.mesh_path + robot.links[link].visual.gparam.get('filename')
                    gparam = trimesh.load_mesh(mesh_name)
                elif robot_gtype == 'cylinder':
                    radius = float(robot.links[link].visual.gparam.get('radius'))
                    length = float(robot.links[link].visual.gparam.get('length'))
                    gparam = (radius, length)
                elif robot_gtype == 'sphere':
                    radius = float(robot.links[link].visual.gparam.get('radius'))
                    gparam = radius
                elif robot_gtype == 'box':
                    size = robot.links[link].visual.gparam.get('size')
                    gparam = size
            else:
                robot_gtype = robot.links[link].collision.gtype
                h_mat = np.dot(transformation.h_mat, robot.links[link].collision.offset.h_mat)

                if robot_gtype is None:
                    continue
                
                if robot_gtype == "mesh":
                    mesh_name = self.mesh_path + robot.links[link].collision.gparam.get('filename')
                    gparam = trimesh.load_mesh(mesh_name)
                elif robot_gtype == 'cylinder':
                    radius = float(robot.links[link].collision.gparam.get('radius'))
                    length = float(robot.links[link].collision.gparam.get('length'))
                    gparam = (radius, length)
                elif robot_gtype == 'sphere':
                    radius = float(robot.links[link].collision.gparam.get('radius'))
                    gparam = radius
                elif robot_gtype == 'box':
                    size = robot.links[link].collision.gparam.get('size')
                    gparam = size

            self.add_object(robot.links[link].name, robot_gtype, gparam, h_mat)

        _, names = self.in_collision_internal(return_names=True)
        self._filter_names = copy.deepcopy(names)

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

    def in_collision_internal(self, return_names=False, return_data=False):
        """
        Check if any pair of objects in the manager collide with one another.

        Args:
            return_names (bool): If true, a set is returned containing the names 
                                 of all pairs of objects in collision.
            return_data (bool): If true, a list of ContactData is returned as well
        
        Returns:
            is_collision (bool): True if a collision occurred between any pair of objects and False otherwise
            names (set of 2-tup): The set of pairwise collisions. Each tuple
                                  contains two names in alphabetical order indicating
                                  that the two corresponding objects are in collision.
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
        for contact in cdata.result.contacts:
            names = (self._extract_name(contact.o1),
                        self._extract_name(contact.o2))
            names = tuple(sorted(names))
            if (names[0], names[1]) in self._filter_names:
                continue
            if return_names:
                objs_in_collision.add(tuple(sorted(names)))
            if return_data:
                contact_data.append(ContactData(names, contact))
                
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

    def in_collision_other(self, other_manager=None, return_names=False, return_data=False): 
        """
        Check if any object from this manager collides with any object
        from another manager.

        Args:
            other_manager (CollisionManager): Another collision manager object
            return_names (bool): If true, a set is returned containing the names 
                                 of all pairs of objects in collision.
            return_data (bool): If true, a list of ContactData is returned as well
        
        Returns:
            is_collision (bool): True if a collision occurred between any pair of objects and False otherwise
            names (set of 2-tup): The set of pairwise collisions. Each tuple
                                  contains two names (first from this manager,
                                  second from the other_manager) indicating
                                  that the two corresponding objects are in collision.
            contacts (list of ContactData): All contacts detected
        """
        
        if other_manager is None:
            if return_names:
                return None, None
            return None
            
        cdata = fcl.CollisionData()
        if return_names or return_data:
            cdata = fcl.CollisionData(request=fcl.CollisionRequest(
                num_max_contacts=100000, enable_contact=False))

        self._manager.collide(other_manager._manager,
                              cdata,
                              fcl.defaultCollisionCallback)
        
        result = cdata.result.is_collision

        objs_in_collision = set()
        contact_data = []

        for contact in cdata.result.contacts:
            reverse = False
            coll_names = (self._extract_name(contact.o1), other_manager._extract_name(contact.o2))
            if (coll_names[0], coll_names[1]) in self._filter_names:
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

        if return_names:
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

    def get_distances_internal(self):
        req = fcl.DistanceRequest()
        res = fcl.DistanceResult()

        result = collections.defaultdict(float)
        for (o1, o2) in list(itertools.permutations(self._objs, 2)):
            if (o1, o2) in self._filter_names:
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
        print(f"*"*20 + f" {name} Collision Info "+ f"*"*20)
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


