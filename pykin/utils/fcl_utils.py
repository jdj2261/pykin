import numpy as np
import collections

try:
    import fcl
except BaseException:
    fcl = None

from pykin.kinematics.transform import Transform
import time

class ContactData:
    """
    Data structure for holding information about a collision contact.
    """

    def __init__(self, names, contact):
        self.names = set(names)
        self._inds = {
            names[0]: contact.b1,
            names[1]: contact.b2
        }
        self._point = contact.pos
        self._depth = contact.penetration_depth

    @property
    def point(self):
        return self._point

    @property
    def depth(self):
        return self._depth

    def index(self, name):
        return self._inds[name]


class FclManager:

    def __init__(self):
        self._objs = {}
        self._names = collections.defaultdict(lambda: None)
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()

    def add_object(self, 
                   name, 
                   gtype=None,
                   gparam=None,
                   transform=None):
        
        if gtype is None:
            return

        if transform is None:
            transform = np.eye(4)
        transform = np.asanyarray(transform, dtype=np.float32)
        if transform.shape != (4, 4):
            raise ValueError('transform must be (4,4)!')

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

    def remove_object(self, name):
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

    def remove_all_object(self):
        self._objs = {}
        self._names = collections.defaultdict(lambda: None)
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()

    def collision_check(self, return_names=False, return_data=False):
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

                if 'lower_forearm' in coll_names[0] and 'wrist' in coll_names[1]:
                    continue
                if 'upper_forearm' in coll_names[0] and 'upper_forearm_visual' in coll_names[1]:
                    continue
                if 'lower_forearm' in coll_names[0] and 'upper_forearm_visual' in coll_names[1]:
                    continue
                if 'lower_elbow' in coll_names[0] and 'upper_elbow_visual' in coll_names[1]:
                    continue
                if 'lower_shoulder' in coll_names[0] and 'upper_elbow' in coll_names[1]:
                    continue
                if 'lower_shoulder' in coll_names[0] and 'upper_shoulder' in coll_names[1]:
                    continue
                if 'upper_elbow' in coll_names[0] and 'upper_elbow_visual' in coll_names[1]:
                    continue
                if 'lower_shoulder' in coll_names[0] and 'upper_elbow_visual' in coll_names[1]:
                    continue
                if 'lower_elbow' in coll_names[0] and 'upper_forearm' in coll_names[1]:
                    continue
                if 'gripper_base' in coll_names[0] and 'hand_accelerometer' in coll_names[1]:
                    continue
                if 'head_link' in coll_names[0] and 'sonar_ring' in coll_names[1]:
                    continue
                if 'head_link' in coll_names[0] and 'head' in coll_names[1]:
                    continue
                if 'head_link' in coll_names[0] and 'screen' in coll_names[1]:
                    continue
                if 'head_link' in coll_names[0] and 'display' in coll_names[1]:
                    continue
                if 'gripper_base' in coll_names[0] and 'hand_accelerometer' in coll_names[1]:
                    continue
                if 'hand' in coll_names[0] and 'hand_accelerometer' in coll_names[1]:
                    continue
                if 'hand' in coll_names[0] and 'wrist' in coll_names[1]:
                    continue
                if 'gripper_base' in coll_names[0] and 'hand' in coll_names[1]:
                    continue
                if 'display' in coll_names[0] and 'screen' in coll_names[1]:
                    continue
                
                if return_names:
                    objs_in_collision.add(coll_names)
                if return_data:
                    contact_data.append(ContactData(coll_names, contact))

        if return_names and return_data:
            if len(objs_in_collision) == 0:
                result = False
            return result, objs_in_collision, contact_data
        elif return_names:
            if len(objs_in_collision) == 0:
                result = False
            return result, objs_in_collision
        elif return_data:
            if len(objs_in_collision) == 0:
                result = False
            return result, contact_data
        else:
            return result

    def _get_geom(self, gtype, gparam):
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
        return geom

    def _extract_name(self, geom):
        return self._names[id(geom)]

    # def _is_unique_collision(self, names):
    #     if any('collision_head_link_1', 'collision_head_link_2') in names:
    #         print(names)
    #     # (names[0] == 'collision_head_link_1' and names[1] == 'sonar_ring') or \
    #     #    (names[0] == 'collision_head_link_2' and names[1] == 'collision_head_link_1') or \
    #     #    (names[0] == 'collision_head_link_2' and names[1] == 'sonar_ring') or \
    #     #    (names[0] == 'collision_head_link_2' and names[1] == 'head') or \
    #     #    (names[0] == 'collision_head_link_2' and names[1] == 'screen') or \
    #     #    (names[0] == 'collision_head_link_1' and names[1] == 'head') or \
    #     #    (names[0] == 'collision_head_link_1' and names[1] == 'screen') or \
    #     #    (names[0] == 'left_upper_forearm' and names[1] == 'left_upper_forearm_visual') or \
    #     #    (names[0] == 'left_lower_elbow'      and names[1] == 'left_upper_forearm') or \
    #     #    (names[0] == 'left_lower_elbow'      and names[1] == 'left_upper_forearm_visual') or \
    #     #    (names[0] == 'left_lower_forearm' and names[1] == 'left_wrist') or \
    #     #    (names[0] == 'left_upper_elbow' and names[1] == 'left_upper_elbow_visual') or \
    #     #    (names[0] == 'left_lower_shoulder'   and names[1] == 'left_upper_elbow') or \
    #     #    (names[0] == 'left_lower_shoulder'   and names[1] == 'left_upper_elbow_visual') or \
    #     #    (names[0] == 'right_upper_forearm_visual' and names[1] == 'right_lower_elbow') or \
    #     #    (names[0] == 'right_upper_forearm'   and names[1] == 'right_upper_forearm_visual') or \
    #     #    (names[0] == 'right_upper_forearm'   and names[1] == 'right_lower_elbow') or \
    #     #    (names[0] == 'right_upper_elbow_visual' and names[1] == 'right_upper_elbow') or \
    #     #    (names[0] == 'right_upper_elbow_visual' and names[1] == 'right_lower_shoulder') or \
    #     #    (names[0] == 'right_upper_elbow'     and names[1] == 'right_lower_shoulder') or \
    #     #    (names[0] == 'right_wrist'           and names[1] == 'right_lower_forearm'):

    #         return True


