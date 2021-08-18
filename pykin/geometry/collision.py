
import os, sys
import warnings

try:
    import fcl
except ImportError:
    warnings.warn(
        "Cannot display mesh. Library 'fcl' not installed.")
from collections import OrderedDict

pykin_path = os.path.abspath(os.path.dirname(__file__)+"../../")
sys.path.append(pykin_path)

from pykin.kinematics.transform import Transform
from pykin.utils import plot as plt


class Collision:
    
    def __init__(self, robot=None, obj=None, fk:dict=None):
        self.robot = robot
        self._obj = obj
        self.fk = fk
        self.objects = []
        self.links = OrderedDict()

        if obj is not None:
            self.objects.append(obj)

    def __repr__(self):
        if self.robot is not None:
            return f"""Robot Collision Info:
            {list(self.links.values())}"""
        else:
            return f"""Object Collision Info: {self.objects}"""

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, obj):
        self._obj = obj

    def append_objects(self, gtype, offset=Transform(), color='k'):
        if self._obj.keys():
            if len(self.objects) != 0:
                objects = [obj[1] for obj in self.objects]
                assert objects[0].keys() != self._obj.keys(), f"Duplicate name. please check again"
            self.objects.append((gtype, self._obj, offset, color))
 
    def collision_check(self, visible=False):
        ax = None
        if visible:
            _, ax = plt.init_3d_figure("Collision")
            plt.plot_basis(ax=ax)
            ax.legend()

        geoms, objs, names = self.convert_fcl(visible, ax)

        geom_id_to_name = {id(geom): name for geom, name in zip(geoms, names)}

        # Create manager
        manager = fcl.DynamicAABBTreeCollisionManager()
        manager.registerObjects(objs)
        manager.setup()

        # Create collision request structure
        crequest = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
        cdata = fcl.CollisionData(crequest, fcl.CollisionResult())

        # Run collision request
        manager.collide(cdata, fcl.defaultCollisionCallback)

        # objects that are in collision
        objs_in_collision = self.extract_collision_data(geom_id_to_name, cdata)
        
        for coll_pair in objs_in_collision:
            print('Object {} in collision with object {}!'.format(
                coll_pair[0], coll_pair[1]))

    def convert_fcl(self, visible=False, ax=None):
        geoms = []
        objs = []
        names = []

        for object in self.objects:
            obj_type = object[0]
            obj_info = object[1]
            obj_pose = object[2]
            obj_color = object[3]

            if obj_type == 'cylinder':
                geom, obj, name = self.fcl_cylinder(obj_info, obj_pose, obj_color, visible, ax)
            if obj_type == 'sphere':
                geom, obj, name = self.fcl_sphere(obj_info, obj_pose, obj_color, visible, ax)
            if obj_type == 'box':
                geom, obj, name = self.fcl_box(obj_info, obj_pose, obj_color, visible, ax)

            geoms.append(geom)
            objs.append(obj)
            names.append(name)
        
        return geoms, objs, names

    def fcl_cylinder(self, info, pose, color, visible, ax=None):
        radius = float(list(info.values())[0][0].get('radius'))
        length = float(list(info.values())[0][1].get('length'))
        rot = pose.R_mat
        pos = pose.pos
        A2B = fcl.Transform(rot, pos)

        geom = fcl.Cylinder(radius, length)
        obj = fcl.CollisionObject(geom, A2B)
        name = list(info.keys())[0]

        if visible:
            plt.plot_cylinder(ax=ax, A2B=pose.matrix(), radius=radius,
                            length=length, alpha=0.5, color=color)
        return geom, obj, name

    def fcl_sphere(self, info, pose, color, visible, ax=None):
        radius = float(list(info.values())[0].get('radius'))
        pos = pose.pos
        A2B = fcl.Transform(pos)

        geom = fcl.Sphere(radius)
        obj = fcl.CollisionObject(geom, A2B)
        name = list(info.keys())[0]

        if visible:
            plt.plot_sphere(ax=ax, p=pose.matrix()[:3, -1],
                            radius=radius, alpha=0.5, color=color)
        return geom, obj, name

    def fcl_box(self, info, pose, color, visible, ax=None):
        size = list(info.values())[0].get('size')
        rot = pose.R_mat
        pos = pose.pos
        A2B = fcl.Transform(rot, pos)

        geom = fcl.Box(*size)
        obj = fcl.CollisionObject(geom, A2B)
        name = list(info.keys())[0]

        if visible:
            plt.plot_box(ax=ax, A2B=pose.matrix(),
                        size=size, alpha=0.5, color=color)
        return geom, obj, name

    def extract_collision_data(self, geom_id_to_name, cdata):
        objs_in_collision = set()
        for contact in cdata.result.contacts:
            # Extract collision geometries that are in contact
            coll_geom_0 = contact.o1
            coll_geom_1 = contact.o2

            # Get their names
            coll_names = [geom_id_to_name[id(coll_geom_0)], geom_id_to_name[id(coll_geom_1)]]
            coll_names = tuple(sorted(coll_names))

            # aboue baxter
            if 'lower_forearm'  in coll_names[0] and 'wrist'                in coll_names[1]: continue
            if 'upper_forearm'  in coll_names[0] and 'upper_forearm_visual' in coll_names[1]: continue
            if 'lower_forearm'  in coll_names[0] and 'upper_forearm_visual' in coll_names[1]: continue
            if 'lower_elbow'    in coll_names[0] and 'upper_elbow_visual'   in coll_names[1]: continue
            if 'lower_shoulder' in coll_names[0] and 'upper_elbow'          in coll_names[1]: continue
            if 'lower_shoulder' in coll_names[0] and 'upper_shoulder'       in coll_names[1]: continue
            if 'upper_elbow'    in coll_names[0] and 'upper_elbow_visual'   in coll_names[1]: continue
            if 'lower_shoulder' in coll_names[0] and 'upper_elbow_visual'   in coll_names[1]: continue
            if 'lower_elbow'    in coll_names[0] and 'upper_forearm'        in coll_names[1]: continue
            if 'gripper_base'   in coll_names[0] and 'hand_accelerometer'   in coll_names[1]: continue
            if 'head_link'      in coll_names[0] and 'sonar_ring'           in coll_names[1]: continue
            if 'head_link'      in coll_names[0] and 'head'                 in coll_names[1]: continue
            if 'head_link'      in coll_names[0] and 'screen'               in coll_names[1]: continue
            if 'head_link'      in coll_names[0] and 'display'              in coll_names[1]: continue
            if 'gripper_base'   in coll_names[0] and 'hand_accelerometer'   in coll_names[1]: continue
            if 'hand'           in coll_names[0] and 'hand_accelerometer'   in coll_names[1]: continue
            if 'hand'           in coll_names[0] and 'wrist'                in coll_names[1]: continue
            if 'gripper_base'   in coll_names[0] and 'hand'                 in coll_names[1]: continue
            if 'display'        in coll_names[0] and 'screen'               in coll_names[1]: continue

            objs_in_collision.add(coll_names)
        return objs_in_collision

    def pairwise_collsion_check(self):
        pass

    def pairwise_distance_check(self):
        pass

    def continous_collision_check(self):
        pass

    def print_collision_result(self, o1_name, o2_name, result):
        print('Collision between {} and {}:'.format(o1_name, o2_name))
        print('-'*30)
        print('Collision?: {}'.format(result.is_collision))
        print('Number of contacts: {}'.format(len(result.contacts)))
        print('')

    def print_distance_result(self, o1_name, o2_name, result):
        print('Distance between {} and {}:'.format(o1_name, o2_name))
        print('-'*30)
        print('Distance: {}'.format(result.min_distance))
        print('Closest Points:')
        print(result.nearest_points[0])
        print(result.nearest_points[1])
        print('')
