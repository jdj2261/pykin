import io
from xml.etree import ElementTree as ET
from collections import OrderedDict

from pykin.models.robot_model import RobotModel
from pykin.geometry.frame import Joint, Link, Frame
from pykin.geometry.geometry import Visual, Collision
from pykin.kinematics.transform import Transform
from pykin.utils.kin_utils import *


class URDFModel(RobotModel):
    def __init__(self, fname):
        super().__init__(fname)
        self.tree_xml = ET.parse(fname)
        self.root = self.tree_xml.getroot()
        self.robot_name = self.root.attrib.get('name')

        self._make_tree()

    def get_urdf(self):
        with io.StringIO() as string:
            string.write(ET.tostring(self.tree_xml.getroot(), encoding="unicode"))
            return string.getvalue()

    def _make_tree(self):
        self._make_links()
        self._make_joints()

        root_name = next(iter(self._links))
        self.root = self._links[root_name]

        root_frame = self._make_root_frame(root_name)
        self.root = root_frame

    def _make_links(self):
        for idx, link_tag in enumerate(self.root.findall('link')):
            link_frame = self._parse_link(link_tag, idx=idx)
            self._links[link_frame.link.name] = link_frame.link

    def _make_joints(self):
        for idx, joint_tag in enumerate(self.root.findall('joint')):
            joint_frame = self._parse_joint(joint_tag, idx=idx)
            self._joints[joint_frame.joint.name] = joint_frame.joint

    def _make_root_frame(self, root_name):
        root_frame = Frame(root_name + "_frame")
        root_frame.joint = Joint()
        root_frame.link = Link(root_name, 
                               offset=convert_transform(root_frame.link.offset), 
                               visual=root_frame.link.visual, 
                               collision=root_frame.link.collision)

        root_frame.children = self._build_chain_recursive(self.root, self._links, self._joints)
        return root_frame

    def _parse_link(self, link_tag, idx):
        attrib = link_tag.attrib
        link_name = attrib.get('name', 'link_' + str(idx))
        frame = Frame(link_name + '_frame',
                      link=Link(link_name, offset=Transform(), visual=Visual(), collision=Collision()))

        self._parse_visual(link_tag, frame)
        self._parse_collision(link_tag, frame)
        return frame

    def _parse_joint(self, joint_tag, idx):
        attrib = joint_tag.attrib
        joint_name = attrib.get('name', 'joint_' + str(idx))

        frame = Frame(joint_name + '_frame',
                      joint=Joint(name=joint_name, offset=Transform(), dtype=attrib['type'], limit=[None, None]))

        parent_tag = joint_tag.find('parent')
        frame.joint.parent = parent_tag.attrib['link']

        child_tag = joint_tag.find('child')
        frame.joint.child = child_tag.attrib['link']

        # origin
        origin_tag = joint_tag.find('origin')
        if origin_tag is not None:
            frame.joint.offset.pos = convert_string_to_narray(origin_tag.attrib.get('xyz'))
            frame.joint.offset.rot = convert_string_to_narray(origin_tag.attrib.get('rpy'))

        # axis
        axis_tag = joint_tag.find('axis')
        self._parse_axis(axis_tag, frame)

        # limit
        limit_tag = joint_tag.find('limit')
        self._parse_limit(limit_tag, frame)

        return frame

    def _parse_visual(self, link_tag, frame):   
        for visual_tag in link_tag.findall('visual'):
            self._parse_visual_origin(visual_tag, frame)
            self._parse_visual_geometry(visual_tag, frame)
            self._parse_visual_color(visual_tag, frame)

    def _parse_visual_origin(self, input_tag, frame):
        for origin_tag in input_tag.findall('origin'):
            frame.link.visual.offset.pos = convert_string_to_narray(origin_tag.attrib.get('xyz'))
            frame.link.visual.offset.rot = convert_string_to_narray(origin_tag.attrib.get('rpy'))

    def _parse_visual_geometry(self, input_tag, frame):
        for geometry_tag in input_tag.findall('geometry'):
            for shape_type in LINK_TYPES:
                for shapes in geometry_tag.findall(shape_type):
                    self._convert_visual(shapes, frame)

    def _convert_visual(self, shapes, frame):
        if shapes.tag == "box":
            frame.link.visual.gtype = shapes.tag
            frame.link.visual.gparam = {"size" : convert_string_to_narray(shapes.attrib.get('size', None))}
        elif shapes.tag == "cylinder":
            frame.link.visual.gtype = shapes.tag
            frame.link.visual.gparam = {"length" : shapes.attrib.get('length', 0),
                                        "radius" : shapes.attrib.get('radius', 0)}
        elif shapes.tag == "sphere":
            frame.link.visual.gtype = shapes.tag
            frame.link.visual.gparam = {"radius" : shapes.attrib.get('radius', 0)}
        elif shapes.tag == "mesh":
            frame.link.visual.gtype = shapes.tag
            frame.link.visual.gparam = {"filename" : shapes.attrib.get('filename', None)}
        else:
            frame.link.visual.gtype = None
            frame.link.visual.gparam = None

    def _parse_visual_color(self, input_tag, frame):
        for material_tag in input_tag.findall('material'):
            for colors in material_tag.findall('color'):
                frame.link.visual.gparam['color'] = {material_tag.get('name'):convert_string_to_narray(colors.attrib.get('rgba'))}

    def _parse_collision(self, link_tag, frame):
        for collision_tag in link_tag.findall('collision'):
            self._parse_collision_origin(collision_tag, frame)
            self._parse_collision_geometry(collision_tag, frame)

    def _parse_collision_origin(self, input_tag, frame):
        for origin_tag in input_tag.findall('origin'):
            frame.link.collision.offset.pos = convert_string_to_narray(origin_tag.attrib.get('xyz'))
            frame.link.collision.offset.rot = convert_string_to_narray(origin_tag.attrib.get('rpy'))

    def _parse_collision_geometry(self, input_tag, frame):
        for geometry_tag in input_tag.findall('geometry'):
            for shape_type in LINK_TYPES:
                for shapes in geometry_tag.findall(shape_type):
                    self._convert_collision(shapes, frame)

    def _convert_collision(self, shapes, frame):
        if shapes.tag == "box":
            frame.link.collision.gtype = shapes.tag
            frame.link.collision.gparam = {"size" : convert_string_to_narray(shapes.attrib.get('size', None))}
        elif shapes.tag == "cylinder":
            frame.link.collision.gtype = shapes.tag
            frame.link.collision.gparam = {"length" : shapes.attrib.get('length', 0),
                                        "radius" : shapes.attrib.get('radius', 0)}
        elif shapes.tag == "sphere":
            frame.link.collision.gtype = shapes.tag
            frame.link.collision.gparam = {"radius" : shapes.attrib.get('radius', 0)}
        elif shapes.tag == "mesh":
            frame.link.collision.gtype = shapes.tag
            frame.link.collision.gparam = {"filename" : shapes.attrib.get('filename', None)}
        else:
            frame.link.collision.gtype = None
            frame.link.collision.gparam = None

    def _parse_axis(self, axis_tag, frame):
        if axis_tag is not None:
            frame.joint.axis = convert_string_to_narray(axis_tag.attrib.get('xyz'))
    
    def _parse_limit(self, limit_tag, frame):
        if limit_tag is not None:
            if "lower" in limit_tag.attrib:
                frame.joint.limit[0] = float(limit_tag.attrib["lower"])
            if "upper" in limit_tag.attrib:
                frame.joint.limit[1] = float(limit_tag.attrib["upper"])

    def _build_chain_recursive(self, root: Link, links: OrderedDict, joints: OrderedDict) -> list:
        children = []
        for joint in joints.values():

            if joint.parent == root.name:
                child_frame = Frame(joint.child + "_frame")
                child_frame.joint = Joint(joint.name, 
                                        offset=convert_transform(joint.offset), 
                                        dtype=JOINT_TYPE_MAP[joint.dtype], 
                                        axis=joint.axis, 
                                        limit=joint.limit)

                chil_link = links[joint.child]
                child_frame.link = Link(chil_link.name, 
                                        offset=convert_transform(chil_link.offset),
                                        visual=chil_link.visual,
                                        collision=chil_link.collision)
                child_frame.children = self._build_chain_recursive(child_frame.link, links, joints)
                children.append(child_frame)

        return children

    def find_frame(self, frame_name):
        if self.root.name == frame_name:
            return self.root
        return self._find_recursive(frame_name, self.root, frame_type="frame")

    def find_link(self, link_name):
        if self.root.link.name == link_name:
            return self.root.link
        return self._find_recursive(link_name, self.root, frame_type="link")

    def find_joint(self, joint_name):
        if self.root.joint.name == joint_name:
            return self.root.joint
        return self._find_recursive(joint_name, self.root, frame_type="joint")

    @staticmethod
    def _find_recursive(name, frame, frame_type):
        for child in frame.children:
            if frame_type == "frame" and child.name == name:
                return child
            if frame_type == "link" and child.link.name == name:
                return child.link
            if frame_type == "joint" and child.joint.name == name:
                return child.joint
            ret = URDFModel._find_recursive(name, child, frame_type)
            if not ret is None:
                return ret

    def _get_actuated_joint_names(self, desired_frames=None):
        if desired_frames is None:
            joint_names = self._get_joint_names(root_frame=self.root)
        else:
            joint_names = self._get_joint_names(desired_frames=desired_frames)
        return joint_names

    def _get_joint_names(self, root_frame=None, desired_frames=None):
        if root_frame is not None:
            joint_names = []
            joint_names =  self._get_all_joint_names_recursive(joint_names, root_frame)

        if desired_frames is not None:
            joint_names = self._get_desired_joint_names(desired_frames)

        return joint_names

    def _get_all_joint_names_recursive(self, joint_names, root_frame):
        if root_frame.joint.num_dof != 0:
            joint_names.append(root_frame.joint.name)
        for child in root_frame.children:
            self._get_all_joint_names_recursive(joint_names, child)
        return joint_names

    def _get_desired_joint_names(self, desired_frames):
        joint_names = []
        for f in desired_frames:
            if f.joint.num_dof != 0:
                joint_names.append(f.joint.name)
        return joint_names

    @property
    def dof(self):
        return sum([joint.num_dof for joint in self.joints.values()])

    @property
    def num_links(self):
        return len(self.links)

    @property
    def num_joints(self):
        return len(self.joints)

    @property
    def num_fixed_joints(self):
        return sum([1 for joint in self.joints.values() if joint.num_dof == 0])

    @property
    def num_actuated_joints(self):
        return sum([1 for joint in self.joints.values() if joint.num_dof != 0])

    @staticmethod
    def generate_desired_frame_recursive(base_frame, eef_name):
        for child in base_frame.children:
            if child.link.name == eef_name:
                return [child]
            else:
                frames = URDFModel.generate_desired_frame_recursive(child, eef_name)
                if frames is not None:
                    return [child] + frames
