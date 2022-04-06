import io, os
from xml.etree import ElementTree as ET
from collections import OrderedDict

from pykin.models.robot_model import RobotModel
from pykin.geometry.frame import Joint, Link, Frame
from pykin.geometry.geometry import Visual, Collision
from pykin.kinematics.transform import Transform, convert_transform
from pykin.models.urdf_joint import URDF_Joint
from pykin.models.urdf_link import URDF_Link
from pykin.utils.kin_utils import JOINT_TYPE_MAP

class URDFModel(RobotModel):
    """
    Initializes a urdf model, as defined by a single corresponding robot URDF

    Args:
        fname (str): path to the urdf file.
    """
    def __init__(self, fname):
        super().__init__()

        if not os.path.isfile(fname):
            raise FileNotFoundError(f'{fname} is not Found..')

        self.tree_xml = ET.parse(fname)
        self.root = self.tree_xml.getroot()
        self.robot_name = self.root.attrib.get('name')

        self._set_links()
        self._set_joints()
        self._set_root()

    def get_urdf(self):
        """
        Reads a string of the urdf file.

        Returns:
            str: xml read in from file
        """
        with io.StringIO() as string:
            string.write(ET.tostring(self.tree_xml.getroot(), encoding="unicode"))
            return string.getvalue()

    def find_frame(self, frame_name):
        """
        Args:
            frame_name (str): frame's name

        Returns:
            Frame: frame with child frames
        """
        if self.root.name == frame_name:
            return self.root
        return self._find_name_recursive(frame_name, self.root, frame_type="frame")

    def find_link(self, link_name):
        """
        Args:
            link_name (str): link's name

        Returns:
            Link: desired robot's link
        """
        if self.root.link.name == link_name:
            return self.root.link
        return self._find_name_recursive(link_name, self.root, frame_type="link")

    def find_joint(self, joint_name):
        """
        Args:
            joint_name (str): joint's name

        Returns:
            Joint: desired robot's joint
        """
        if self.root.joint.name == joint_name:
            return self.root.joint
        return self._find_name_recursive(joint_name, self.root, frame_type="joint")

    def get_all_active_joint_names(self):
        """
        Returns actuated(revolute, prismatic) joint names

        Args:
            desired_frames (list): If is not empty, will get desired actuated joint names

        Returns:
            list: actuated joint names
        """

        joint_names = []
        if self.root is not None:
            joint_names = self._get_all_active_joint_names_recursive(joint_names, self.root)

        for i, joint in enumerate(joint_names):
            if "head" in joint:
                head_joint = joint_names.pop(i)
                joint_names.insert(0, head_joint)

        return joint_names

    @staticmethod
    def _get_all_active_joint_names_recursive(joint_names, frame):
        """
        Return the name of all actuated joint(revolute, prismatic)

        Args:
            joint_names (list): all actuated joint names
            root_frame (Frame): root frame

        Returns:
            list: Append joint if joint's dof is not zero
        """
        if frame.joint.num_dof != 0:
            joint_names.append(frame.joint.name)
        for child in frame.children:
            URDFModel._get_all_active_joint_names_recursive(joint_names, child)
        return joint_names

    def get_revolute_joint_names(self, desired_frames=None):
        """
        Returns revolute joint names

        Args:
            desired_frames (list): If is not empty, will get desired actuated joint names

        Returns:
            list: revolute joint names
        """
        if desired_frames is None:
            joint_names = self._get_revolute_joint_names(self.root)
        else:
            joint_names = self._get_revolute_joint_names(desired_frames)

        for i, joint in enumerate(joint_names):
            if "head" in joint:
                head_joint = joint_names.pop(i)
                joint_names.insert(0, head_joint)

        return joint_names

    def _set_links(self):
        """
        Set all links from urdf file
        """
        for idx, elem_link in enumerate(self.root.findall('link')):
            link_frame = self._get_link_frame(idx, elem_link)
            self._links[link_frame.link.name] = link_frame.link

    def _set_joints(self):
        """
        Set all joints from urdf file
        """
        for idx, elem_joint in enumerate(self.root.findall('joint')):
            joint_frame = self._get_joint_frame(idx, elem_joint)
            self._joints[joint_frame.joint.name] = joint_frame.joint

    def _set_root(self):
        """
        Set root from urdf file
        """
        root_name = next(iter(self._links))
        self._root_link = self._links[root_name]

        root_frame = self._generate_root_frame(root_name)
        self.root = root_frame

    def _generate_root_frame(self, root_name):
        """
        Generates root frame with all child frames

        Args:
            root_name (str): root name

        Returns:
            Frame: root frame with all child frames
        """
        root_frame = Frame(root_name + "_frame")
        root_frame.joint = Joint()
        root_frame.link = Link(root_name)
        root_frame.children = self._generate_children_recursive(self._root_link, self._links, self._joints)
        return root_frame

    def _get_link_frame(self, idx, elem_link):
        """
        Returns link frame from urdf file

        Args:
            idx (int): index of link parsed from urdf file
            elem_link (xml.etree.ElementTree.Element): element of link parsed from urdf file

        Returns:
            Frame: link frame with all child frames
        """
        attrib = elem_link.attrib
        link_name = attrib.get('name', 'link_' + str(idx))
        link_frame = Frame(name=link_name + '_frame',
                           link=Link(
                           name=link_name, 
                           offset=Transform(), 
                           visual=Visual(), 
                           collision=Collision()))

        URDF_Link._set_visual(elem_link, link_frame)
        URDF_Link._set_collision(elem_link, link_frame)
        
        return link_frame

    def _get_joint_frame(self, idx, elem_joint):
        """
        Returns joint frame from urdf file

        Args:
            idx (int): index of joint parsed from urdf file
            elem_joint (xml.etree.ElementTree.Element): element of joint parsed from urdf file

        Returns:
            Frame: joint frame with all child frames
        """
        attrib = elem_joint.attrib
        joint_name = attrib.get('name', 'joint_' + str(idx))
        joint_frame = Frame(name=joint_name + '_frame',
                            joint=Joint(
                            name=joint_name, 
                            offset=Transform(), 
                            dtype=attrib['type'], 
                            limit=[None, None]))

        parent_tag = elem_joint.find('parent')
        joint_frame.joint.parent = parent_tag.attrib['link']

        child_tag = elem_joint.find('child')
        joint_frame.joint.child = child_tag.attrib['link']

        URDF_Joint._set_origin(elem_joint, joint_frame)
        URDF_Joint._set_axis(elem_joint, joint_frame)
        URDF_Joint._set_limit(elem_joint, joint_frame)

        return joint_frame
    
    @staticmethod
    def _generate_children_recursive(root_link: Link, links: OrderedDict, joints: OrderedDict) -> list:
        """
        Generates child frame recursive from current joint

        Args:
            root_link (Link): root link
            links (OrderedDict): element of joint parsed from urdf file
            joints (OrderedDict): element of joint parsed from urdf file

        Returns:
            list: Append list If current joint's parent link is root link
        """
        children = []
        for joint in joints.values():
            if joint.parent == root_link.name:
                child_frame = Frame(joint.child + "_frame")
                child_frame.joint = Joint(joint.name, 
                                        offset=convert_transform(joint.offset), 
                                        dtype=JOINT_TYPE_MAP[joint.dtype], 
                                        axis=joint.axis, 
                                        limit=joint.limit)

                child_link = links[joint.child]
                child_frame.link = Link(child_link.name, 
                                        offset=convert_transform(child_link.offset),
                                        visual=child_link.visual,
                                        collision=child_link.collision)

                child_frame.children = URDFModel._generate_children_recursive(child_frame.link, links, joints)
                children.append(child_frame)

        return children

    @staticmethod
    def _find_name_recursive(name, frame, frame_type):
        """
        Return the name of the frame, link, or joint you want to find.

        Args:
            name (str): name you want to find
            frame (Frame): frame from root until it finds the desired name
            frame_type (str): 3 frame types, frame or link or joint

        Returns:
            3 types: Frame, Link, Joint
        """
        for frame in frame.children:
            if frame_type == "frame" and frame.name == name:
                return frame
            if frame_type == "link" and frame.link.name == name:
                return frame.link
            if frame_type == "joint" and frame.joint.name == name:
                return frame.joint
            ret = URDFModel._find_name_recursive(name, frame, frame_type)

            assert (ret != None), f"Not Found {name}, please check the name again"
            return ret

    def _get_revolute_joint_names(self, frame):
        """
        Return the name of the actuated joint(revolute, prismatic)

        Args:
            root_frame (str): root frame
            desired_frames (Frame): frames from root until it finds the desired name

        Returns:
            list: Append joint if joint's dof is not zero
        """
        if not isinstance(frame, list):
            joint_names = []
            joint_names =  self._get_all_revolute_joint_names_recursive(frame, joint_names)
        else:
            joint_names = self._get_desired_revolute_joint_names(frame)

        return joint_names

    @staticmethod
    def _get_all_revolute_joint_names_recursive(root_frame, joint_names):
        """
        Return the name of all revolute joint

        Args:
            root_frame (Frame): root frame
            joint_names (list): all actuated joint names
            
        Returns:
            list: Append joint if joint's dof is not zero
        """
        if root_frame.joint.dtype == 'revolute':
            joint_names.append(root_frame.joint.name)
        for child in root_frame.children:
            URDFModel._get_all_revolute_joint_names_recursive(child, joint_names)
        return joint_names

    @staticmethod
    def _get_desired_revolute_joint_names(desired_frames):
        """
        Return the name of desired actuated joint(revolute, prismatic)

        Args:
            desired_frames (list): desired actuated joint names

        Returns:
            list: Append joint if joint's dof is not zero
        """
        joint_names = []
        for frame in desired_frames:
            if frame.joint.dtype == 'revolute':
                joint_names.append(frame.joint.name)
        return joint_names

    @property
    def dof(self):
        """
        Returns:
            int: robot's dof
        """
        return sum([joint.num_dof for joint in self.joints.values()])

    @property
    def num_links(self):
        """
        Returns:
            int: number of links
        """
        return len(self.links)

    @property
    def num_joints(self):
        """
        Returns:
            int: number of joints
        """
        return len(self.joints)

    @property
    def num_fixed_joints(self):
        """
        Returns:
            int: number of fixed joints
        """
        return sum([1 for joint in self.joints.values() if joint.num_dof == 0])

    @property
    def num_actuated_joints(self):
        """
        Returns:
            int: number of actuated joints
        """
        return sum([1 for joint in self.joints.values() if joint.num_dof != 0])

    @property
    def num_revolute_joints(self):
        """
        Returns:
            int: number of actuated joints
        """
        return len(self.get_revolute_joint_names())

    @staticmethod
    def generate_desired_frame_recursive(base_frame, eef_name):
        """
        Return frames from base_frame to eef_frame you want to find

        Args:
            base_frame (list): reference frame
            eef_name (str): end effector name

        Returns:
            list: Append frame until child link name is eef name
        """
        for child in base_frame.children:
            if child.link.name == eef_name:
                return [child]
            else:
                frames = URDFModel.generate_desired_frame_recursive(child, eef_name)
                if frames is not None:
                    return [child] + frames