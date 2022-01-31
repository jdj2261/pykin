from pykin.utils.kin_utils import convert_string_to_narray

class URDF_Joint:
    """
    Class of parsing joint info described in URDF
    """
    @staticmethod
    def _set_origin(elem_joint, joint_frame):
        """
        Set joint's origin
        
        Args:
            elem_joint (xml.etree.ElementTree.Element): element of link parsed from urdf file
            joint_frame (Frame): joint frame
        """ 
        elem_origin = elem_joint.find('origin')
        if elem_origin is not None:
            joint_frame.joint.offset.pos = convert_string_to_narray(
                elem_origin.attrib.get('xyz'))
            joint_frame.joint.offset.rot = convert_string_to_narray(
                elem_origin.attrib.get('rpy'))
    
    @staticmethod
    def _set_axis(elem_joint, joint_frame):
        """
        Set joint's axis
        
        Args:
            elem_joint (xml.etree.ElementTree.Element): element of link parsed from urdf file
            joint_frame (Frame): joint frame
        """ 
        elem_axis = elem_joint.find('axis')
        if elem_axis is not None:
            joint_frame.joint.axis = convert_string_to_narray(
                elem_axis.attrib.get('xyz'))

    @staticmethod
    def _set_limit(elem_joint, joint_frame):
        """
        Set joint's limit
        
        Args:
            elem_joint (xml.etree.ElementTree.Element): element of link parsed from urdf file
            joint_frame (Frame): joint frame
        """ 
        elem_limit = elem_joint.find('limit')
        if elem_limit is not None:
            if "lower" in elem_limit.attrib:
                joint_frame.joint.limit[0] = float(elem_limit.attrib["lower"])
            if "upper" in elem_limit.attrib:
                joint_frame.joint.limit[1] = float(elem_limit.attrib["upper"])