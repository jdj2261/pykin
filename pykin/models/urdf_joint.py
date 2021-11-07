from pykin.utils.kin_utils import convert_string_to_narray

class URDF_Joint:
    """
    Class of parsing joint info described in URDF
    """
    @staticmethod
    def set_origin(elem_joint, frame):
        """
        Set joint's origin
        """
        elem_origin = elem_joint.find('origin')
        if elem_origin is not None:
            frame.joint.offset.pos = convert_string_to_narray(
                elem_origin.attrib.get('xyz'))
            frame.joint.offset.rot = convert_string_to_narray(
                elem_origin.attrib.get('rpy'))
    
    @staticmethod
    def set_axis(elem_joint, frame):
        """
        Set joint's axis
        """
        elem_axis = elem_joint.find('axis')
        if elem_axis is not None:
            frame.joint.axis = convert_string_to_narray(
                elem_axis.attrib.get('xyz'))

    @staticmethod
    def set_limit(elem_joint, frame):
        """
        Set joint's limit
        """
        elem_limit = elem_joint.find('limit')
        if elem_limit is not None:
            if "lower" in elem_limit.attrib:
                frame.joint.limit[0] = float(elem_limit.attrib["lower"])
            if "upper" in elem_limit.attrib:
                frame.joint.limit[1] = float(elem_limit.attrib["upper"])