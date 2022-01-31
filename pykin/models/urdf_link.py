from pykin.utils.kin_utils import convert_string_to_narray, LINK_TYPES

class URDF_Link:
    """
    Class of parsing link info described in URDF
    """
    @staticmethod
    def _set_visual(elem_link, link_frame):
        """
        Set link visual

        Args:
            elem_link (xml.etree.ElementTree.Element): element of link parsed from urdf file
            link_frame (Frame): link frame
        """ 
        for elem_visual in elem_link.findall('visual'):
            URDF_Link._set_visual_origin(elem_visual, link_frame)
            URDF_Link._set_visual_geometry(elem_visual, link_frame)
            URDF_Link._set_visual_color(elem_visual, link_frame)

    @staticmethod
    def _set_collision(elem_link, link_frame):
        """
        Set link collision

        Args:
            elem_link (xml.etree.ElementTree.Element): element of link parsed from urdf file
            link_frame (Frame): link frame
        """ 
        for elem_collision in elem_link.findall('collision'):
            URDF_Link._set_collision_origin(elem_collision, link_frame)
            URDF_Link._set_collision_geometry(elem_collision, link_frame)
            URDF_Link._set_collision_color(elem_collision, link_frame)
            
    @staticmethod
    def _set_visual_origin(elem_visual, link_frame):
        """
        Set link visual's origin
        
        Args:
            elem_visual (xml.etree.ElementTree.Element): element of link's visual parsed from urdf file
            link_frame (Frame): link frame
        """ 
        for elem_origin in elem_visual.findall('origin'):
            link_frame.link.visual.offset.pos = convert_string_to_narray(elem_origin.attrib.get('xyz'))
            link_frame.link.visual.offset.rot = convert_string_to_narray(elem_origin.attrib.get('rpy'))

    @staticmethod
    def _set_visual_geometry(elem_visual, link_frame):
        """
        Set link visual's geometry

        Args:
            elem_visual (xml.etree.ElementTree.Element): element of link's visual parsed from urdf file
            link_frame (Frame): link frame
        """ 

        def _set_link_visual_geom(shapes, link_frame):
            if shapes.tag == "box":
                link_frame.link.visual.gtype = shapes.tag
                link_frame.link.visual.gparam = {"size" : convert_string_to_narray(shapes.attrib.get('size', None))}
            elif shapes.tag == "cylinder":
                link_frame.link.visual.gtype = shapes.tag
                link_frame.link.visual.gparam = {"length" : shapes.attrib.get('length', 0),
                                            "radius" : shapes.attrib.get('radius', 0)}
            elif shapes.tag == "sphere":
                link_frame.link.visual.gtype = shapes.tag
                link_frame.link.visual.gparam = {"radius" : shapes.attrib.get('radius', 0)}
            elif shapes.tag == "mesh":
                link_frame.link.visual.gtype = shapes.tag
                link_frame.link.visual.gparam = {"filename" : shapes.attrib.get('filename', None)}
            else:
                link_frame.link.visual.gtype = None
                link_frame.link.visual.gparam = None

        for elem_geometry in elem_visual.findall('geometry'):
            for shape_type in LINK_TYPES:
                for shapes in elem_geometry.findall(shape_type):
                    _set_link_visual_geom(shapes, link_frame)

    @staticmethod
    def _set_visual_color(elem_visual, link_frame):
        """
        Set link visual's color
        
        Args:
            elem_visual (xml.etree.ElementTree.Element): element of link's visual parsed from urdf file
            link_frame (Frame): link frame
        """ 
        for elem_matrial in elem_visual.findall('material'):
            for elem_color in elem_matrial.findall('color'):
                rgba = convert_string_to_narray(elem_color.attrib.get('rgba'))
                link_frame.link.visual.gparam['color'] = {elem_matrial.get('name') : rgba}
    
    @staticmethod
    def _set_collision_origin(elem_collision, link_frame):
        """
        Set link collision's origin

        Args:
            elem_collision (xml.etree.ElementTree.Element): element of link's collision parsed from urdf file
            link_frame (Frame): link frame
        """  
        for elem_origin in elem_collision.findall('origin'):
            link_frame.link.collision.offset.pos = convert_string_to_narray(elem_origin.attrib.get('xyz'))
            link_frame.link.collision.offset.rot = convert_string_to_narray(elem_origin.attrib.get('rpy'))

    @staticmethod
    def _set_collision_geometry(elem_collision, link_frame):
        """
        Set link collision's geometry
        
        Args:
            elem_collision (xml.etree.ElementTree.Element): element of link's collision parsed from urdf file
            link_frame (Frame): link frame
        """     
        def _set_link_collision_geom(shapes, link_frame):
            if shapes.tag == "box":
                link_frame.link.collision.gtype = shapes.tag
                link_frame.link.collision.gparam = {"size" : convert_string_to_narray(shapes.attrib.get('size', None))}
            elif shapes.tag == "cylinder":
                link_frame.link.collision.gtype = shapes.tag
                link_frame.link.collision.gparam = {"length" : shapes.attrib.get('length', 0),
                                            "radius" : shapes.attrib.get('radius', 0)}
            elif shapes.tag == "sphere":
                link_frame.link.collision.gtype = shapes.tag
                link_frame.link.collision.gparam = {"radius" : shapes.attrib.get('radius', 0)}
            elif shapes.tag == "mesh":
                link_frame.link.collision.gtype = shapes.tag
                link_frame.link.collision.gparam = {"filename" : shapes.attrib.get('filename', None)}
            else:
                link_frame.link.collision.gtype = None
                link_frame.link.collision.gparam = None

        elem_geometry = elem_collision.find('geometry')
        for shape_type in LINK_TYPES:
            for shapes in elem_geometry.findall(shape_type):
                _set_link_collision_geom(shapes, link_frame)

    @staticmethod
    def _set_collision_color(elem_collision, link_frame):
        """
        Set link visual's color

        Args:
            elem_collision (xml.etree.ElementTree.Element): element of link's collision parsed from urdf file
            link_frame (Frame): link frame
        """
        for elem_matrial in elem_collision.findall('material'):
            for elem_color in elem_matrial.findall('color'):
                rgba = convert_string_to_narray(elem_color.attrib.get('rgba'))
                link_frame.link.collision.gparam['color'] = {elem_matrial.get('name') : rgba}