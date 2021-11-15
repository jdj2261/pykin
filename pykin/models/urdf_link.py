from pykin.utils.kin_utils import convert_string_to_narray, LINK_TYPES

class URDF_Link:
    """
    Class of parsing link info described in URDF
    """
    @staticmethod
    def set_visual(elem_link, link_frame):
        """
        Set link visual
        """ 
        for elem_visual in elem_link.findall('visual'):
            URDF_Link.set_visual_origin(elem_visual, link_frame)
            URDF_Link.set_visual_geometry(elem_visual, link_frame)
            URDF_Link.set_visual_color(elem_visual, link_frame)

    @staticmethod
    def set_collision(elem_link, link_frame):
        """
        Set link collision
        """ 
        for elem_collision in elem_link.findall('collision'):
            URDF_Link.set_collision_origin(elem_collision, link_frame)
            URDF_Link.set_collision_geometry(elem_collision, link_frame)
            URDF_Link.set_collision_color(elem_collision, link_frame)
            
    @staticmethod
    def set_visual_origin(elem_visual, frame):
        """
        Set link visual's origin
        """ 
        for elem_origin in elem_visual.findall('origin'):
            frame.link.visual.offset.pos = convert_string_to_narray(elem_origin.attrib.get('xyz'))
            frame.link.visual.offset.rot = convert_string_to_narray(elem_origin.attrib.get('rpy'))

    @staticmethod
    def set_visual_geometry(elem_visual, frame):
        """
        Set link visual's geometry
        """ 

        def _set_link_visual_geom(shapes, frame):
            """
            Set link visual's geometry
            """ 
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

        for elem_geometry in elem_visual.findall('geometry'):
            for shape_type in LINK_TYPES:
                for shapes in elem_geometry.findall(shape_type):
                    _set_link_visual_geom(shapes, frame)

    @staticmethod
    def set_visual_color(elem_visual, frame):
        """
        Set link visual's color
        """ 
        for elem_matrial in elem_visual.findall('material'):
            for elem_color in elem_matrial.findall('color'):
                rgba = convert_string_to_narray(elem_color.attrib.get('rgba'))
                frame.link.visual.gparam['color'] = {elem_matrial.get('name') : rgba}
    
    @staticmethod
    def set_collision_origin(elem_collision, frame):
        """
        Set link collision's origin
        """ 
        for elem_origin in elem_collision.findall('origin'):
            frame.link.collision.offset.pos = convert_string_to_narray(elem_origin.attrib.get('xyz'))
            frame.link.collision.offset.rot = convert_string_to_narray(elem_origin.attrib.get('rpy'))

    @staticmethod
    def set_collision_geometry(elem_collision, frame):
        """
        Set link collision's geometry
        """ 
        
        def _set_link_collision_geom(shapes, frame):
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

        elem_geometry = elem_collision.find('geometry')
        for shape_type in LINK_TYPES:
            for shapes in elem_geometry.findall(shape_type):
                _set_link_collision_geom(shapes, frame)

    @staticmethod
    def set_collision_color(elem_collision, frame):
        """
        Set link visual's color
        """ 
        for elem_matrial in elem_collision.findall('material'):
            for elem_color in elem_matrial.findall('color'):
                rgba = convert_string_to_narray(elem_color.attrib.get('rgba'))
                frame.link.collision.gparam['color'] = {elem_matrial.get('name') : rgba}
    