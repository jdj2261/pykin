
import trimesh
import numpy as np

def apply_robot_to_collision_manager(c_manager, robot, fk=None, geom="visual"):
    if fk is None:
        fk = robot.init_fk

    for link, transformation in fk.items():
        if geom == "visual":
            if robot.links[link].visual.gtype == "mesh":
                mesh_name = robot.links[link].visual.gparam.get('filename')
                file_name = c_manager.mesh_path + mesh_name
                mesh = trimesh.load_mesh(file_name)
                h_mat = np.dot(transformation.h_mat, robot.links[link].visual.offset.h_mat)
                if c_manager._filter_names:
                    c_manager.set_transform(name=robot.links[link].name, h_mat=h_mat)
                else:
                    c_manager.add_object(
                        name=robot.links[link].name, 
                        gtype="mesh",
                        gparam=mesh,
                        transform=h_mat)

        if geom == "collision":
            if robot.links[link].collision.gtype == "mesh":
                mesh_name = robot.links[link].collision.gparam.get('filename')
                file_name = c_manager.mesh_path + mesh_name
                mesh = trimesh.load_mesh(file_name)
                h_mat = np.dot(transformation.h_mat, robot.links[link].collision.offset.h_mat)
                if c_manager._filter_names:
                    c_manager.set_transform(name=robot.links[link].name, h_mat=h_mat)
                else:
                    c_manager.add_object(
                        name=robot.links[link].name, 
                        gtype="mesh",
                        gparam=mesh,
                        transform=h_mat)
    return c_manager


def apply_robot_to_scene(mesh_path=None, scene=None, robot=None, fk=None, geom="visual"):
    if scene is None:
        scene = trimesh.Scene()

    if fk is None:
        fk = robot.init_fk

    for link, transformation in fk.items():
        if geom == "visual":
            if robot.links[link].visual.gtype == "mesh":
                mesh_name = robot.links[link].visual.gparam.get('filename')
                file_name = mesh_path + mesh_name
                mesh = trimesh.load_mesh(file_name)
                h_mat = np.dot(transformation.h_mat, robot.links[link].visual.offset.h_mat)
                color = robot.links[link].visual.gparam.get('color')

                if color is None:
                    color = np.array([0.2, 0, 0])
                else:
                    color = np.array([color for color in color.values()]).flatten()

                mesh.visual.face_colors = color
                scene.add_geometry(mesh, transform=h_mat)  

        if geom == "collision":
            if robot.links[link].collision.gtype == "mesh":
                mesh_name = robot.links[link].collision.gparam.get('filename')
                file_name = mesh_path + mesh_name
                mesh = trimesh.load_mesh(file_name)
                h_mat = np.dot(transformation.h_mat, robot.links[link].collision.offset.h_mat)
                color = robot.links[link].collision.gparam.get('color')

                if color is None:
                    color = np.array([0.2, 0, 0])
                else:
                    color = np.array([color for color in color.values()]).flatten()
                
                mesh.visual.face_colors = color
                scene.add_geometry(mesh, transform=h_mat)      
    scene.set_camera(np.array([np.pi/2, 0, np.pi/2]), 5, resolution=(1024, 512))
    return scene


def get_mesh_param(link_type):
    file_name = str(link_type.gparam.get('filename'))
    color = link_type.gparam.get('color')
    color = np.array([color for color in color.values()]).flatten()
    return (file_name, color)

def get_cylinder_param(link_type):
    radius = float(link_type.gparam.get('radius'))
    length = float(link_type.gparam.get('length'))
    return (radius, length)

def get_spehre_param(link_type):
    radius = float(link_type.gparam.get('radius'))
    return radius

def get_box_param(link_type):
    size = list(link_type.gparam.get('size'))
    return size

def get_robot_collision_geom(link):
    """
    Get robot collision geometry from link

    Args:
        link (Link): robot's link

    Returns:
        name (str): geom's name
        gtype: geom's type
        gparam: geom's param
    """
    name = None
    gtype = None
    gparam = None

    if link.collision.gtype == "cylinder":
        name = link.name
        gtype = link.collision.gtype
        gparam = get_cylinder_param(link.collision)
    elif link.collision.gtype == "sphere":
        name = link.name
        gtype = link.collision.gtype
        gparam = get_spehre_param(link.collision)
    elif link.collision.gtype == "box":
        name = link.name
        gtype = link.collision.gtype
        gparam = get_box_param(link.collision)

    return name, gtype, gparam

def get_robot_visual_geom(link):
    """
    Get robot visual geometry from link

    Args:
        link (Link): robot's link

    Returns:
        name (str): geom's name
        gtype: geom's type
        gparam: geom's param
    """

    name = None
    gtype = None
    gparam = None

    if link.visual.gtype == "mesh":
        name = link.name
        gtype = link.visual.gtype
        gparam = get_cylinder_param(link.visual)
    if link.visual.gtype == "cylinder":
        name = link.name
        gtype = link.visual.gtype
        gparam = get_cylinder_param(link.visual)
    elif link.visual.gtype == "sphere":
        name = link.name
        gtype = link.visual.gtype
        gparam = get_spehre_param(link.visual)
    elif link.visual.gtype == "box":
        name = link.name
        gtype = link.visual.gtype
        gparam = get_box_param(link.visual)

    return name, gtype, gparam




