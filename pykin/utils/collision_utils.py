
import trimesh
import numpy as np

def apply_robot_to_collision_manager(c_manager, robot, fk=None):
    if fk is None:
        fk = robot.init_transformations

    for link, transformation in fk.items():
        if robot.links[link].visual.gtype == "mesh":
            mesh_name = robot.links[link].visual.gparam.get('filename')
            file_name = c_manager.mesh_path + mesh_name
            mesh = trimesh.load_mesh(file_name)
            A2B = np.dot(transformation.h_mat, robot.links[link].visual.offset.h_mat)
            c_manager.add_object(name=robot.links[link].name, gtype="mesh", gparam=mesh, transform=A2B)
    return c_manager


def apply_robot_to_scene(mesh_path=None, scene=None, robot=None, fk=None):
    if scene is None:
        scene = trimesh.Scene()

    if fk is None:
        fk = robot.init_transformations

    for link, transformation in fk.items():
        if robot.links[link].visual.gtype == "mesh":
            mesh_name = robot.links[link].visual.gparam.get('filename')
            file_name = mesh_path + mesh_name
            mesh = trimesh.load_mesh(file_name)
            A2B = np.dot(transformation.h_mat, robot.links[link].visual.offset.h_mat)

            color = robot.links[link].visual.gparam.get('color')
            color = np.array([color for color in color.values()]).flatten()
            mesh.visual.face_colors = color
            scene.add_geometry(mesh, transform=A2B)
            # scene = convert_trimesh_scene(scene, file_name, A2B, color)
           
    return scene
