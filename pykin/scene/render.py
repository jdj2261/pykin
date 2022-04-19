import numpy as np
import trimesh
from abc import abstractclassmethod, ABCMeta

import pykin.utils.plot_utils as plt
from pykin.utils.kin_utils import apply_robot_to_scene

class SceneRender(metaclass=ABCMeta):

    @abstractclassmethod
    def render_all_scene():
        raise NotImplementedError

    @abstractclassmethod
    def render_object_and_gripper():
        raise NotImplementedError
    
    @abstractclassmethod
    def render_object():
        raise NotImplementedError

    @abstractclassmethod
    def render_gripper():
        raise NotImplementedError

    @abstractclassmethod
    def show():
        raise NotImplementedError

class RenderTriMesh(SceneRender):

    def __init__(self):
        self.scene = trimesh.Scene()
        # self.scene.set_camera(
        #     np.array([np.pi/2, 0, np.pi/2]), 5, resolution=(1024, 512))
    
    def render_all_scene(self, objs, robot, geom):
        self.render_object(objs)
        self.render_robot(robot, geom)
    
    def render_object_and_gripper(self, objs, robot):
        self.render_object(objs)
        self.render_gripper(robot)

    def render_object(self, objs):
        pass

    def render_robot(self, robot, geom):
        for link, info in robot.info[geom].items():
            if info[1] == "mesh":
                file_name = robot.mesh_path + robot.links[link].visual.gparam.get('filename')
                mesh = trimesh.load_mesh(file_name)
                color = robot.links[link].visual.gparam.get('color')

                if color is None:
                    color = np.array([0.2, 0, 0])
                else:
                    color = np.array([color for color in color.values()]).flatten()
                
                mesh.visual.face_colors = color
                self.scene.add_geometry(mesh, transform=info[3])

    def render_gripper(self, robot):
        pass

    def show(self):
        self.scene.show()

class RenderPyPlot(SceneRender):

    @staticmethod
    def render_all_scene(ax, objs, robot, geom, alpha, robot_color):
        RenderPyPlot.render_object(ax, objs, alpha)
        RenderPyPlot.render_robot(ax, robot, geom, alpha, robot_color)

    @staticmethod
    def render_object_and_gripper(ax, objs, robot, alpha, robot_color, visible_tcp):
        RenderPyPlot.render_object(ax, objs, alpha)
        RenderPyPlot.render_gripper(ax, robot, alpha, robot_color, visible_tcp)

    @staticmethod
    def render_object(ax, objs, alpha):
        plt.plot_basis(ax)
        for info in objs.values():
            plt.plot_mesh(
                ax=ax, 
                mesh=info.gparam, 
                h_mat=info.h_mat, 
                color=info.color,
                alpha=alpha,
            )

    @staticmethod
    def render_robot(ax, robot, geom, alpha, color):
        plt.plot_basis(ax)
        for link, info in robot.info[geom].items():            
            if info[1] == 'mesh':
                mesh_color = color
                if color is None:
                    mesh_color = robot.links[link].collision.gparam.get('color')
                    mesh_color = np.array([color for color in mesh_color.values()]).flatten()
                if "finger" in link:
                    alpha = 1
                plt.plot_mesh(ax, mesh=info[2], h_mat=info[3], alpha=alpha, color=mesh_color)
            
            if info[1] == 'cylinder':
                cylinder_color = plt.get_color(robot.links[link].visual.gparam)
                length = float(robot.links[link].collision.gparam.get('length'))
                radius = float(robot.links[link].collision.gparam.get('radius'))
                plt.plot_cylinder(ax, length=length, radius=radius, h_mat=info[3], alpha=alpha, color=cylinder_color)

            if info[1] == 'sphere':
                sphere_color = plt.get_color(robot.links[link].visual.gparam)
                radius = float(robot.links[link].collision.gparam.get('radius'))
                pos = info[3][:3,-1]
                plt.plot_sphere(ax, radius=radius, center_point=pos, n_steps=20, alpha=alpha, color=sphere_color)
    
            if info[1]== 'box':
                box_color = plt.get_color(robot.links[link].visual.gparam)
                size = robot.links[link].collision.gparam.get('size')
                plt.plot_box(ax, size, h_mat=info[3], alpha=alpha, color=box_color)

    @staticmethod
    def render_gripper(ax, robot, alpha=0.3, color='b', visible_tcp=True):
        plt.plot_basis(ax)
        for _, info in robot.gripper.info.items():
            if info[1] == 'mesh':
                plt.plot_mesh(ax, mesh=info[2], h_mat=info[3], alpha=alpha, color=color)

        if visible_tcp:
            ax.scatter(
                robot.gripper.info["tcp"][3][0,3], 
                robot.gripper.info["tcp"][3][1,3], 
                robot.gripper.info["tcp"][3][2,3], s=5, c='r')

    @staticmethod
    def show():
        plt.show_figure()