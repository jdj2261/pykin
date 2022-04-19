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
    
    def render_all_scene(self, objs, robot, geom):
        self.render_object(objs)
        self.render_robot(robot, geom)
    
    def render_object_and_gripper(self, objs, robot):
        self.render_object(objs)
        self.render_gripper(robot)

    def render_object(self, objs):
        pass

    def render_robot(self, robot, geom="collision"):
        self.scene = apply_robot_to_scene(trimesh_scene=self.scene, robot=robot, geom=geom)
        self.scene.set_camera(np.array([np.pi/2, 0, np.pi]), 5, resolution=(1024, 512))

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
        plt.plot_geom(ax, robot, geom, alpha, color)

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